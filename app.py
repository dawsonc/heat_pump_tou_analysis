"""Streamlit UI for the Heat Pump TOU Electricity Cost Calculator.

Provides two tabs:
- Results: summary metrics, monthly cost comparison, electricity breakdown, cost heatmap.
- Advanced Diagnostics: daily drill-down with hour-by-hour plots.

Sidebar controls allow selection of rate schedule, building parameters,
heat pump preset, and gas furnace comparison settings.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from diagnostics import (
    MONTH_NAMES,
    build_hourly_table,
    extract_day_data,
    plot_cop_curve,
    plot_daily_cop,
    plot_daily_cost,
    plot_daily_energy,
    plot_daily_load,
    plot_daily_temperature,
    plot_load_duration,
)
from model import (
    BTU_PER_KWH,
    aggregate_annual,
    aggregate_monthly,
    compute_capacity,
    compute_cooling_load,
    compute_cop,
    compute_electric_cost,
    compute_gas_cost,
    compute_gas_energy,
    compute_heating_load,
    compute_hp_energy,
)
from presets import (
    BUILDING_PRESETS,
    BUILDING_SIZES,
    DEFAULT_BUILDING_INSULATION,
    DEFAULT_BUILDING_SIZE,
    DEFAULT_GAS_FURNACE_PRESET,
    DEFAULT_GAS_MONTHLY_CHARGE,
    DEFAULT_GAS_SUMMER_RATE_PER_THERM,
    DEFAULT_GAS_WINTER_RATE_PER_THERM,
    DEFAULT_HP_PRESET,
    DEFAULT_T_SET_COOL_F,
    DEFAULT_T_SET_HEAT_F,
    GAS_FURNACE_PRESETS,
    HP_PRESETS,
    INSULATION_LEVELS,
)
from rates import (
    RATE_PRESETS,
    build_flat_schedule,
    build_seasonal_flat_schedule,
    build_tou_schedule,
    extract_on_peak_hours,
    extract_tou_prices,
    gas_rate_lookup,
    schedule_type,
    tou_lookup,
)
from weather import load_weather

st.set_page_config(page_title="Heat Pump TOU Calculator", layout="wide")


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------


@st.cache_data
def get_weather():
    """Load and cache TMY3 weather data (called once, cached across reruns)."""
    df = load_weather()
    return (
        df["T_drybulb_F"].to_numpy(),
        df["month"].to_numpy(),
        df["day"].to_numpy(),
        df["hour_of_day"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_rate_schedule_sidebar() -> tuple[str, dict]:
    """Render rate schedule controls and return (rate_name, schedule)."""
    st.sidebar.subheader("Rate Schedule")
    rate_names = list(RATE_PRESETS.keys())
    rate_name = st.sidebar.selectbox(
        "Rate preset",
        options=rate_names,
        index=rate_names.index("Eversource R-1HP"),
        help="Select a preset to populate the fields below. Edit any value.",
    )
    preset = RATE_PRESETS[rate_name]

    # Customer charge (editable)
    customer_charge = st.sidebar.number_input(
        "Customer charge ($/month)",
        min_value=0.00,
        max_value=50.00,
        value=preset["customer_charge"],
        step=1.00,
        format="%.2f",
        key=f"elec_cust_{rate_name}",
    )

    stype = schedule_type(preset)

    if stype == "flat":
        # Single-tier per season — show summer/winter rate inputs
        s_price = list(preset["summer"]["tiers"].values())[0]["price"]
        w_price = list(preset["winter"]["tiers"].values())[0]["price"]
        is_truly_flat = abs(s_price - w_price) < 0.001

        if is_truly_flat:
            rate_val = st.sidebar.number_input(
                "Rate ($/kWh)",
                min_value=0.00,
                max_value=2.00,
                value=s_price,
                step=0.01,
                format="%.3f",
                key=f"flat_rate_{rate_name}",
            )
            schedule = build_flat_schedule(rate_val, customer_charge, rate_name)
        else:
            col_s, col_w = st.sidebar.columns(2)
            with col_s:
                summer_rate = st.number_input(
                    "Summer ($/kWh)",
                    min_value=0.00,
                    max_value=2.00,
                    value=s_price,
                    step=0.01,
                    format="%.3f",
                    key=f"summer_rate_{rate_name}",
                )
            with col_w:
                winter_rate = st.number_input(
                    "Winter ($/kWh)",
                    min_value=0.00,
                    max_value=2.00,
                    value=w_price,
                    step=0.01,
                    format="%.3f",
                    key=f"winter_rate_{rate_name}",
                )
            schedule = build_seasonal_flat_schedule(
                summer_rate, winter_rate, customer_charge, rate_name,
            )
    else:
        # TOU: show on-peak hours and per-season rates
        on_peak_hours = extract_on_peak_hours(preset)
        peak_start_default = on_peak_hours[0] if on_peak_hours else 16
        peak_end_default = on_peak_hours[-1] if on_peak_hours else 20

        col_start, col_end = st.sidebar.columns(2)
        with col_start:
            peak_start = st.number_input(
                "On-peak start (hour)",
                min_value=0,
                max_value=23,
                value=peak_start_default,
                step=1,
                key=f"peak_start_{rate_name}",
            )
        with col_end:
            peak_end = st.number_input(
                "On-peak end (hour)",
                min_value=0,
                max_value=23,
                value=peak_end_default,
                step=1,
                key=f"peak_end_{rate_name}",
            )

        s_on, s_off = extract_tou_prices(preset, "summer")
        w_on, w_off = extract_tou_prices(preset, "winter")

        st.sidebar.markdown("**Summer rates ($/kWh)**")
        col_son, col_soff = st.sidebar.columns(2)
        with col_son:
            summer_on = st.number_input(
                "On-peak",
                min_value=0.00,
                max_value=2.00,
                value=s_on,
                step=0.01,
                format="%.3f",
                key=f"s_on_{rate_name}",
            )
        with col_soff:
            summer_off = st.number_input(
                "Off-peak",
                min_value=0.00,
                max_value=2.00,
                value=s_off,
                step=0.01,
                format="%.3f",
                key=f"s_off_{rate_name}",
            )

        st.sidebar.markdown("**Winter rates ($/kWh)**")
        col_won, col_woff = st.sidebar.columns(2)
        with col_won:
            winter_on = st.number_input(
                "On-peak",
                min_value=0.00,
                max_value=2.00,
                value=w_on,
                step=0.01,
                format="%.3f",
                key=f"w_on_{rate_name}",
            )
        with col_woff:
            winter_off = st.number_input(
                "Off-peak",
                min_value=0.00,
                max_value=2.00,
                value=w_off,
                step=0.01,
                format="%.3f",
                key=f"w_off_{rate_name}",
            )

        try:
            schedule = build_tou_schedule(
                peak_start,
                peak_end,
                summer_on,
                summer_off,
                winter_on,
                winter_off,
                customer_charge,
                rate_name,
            )
        except ValueError as exc:
            st.sidebar.error(str(exc))
            schedule = preset  # fall back to preset on validation error

    return rate_name, schedule


def render_sidebar() -> dict:
    """Render all sidebar controls and return selected parameters."""
    st.sidebar.header("Settings")

    # --- Rate Schedule ---
    rate_name, schedule = _render_rate_schedule_sidebar()

    # --- Building ---
    st.sidebar.subheader("Building")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        building_size = st.selectbox(
            "Size",
            options=BUILDING_SIZES,
            index=BUILDING_SIZES.index(DEFAULT_BUILDING_SIZE),
        )
    with col2:
        insulation = st.selectbox(
            "Insulation",
            options=INSULATION_LEVELS,
            index=INSULATION_LEVELS.index(DEFAULT_BUILDING_INSULATION),
        )
    ua = BUILDING_PRESETS[insulation][building_size]
    st.sidebar.metric(
        "UA (BTU/h per \u00b0F)",
        f"{ua:,}",
        help=(
            "Overall heat-loss coefficient (UA). Measures the rate of "
            "heat loss from the building in BTU/h for each \u00b0F of "
            "temperature difference between indoors and outdoors. "
            "Higher UA = leakier building."
        ),
    )

    # --- Heat Pump ---
    st.sidebar.subheader("Heat Pump")
    hp_names = list(HP_PRESETS.keys())
    hp_name = st.sidebar.selectbox(
        "Heat pump preset",
        options=hp_names,
        index=hp_names.index(DEFAULT_HP_PRESET),
    )
    hp = HP_PRESETS[hp_name]
    st.sidebar.caption(hp["notes"])

    has_backup = st.sidebar.checkbox(
        "Electric resistance backup",
        value=hp["has_backup_heat"],
        key=f"backup_{hp_name}",
        help=(
            "When checked, electric resistance backup (COP=1.0) covers "
            "any heating load the heat pump cannot serve. When unchecked, "
            "that load becomes unmet demand."
        ),
    )

    # --- Gas Furnace ---
    st.sidebar.subheader("Gas Furnace (comparison)")
    col_gw, col_gs = st.sidebar.columns(2)
    with col_gw:
        gas_rate_winter = st.number_input(
            "Winter gas ($/therm)",
            min_value=0.00,
            max_value=10.00,
            value=DEFAULT_GAS_WINTER_RATE_PER_THERM,
            step=0.10,
            format="%.2f",
            help="Gas rate for Nov-Apr (heating season).",
        )
    with col_gs:
        gas_rate_summer = st.number_input(
            "Summer gas ($/therm)",
            min_value=0.00,
            max_value=10.00,
            value=DEFAULT_GAS_SUMMER_RATE_PER_THERM,
            step=0.10,
            format="%.2f",
            help="Gas rate for May-Oct.",
        )
    gas_customer_charge = st.sidebar.number_input(
        "Gas customer charge ($/month)",
        min_value=0.00,
        max_value=50.00,
        value=DEFAULT_GAS_MONTHLY_CHARGE,
        step=1.00,
        format="%.2f",
        help=(
            "Monthly fixed charge for gas service. "
            "Added to the annual gas cost (12 months)."
        ),
    )
    furnace_names = list(GAS_FURNACE_PRESETS.keys())
    furnace_name = st.sidebar.selectbox(
        "Furnace efficiency (AFUE)",
        options=furnace_names,
        index=furnace_names.index(DEFAULT_GAS_FURNACE_PRESET),
    )
    afue = GAS_FURNACE_PRESETS[furnace_name]["afue"]

    return {
        "rate_name": rate_name,
        "schedule": schedule,
        "building_size": building_size,
        "insulation": insulation,
        "ua": ua,
        "hp_name": hp_name,
        "hp": hp,
        "has_backup": has_backup,
        "gas_rate_winter": gas_rate_winter,
        "gas_rate_summer": gas_rate_summer,
        "gas_customer_charge": gas_customer_charge,
        "furnace_name": furnace_name,
        "afue": afue,
    }


# ---------------------------------------------------------------------------
# Computation pipeline
# ---------------------------------------------------------------------------


def run_pipeline(T_out, months, days, hours_of_day, params):
    """Run the full 8760-hour computation pipeline.

    Returns a dict with all hourly, monthly, and annual results.
    """
    hp = params["hp"]
    ua = params["ua"]
    schedule = params["schedule"]

    # Shared: building load
    heat_load = compute_heating_load(T_out, ua, DEFAULT_T_SET_HEAT_F)
    cool_load = compute_cooling_load(T_out, ua, DEFAULT_T_SET_COOL_F)

    # Path A: heat pump
    cop, lockout_mask = compute_cop(
        T_out, hp["cop_curve"], hp["lockout_temp_f"]
    )
    capacity = compute_capacity(
        T_out, hp["capacity_curve"], hp["rated_capacity_btu_h"]
    )
    hp_energy = compute_hp_energy(
        heat_load, cool_load, cop, hp["cop_cooling"], capacity, lockout_mask,
        has_backup=params["has_backup"],
    )

    # Rates
    rates = tou_lookup(months, hours_of_day, schedule)

    # Costs — total HP electricity
    cost_elec_total = compute_electric_cost(hp_energy.kwh_total, rates)
    # Costs — heating-only HP electricity (for comparison with gas)
    cost_elec_heat = compute_electric_cost(hp_energy.kwh_heat, rates)
    # Costs — cooling-only HP electricity
    cost_elec_cool = cost_elec_total - cost_elec_heat

    # Path B: gas furnace
    therms = compute_gas_energy(heat_load, params["afue"])
    gas_rates = gas_rate_lookup(
        months,
        params["gas_rate_summer"],
        params["gas_rate_winter"],
    )
    cost_gas = compute_gas_cost(therms, gas_rates)

    # Aggregation
    customer_charge = schedule["customer_charge"]
    backup_kwh = hp_energy.load_backup / BTU_PER_KWH

    monthly_cost_elec = (
        aggregate_monthly(cost_elec_total, months) + customer_charge
    )
    monthly_cost_heat_elec = aggregate_monthly(cost_elec_heat, months)
    monthly_cost_cool_elec = aggregate_monthly(cost_elec_cool, months)
    gas_customer_charge = params.get("gas_customer_charge", 0.0)
    monthly_cost_gas = aggregate_monthly(cost_gas, months) + gas_customer_charge
    monthly_kwh_heat = aggregate_monthly(hp_energy.kwh_heat, months)
    monthly_kwh_cool = aggregate_monthly(hp_energy.kwh_cool, months)
    monthly_backup_kwh = aggregate_monthly(backup_kwh, months)

    annual_cost_elec = (
        aggregate_annual(cost_elec_total) + 12 * customer_charge
    )
    annual_cost_heat_elec = aggregate_annual(cost_elec_heat)
    annual_cost_gas = aggregate_annual(cost_gas) + 12 * gas_customer_charge
    annual_kwh = aggregate_annual(hp_energy.kwh_total)
    annual_backup_kwh = aggregate_annual(backup_kwh)
    annual_heat_kwh = aggregate_annual(hp_energy.kwh_heat)

    return {
        # Hourly arrays (for Tab 2 and heatmap)
        "T_out": T_out,
        "heat_load": heat_load,
        "cool_load": cool_load,
        "cop": cop,
        "lockout_mask": lockout_mask,
        "capacity": capacity,
        "hp_energy": hp_energy,
        "therms": therms,
        "rates": rates,
        "cost_elec_total": cost_elec_total,
        "cost_elec_heat": cost_elec_heat,
        "cost_gas": cost_gas,
        # Monthly (12,) arrays
        "monthly_cost_elec": monthly_cost_elec,
        "monthly_cost_heat_elec": monthly_cost_heat_elec,
        "monthly_cost_cool_elec": monthly_cost_cool_elec,
        "monthly_cost_gas": monthly_cost_gas,
        "monthly_kwh_heat": monthly_kwh_heat,
        "monthly_kwh_cool": monthly_kwh_cool,
        "monthly_backup_kwh": monthly_backup_kwh,
        # Annual scalars
        "annual_cost_elec": annual_cost_elec,
        "annual_cost_heat_elec": annual_cost_heat_elec,
        "annual_cost_gas": annual_cost_gas,
        "annual_kwh": annual_kwh,
        "annual_backup_kwh": annual_backup_kwh,
        "annual_heat_kwh": annual_heat_kwh,
    }


# ---------------------------------------------------------------------------
# Tab 1: Results
# ---------------------------------------------------------------------------


def _add_month_boundaries(fig: go.Figure) -> None:
    """Add month boundary lines and labels to the heatmap x-axis."""
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cumulative = 0
    for i, days in enumerate(days_per_month):
        mid = cumulative + days / 2
        fig.add_annotation(
            x=mid, y=-1.5, text=MONTH_NAMES[i],
            showarrow=False, font=dict(size=10),
            xref="x", yref="y",
        )
        if i > 0:
            fig.add_vline(
                x=cumulative + 0.5,
                line=dict(color="white", width=0.5),
            )
        cumulative += days


def render_tab_results(results, params):
    """Render the Results tab with summary cards and charts."""

    # --- Summary Metric Cards ---
    st.subheader("Annual Summary")

    annual_hp_cost = results["annual_cost_elec"]
    annual_gas_cost = results["annual_cost_gas"]
    annual_heat_elec = results["annual_cost_heat_elec"]
    savings_dollars = annual_gas_cost - annual_heat_elec
    savings_pct = (
        (savings_dollars / annual_gas_cost * 100)
        if annual_gas_cost > 0 else 0.0
    )
    backup_share = (
        (results["annual_backup_kwh"] / results["annual_heat_kwh"] * 100)
        if results["annual_heat_kwh"] > 0 else 0.0
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Annual HP Cost", f"${annual_hp_cost:,.0f}")
    with c2:
        st.metric("Annual Gas Heating Cost", f"${annual_gas_cost:,.0f}")
    with c3:
        st.metric(
            "Heating Savings (HP vs Gas)",
            f"${savings_dollars:,.0f}",
            delta=f"{savings_pct:+.1f}%",
            delta_color="normal" if savings_dollars >= 0 else "inverse",
        )
    with c4:
        st.metric("Total kWh (HP)", f"{results['annual_kwh']:,.0f}")
    with c5:
        backup_label = (
            "Backup Resistance Share"
            if params.get("has_backup", True)
            else "Unmet Demand Share"
        )
        st.metric(backup_label, f"{backup_share:.1f}%")

    # --- Hero Chart: Monthly Heating Cost Comparison ---
    st.subheader("Monthly Heating Cost: Heat Pump vs Gas Furnace")

    fig_hero = go.Figure()
    fig_hero.add_trace(go.Bar(
        name="Heat Pump (electricity)",
        x=MONTH_NAMES,
        y=results["monthly_cost_heat_elec"],
        marker_color="#1f77b4",
        text=[f"${v:.0f}" for v in results["monthly_cost_heat_elec"]],
        textposition="outside",
    ))
    fig_hero.add_trace(go.Bar(
        name="Gas Furnace",
        x=MONTH_NAMES,
        y=results["monthly_cost_gas"],
        marker_color="#d62728",
        text=[f"${v:.0f}" for v in results["monthly_cost_gas"]],
        textposition="outside",
    ))
    fig_hero.update_layout(
        barmode="group",
        yaxis_title="Heating Cost ($)",
        xaxis_title="Month",
        height=450,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_hero, use_container_width=True)

    # --- Monthly Total Electricity Cost (Heating + Cooling) ---
    st.subheader("Monthly Total Electricity Cost (Heating + Cooling)")

    monthly_heat_cost = results["monthly_cost_heat_elec"]
    monthly_cool_cost = results["monthly_cost_cool_elec"]

    fig_total_cost = go.Figure()
    fig_total_cost.add_trace(go.Bar(
        name="Heating",
        x=MONTH_NAMES,
        y=monthly_heat_cost,
        marker_color="#1f77b4",
    ))
    fig_total_cost.add_trace(go.Bar(
        name="Cooling",
        x=MONTH_NAMES,
        y=monthly_cool_cost,
        marker_color="#17becf",
    ))
    # Add total labels on top of stacked bars
    monthly_total_cost = monthly_heat_cost + monthly_cool_cost
    fig_total_cost.add_trace(go.Scatter(
        x=MONTH_NAMES,
        y=monthly_total_cost,
        text=[f"${v:.0f}" for v in monthly_total_cost],
        mode="text",
        textposition="top center",
        showlegend=False,
    ))
    fig_total_cost.update_layout(
        barmode="stack",
        yaxis_title="Electricity Cost ($)",
        xaxis_title="Month",
        height=450,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_total_cost, use_container_width=True)

    # --- Monthly Electricity Breakdown ---
    st.subheader("Monthly Electricity Consumption Breakdown")

    monthly_hp_only_kwh = (
        results["monthly_kwh_heat"] - results["monthly_backup_kwh"]
    )

    fig_breakdown = go.Figure()
    fig_breakdown.add_trace(go.Bar(
        name="HP Heating (kWh)",
        x=MONTH_NAMES,
        y=monthly_hp_only_kwh,
        marker_color="#1f77b4",
    ))
    backup_trace_name = (
        "Backup Resistance (kWh)"
        if params.get("has_backup", True)
        else "Unmet Demand (kWh-equiv)"
    )
    fig_breakdown.add_trace(go.Bar(
        name=backup_trace_name,
        x=MONTH_NAMES,
        y=results["monthly_backup_kwh"],
        marker_color="#ff7f0e",
    ))
    fig_breakdown.add_trace(go.Bar(
        name="Cooling (kWh)",
        x=MONTH_NAMES,
        y=results["monthly_kwh_cool"],
        marker_color="#17becf",
    ))
    fig_breakdown.update_layout(
        barmode="stack",
        yaxis_title="Electricity (kWh)",
        xaxis_title="Month",
        height=400,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)

    # --- Hourly Cost Heatmap ---
    st.subheader("Hourly Electricity Cost Heatmap")

    cost_arr = results["cost_elec_total"]
    assert cost_arr.shape[0] == 8760, "Expected 8760 hourly cost values"
    cost_matrix = cost_arr.reshape(365, 24).T  # (24 hours, 365 days)

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=cost_matrix,
        x=list(range(1, 366)),
        y=list(range(24)),
        colorscale="YlOrRd",
        colorbar=dict(title="$/hour"),
        hovertemplate=(
            "Day %{x}, Hour %{y}<br>Cost: $%{z:.3f}<extra></extra>"
        ),
    ))
    fig_heatmap.update_layout(
        xaxis_title="Day of Year",
        yaxis_title="Hour of Day",
        yaxis=dict(dtick=2),
        height=400,
        margin=dict(t=20, b=40),
    )
    _add_month_boundaries(fig_heatmap)
    st.plotly_chart(fig_heatmap, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Advanced Diagnostics
# ---------------------------------------------------------------------------


def render_tab_diagnostics(results, params, months, days, hours_of_day):
    """Render the Advanced Diagnostics tab."""
    hp = params["hp"]

    # --- Date Picker ---
    st.subheader("Daily Drill-Down")

    unique_days = sorted(set(zip(months.tolist(), days.tolist())))

    col1, col2 = st.columns(2)
    with col1:
        selected_month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda m: MONTH_NAMES[m - 1],
            index=0,
        )
    with col2:
        valid_days = sorted(d for m, d in unique_days if m == selected_month)
        selected_day = st.selectbox(
            "Day",
            options=valid_days,
            index=min(14, len(valid_days) - 1),
        )

    # --- Extract day data ---
    day_data = extract_day_data(
        selected_month,
        selected_day,
        weather_months=months,
        weather_days=days,
        weather_hours=hours_of_day,
        T_out=results["T_out"],
        heat_load=results["heat_load"],
        cool_load=results["cool_load"],
        cop=results["cop"],
        lockout_mask=results["lockout_mask"],
        capacity=results["capacity"],
        kwh_heat=results["hp_energy"].kwh_heat,
        kwh_cool=results["hp_energy"].kwh_cool,
        kwh_total=results["hp_energy"].kwh_total,
        load_hp=results["hp_energy"].load_hp,
        load_backup=results["hp_energy"].load_backup,
        therms=results["therms"],
        rates=results["rates"],
        cost_elec=results["cost_elec_total"],
        cost_gas=results["cost_gas"],
    )

    # --- Daily Plots ---
    st.markdown(
        f"**{MONTH_NAMES[selected_month - 1]} {selected_day}**"
    )

    st.plotly_chart(
        plot_daily_temperature(day_data, DEFAULT_T_SET_HEAT_F, DEFAULT_T_SET_COOL_F),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_daily_load(day_data),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_daily_cop(day_data, hp["lockout_temp_f"]),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_daily_energy(day_data),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_daily_cost(day_data, params["rate_name"]),
        use_container_width=True,
    )

    # --- Hourly Data Table ---
    st.subheader("Hourly Data")
    table_data = build_hourly_table(day_data)
    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    csv = df_table.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=(
            f"hourly_data_{MONTH_NAMES[selected_month - 1]}"
            f"_{selected_day}.csv"
        ),
        mime="text/csv",
    )

    # --- Full-Year Diagnostic Plots ---
    st.divider()
    st.subheader("System Performance Curves")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            plot_cop_curve(
                hp["cop_curve"],
                hp["lockout_temp_f"],
                results["T_out"],
            ),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            plot_load_duration(results["heat_load"], results["capacity"]),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    st.title("Heat Pump TOU Electricity Cost Calculator")

    T_out, months, days, hours_of_day = get_weather()
    params = render_sidebar()
    results = run_pipeline(T_out, months, days, hours_of_day, params)

    tab1, tab2 = st.tabs(["Results", "Advanced Diagnostics"])

    with tab1:
        render_tab_results(results, params)

    with tab2:
        render_tab_diagnostics(results, params, months, days, hours_of_day)


main()
