"""Daily drill-down data extraction and diagnostic plot generation.

Supports the Advanced Diagnostics tab by extracting 24-hour slices
from the full 8,760-hour computation results and generating Plotly
figures for temperature, load, COP, energy, and cost.

All functions return plotly.graph_objects.Figure instances.
No Streamlit dependency -- this module is pure Plotly.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BTU_PER_KWH = 3412  # duplicated from model.py to keep this module independent

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# Consistent color palette across all plots
COLOR_HP = "#1f77b4"        # Blue — heat pump electricity
COLOR_BACKUP = "#ff7f0e"    # Orange — backup resistance
COLOR_GAS = "#d62728"       # Red — gas furnace
COLOR_HEATING = "#e377c2"   # Pink — heating load
COLOR_COOLING = "#17becf"   # Cyan — cooling load
COLOR_COP = "#2ca02c"       # Green — COP line
COLOR_SETPOINT = "#7f7f7f"  # Gray — setpoint reference lines
COLOR_LOCKOUT = "#d62728"   # Red — lockout zones


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


class DayData(NamedTuple):
    """24-hour slice of all computation arrays for a single day."""

    hours: npt.NDArray[np.int32]
    T_out: npt.NDArray[np.float64]
    heat_load: npt.NDArray[np.float64]
    cool_load: npt.NDArray[np.float64]
    cop: npt.NDArray[np.float64]
    lockout_mask: npt.NDArray[np.bool_]
    capacity: npt.NDArray[np.float64]
    kwh_heat: npt.NDArray[np.float64]
    kwh_cool: npt.NDArray[np.float64]
    kwh_total: npt.NDArray[np.float64]
    load_hp: npt.NDArray[np.float64]
    load_backup: npt.NDArray[np.float64]
    therms: npt.NDArray[np.float64]
    rates: npt.NDArray[np.float64]
    cost_elec: npt.NDArray[np.float64]
    cost_gas: npt.NDArray[np.float64]


def extract_day_data(
    month: int,
    day: int,
    *,
    weather_months: npt.NDArray[np.integer],
    weather_days: npt.NDArray[np.integer],
    weather_hours: npt.NDArray[np.integer],
    T_out: npt.NDArray[np.float64],
    heat_load: npt.NDArray[np.float64],
    cool_load: npt.NDArray[np.float64],
    cop: npt.NDArray[np.float64],
    lockout_mask: npt.NDArray[np.bool_],
    capacity: npt.NDArray[np.float64],
    kwh_heat: npt.NDArray[np.float64],
    kwh_cool: npt.NDArray[np.float64],
    kwh_total: npt.NDArray[np.float64],
    load_hp: npt.NDArray[np.float64],
    load_backup: npt.NDArray[np.float64],
    therms: npt.NDArray[np.float64],
    rates: npt.NDArray[np.float64],
    cost_elec: npt.NDArray[np.float64],
    cost_gas: npt.NDArray[np.float64],
) -> DayData:
    """Extract 24-hour slice from 8760-hour arrays for a given month/day.

    Parameters
    ----------
    month : int
        Month number (1-12).
    day : int
        Day of month.
    All other parameters are keyword-only (8760,) arrays.

    Returns
    -------
    DayData with all 24-hour sliced arrays.

    Raises
    ------
    ValueError
        If the specified month/day does not match exactly 24 hours.
    """
    mask = (weather_months == month) & (weather_days == day)
    n = int(mask.sum())
    if n != 24:
        raise ValueError(
            f"Month {month}, day {day}: expected 24 hours, found {n}"
        )
    return DayData(
        hours=weather_hours[mask],
        T_out=T_out[mask],
        heat_load=heat_load[mask],
        cool_load=cool_load[mask],
        cop=cop[mask],
        lockout_mask=lockout_mask[mask],
        capacity=capacity[mask],
        kwh_heat=kwh_heat[mask],
        kwh_cool=kwh_cool[mask],
        kwh_total=kwh_total[mask],
        load_hp=load_hp[mask],
        load_backup=load_backup[mask],
        therms=therms[mask],
        rates=rates[mask],
        cost_elec=cost_elec[mask],
        cost_gas=cost_gas[mask],
    )


# ---------------------------------------------------------------------------
# Daily plot functions
# ---------------------------------------------------------------------------

_HOUR_TICK = dict(
    tickmode="array",
    tickvals=list(range(0, 24, 2)),
    ticktext=[f"{h}:00" for h in range(0, 24, 2)],
)
_LAYOUT_DEFAULTS = dict(
    margin=dict(l=50, r=20, t=36, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def plot_daily_temperature(
    day_data: DayData,
    T_set_heat: float,
    T_set_cool: float,
) -> go.Figure:
    """Outdoor temperature with heating/cooling setpoint reference lines."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=day_data.hours, y=day_data.T_out,
        mode="lines+markers", name="Outdoor Temp",
        line=dict(color=COLOR_HP, width=2),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 23], y=[T_set_heat, T_set_heat],
        mode="lines", name=f"Heat Setpoint ({T_set_heat}\u00b0F)",
        line=dict(color=COLOR_HEATING, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 23], y=[T_set_cool, T_set_cool],
        mode="lines", name=f"Cool Setpoint ({T_set_cool}\u00b0F)",
        line=dict(color=COLOR_COOLING, dash="dash"),
    ))
    fig.update_layout(
        title="Outdoor Temperature",
        xaxis=dict(title="Hour", **_HOUR_TICK),
        yaxis_title="Temperature (\u00b0F)",
        height=300,
        **_LAYOUT_DEFAULTS,
    )
    return fig


def plot_daily_load(day_data: DayData) -> go.Figure:
    """Heating and cooling load as filled area plots (BTU/h)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=day_data.hours, y=day_data.heat_load,
        mode="lines", name="Heating Load",
        line=dict(color=COLOR_HEATING, width=2),
        fill="tozeroy", fillcolor="rgba(227, 119, 194, 0.3)",
    ))
    fig.add_trace(go.Scatter(
        x=day_data.hours, y=day_data.cool_load,
        mode="lines", name="Cooling Load",
        line=dict(color=COLOR_COOLING, width=2),
        fill="tozeroy", fillcolor="rgba(23, 190, 207, 0.3)",
    ))
    fig.update_layout(
        title="Thermal Load",
        xaxis=dict(title="Hour", **_HOUR_TICK),
        yaxis_title="Load (BTU/h)",
        height=300,
        **_LAYOUT_DEFAULTS,
    )
    return fig


def plot_daily_cop(
    day_data: DayData,
    lockout_temp: float,
) -> go.Figure:
    """COP at each hour with lockout zones highlighted in red."""
    fig = go.Figure()

    # COP line
    fig.add_trace(go.Scatter(
        x=day_data.hours, y=day_data.cop,
        mode="lines+markers", name="COP",
        line=dict(color=COLOR_COP, width=2),
        marker=dict(size=5),
    ))

    # Highlight lockout hours with red vertical bands
    if np.any(day_data.lockout_mask):
        lockout_hours = day_data.hours[day_data.lockout_mask]
        for h in lockout_hours:
            fig.add_vrect(
                x0=h - 0.4, x1=h + 0.4,
                fillcolor=COLOR_LOCKOUT, opacity=0.15,
                line_width=0,
            )
        fig.add_trace(go.Scatter(
            x=lockout_hours,
            y=day_data.cop[day_data.lockout_mask],
            mode="markers", name="Lockout (COP=1.0)",
            marker=dict(color=COLOR_LOCKOUT, size=8, symbol="x"),
        ))

    # COP = 1.0 reference line (resistance baseline)
    fig.add_hline(
        y=1.0, line_dash="dot", line_color=COLOR_SETPOINT,
        annotation_text="COP = 1.0 (resistance)",
        annotation_position="top left",
    )

    y_max = max(float(np.max(day_data.cop)) * 1.2, 2.0)
    fig.update_layout(
        title=f"Heat Pump COP (lockout below {lockout_temp}\u00b0F)",
        xaxis=dict(title="Hour", **_HOUR_TICK),
        yaxis=dict(title="COP", range=[0, y_max]),
        height=300,
        **_LAYOUT_DEFAULTS,
    )
    return fig


def plot_daily_energy(day_data: DayData) -> go.Figure:
    """HP kWh + backup kWh stacked bars; gas therms on secondary y-axis."""
    kwh_backup = day_data.load_backup / BTU_PER_KWH
    kwh_hp_only = day_data.kwh_heat - kwh_backup

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=day_data.hours, y=kwh_hp_only,
        name="HP Heating (kWh)", marker_color=COLOR_HP,
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=day_data.hours, y=kwh_backup,
        name="Backup (kWh)", marker_color=COLOR_BACKUP,
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=day_data.hours, y=day_data.kwh_cool,
        name="Cooling (kWh)", marker_color=COLOR_COOLING,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=day_data.hours, y=day_data.therms,
        mode="lines+markers", name="Gas (therms)",
        line=dict(color=COLOR_GAS, width=2),
        marker=dict(size=5),
    ), secondary_y=True)

    fig.update_layout(
        title="Energy Consumption",
        barmode="stack",
        xaxis=dict(title="Hour", **_HOUR_TICK),
        height=350,
        **_LAYOUT_DEFAULTS,
    )
    fig.update_yaxes(title_text="Electricity (kWh)", secondary_y=False)
    fig.update_yaxes(title_text="Gas (therms)", secondary_y=True)
    return fig


def plot_daily_cost(
    day_data: DayData,
    rate_schedule_name: str,
) -> go.Figure:
    """HP vs gas cost with TOU rate tier as background color bands."""
    fig = go.Figure()

    # TOU rate background bands — group contiguous hours at the same rate
    unique_rates = sorted(set(day_data.rates.tolist()))
    rate_colors = _rate_tier_colors(unique_rates)
    i = 0
    shown_rates: set[float] = set()
    while i < len(day_data.rates):
        rate = day_data.rates[i]
        j = i
        while j < len(day_data.rates) and day_data.rates[j] == rate:
            j += 1
        show_legend = rate not in shown_rates
        shown_rates.add(rate)
        fig.add_vrect(
            x0=i - 0.5, x1=j - 0.5,
            fillcolor=rate_colors[rate], opacity=0.15,
            line_width=0,
            annotation_text=f"${rate:.2f}" if show_legend else None,
            annotation_position="top left" if show_legend else None,
        )
        i = j

    fig.add_trace(go.Scatter(
        x=day_data.hours, y=day_data.cost_elec,
        mode="lines+markers", name="HP Electricity Cost",
        line=dict(color=COLOR_HP, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=day_data.hours, y=day_data.cost_gas,
        mode="lines+markers", name="Gas Furnace Cost",
        line=dict(color=COLOR_GAS, width=2),
    ))

    fig.update_layout(
        title=f"Hourly Cost ({rate_schedule_name})",
        xaxis=dict(title="Hour", **_HOUR_TICK),
        yaxis_title="Cost ($)",
        height=350,
        **_LAYOUT_DEFAULTS,
    )
    return fig


def _rate_tier_colors(unique_rates: list[float]) -> dict[float, str]:
    """Assign colors to rate tiers — lower rates get cooler colors."""
    palette = ["#2ca02c", "#bcbd22", "#ff7f0e", "#d62728"]
    n = len(unique_rates)
    if n == 1:
        return {unique_rates[0]: palette[0]}
    return {
        rate: palette[min(i, len(palette) - 1)]
        for i, rate in enumerate(unique_rates)
    }


# ---------------------------------------------------------------------------
# Full-year diagnostic plots (not day-specific)
# ---------------------------------------------------------------------------


def plot_cop_curve(
    cop_curve: list[tuple[float, float]],
    lockout_temp: float,
    T_out_all: npt.NDArray[np.float64],
) -> go.Figure:
    """COP vs outdoor temp with TMY temperature histogram overlay."""
    # Dense temp grid for smooth COP curve
    T_dense = np.linspace(-20, 110, 500)
    temps = np.array([pt[0] for pt in cop_curve], dtype=np.float64)
    cops = np.array([pt[1] for pt in cop_curve], dtype=np.float64)
    cop_dense = np.interp(T_dense, temps, cops)
    cop_dense = np.where(T_dense < lockout_temp, 1.0, cop_dense)
    cop_dense = np.maximum(cop_dense, 1.0)

    # TMY temperature histogram
    bin_edges = np.arange(-25, 115, 5)
    hist_counts, _ = np.histogram(T_out_all, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Histogram on secondary axis (draw first so COP line is on top)
    fig.add_trace(go.Bar(
        x=bin_centers, y=hist_counts,
        name="TMY Hours",
        marker_color="rgba(180, 180, 180, 0.5)",
        width=4.5,
    ), secondary_y=True)

    # COP curve on primary axis
    fig.add_trace(go.Scatter(
        x=T_dense, y=cop_dense,
        mode="lines", name="COP",
        line=dict(color=COLOR_COP, width=3),
    ), secondary_y=False)

    # Reference COP points
    ref_temps = [pt[0] for pt in cop_curve]
    ref_cops = [pt[1] for pt in cop_curve]
    fig.add_trace(go.Scatter(
        x=ref_temps, y=ref_cops,
        mode="markers", name="Reference Points",
        marker=dict(color=COLOR_COP, size=10, symbol="circle"),
    ), secondary_y=False)

    # Lockout line
    fig.add_vline(
        x=lockout_temp, line_dash="dash", line_color=COLOR_LOCKOUT,
        annotation_text=f"Lockout {lockout_temp}\u00b0F",
        annotation_position="top right",
    )

    fig.update_layout(
        title="COP Curve with TMY Temperature Distribution",
        xaxis_title="Outdoor Temperature (\u00b0F)",
        height=400,
        **_LAYOUT_DEFAULTS,
    )
    fig.update_yaxes(title_text="COP", secondary_y=False)
    fig.update_yaxes(title_text="Hours per Year", secondary_y=True)
    return fig


def plot_load_duration(
    heat_load: npt.NDArray[np.float64],
    capacity: npt.NDArray[np.float64],
) -> go.Figure:
    """Sorted heating load (descending) with HP capacity overlay."""
    sort_idx = np.argsort(-heat_load)
    sorted_load = heat_load[sort_idx]
    sorted_cap = capacity[sort_idx]

    # Find last non-zero load hour for cleaner display
    nonzero = np.nonzero(sorted_load > 0)[0]
    n_display = int(nonzero[-1] + 1) if len(nonzero) > 0 else len(sorted_load)
    x_range = np.arange(n_display)

    backup_hours = int(np.sum(heat_load > capacity))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_range, y=sorted_load[:n_display],
        mode="lines", name="Heating Load",
        line=dict(color=COLOR_HEATING, width=2),
        fill="tozeroy", fillcolor="rgba(227, 119, 194, 0.3)",
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=sorted_cap[:n_display],
        mode="lines", name="HP Capacity",
        line=dict(color=COLOR_HP, width=2, dash="dash"),
    ))

    fig.add_annotation(
        x=0.5, y=0.95, xref="paper", yref="paper",
        text=f"{backup_hours:,} hours require backup resistance",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor=COLOR_BACKUP, borderwidth=1, borderpad=4,
    )

    fig.update_layout(
        title="Load Duration Curve",
        xaxis_title="Hours (sorted by load)",
        yaxis_title="BTU/h",
        height=400,
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Hourly data table builder
# ---------------------------------------------------------------------------


def build_hourly_table(day_data: DayData) -> dict[str, list]:
    """Build a dict suitable for pd.DataFrame from DayData.

    Values are rounded for display. Suitable for st.dataframe and CSV export.
    """
    kwh_backup = day_data.load_backup / BTU_PER_KWH
    kwh_hp_only = day_data.kwh_heat - kwh_backup

    return {
        "Hour": day_data.hours.tolist(),
        "T_out (\u00b0F)": [round(v, 1) for v in day_data.T_out],
        "Heat Load (BTU/h)": [round(v, 0) for v in day_data.heat_load],
        "Cool Load (BTU/h)": [round(v, 0) for v in day_data.cool_load],
        "COP": [round(v, 2) for v in day_data.cop],
        "Lockout": day_data.lockout_mask.tolist(),
        "HP Capacity (BTU/h)": [round(v, 0) for v in day_data.capacity],
        "kWh HP": [round(v, 3) for v in kwh_hp_only],
        "kWh Backup": [round(v, 3) for v in kwh_backup],
        "kWh Cool": [round(v, 3) for v in day_data.kwh_cool],
        "kWh Total": [round(v, 3) for v in day_data.kwh_total],
        "Therms Gas": [round(v, 4) for v in day_data.therms],
        "Rate ($/kWh)": [round(v, 4) for v in day_data.rates],
        "Cost Elec ($)": [round(v, 4) for v in day_data.cost_elec],
        "Cost Gas ($)": [round(v, 4) for v in day_data.cost_gas],
    }
