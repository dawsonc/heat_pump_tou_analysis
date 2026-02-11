"""Tests for cost calculation and aggregation functions.

Covers:
- Electric cost: kwh * rate (element-wise)
- Monthly aggregation: sum by month into (12,) array
- Annual aggregation: scalar sum
- Peak demand: max hourly kW per month
- Customer charge integration
- End-to-end pipeline on Boston TMY data
"""

from __future__ import annotations

import numpy as np
import pytest

from model import (
    BTU_PER_KWH,
    aggregate_annual,
    aggregate_monthly,
    compute_capacity,
    compute_cop,
    compute_cooling_load,
    compute_electric_cost,
    compute_gas_cost,
    compute_gas_energy,
    compute_heating_load,
    compute_hp_energy,
    compute_peak_demand_monthly,
)
from presets import (
    BUILDING_PRESETS,
    DEFAULT_BUILDING_INSULATION,
    DEFAULT_BUILDING_SIZE,
    DEFAULT_GAS_FURNACE_PRESET,
    DEFAULT_GAS_SUMMER_RATE_PER_THERM,
    DEFAULT_GAS_WINTER_RATE_PER_THERM,
    DEFAULT_HP_PRESET,
    DEFAULT_T_SET_COOL_F,
    DEFAULT_T_SET_HEAT_F,
    GAS_FURNACE_PRESETS,
    HP_PRESETS,
)
from rates import EVERSOURCE_R1HP, FLAT_RATE, gas_rate_lookup, tou_lookup


# ---------------------------------------------------------------------------
# TestComputeElectricCost
# ---------------------------------------------------------------------------


class TestComputeElectricCost:
    """Tests for compute_electric_cost."""

    def test_basic_multiplication(self) -> None:
        """kwh * rate produces correct cost."""
        kwh = np.array([2.0, 5.0])
        rates = np.array([0.30, 0.20])
        cost = compute_electric_cost(kwh, rates)
        np.testing.assert_array_almost_equal(cost, [0.60, 1.00])

    def test_shape_mismatch_raises(self) -> None:
        """Different-length arrays raise ValueError."""
        kwh = np.ones(10)
        rates = np.ones(20)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_electric_cost(kwh, rates)

    def test_tou_rates_applied_hourly(self) -> None:
        """Different rates at different hours produce different costs."""
        kwh = np.array([1.0, 1.0, 1.0])
        rates = np.array([0.20, 0.30, 0.40])
        cost = compute_electric_cost(kwh, rates)
        np.testing.assert_array_almost_equal(cost, [0.20, 0.30, 0.40])

    def test_heating_only_cost(self) -> None:
        """Using kwh_heat (not kwh_total) gives heating-only cost."""
        kwh_heat = np.array([3.0, 0.0])
        kwh_cool = np.array([0.0, 1.0])
        rates = np.array([0.25, 0.25])
        cost_heat = compute_electric_cost(kwh_heat, rates)
        cost_cool = compute_electric_cost(kwh_cool, rates)
        np.testing.assert_array_almost_equal(cost_heat, [0.75, 0.0])
        np.testing.assert_array_almost_equal(cost_cool, [0.0, 0.25])


# ---------------------------------------------------------------------------
# TestAggregateMonthly
# ---------------------------------------------------------------------------


class TestAggregateMonthly:
    """Tests for aggregate_monthly."""

    def test_uniform_values(self, months_8760) -> None:
        """All-ones array gives per-month hour counts."""
        values = np.ones(8760)
        monthly = aggregate_monthly(values, months_8760)
        # Hours per month for standard non-leap year
        expected_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
        np.testing.assert_array_equal(monthly, expected_hours)

    def test_known_monthly_totals(self, months_8760) -> None:
        """January values=1.0, July values=2.0, rest zero."""
        values = np.zeros(8760)
        values[months_8760 == 1] = 1.0
        values[months_8760 == 7] = 2.0
        monthly = aggregate_monthly(values, months_8760)
        assert monthly[0] == pytest.approx(744.0)  # Jan: 744 * 1.0
        assert monthly[6] == pytest.approx(744.0 * 2.0)  # Jul: 744 * 2.0
        # All other months should be zero
        for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]:
            assert monthly[i] == 0.0


# ---------------------------------------------------------------------------
# TestAggregateAnnual
# ---------------------------------------------------------------------------


class TestAggregateAnnual:
    """Tests for aggregate_annual."""

    def test_basic_sum(self) -> None:
        """Known array sums correctly."""
        values = np.array([1.0, 2.0, 3.0])
        assert aggregate_annual(values) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# TestComputePeakDemandMonthly
# ---------------------------------------------------------------------------


class TestComputePeakDemandMonthly:
    """Tests for compute_peak_demand_monthly."""

    def test_uniform_values(self, months_8760) -> None:
        """All-ones => peak is 1.0 for every month."""
        kwh = np.ones(8760)
        peak = compute_peak_demand_monthly(kwh, months_8760)
        np.testing.assert_array_equal(peak, 1.0)

    def test_known_spike_in_january(self, months_8760) -> None:
        """January has a spike of 10.0, rest are 1.0."""
        kwh = np.ones(8760)
        # Set a single hour in January to 10.0
        kwh[0] = 10.0
        peak = compute_peak_demand_monthly(kwh, months_8760)
        assert peak[0] == 10.0  # January peak
        # All other months should be 1.0
        for i in range(1, 12):
            assert peak[i] == 1.0


# ---------------------------------------------------------------------------
# TestCostWithCustomerCharge
# ---------------------------------------------------------------------------


class TestCostWithCustomerCharge:
    """Tests for customer charge integration pattern."""

    def test_monthly_with_customer_charge(self, months_8760) -> None:
        """Monthly total includes customer charge added per month."""
        rng = np.random.default_rng(7)
        values = rng.random(8760) * 0.50  # random hourly costs
        monthly_variable = aggregate_monthly(values, months_8760)
        customer_charge = FLAT_RATE["customer_charge"]
        monthly_total = monthly_variable + customer_charge
        # Each month's total should be variable + charge
        for i in range(12):
            assert monthly_total[i] == pytest.approx(
                monthly_variable[i] + customer_charge
            )

    def test_annual_with_customer_charge(self, months_8760) -> None:
        """Annual total includes 12 * customer charge."""
        values = np.ones(8760) * 0.10  # $0.10/hour
        annual_variable = aggregate_annual(values)
        customer_charge = FLAT_RATE["customer_charge"]
        annual_total = annual_variable + 12 * customer_charge
        expected = 8760 * 0.10 + 12 * customer_charge
        assert annual_total == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestEndToEndCostIntegration
# ---------------------------------------------------------------------------


class TestEndToEndCostIntegration:
    """End-to-end cost integration tests using Boston TMY data."""

    @pytest.fixture
    def pipeline_results(self, boston_weather_df, months_8760, hours_of_day_8760):
        """Run full HP + gas pipeline, return dict of all results."""
        T_out = boston_weather_df["T_drybulb_F"].values
        hp = HP_PRESETS[DEFAULT_HP_PRESET]
        UA = BUILDING_PRESETS[DEFAULT_BUILDING_INSULATION][DEFAULT_BUILDING_SIZE]
        afue = GAS_FURNACE_PRESETS[DEFAULT_GAS_FURNACE_PRESET]["afue"]

        # Shared: building load
        heat_load = compute_heating_load(T_out, UA, DEFAULT_T_SET_HEAT_F)
        cool_load = compute_cooling_load(T_out, UA, DEFAULT_T_SET_COOL_F)

        # Path A: heat pump
        cop, lockout_mask = compute_cop(T_out, hp["cop_curve"], hp["lockout_temp_f"])
        capacity = compute_capacity(T_out, hp["capacity_curve"], hp["rated_capacity_btu_h"])
        hp_energy = compute_hp_energy(heat_load, cool_load, cop, hp["cop_cooling"], capacity, lockout_mask)

        # Rates
        flat_rates = tou_lookup(months_8760, hours_of_day_8760, FLAT_RATE)
        ever_rates = tou_lookup(months_8760, hours_of_day_8760, EVERSOURCE_R1HP)

        # Costs
        cost_elec_flat = compute_electric_cost(hp_energy.kwh_total, flat_rates)
        cost_elec_ever = compute_electric_cost(hp_energy.kwh_total, ever_rates)
        cost_heat_flat = compute_electric_cost(hp_energy.kwh_heat, flat_rates)
        cost_heat_ever = compute_electric_cost(hp_energy.kwh_heat, ever_rates)

        # Path B: gas
        therms = compute_gas_energy(heat_load, afue)
        gas_rates = gas_rate_lookup(
            months_8760,
            DEFAULT_GAS_SUMMER_RATE_PER_THERM,
            DEFAULT_GAS_WINTER_RATE_PER_THERM,
        )
        cost_gas = compute_gas_cost(therms, gas_rates)

        return {
            "hp_energy": hp_energy,
            "therms": therms,
            "cost_elec_flat": cost_elec_flat,
            "cost_elec_ever": cost_elec_ever,
            "cost_heat_flat": cost_heat_flat,
            "cost_heat_ever": cost_heat_ever,
            "cost_gas": cost_gas,
            "gas_rates": gas_rates,
            "flat_rates": flat_rates,
            "ever_rates": ever_rates,
            "months": months_8760,
            "heat_load": heat_load,
            "cool_load": cool_load,
        }

    def test_hp_annual_cost_plausible(self, pipeline_results) -> None:
        """Annual HP electricity cost (Flat Rate) is in plausible range."""
        annual = aggregate_annual(pipeline_results["cost_elec_flat"])
        assert 900 < annual < 9000, f"Annual HP cost ${annual:.0f} outside plausible range"

    def test_gas_annual_cost_plausible(self, pipeline_results) -> None:
        """Annual gas cost is in plausible range."""
        annual = aggregate_annual(pipeline_results["cost_gas"])
        assert 500 < annual < 5000, f"Annual gas cost ${annual:.0f} outside plausible range"

    def test_winter_heating_higher_than_summer(self, pipeline_results) -> None:
        """Winter months have higher heating costs than summer months."""
        months = pipeline_results["months"]
        cost_heat = pipeline_results["cost_heat_flat"]
        monthly_heat = aggregate_monthly(cost_heat, months)
        # January (index 0) should have higher heating cost than July (index 6)
        assert monthly_heat[0] > monthly_heat[6]

    def test_eversource_cheaper_winter_heating(self, pipeline_results) -> None:
        """Eversource R-1HP has lower winter heating cost than Flat Rate."""
        months = pipeline_results["months"]
        heat_flat = aggregate_monthly(pipeline_results["cost_heat_flat"], months)
        heat_ever = aggregate_monthly(pipeline_results["cost_heat_ever"], months)
        # January (winter): Eversource winter rate < Flat rate
        assert heat_ever[0] < heat_flat[0]

    def test_plausibility_bounds(self, pipeline_results) -> None:
        """Comprehensive plausibility check on all metrics."""
        hp = pipeline_results["hp_energy"]
        months = pipeline_results["months"]

        # Annual kWh
        annual_kwh = aggregate_annual(hp.kwh_total)
        assert 3000 < annual_kwh < 30000

        # Annual therms
        annual_therms = aggregate_annual(pipeline_results["therms"])
        assert 200 < annual_therms < 2000

        # Peak demand should be reasonable (< 50 kW)
        peak = compute_peak_demand_monthly(hp.kwh_total, months)
        assert np.all(peak < 50)
        assert np.all(peak >= 0)

    def test_seasonal_gas_rates_applied(self, pipeline_results) -> None:
        """Gas costs reflect seasonal rate variation."""
        months = pipeline_results["months"]
        gas_rates = pipeline_results["gas_rates"]

        # Winter months should use winter rate
        winter_mask = (months <= 4) | (months >= 11)
        assert np.all(
            gas_rates[winter_mask] == pytest.approx(DEFAULT_GAS_WINTER_RATE_PER_THERM)
        )

        # Summer months should use summer rate
        summer_mask = (months >= 5) & (months <= 10)
        assert np.all(
            gas_rates[summer_mask] == pytest.approx(DEFAULT_GAS_SUMMER_RATE_PER_THERM)
        )
