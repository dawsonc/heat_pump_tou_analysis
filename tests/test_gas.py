"""Tests for gas furnace cost calculations.

Edge cases from spec:
- Gas calc at zero load (should be zero therms)
- AFUE = 100% (therms = load / 100,000)
- Various AFUE presets produce expected relative costs
"""

from __future__ import annotations

import numpy as np
import pytest

from model import BTU_PER_THERM, compute_gas_cost, compute_gas_energy
from presets import (
    DEFAULT_GAS_SUMMER_RATE_PER_THERM,
    DEFAULT_GAS_WINTER_RATE_PER_THERM,
    GAS_FURNACE_PRESETS,
)
from rates import gas_rate_lookup


# ---------------------------------------------------------------------------
# TestComputeGasEnergy
# ---------------------------------------------------------------------------


class TestComputeGasEnergy:
    """Tests for compute_gas_energy."""

    def test_zero_load_zero_therms(self) -> None:
        """Zero heating load produces zero gas consumption."""
        heat_load = np.zeros(8760)
        therms = compute_gas_energy(heat_load, afue=0.80)
        np.testing.assert_array_equal(therms, 0.0)

    def test_afue_100_percent(self) -> None:
        """AFUE=1.0: therms = load / 100,000 exactly."""
        # 100,000 BTU = 1 therm at 100% efficiency
        heat_load = np.array([100_000.0, 200_000.0, 50_000.0])
        therms = compute_gas_energy(heat_load, afue=1.0)
        np.testing.assert_array_almost_equal(therms, [1.0, 2.0, 0.5])

    def test_afue_80_percent(self) -> None:
        """AFUE=0.80: needs more gas to deliver same heat."""
        # therms = 100,000 / (0.80 * 100,000) = 1.25
        heat_load = np.array([100_000.0])
        therms = compute_gas_energy(heat_load, afue=0.80)
        assert therms[0] == pytest.approx(1.25)

    def test_afue_96_percent(self) -> None:
        """AFUE=0.96: high-efficiency condensing furnace."""
        # therms = 100,000 / (0.96 * 100,000) = 1.04167
        heat_load = np.array([100_000.0])
        therms = compute_gas_energy(heat_load, afue=0.96)
        assert therms[0] == pytest.approx(100_000 / (0.96 * BTU_PER_THERM))

    def test_lower_afue_more_therms(self) -> None:
        """Lower AFUE uses more gas for same load."""
        heat_load = np.array([50_000.0])
        therms_80 = compute_gas_energy(heat_load, afue=0.80)
        therms_96 = compute_gas_energy(heat_load, afue=0.96)
        assert therms_80[0] > therms_96[0]

    def test_all_presets_ranking(self) -> None:
        """All AFUE presets produce correct relative gas consumption."""
        heat_load = np.array([100_000.0])
        results = {}
        for name, preset in GAS_FURNACE_PRESETS.items():
            therms = compute_gas_energy(heat_load, afue=preset["afue"])
            results[name] = therms[0]

        # Lower AFUE -> more therms
        assert results["Older furnace"] > results["Standard high-efficiency"]
        assert results["Standard high-efficiency"] > results["High-efficiency condensing"]

    def test_output_shape(self) -> None:
        """Output shape matches input."""
        heat_load = np.full(8760, 30_000.0)
        therms = compute_gas_energy(heat_load, afue=0.80)
        assert therms.shape == (8760,)

    def test_output_dtype(self) -> None:
        """Output dtype is float64."""
        heat_load = np.array([50_000.0])
        therms = compute_gas_energy(heat_load, afue=0.80)
        assert therms.dtype == np.float64


# ---------------------------------------------------------------------------
# TestGasCostComparison
# ---------------------------------------------------------------------------


class TestGasCostComparison:
    """Tests verifying gas cost calculation (therms * rate)."""

    def test_gas_cost_formula(self) -> None:
        """Gas cost = therms * gas_rate per spec."""
        heat_load = np.array([80_000.0])
        therms = compute_gas_energy(heat_load, afue=0.80)
        # therms = 80000 / 80000 = 1.0
        gas_rate = 2.50
        cost = therms * gas_rate
        assert cost[0] == pytest.approx(2.50)

    def test_gas_cost_zero_load(self) -> None:
        """Zero load means zero gas cost."""
        heat_load = np.zeros(24)
        therms = compute_gas_energy(heat_load, afue=0.80)
        cost = therms * 2.50
        np.testing.assert_array_equal(cost, 0.0)


# ---------------------------------------------------------------------------
# TestComputeGasCost
# ---------------------------------------------------------------------------


class TestComputeGasCost:
    """Tests for the formal compute_gas_cost function."""

    def test_basic_cost(self) -> None:
        """Known therms and rate produce correct cost."""
        therms = np.array([1.0])
        cost = compute_gas_cost(therms, gas_rate=2.50)
        assert cost[0] == pytest.approx(2.50)

    def test_zero_therms(self) -> None:
        """Zero therms produces zero cost."""
        therms = np.zeros(24)
        cost = compute_gas_cost(therms, gas_rate=2.50)
        np.testing.assert_array_equal(cost, 0.0)

    def test_zero_rate(self) -> None:
        """Zero gas rate produces zero cost."""
        therms = np.array([1.0, 2.0])
        cost = compute_gas_cost(therms, gas_rate=0.0)
        np.testing.assert_array_equal(cost, 0.0)

    def test_shape_preserved(self) -> None:
        """Output shape matches input."""
        therms = np.ones(8760)
        cost = compute_gas_cost(therms, gas_rate=2.50)
        assert cost.shape == (8760,)

    def test_dtype_float64(self) -> None:
        """Output dtype is float64."""
        therms = np.array([1.0])
        cost = compute_gas_cost(therms, gas_rate=2.50)
        assert cost.dtype == np.float64

    def test_full_pipeline(self) -> None:
        """End-to-end: compute_gas_energy -> compute_gas_cost."""
        heat_load = np.array([80_000.0])
        therms = compute_gas_energy(heat_load, afue=0.80)
        cost = compute_gas_cost(therms, gas_rate=2.50)
        # therms = 80000 / (0.80 * 100000) = 1.0
        # cost = 1.0 * 2.50 = 2.50
        assert cost[0] == pytest.approx(2.50)

    def test_higher_rate_higher_cost(self) -> None:
        """Increasing gas rate increases cost proportionally."""
        therms = np.array([1.0])
        cost_low = compute_gas_cost(therms, gas_rate=1.50)
        cost_high = compute_gas_cost(therms, gas_rate=3.00)
        assert cost_high[0] == pytest.approx(2.0 * cost_low[0])

    def test_matches_inline_formula(self) -> None:
        """compute_gas_cost matches inline therms * rate."""
        therms = np.array([0.5, 1.0, 1.5])
        gas_rate = 2.50
        cost_func = compute_gas_cost(therms, gas_rate)
        cost_inline = therms * gas_rate
        np.testing.assert_array_equal(cost_func, cost_inline)


# ---------------------------------------------------------------------------
# TestSeasonalGasCost
# ---------------------------------------------------------------------------


class TestSeasonalGasCost:
    """Tests for compute_gas_cost with seasonal (array) gas rates."""

    def test_array_rate_accepted(self) -> None:
        """compute_gas_cost works with an (N,) rate array."""
        therms = np.array([1.0, 1.0])
        rates = np.array([2.50, 1.80])
        cost = compute_gas_cost(therms, rates)
        np.testing.assert_array_almost_equal(cost, [2.50, 1.80])

    def test_scalar_still_works(self) -> None:
        """compute_gas_cost still works with a scalar rate (backward compat)."""
        therms = np.array([1.0, 2.0])
        cost = compute_gas_cost(therms, gas_rate=2.50)
        np.testing.assert_array_almost_equal(cost, [2.50, 5.00])

    def test_seasonal_pipeline(self, months_8760: np.ndarray) -> None:
        """End-to-end: gas_rate_lookup -> compute_gas_cost with seasonal rates."""
        therms = np.ones(8760)  # 1 therm/hour everywhere
        rates = gas_rate_lookup(months_8760, summer_rate=1.80, winter_rate=2.50)
        cost = compute_gas_cost(therms, rates)

        summer_mask = (months_8760 >= 5) & (months_8760 <= 10)
        np.testing.assert_array_almost_equal(cost[summer_mask], 1.80)
        np.testing.assert_array_almost_equal(cost[~summer_mask], 2.50)

    def test_winter_more_expensive_than_summer(self, months_8760: np.ndarray) -> None:
        """With typical MA rates, winter gas cost exceeds summer for same therms."""
        therms = np.ones(8760)
        rates = gas_rate_lookup(
            months_8760,
            summer_rate=DEFAULT_GAS_SUMMER_RATE_PER_THERM,
            winter_rate=DEFAULT_GAS_WINTER_RATE_PER_THERM,
        )
        cost = compute_gas_cost(therms, rates)
        winter_mask = (months_8760 <= 4) | (months_8760 >= 11)
        assert cost[winter_mask].mean() > cost[~winter_mask].mean()

    def test_flat_rate_matches_scalar(self, months_8760: np.ndarray) -> None:
        """When summer == winter, array rate gives same result as scalar."""
        therms = np.random.default_rng(42).uniform(0, 2, 8760)
        rate_val = 2.50
        cost_scalar = compute_gas_cost(therms, gas_rate=rate_val)
        rates_array = gas_rate_lookup(months_8760, summer_rate=rate_val, winter_rate=rate_val)
        cost_array = compute_gas_cost(therms, rates_array)
        np.testing.assert_array_almost_equal(cost_scalar, cost_array)

    def test_shape_preserved(self, months_8760: np.ndarray) -> None:
        """Output shape is (8760,) with seasonal rates."""
        therms = np.ones(8760)
        rates = gas_rate_lookup(months_8760, summer_rate=1.80, winter_rate=2.50)
        cost = compute_gas_cost(therms, rates)
        assert cost.shape == (8760,)

    def test_default_rates_exist(self) -> None:
        """Default seasonal gas rate constants are defined and reasonable."""
        assert DEFAULT_GAS_WINTER_RATE_PER_THERM > 0
        assert DEFAULT_GAS_SUMMER_RATE_PER_THERM > 0
        assert DEFAULT_GAS_WINTER_RATE_PER_THERM > DEFAULT_GAS_SUMMER_RATE_PER_THERM
