"""Tests for gas furnace cost calculations.

Edge cases from spec:
- Gas calc at zero load (should be zero therms)
- AFUE = 100% (therms = load / 100,000)
- Various AFUE presets produce expected relative costs
"""

from __future__ import annotations

import numpy as np
import pytest

from model import BTU_PER_THERM, compute_gas_energy
from presets import GAS_FURNACE_PRESETS


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
