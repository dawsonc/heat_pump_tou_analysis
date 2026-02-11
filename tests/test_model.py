"""Tests for model.py: thermal load, COP interpolation, energy calculations.

Edge cases from spec:
- Sub-zero temps (COP at/below lockout)
- Capacity derating and backup resistance handoff
- Zero-load hours (between heating and cooling setpoints)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from model import (
    BTU_PER_KWH,
    HPEnergy,
    compute_capacity,
    compute_cooling_load,
    compute_cop,
    compute_gas_energy,
    compute_heating_load,
    compute_hp_energy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_cop_curve() -> list[tuple[float, float]]:
    """Cold-climate Hyper-Heat COP curve from presets."""
    return [(5, 1.75), (17, 2.5), (47, 3.5)]


@pytest.fixture
def default_capacity_curve() -> list[tuple[float, float]]:
    """Cold-climate Hyper-Heat capacity curve from presets."""
    return [(5, 0.70), (17, 0.85), (47, 1.00)]


@pytest.fixture
def baseline_cop_curve() -> list[tuple[float, float]]:
    """Baseline (non-cold-climate) COP curve — only 2 points."""
    return [(17, 1.8), (47, 3.0)]


# ---------------------------------------------------------------------------
# TestComputeHeatingLoad
# ---------------------------------------------------------------------------


class TestComputeHeatingLoad:
    """Tests for compute_heating_load."""

    def test_zero_load_above_setpoint(self) -> None:
        """No heating needed when outdoor temp exceeds setpoint."""
        T_out = np.array([75.0, 80.0, 90.0])
        result = compute_heating_load(T_out, UA=630, T_set_heat=68)
        np.testing.assert_array_equal(result, 0.0)

    def test_zero_load_at_setpoint(self) -> None:
        """No heating needed when outdoor temp equals setpoint."""
        T_out = np.array([68.0, 68.0])
        result = compute_heating_load(T_out, UA=630, T_set_heat=68)
        np.testing.assert_array_equal(result, 0.0)

    def test_positive_load_below_setpoint(self) -> None:
        """Heating load for typical winter conditions."""
        # UA=630, T_set=68, T_out=30 -> 630 * (68-30) = 630 * 38 = 23,940
        T_out = np.array([30.0])
        result = compute_heating_load(T_out, UA=630, T_set_heat=68)
        np.testing.assert_array_almost_equal(result, [23_940.0])

    def test_sub_zero_temps(self) -> None:
        """Heating load at sub-zero outdoor temperature."""
        # UA=630, T_set=68, T_out=-10 -> 630 * (68-(-10)) = 630 * 78 = 49,140
        T_out = np.array([-10.0])
        result = compute_heating_load(T_out, UA=630, T_set_heat=68)
        np.testing.assert_array_almost_equal(result, [49_140.0])

    def test_mixed_temps(self) -> None:
        """Array with temps above and below setpoint."""
        T_out = np.array([30.0, 68.0, 80.0, 0.0])
        result = compute_heating_load(T_out, UA=630, T_set_heat=68)
        expected = np.array([23_940.0, 0.0, 0.0, 42_840.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_shape_8760(self) -> None:
        """Output shape matches input for full-year array."""
        T_out = np.full(8760, 30.0)
        result = compute_heating_load(T_out, UA=630, T_set_heat=68)
        assert result.shape == (8760,)

    def test_output_dtype(self) -> None:
        """Output dtype is float64."""
        T_out = np.array([30.0, 50.0])
        result = compute_heating_load(T_out, UA=630, T_set_heat=68)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# TestComputeCoolingLoad
# ---------------------------------------------------------------------------


class TestComputeCoolingLoad:
    """Tests for compute_cooling_load."""

    def test_zero_load_below_setpoint(self) -> None:
        """No cooling needed when outdoor temp is below setpoint."""
        T_out = np.array([60.0, 70.0, 74.0])
        result = compute_cooling_load(T_out, UA=630, T_set_cool=75)
        np.testing.assert_array_equal(result, 0.0)

    def test_zero_load_at_setpoint(self) -> None:
        """No cooling needed when outdoor temp equals setpoint."""
        T_out = np.array([75.0, 75.0])
        result = compute_cooling_load(T_out, UA=630, T_set_cool=75)
        np.testing.assert_array_equal(result, 0.0)

    def test_positive_load_above_setpoint(self) -> None:
        """Cooling load for hot day."""
        # UA=630, T_set=75, T_out=95 -> 630 * (95-75) = 630 * 20 = 12,600
        T_out = np.array([95.0])
        result = compute_cooling_load(T_out, UA=630, T_set_cool=75)
        np.testing.assert_array_almost_equal(result, [12_600.0])

    def test_mixed_temps(self) -> None:
        """Array with temps above and below cooling setpoint."""
        T_out = np.array([60.0, 75.0, 85.0, 100.0])
        result = compute_cooling_load(T_out, UA=630, T_set_cool=75)
        expected = np.array([0.0, 0.0, 6_300.0, 15_750.0])
        np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# TestComputeCop
# ---------------------------------------------------------------------------


class TestComputeCop:
    """Tests for compute_cop with piecewise-linear interpolation."""

    def test_at_reference_points(self, default_cop_curve) -> None:
        """COP matches curve exactly at reference temperatures."""
        T_out = np.array([5.0, 17.0, 47.0])
        cop, _ = compute_cop(T_out, default_cop_curve, lockout_temp=-15)
        np.testing.assert_array_almost_equal(cop, [1.75, 2.5, 3.5])

    def test_interpolation_between_points(self, default_cop_curve) -> None:
        """Linear interpolation between reference points."""
        # T_out=32 is midpoint of [17, 47] segment
        # COP = 2.5 + (3.5-2.5) * (32-17)/(47-17) = 2.5 + 0.5 = 3.0
        T_out = np.array([32.0])
        cop, _ = compute_cop(T_out, default_cop_curve, lockout_temp=-15)
        assert cop[0] == pytest.approx(3.0)

    def test_interpolation_lower_segment(self, default_cop_curve) -> None:
        """Linear interpolation in the lower segment (5-17 degF)."""
        # T_out=11 is midpoint of [5, 17] segment
        # COP = 1.75 + (2.5-1.75) * (11-5)/(17-5) = 1.75 + 0.375 = 2.125
        T_out = np.array([11.0])
        cop, _ = compute_cop(T_out, default_cop_curve, lockout_temp=-15)
        assert cop[0] == pytest.approx(2.125)

    def test_extrapolation_above_highest(self, default_cop_curve) -> None:
        """COP clamped at highest reference value above max temp."""
        T_out = np.array([60.0, 80.0, 100.0])
        cop, _ = compute_cop(T_out, default_cop_curve, lockout_temp=-15)
        np.testing.assert_array_almost_equal(cop, [3.5, 3.5, 3.5])

    def test_extrapolation_below_lowest_above_lockout(
        self, default_cop_curve
    ) -> None:
        """COP clamped at lowest reference value below min temp but above lockout."""
        # T_out=0 is below 5 (lowest ref) but above -15 (lockout)
        T_out = np.array([0.0, -5.0, -14.9])
        cop, lockout_mask = compute_cop(
            T_out, default_cop_curve, lockout_temp=-15
        )
        np.testing.assert_array_almost_equal(cop, [1.75, 1.75, 1.75])
        np.testing.assert_array_equal(lockout_mask, [False, False, False])

    def test_lockout_sets_cop_to_one(self, default_cop_curve) -> None:
        """COP = 1.0 (resistance backup) below lockout temperature."""
        T_out = np.array([-20.0, -30.0, -50.0])
        cop, lockout_mask = compute_cop(
            T_out, default_cop_curve, lockout_temp=-15
        )
        np.testing.assert_array_almost_equal(cop, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(lockout_mask, [True, True, True])

    def test_lockout_at_exact_boundary(self, default_cop_curve) -> None:
        """HP still runs at exactly the lockout temp (strict less-than)."""
        T_out = np.array([-15.0])
        cop, lockout_mask = compute_cop(
            T_out, default_cop_curve, lockout_temp=-15
        )
        # At -15 exactly: not locked out, COP clamped at lowest ref (1.75)
        assert lockout_mask[0] == False
        assert cop[0] == pytest.approx(1.75)

    def test_cop_never_below_one(self, default_cop_curve) -> None:
        """COP floor is 1.0 across all temperatures."""
        T_out = np.linspace(-50, 120, 1000)
        cop, _ = compute_cop(T_out, default_cop_curve, lockout_temp=-15)
        assert np.all(cop >= 1.0)

    def test_two_point_curve(self, baseline_cop_curve) -> None:
        """Correct interpolation with a 2-point COP curve."""
        # Baseline: [(17, 1.8), (47, 3.0)]
        # At 32 (midpoint): COP = 1.8 + (3.0-1.8)*(32-17)/(47-17) = 1.8 + 0.6 = 2.4
        T_out = np.array([17.0, 32.0, 47.0, 60.0, 10.0])
        cop, _ = compute_cop(T_out, baseline_cop_curve, lockout_temp=15)
        expected = [1.8, 2.4, 3.0, 3.0, 1.0]  # 10 < 15 lockout -> 1.0
        np.testing.assert_array_almost_equal(cop, expected)

    def test_output_shapes(self, default_cop_curve) -> None:
        """Both cop and lockout_mask have same shape as input."""
        T_out = np.full(8760, 30.0)
        cop, lockout_mask = compute_cop(
            T_out, default_cop_curve, lockout_temp=-15
        )
        assert cop.shape == (8760,)
        assert lockout_mask.shape == (8760,)

    def test_lockout_mask_dtype(self, default_cop_curve) -> None:
        """lockout_mask is boolean dtype."""
        T_out = np.array([30.0, -20.0])
        _, lockout_mask = compute_cop(
            T_out, default_cop_curve, lockout_temp=-15
        )
        assert lockout_mask.dtype == np.bool_


# ---------------------------------------------------------------------------
# TestComputeCapacity
# ---------------------------------------------------------------------------


class TestComputeCapacity:
    """Tests for compute_capacity with piecewise-linear derating."""

    def test_at_rated_temp(self, default_capacity_curve) -> None:
        """Full rated capacity at 47 degF."""
        T_out = np.array([47.0])
        result = compute_capacity(T_out, default_capacity_curve, 36_000)
        np.testing.assert_array_almost_equal(result, [36_000.0])

    def test_at_lowest_point(self, default_capacity_curve) -> None:
        """Capacity at lowest reference point (5 degF)."""
        # 0.70 * 36000 = 25,200
        T_out = np.array([5.0])
        result = compute_capacity(T_out, default_capacity_curve, 36_000)
        np.testing.assert_array_almost_equal(result, [25_200.0])

    def test_at_midpoint(self, default_capacity_curve) -> None:
        """Capacity at 17 degF reference point."""
        # 0.85 * 36000 = 30,600
        T_out = np.array([17.0])
        result = compute_capacity(T_out, default_capacity_curve, 36_000)
        np.testing.assert_array_almost_equal(result, [30_600.0])

    def test_interpolation(self, default_capacity_curve) -> None:
        """Linear interpolation between reference points."""
        # T_out=32: midpoint of [17, 47] -> fraction = 0.85 + 0.5*(1.0-0.85) = 0.925
        # capacity = 0.925 * 36000 = 33,300
        T_out = np.array([32.0])
        result = compute_capacity(T_out, default_capacity_curve, 36_000)
        np.testing.assert_array_almost_equal(result, [33_300.0])

    def test_below_lowest_point(self, default_capacity_curve) -> None:
        """Capacity clamped at lowest fraction below min reference temp."""
        # Below 5 degF: fraction = 0.70; capacity = 25,200
        T_out = np.array([-10.0, -20.0])
        result = compute_capacity(T_out, default_capacity_curve, 36_000)
        np.testing.assert_array_almost_equal(result, [25_200.0, 25_200.0])

    def test_above_highest_point(self, default_capacity_curve) -> None:
        """Capacity clamped at highest fraction above max reference temp."""
        T_out = np.array([60.0, 100.0])
        result = compute_capacity(T_out, default_capacity_curve, 36_000)
        np.testing.assert_array_almost_equal(result, [36_000.0, 36_000.0])

    def test_different_rated_capacity(self, default_capacity_curve) -> None:
        """Capacity scales with rated_capacity_btu_h."""
        T_out = np.array([47.0, 5.0])
        result = compute_capacity(T_out, default_capacity_curve, 48_000)
        np.testing.assert_array_almost_equal(result, [48_000.0, 33_600.0])


# ---------------------------------------------------------------------------
# TestComputeHpEnergy
# ---------------------------------------------------------------------------


class TestComputeHpEnergy:
    """Tests for compute_hp_energy."""

    def test_zero_load_zero_energy(self) -> None:
        """Zero heating and cooling load produces zero kWh."""
        n = 10
        result = compute_hp_energy(
            heat_load=np.zeros(n),
            cool_load=np.zeros(n),
            cop=np.full(n, 3.5),
            cop_cool=3.8,
            capacity=np.full(n, 36_000.0),
            lockout_mask=np.zeros(n, dtype=bool),
        )
        np.testing.assert_array_equal(result.kwh_heat, 0.0)
        np.testing.assert_array_equal(result.kwh_cool, 0.0)
        np.testing.assert_array_equal(result.kwh_total, 0.0)

    def test_heating_only_no_lockout(self) -> None:
        """Heating with known COP, no lockout, capacity > load."""
        # heat_load = 10,236 BTU/h, COP = 3.0 -> kwh = 10236 / (3.0*3412) = 1.0
        heat_load = np.array([10_236.0])
        result = compute_hp_energy(
            heat_load=heat_load,
            cool_load=np.zeros(1),
            cop=np.array([3.0]),
            cop_cool=3.8,
            capacity=np.array([36_000.0]),
            lockout_mask=np.array([False]),
        )
        assert result.kwh_heat[0] == pytest.approx(1.0)
        assert result.kwh_cool[0] == pytest.approx(0.0)
        assert result.kwh_total[0] == pytest.approx(1.0)

    def test_cooling_only(self) -> None:
        """Cooling with known COP, no heating."""
        # cool_load = 12,600, cop_cool = 3.8 -> kwh = 12600 / (3.8*3412) = 0.9717...
        cool_load = np.array([12_600.0])
        result = compute_hp_energy(
            heat_load=np.zeros(1),
            cool_load=cool_load,
            cop=np.array([3.5]),
            cop_cool=3.8,
            capacity=np.array([36_000.0]),
            lockout_mask=np.array([False]),
        )
        expected_cool = 12_600.0 / (3.8 * BTU_PER_KWH)
        assert result.kwh_cool[0] == pytest.approx(expected_cool)
        assert result.kwh_heat[0] == pytest.approx(0.0)

    def test_lockout_full_backup(self) -> None:
        """During lockout, all heating goes through resistance (COP=1.0)."""
        heat_load = np.array([23_940.0])
        result = compute_hp_energy(
            heat_load=heat_load,
            cool_load=np.zeros(1),
            cop=np.array([1.0]),  # lockout-adjusted COP
            cop_cool=3.8,
            capacity=np.array([36_000.0]),
            lockout_mask=np.array([True]),
        )
        # All load goes to backup: kwh = 23940 / 3412
        expected = 23_940.0 / BTU_PER_KWH
        assert result.kwh_heat[0] == pytest.approx(expected)
        assert result.load_hp[0] == pytest.approx(0.0)
        assert result.load_backup[0] == pytest.approx(23_940.0)

    def test_capacity_derating_partial_backup(self) -> None:
        """When load exceeds capacity, backup covers the difference."""
        # heat_load=40000, capacity=25200, COP=1.75
        # load_hp=25200, load_backup=14800
        # kwh_heat = 25200/(1.75*3412) + 14800/3412
        heat_load = np.array([40_000.0])
        result = compute_hp_energy(
            heat_load=heat_load,
            cool_load=np.zeros(1),
            cop=np.array([1.75]),
            cop_cool=3.8,
            capacity=np.array([25_200.0]),
            lockout_mask=np.array([False]),
        )
        expected_hp_kwh = 25_200.0 / (1.75 * BTU_PER_KWH)
        expected_backup_kwh = 14_800.0 / BTU_PER_KWH
        assert result.kwh_heat[0] == pytest.approx(
            expected_hp_kwh + expected_backup_kwh
        )
        assert result.load_hp[0] == pytest.approx(25_200.0)
        assert result.load_backup[0] == pytest.approx(14_800.0)

    def test_no_backup_needed(self) -> None:
        """When capacity exceeds load, no backup is used."""
        heat_load = np.array([20_000.0])
        result = compute_hp_energy(
            heat_load=heat_load,
            cool_load=np.zeros(1),
            cop=np.array([3.0]),
            cop_cool=3.8,
            capacity=np.array([36_000.0]),
            lockout_mask=np.array([False]),
        )
        assert result.load_backup[0] == pytest.approx(0.0)
        assert result.load_hp[0] == pytest.approx(20_000.0)

    def test_total_is_sum(self) -> None:
        """kwh_total equals kwh_heat + kwh_cool."""
        n = 100
        result = compute_hp_energy(
            heat_load=np.random.default_rng(42).uniform(0, 50_000, n),
            cool_load=np.random.default_rng(43).uniform(0, 15_000, n),
            cop=np.full(n, 2.5),
            cop_cool=3.5,
            capacity=np.full(n, 30_000.0),
            lockout_mask=np.zeros(n, dtype=bool),
        )
        np.testing.assert_array_almost_equal(
            result.kwh_total, result.kwh_heat + result.kwh_cool
        )

    def test_returns_named_tuple(self) -> None:
        """Result is an HPEnergy named tuple with expected fields."""
        result = compute_hp_energy(
            heat_load=np.zeros(1),
            cool_load=np.zeros(1),
            cop=np.array([3.5]),
            cop_cool=3.8,
            capacity=np.array([36_000.0]),
            lockout_mask=np.array([False]),
        )
        assert isinstance(result, HPEnergy)
        assert hasattr(result, "kwh_heat")
        assert hasattr(result, "kwh_cool")
        assert hasattr(result, "kwh_total")
        assert hasattr(result, "load_hp")
        assert hasattr(result, "load_backup")

    def test_mixed_lockout_and_normal(self) -> None:
        """Array with some hours locked out, some normal."""
        heat_load = np.array([20_000.0, 20_000.0, 0.0])
        lockout = np.array([True, False, False])
        cop = np.array([1.0, 3.0, 3.5])
        capacity = np.array([36_000.0, 36_000.0, 36_000.0])

        result = compute_hp_energy(
            heat_load=heat_load,
            cool_load=np.zeros(3),
            cop=cop,
            cop_cool=3.8,
            capacity=capacity,
            lockout_mask=lockout,
        )

        # Hour 0: locked out -> all backup
        assert result.load_hp[0] == pytest.approx(0.0)
        assert result.load_backup[0] == pytest.approx(20_000.0)
        assert result.kwh_heat[0] == pytest.approx(20_000.0 / BTU_PER_KWH)

        # Hour 1: normal operation, capacity > load
        assert result.load_hp[1] == pytest.approx(20_000.0)
        assert result.load_backup[1] == pytest.approx(0.0)
        assert result.kwh_heat[1] == pytest.approx(
            20_000.0 / (3.0 * BTU_PER_KWH)
        )

        # Hour 2: zero load
        assert result.kwh_heat[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestIntegrationPipeline
# ---------------------------------------------------------------------------


class TestComputeHpEnergyNoBackup:
    """Tests for compute_hp_energy with has_backup=False."""

    def test_no_backup_load_still_computed(self) -> None:
        """load_backup is still non-zero even when has_backup=False."""
        heat_load = np.array([40_000.0])
        result = compute_hp_energy(
            heat_load=heat_load,
            cool_load=np.zeros(1),
            cop=np.array([1.75]),
            cop_cool=3.8,
            capacity=np.array([25_200.0]),
            lockout_mask=np.array([False]),
            has_backup=False,
        )
        assert result.load_backup[0] == pytest.approx(14_800.0)

    def test_no_backup_kwh_excludes_backup(self) -> None:
        """kwh_heat does not include backup kWh when has_backup=False."""
        heat_load = np.array([40_000.0])
        result = compute_hp_energy(
            heat_load=heat_load,
            cool_load=np.zeros(1),
            cop=np.array([1.75]),
            cop_cool=3.8,
            capacity=np.array([25_200.0]),
            lockout_mask=np.array([False]),
            has_backup=False,
        )
        expected_hp_kwh = 25_200.0 / (1.75 * BTU_PER_KWH)
        assert result.kwh_heat[0] == pytest.approx(expected_hp_kwh)
        assert result.kwh_total[0] == pytest.approx(expected_hp_kwh)

    def test_default_has_backup_true(self) -> None:
        """Default behavior (no argument) still includes backup."""
        heat_load = np.array([40_000.0])
        result = compute_hp_energy(
            heat_load=heat_load,
            cool_load=np.zeros(1),
            cop=np.array([1.75]),
            cop_cool=3.8,
            capacity=np.array([25_200.0]),
            lockout_mask=np.array([False]),
        )
        expected = 25_200.0 / (1.75 * BTU_PER_KWH) + 14_800.0 / BTU_PER_KWH
        assert result.kwh_heat[0] == pytest.approx(expected)

    def test_lockout_no_backup_zero_kwh(self) -> None:
        """During lockout with no backup, kwh_heat is zero."""
        heat_load = np.array([20_000.0])
        result = compute_hp_energy(
            heat_load=heat_load,
            cool_load=np.zeros(1),
            cop=np.array([1.0]),
            cop_cool=3.8,
            capacity=np.array([36_000.0]),
            lockout_mask=np.array([True]),
            has_backup=False,
        )
        assert result.kwh_heat[0] == pytest.approx(0.0)
        assert result.load_backup[0] == pytest.approx(20_000.0)


class TestIntegrationPipeline:
    """Integration tests exercising the full computation chain."""

    def test_deadband_produces_zero(self) -> None:
        """Temps in deadband (68-75) produce zero load and zero kWh."""
        T_out = np.full(8760, 71.0)  # between 68 (heat) and 75 (cool)
        heat_load = compute_heating_load(T_out, UA=630, T_set_heat=68)
        cool_load = compute_cooling_load(T_out, UA=630, T_set_cool=75)

        np.testing.assert_array_equal(heat_load, 0.0)
        np.testing.assert_array_equal(cool_load, 0.0)

        cop_curve = [(5, 1.75), (17, 2.5), (47, 3.5)]
        cap_curve = [(5, 0.70), (17, 0.85), (47, 1.00)]
        cop, lockout_mask = compute_cop(T_out, cop_curve, lockout_temp=-15)
        capacity = compute_capacity(T_out, cap_curve, 36_000)

        result = compute_hp_energy(
            heat_load, cool_load, cop, 3.8, capacity, lockout_mask
        )
        np.testing.assert_array_equal(result.kwh_total, 0.0)

    def test_full_pipeline_cold_day(self) -> None:
        """Full pipeline on a cold day (T_out=0) with default preset."""
        T_out = np.full(24, 0.0)
        UA = 630
        heat_load = compute_heating_load(T_out, UA, T_set_heat=68)
        cool_load = compute_cooling_load(T_out, UA, T_set_cool=75)

        # heat_load = 630 * 68 = 42,840 per hour
        np.testing.assert_array_almost_equal(heat_load, 42_840.0)
        np.testing.assert_array_equal(cool_load, 0.0)

        cop_curve = [(5, 1.75), (17, 2.5), (47, 3.5)]
        cap_curve = [(5, 0.70), (17, 0.85), (47, 1.00)]
        cop, lockout_mask = compute_cop(T_out, cop_curve, lockout_temp=-15)
        capacity = compute_capacity(T_out, cap_curve, 36_000)

        # COP at 0 degF: clamped at 1.75 (below lowest ref, above lockout)
        np.testing.assert_array_almost_equal(cop, 1.75)
        np.testing.assert_array_equal(lockout_mask, False)
        # Capacity at 0 degF: clamped at 0.70 * 36000 = 25,200
        np.testing.assert_array_almost_equal(capacity, 25_200.0)

        result = compute_hp_energy(
            heat_load, cool_load, cop, 3.8, capacity, lockout_mask
        )

        # load_hp=25200, load_backup=42840-25200=17640
        np.testing.assert_array_almost_equal(result.load_hp, 25_200.0)
        np.testing.assert_array_almost_equal(result.load_backup, 17_640.0)

        # kwh_heat per hour = 25200/(1.75*3412) + 17640/3412
        expected_kwh = 25_200 / (1.75 * 3412) + 17_640 / 3412
        np.testing.assert_array_almost_equal(result.kwh_heat, expected_kwh)
        assert result.kwh_heat[0] > 0

        # Gas comparison
        therms = compute_gas_energy(heat_load, afue=0.80)
        # therms = 42840 / (0.80 * 100000) = 0.5355
        np.testing.assert_array_almost_equal(therms, 42_840 / 80_000)

    def test_full_pipeline_hot_day(self) -> None:
        """Full pipeline on a hot day: no heating, only cooling."""
        T_out = np.full(24, 95.0)
        UA = 630
        heat_load = compute_heating_load(T_out, UA, T_set_heat=68)
        cool_load = compute_cooling_load(T_out, UA, T_set_cool=75)

        np.testing.assert_array_equal(heat_load, 0.0)
        # cool_load = 630 * 20 = 12,600
        np.testing.assert_array_almost_equal(cool_load, 12_600.0)

        cop_curve = [(5, 1.75), (17, 2.5), (47, 3.5)]
        cap_curve = [(5, 0.70), (17, 0.85), (47, 1.00)]
        cop, lockout_mask = compute_cop(T_out, cop_curve, lockout_temp=-15)
        capacity = compute_capacity(T_out, cap_curve, 36_000)

        result = compute_hp_energy(
            heat_load, cool_load, cop, 3.8, capacity, lockout_mask
        )

        assert np.all(result.kwh_heat == 0.0)
        expected_cool = 12_600.0 / (3.8 * BTU_PER_KWH)
        np.testing.assert_array_almost_equal(result.kwh_cool, expected_cool)
        np.testing.assert_array_almost_equal(result.kwh_total, expected_cool)

        # Gas: zero heating load -> zero therms
        therms = compute_gas_energy(heat_load, afue=0.80)
        np.testing.assert_array_equal(therms, 0.0)

    def test_full_pipeline_boston_tmy(self, boston_weather_df) -> None:
        """Integration with real Boston TMY3 data."""
        T_out = boston_weather_df["T_drybulb_F"].to_numpy()
        assert T_out.shape == (8760,)

        UA = 630
        heat_load = compute_heating_load(T_out, UA, T_set_heat=68)
        cool_load = compute_cooling_load(T_out, UA, T_set_cool=75)

        cop_curve = [(5, 1.75), (17, 2.5), (47, 3.5)]
        cap_curve = [(5, 0.70), (17, 0.85), (47, 1.00)]
        cop, lockout_mask = compute_cop(T_out, cop_curve, lockout_temp=-15)
        capacity = compute_capacity(T_out, cap_curve, 36_000)

        result = compute_hp_energy(
            heat_load, cool_load, cop, 3.8, capacity, lockout_mask
        )

        # Basic sanity checks
        assert result.kwh_total.shape == (8760,)
        assert np.all(result.kwh_total >= 0)
        assert np.all(result.kwh_heat >= 0)
        assert np.all(result.kwh_cool >= 0)
        assert np.all(result.load_hp >= 0)
        assert np.all(result.load_backup >= 0)

        # Annual kWh should be plausible for a Boston home
        annual_kwh = result.kwh_total.sum()
        assert 3_000 < annual_kwh < 30_000, f"Unexpected annual kWh: {annual_kwh}"

        # Gas comparison
        therms = compute_gas_energy(heat_load, afue=0.80)
        assert therms.shape == (8760,)
        assert np.all(therms >= 0)
        annual_therms = therms.sum()
        assert 200 < annual_therms < 2_000, (
            f"Unexpected annual therms: {annual_therms}"
        )
