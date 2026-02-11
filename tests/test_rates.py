"""Tests for rates.py: rate lookup, season boundaries, tier assignment.

Edge cases from spec:
- Season boundary (Apr 30 -> May 1)
- Midnight tier transitions
- Flat rate returns same value for all hours
"""

import numpy as np
import pytest

from rates import (
    CUSTOM_TOU_TEMPLATE,
    EVERSOURCE_R1HP,
    FLAT_RATE,
    RATE_PRESETS,
    SUMMER_MONTHS,
    TOU_PEAK_SAVER,
    RateSchedule,
    _build_hour_rate_map,
    build_flat_schedule,
    build_seasonal_flat_schedule,
    build_tou_schedule,
    extract_on_peak_hours,
    extract_tou_prices,
    gas_rate_lookup,
    get_season,
    schedule_type,
    tou_lookup,
)


# -------------------------------------------------------------------
# Tests for get_season
# -------------------------------------------------------------------


class TestGetSeason:
    @pytest.mark.parametrize(
        "month,expected",
        [
            (1, "winter"),
            (2, "winter"),
            (3, "winter"),
            (4, "winter"),
            (5, "summer"),
            (6, "summer"),
            (7, "summer"),
            (8, "summer"),
            (9, "summer"),
            (10, "summer"),
            (11, "winter"),
            (12, "winter"),
        ],
    )
    def test_scalar_all_months(self, month: int, expected: str) -> None:
        """Every month maps to the correct season."""
        assert get_season(month) == expected

    def test_summer_boundary_april_to_may(self) -> None:
        """April is winter, May is summer -- the season boundary."""
        assert get_season(4) == "winter"
        assert get_season(5) == "summer"

    def test_summer_boundary_october_to_november(self) -> None:
        """October is summer, November is winter."""
        assert get_season(10) == "summer"
        assert get_season(11) == "winter"

    def test_array_input(self) -> None:
        """get_season handles NumPy arrays."""
        months = np.array([1, 4, 5, 10, 11, 12])
        result = get_season(months)
        expected = np.array(["winter", "winter", "summer",
                             "summer", "winter", "winter"])
        np.testing.assert_array_equal(result, expected)

    def test_full_year_array(self, months_8760: np.ndarray) -> None:
        """Season labels for full 8760-hour year array."""
        result = get_season(months_8760)
        assert result.shape == (8760,)
        assert np.all(result[months_8760 == 1] == "winter")
        assert np.all(result[months_8760 == 7] == "summer")

    def test_invalid_month_scalar(self) -> None:
        """Month 0 or 13 raises ValueError."""
        with pytest.raises(ValueError):
            get_season(0)
        with pytest.raises(ValueError):
            get_season(13)

    def test_invalid_month_array(self) -> None:
        """Array containing invalid month raises ValueError."""
        with pytest.raises(ValueError):
            get_season(np.array([1, 0, 5]))


# -------------------------------------------------------------------
# Tests for _build_hour_rate_map (private helper)
# -------------------------------------------------------------------


class TestBuildHourRateMap:
    def test_single_tier_all_hours(self) -> None:
        """A single tier covering 0-23 produces uniform (24,) array."""
        season = {"tiers": {"all": {"price": 0.30, "hours": list(range(24))}}}
        result = _build_hour_rate_map(season)
        assert result.shape == (24,)
        np.testing.assert_array_almost_equal(result, 0.30)

    def test_two_tiers_cover_all_hours(self) -> None:
        """Two tiers dividing the day produce correct per-hour rates."""
        season = {
            "tiers": {
                "on_peak": {"price": 0.40, "hours": [16, 17, 18, 19, 20]},
                "off_peak": {
                    "price": 0.20,
                    "hours": [h for h in range(24) if h not in range(16, 21)],
                },
            },
        }
        result = _build_hour_rate_map(season)
        assert result[16] == pytest.approx(0.40)
        assert result[0] == pytest.approx(0.20)
        assert result[23] == pytest.approx(0.20)

    def test_gap_raises_error(self) -> None:
        """If any hour 0-23 is not covered, raises ValueError."""
        season = {
            "tiers": {
                "partial": {"price": 0.30, "hours": list(range(20))},
            },
        }
        with pytest.raises(ValueError, match="not assigned"):
            _build_hour_rate_map(season)

    def test_overlap_raises_error(self) -> None:
        """If the same hour is assigned to two tiers, raises ValueError."""
        season = {
            "tiers": {
                "tier_a": {"price": 0.30, "hours": list(range(12))},
                "tier_b": {"price": 0.20, "hours": list(range(10, 24))},
            },
        }
        with pytest.raises(ValueError, match="multiple tiers"):
            _build_hour_rate_map(season)

    def test_invalid_hour_raises_error(self) -> None:
        """Hour outside 0-23 raises ValueError."""
        season = {
            "tiers": {
                "all": {"price": 0.30, "hours": list(range(24)) + [24]},
            },
        }
        with pytest.raises(ValueError):
            _build_hour_rate_map(season)


# -------------------------------------------------------------------
# Tests for tou_lookup -- Flat Rate
# -------------------------------------------------------------------


class TestTouLookupFlatRate:
    def test_flat_rate_all_identical(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        """Flat rate schedule returns 0.30 for every hour of the year."""
        rates = tou_lookup(months_8760, hours_of_day_8760, FLAT_RATE)
        assert rates.shape == (8760,)
        np.testing.assert_array_almost_equal(rates, 0.30)

    def test_flat_rate_dtype_float(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        """Output dtype is float64."""
        rates = tou_lookup(months_8760, hours_of_day_8760, FLAT_RATE)
        assert rates.dtype == np.float64

    def test_flat_rate_small_array(self) -> None:
        """Flat rate works on small test arrays."""
        months = np.array([1, 7, 12])
        hours = np.array([0, 12, 23])
        rates = tou_lookup(months, hours, FLAT_RATE)
        np.testing.assert_array_almost_equal(rates, 0.30)


# -------------------------------------------------------------------
# Tests for tou_lookup -- Eversource R-1HP
# -------------------------------------------------------------------


class TestTouLookupEversourceR1HP:
    def test_winter_months_get_lower_rate(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        """Winter months (Nov-Apr) get $0.23/kWh."""
        rates = tou_lookup(months_8760, hours_of_day_8760, EVERSOURCE_R1HP)
        winter_mask = (months_8760 <= 4) | (months_8760 >= 11)
        np.testing.assert_array_almost_equal(rates[winter_mask], 0.23)

    def test_summer_months_get_standard_rate(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        """Summer months (May-Oct) get $0.30/kWh."""
        rates = tou_lookup(months_8760, hours_of_day_8760, EVERSOURCE_R1HP)
        summer_mask = (months_8760 >= 5) & (months_8760 <= 10)
        np.testing.assert_array_almost_equal(rates[summer_mask], 0.30)

    def test_season_boundary_april_may(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        """Last hour of April uses winter rate, first hour of May uses summer."""
        rates = tou_lookup(months_8760, hours_of_day_8760, EVERSOURCE_R1HP)

        april_indices = np.where(months_8760 == 4)[0]
        may_indices = np.where(months_8760 == 5)[0]

        last_april_hour = april_indices[-1]
        first_may_hour = may_indices[0]

        assert rates[last_april_hour] == pytest.approx(0.23)
        assert rates[first_may_hour] == pytest.approx(0.30)
        # These should be consecutive hours
        assert first_may_hour == last_april_hour + 1

    def test_season_boundary_october_november(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        """Last hour of October uses summer rate, first of November uses winter."""
        rates = tou_lookup(months_8760, hours_of_day_8760, EVERSOURCE_R1HP)

        oct_indices = np.where(months_8760 == 10)[0]
        nov_indices = np.where(months_8760 == 11)[0]

        assert rates[oct_indices[-1]] == pytest.approx(0.30)
        assert rates[nov_indices[0]] == pytest.approx(0.23)
        assert nov_indices[0] == oct_indices[-1] + 1

    def test_output_shape_and_dtype(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        """Output is (8760,) float64."""
        rates = tou_lookup(months_8760, hours_of_day_8760, EVERSOURCE_R1HP)
        assert rates.shape == (8760,)
        assert rates.dtype == np.float64

    def test_only_two_distinct_rates(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        """R-1HP has exactly two distinct rate values."""
        rates = tou_lookup(months_8760, hours_of_day_8760, EVERSOURCE_R1HP)
        unique = np.unique(rates)
        assert len(unique) == 2
        np.testing.assert_array_almost_equal(sorted(unique), [0.23, 0.30])


# -------------------------------------------------------------------
# Tests for tou_lookup -- Custom TOU (multi-tier)
# -------------------------------------------------------------------


class TestTouLookupCustomTOU:
    @pytest.fixture
    def three_tier_schedule(self) -> RateSchedule:
        """Custom schedule with 3 tiers and different summer/winter prices."""
        return {
            "name": "Test Three-Tier",
            "customer_charge": 10.00,
            "summer": {
                "tiers": {
                    "on_peak": {
                        "price": 0.45,
                        "hours": [16, 17, 18, 19, 20],
                    },
                    "off_peak": {
                        "price": 0.20,
                        "hours": [7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23],
                    },
                    "super_off_peak": {
                        "price": 0.10,
                        "hours": [0, 1, 2, 3, 4, 5, 6],
                    },
                },
            },
            "winter": {
                "tiers": {
                    "on_peak": {
                        "price": 0.40,
                        "hours": [16, 17, 18, 19, 20],
                    },
                    "off_peak": {
                        "price": 0.18,
                        "hours": [7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23],
                    },
                    "super_off_peak": {
                        "price": 0.08,
                        "hours": [0, 1, 2, 3, 4, 5, 6],
                    },
                },
            },
        }

    def test_summer_on_peak(self, three_tier_schedule: RateSchedule) -> None:
        """Summer on-peak hours get $0.45."""
        months = np.array([7, 7, 7])
        hours = np.array([16, 18, 20])
        rates = tou_lookup(months, hours, three_tier_schedule)
        np.testing.assert_array_almost_equal(rates, 0.45)

    def test_winter_super_off_peak(
        self, three_tier_schedule: RateSchedule
    ) -> None:
        """Winter super-off-peak hours (midnight) get $0.08."""
        months = np.array([1, 1, 1])
        hours = np.array([0, 3, 6])
        rates = tou_lookup(months, hours, three_tier_schedule)
        np.testing.assert_array_almost_equal(rates, 0.08)

    def test_midnight_tier_transition(
        self, three_tier_schedule: RateSchedule
    ) -> None:
        """Hour 23 is off-peak, hour 0 is super-off-peak (different tiers)."""
        months = np.array([7, 7])
        hours = np.array([23, 0])
        rates = tou_lookup(months, hours, three_tier_schedule)
        assert rates[0] == pytest.approx(0.20)  # hour 23 = off_peak
        assert rates[1] == pytest.approx(0.10)  # hour 0 = super_off_peak

    def test_full_year_shape(
        self,
        three_tier_schedule: RateSchedule,
        months_8760: np.ndarray,
        hours_of_day_8760: np.ndarray,
    ) -> None:
        """Full year produces (8760,) output with no NaN values."""
        rates = tou_lookup(months_8760, hours_of_day_8760, three_tier_schedule)
        assert rates.shape == (8760,)
        assert not np.any(np.isnan(rates))

    def test_full_year_six_distinct_rates(
        self,
        three_tier_schedule: RateSchedule,
        months_8760: np.ndarray,
        hours_of_day_8760: np.ndarray,
    ) -> None:
        """3 tiers x 2 seasons = 6 distinct rate values."""
        rates = tou_lookup(months_8760, hours_of_day_8760, three_tier_schedule)
        unique = np.unique(rates)
        assert len(unique) == 6


# -------------------------------------------------------------------
# Tests for tou_lookup -- Input validation
# -------------------------------------------------------------------


class TestTouLookupValidation:
    def test_shape_mismatch_raises(self) -> None:
        """months and hours_of_day with different shapes raises ValueError."""
        months = np.array([1, 2, 3])
        hours = np.array([0, 1])
        with pytest.raises(ValueError, match="Shape mismatch"):
            tou_lookup(months, hours, FLAT_RATE)


# -------------------------------------------------------------------
# Tests for preset schedule data integrity
# -------------------------------------------------------------------


class TestPresetIntegrity:
    def test_rate_presets_dict_contains_all(self) -> None:
        """RATE_PRESETS has Flat Rate, Eversource R-1HP, and TOU Peak Saver."""
        assert "Flat Rate" in RATE_PRESETS
        assert "Eversource R-1HP" in RATE_PRESETS
        assert "TOU Peak Saver" in RATE_PRESETS

    def test_flat_rate_summer_winter_same_price(self) -> None:
        """Flat rate has identical prices in summer and winter."""
        s_price = FLAT_RATE["summer"]["tiers"]["all"]["price"]
        w_price = FLAT_RATE["winter"]["tiers"]["all"]["price"]
        assert s_price == w_price

    def test_eversource_winter_cheaper_than_summer(self) -> None:
        """R-1HP winter rate is lower than summer rate."""
        s_price = EVERSOURCE_R1HP["summer"]["tiers"]["standard"]["price"]
        w_price = EVERSOURCE_R1HP["winter"]["tiers"]["heat_pump"]["price"]
        assert w_price < s_price

    def test_all_presets_cover_24_hours(self) -> None:
        """Every preset's tier hours cover exactly hours 0-23."""
        for name, schedule in RATE_PRESETS.items():
            for season_key in ("summer", "winter"):
                all_hours: list[int] = []
                for tier in schedule[season_key]["tiers"].values():
                    all_hours.extend(tier["hours"])
                assert sorted(all_hours) == list(range(24)), (
                    f"{name}/{season_key} does not cover hours 0-23"
                )

    def test_custom_template_valid(self) -> None:
        """The CUSTOM_TOU_TEMPLATE is a valid schedule (no gaps/overlaps)."""
        _build_hour_rate_map(CUSTOM_TOU_TEMPLATE["summer"])
        _build_hour_rate_map(CUSTOM_TOU_TEMPLATE["winter"])

    def test_all_presets_have_customer_charge(self) -> None:
        """Every preset has a customer_charge field."""
        for name, schedule in RATE_PRESETS.items():
            assert "customer_charge" in schedule, (
                f"{name} missing customer_charge"
            )
            assert isinstance(schedule["customer_charge"], (int, float))

    def test_customer_charges_non_negative(self) -> None:
        """Customer charges are non-negative for all presets and template."""
        for schedule in [FLAT_RATE, EVERSOURCE_R1HP, CUSTOM_TOU_TEMPLATE, TOU_PEAK_SAVER]:
            assert schedule["customer_charge"] >= 0


# -------------------------------------------------------------------
# Tests for TOU Peak Saver preset
# -------------------------------------------------------------------


class TestTouLookupPeakSaver:
    def test_summer_on_peak_rate(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        """Summer on-peak hours get $0.731."""
        rates = tou_lookup(months_8760, hours_of_day_8760, TOU_PEAK_SAVER)
        summer_on_mask = (
            ((months_8760 >= 5) & (months_8760 <= 10))
            & ((hours_of_day_8760 >= 16) & (hours_of_day_8760 <= 20))
        )
        np.testing.assert_array_almost_equal(rates[summer_on_mask], 0.731)

    def test_summer_off_peak_rate(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        """Summer off-peak hours get $0.285."""
        rates = tou_lookup(months_8760, hours_of_day_8760, TOU_PEAK_SAVER)
        summer_off_mask = (
            ((months_8760 >= 5) & (months_8760 <= 10))
            & ~((hours_of_day_8760 >= 16) & (hours_of_day_8760 <= 20))
        )
        np.testing.assert_array_almost_equal(rates[summer_off_mask], 0.285)

    def test_winter_on_peak_rate(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        """Winter on-peak hours get $0.478."""
        rates = tou_lookup(months_8760, hours_of_day_8760, TOU_PEAK_SAVER)
        winter_on_mask = (
            ((months_8760 <= 4) | (months_8760 >= 11))
            & ((hours_of_day_8760 >= 16) & (hours_of_day_8760 <= 20))
        )
        np.testing.assert_array_almost_equal(rates[winter_on_mask], 0.478)

    def test_winter_off_peak_rate(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        """Winter off-peak hours get $0.286."""
        rates = tou_lookup(months_8760, hours_of_day_8760, TOU_PEAK_SAVER)
        winter_off_mask = (
            ((months_8760 <= 4) | (months_8760 >= 11))
            & ~((hours_of_day_8760 >= 16) & (hours_of_day_8760 <= 20))
        )
        np.testing.assert_array_almost_equal(rates[winter_off_mask], 0.286)

    def test_four_distinct_rates(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        """TOU Peak Saver has 4 distinct rate values."""
        rates = tou_lookup(months_8760, hours_of_day_8760, TOU_PEAK_SAVER)
        unique = np.unique(np.round(rates, 3))
        assert len(unique) == 4
        expected = sorted([0.285, 0.286, 0.478, 0.731])
        np.testing.assert_array_almost_equal(sorted(unique), expected)


# -------------------------------------------------------------------
# Tests for schedule builder helpers
# -------------------------------------------------------------------


class TestScheduleType:
    def test_flat_rate_is_flat(self) -> None:
        assert schedule_type(FLAT_RATE) == "flat"

    def test_eversource_is_flat(self) -> None:
        assert schedule_type(EVERSOURCE_R1HP) == "flat"

    def test_tou_peak_saver_is_tou(self) -> None:
        assert schedule_type(TOU_PEAK_SAVER) == "tou"

    def test_custom_template_is_tou(self) -> None:
        assert schedule_type(CUSTOM_TOU_TEMPLATE) == "tou"


class TestBuildFlatSchedule:
    def test_builds_valid_schedule(self) -> None:
        schedule = build_flat_schedule(0.30, 10.00)
        _build_hour_rate_map(schedule["summer"])
        _build_hour_rate_map(schedule["winter"])

    def test_rate_values(self) -> None:
        schedule = build_flat_schedule(0.25, 5.00)
        assert schedule["customer_charge"] == 5.00
        rates = _build_hour_rate_map(schedule["summer"])
        np.testing.assert_array_almost_equal(rates, 0.25)

    def test_tou_lookup_integration(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        schedule = build_flat_schedule(0.30, 10.00)
        rates = tou_lookup(months_8760, hours_of_day_8760, schedule)
        np.testing.assert_array_almost_equal(rates, 0.30)


class TestBuildSeasonalFlatSchedule:
    def test_builds_valid_schedule(self) -> None:
        schedule = build_seasonal_flat_schedule(0.30, 0.23, 10.00)
        _build_hour_rate_map(schedule["summer"])
        _build_hour_rate_map(schedule["winter"])

    def test_seasonal_rates(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        schedule = build_seasonal_flat_schedule(0.30, 0.23, 10.00)
        rates = tou_lookup(months_8760, hours_of_day_8760, schedule)
        summer_mask = (months_8760 >= 5) & (months_8760 <= 10)
        np.testing.assert_array_almost_equal(rates[summer_mask], 0.30)
        np.testing.assert_array_almost_equal(rates[~summer_mask], 0.23)


class TestBuildTouSchedule:
    def test_builds_valid_schedule(self) -> None:
        schedule = build_tou_schedule(16, 20, 0.40, 0.20, 0.35, 0.18, 10.00)
        _build_hour_rate_map(schedule["summer"])
        _build_hour_rate_map(schedule["winter"])

    def test_on_peak_hours_correct(self) -> None:
        schedule = build_tou_schedule(16, 20, 0.40, 0.20, 0.35, 0.18, 10.00)
        assert schedule["summer"]["tiers"]["on_peak"]["hours"] == [16, 17, 18, 19, 20]

    def test_off_peak_covers_remaining(self) -> None:
        schedule = build_tou_schedule(16, 20, 0.40, 0.20, 0.35, 0.18, 10.00)
        off_hours = schedule["summer"]["tiers"]["off_peak"]["hours"]
        assert sorted(off_hours) == [h for h in range(24) if h not in range(16, 21)]

    def test_invalid_start_gt_end(self) -> None:
        with pytest.raises(ValueError):
            build_tou_schedule(20, 16, 0.40, 0.20, 0.35, 0.18, 10.00)

    def test_full_day_on_peak(self) -> None:
        """All 24 hours on-peak is valid."""
        schedule = build_tou_schedule(0, 23, 0.40, 0.20, 0.35, 0.18, 10.00)
        _build_hour_rate_map(schedule["summer"])

    def test_tou_lookup_integration(
        self, months_8760: np.ndarray, hours_of_day_8760: np.ndarray
    ) -> None:
        schedule = build_tou_schedule(16, 20, 0.40, 0.20, 0.35, 0.18, 10.00)
        rates = tou_lookup(months_8760, hours_of_day_8760, schedule)
        assert rates.shape == (8760,)
        assert not np.any(np.isnan(rates))


class TestExtractHelpers:
    def test_extract_on_peak_hours_named(self) -> None:
        hours = extract_on_peak_hours(TOU_PEAK_SAVER)
        assert hours == [16, 17, 18, 19, 20]

    def test_extract_on_peak_hours_fallback(self) -> None:
        """Falls back to highest-price tier when no 'on_peak' name."""
        schedule: RateSchedule = {
            "name": "test",
            "customer_charge": 0.0,
            "summer": {
                "tiers": {
                    "high": {"price": 0.50, "hours": [12, 13, 14]},
                    "low": {"price": 0.10, "hours": list(range(12)) + list(range(15, 24))},
                },
            },
            "winter": {"tiers": {"all": {"price": 0.20, "hours": list(range(24))}}},
        }
        assert extract_on_peak_hours(schedule) == [12, 13, 14]

    def test_extract_tou_prices_named(self) -> None:
        on, off = extract_tou_prices(TOU_PEAK_SAVER, "summer")
        assert on == pytest.approx(0.731)
        assert off == pytest.approx(0.285)

    def test_extract_tou_prices_winter(self) -> None:
        on, off = extract_tou_prices(TOU_PEAK_SAVER, "winter")
        assert on == pytest.approx(0.478)
        assert off == pytest.approx(0.286)


# -------------------------------------------------------------------
# Tests for gas_rate_lookup
# -------------------------------------------------------------------


class TestGasRateLookup:
    """Tests for gas_rate_lookup seasonal gas rate mapping."""

    def test_summer_months_get_summer_rate(self, months_8760: np.ndarray) -> None:
        """Summer months (May-Oct) return the summer gas rate."""
        rates = gas_rate_lookup(months_8760, summer_rate=1.80, winter_rate=2.50)
        summer_mask = (months_8760 >= 5) & (months_8760 <= 10)
        np.testing.assert_array_almost_equal(rates[summer_mask], 1.80)

    def test_winter_months_get_winter_rate(self, months_8760: np.ndarray) -> None:
        """Winter months (Nov-Apr) return the winter gas rate."""
        rates = gas_rate_lookup(months_8760, summer_rate=1.80, winter_rate=2.50)
        winter_mask = (months_8760 <= 4) | (months_8760 >= 11)
        np.testing.assert_array_almost_equal(rates[winter_mask], 2.50)

    def test_season_boundary_april_may(self, months_8760: np.ndarray) -> None:
        """Last hour of April uses winter rate, first hour of May uses summer."""
        rates = gas_rate_lookup(months_8760, summer_rate=1.80, winter_rate=2.50)
        april_indices = np.where(months_8760 == 4)[0]
        may_indices = np.where(months_8760 == 5)[0]
        assert rates[april_indices[-1]] == pytest.approx(2.50)
        assert rates[may_indices[0]] == pytest.approx(1.80)
        assert may_indices[0] == april_indices[-1] + 1

    def test_season_boundary_october_november(self, months_8760: np.ndarray) -> None:
        """Last hour of October uses summer rate, first of November uses winter."""
        rates = gas_rate_lookup(months_8760, summer_rate=1.80, winter_rate=2.50)
        oct_indices = np.where(months_8760 == 10)[0]
        nov_indices = np.where(months_8760 == 11)[0]
        assert rates[oct_indices[-1]] == pytest.approx(1.80)
        assert rates[nov_indices[0]] == pytest.approx(2.50)
        assert nov_indices[0] == oct_indices[-1] + 1

    def test_equal_rates_produces_flat(self, months_8760: np.ndarray) -> None:
        """When summer == winter rate, all hours have the same rate."""
        rates = gas_rate_lookup(months_8760, summer_rate=2.50, winter_rate=2.50)
        np.testing.assert_array_almost_equal(rates, 2.50)

    def test_output_shape_and_dtype(self, months_8760: np.ndarray) -> None:
        """Output is (8760,) float64."""
        rates = gas_rate_lookup(months_8760, summer_rate=1.80, winter_rate=2.50)
        assert rates.shape == (8760,)
        assert rates.dtype == np.float64

    def test_two_distinct_rates(self, months_8760: np.ndarray) -> None:
        """Seasonal gas rates produce exactly two distinct values."""
        rates = gas_rate_lookup(months_8760, summer_rate=1.80, winter_rate=2.50)
        unique = np.unique(rates)
        assert len(unique) == 2
        np.testing.assert_array_almost_equal(sorted(unique), [1.80, 2.50])

    def test_small_array(self) -> None:
        """Works on small test arrays."""
        months = np.array([1, 5, 10, 11])
        rates = gas_rate_lookup(months, summer_rate=1.80, winter_rate=2.50)
        np.testing.assert_array_almost_equal(rates, [2.50, 1.80, 1.80, 2.50])

    def test_uses_summer_months_constant(self) -> None:
        """Verify alignment with SUMMER_MONTHS constant."""
        months = np.arange(1, 13)
        rates = gas_rate_lookup(months, summer_rate=1.00, winter_rate=2.00)
        for m in range(1, 13):
            expected = 1.00 if m in SUMMER_MONTHS else 2.00
            assert rates[m - 1] == pytest.approx(expected), f"Month {m} mismatch"
