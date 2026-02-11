"""TOU (Time-of-Use) electricity rate schedule definition and lookup.

Maps every hour of the year to a $/kWh rate based on:
- Rate tiers (on-peak, off-peak, etc.) each with a price.
- Time-of-day assignments per tier.
- Seasonal variation: summer (May-Oct) vs. winter (Nov-Apr).

Provides preset schedules (Flat Rate, Eversource R-1HP) and
supports user-defined custom TOU schedules.

Public API:
- get_season(month) -> 'summer' or 'winter'
- tou_lookup(months, hours_of_day, schedule) -> (N,) rate array in $/kWh
- FLAT_RATE, EVERSOURCE_R1HP, TOU_ILLUSTRATIVE_TOU -- preset RateSchedule dicts
- RATE_PRESETS -- name -> RateSchedule lookup
- CUSTOM_TOU_TEMPLATE -- example multi-tier schedule
- schedule_type(schedule) -> 'flat' or 'tou'
- gas_rate_lookup(months, summer_rate, winter_rate) -> (N,) rate array in $/therm
- build_flat_schedule, build_seasonal_flat_schedule, build_tou_schedule
- extract_on_peak_hours, extract_tou_prices
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

class TierDef(TypedDict):
    """Definition of a single rate tier."""

    price: float       # $/kWh
    hours: list[int]   # hour-of-day values (0-23) this tier applies to


class SeasonSchedule(TypedDict):
    """Hour-of-day tier assignments for one season."""

    tiers: dict[str, TierDef]  # tier_name -> TierDef


class RateSchedule(TypedDict):
    """Complete TOU rate schedule with seasonal variation.

    Attributes:
        name: Human-readable schedule name.
        customer_charge: Monthly fixed charge in $/month.
        summer: Tier assignments for summer months (May-Oct, months 5-10).
        winter: Tier assignments for winter months (Nov-Apr, months 11,12,1-4).
    """

    name: str
    customer_charge: float  # $/month
    summer: SeasonSchedule
    winter: SeasonSchedule


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUMMER_MONTHS = {5, 6, 7, 8, 9, 10}

_ALL_HOURS = list(range(24))


# ---------------------------------------------------------------------------
# Preset schedules
# ---------------------------------------------------------------------------

# Approximate MA all-in residential rate
FLAT_RATE: RateSchedule = {
    "name": "Flat Rate",
    "customer_charge": 7.50,
    "summer": {
        "tiers": {
            "all": {"price": 0.15065 + 0.09591, "hours": _ALL_HOURS},
        },
    },
    "winter": {
        "tiers": {
            "all": {"price": 0.15065 + 0.09591, "hours": _ALL_HOURS},
        },
    },
}

# Eversource R-1HP seasonal heat pump rate (Eastern MA).
EVERSOURCE_R1HP: RateSchedule = {
    "name": "Eversource R-1HP",
    "customer_charge": 7.50,
    "summer": {
        "tiers": {
            "standard": {"price": 0.15065 + 0.17997, "hours": _ALL_HOURS},
        },
    },
    "winter": {
        "tiers": {
            "heat_pump": {"price": 0.15065 + 0.03792, "hours": _ALL_HOURS},
        },
    },
}

# Example multi-tier TOU schedule for the custom schedule editor.
CUSTOM_TOU_TEMPLATE: RateSchedule = {
    "name": "Custom TOU",
    "customer_charge": 7.50,
    "summer": {
        "tiers": {
            "on_peak": {
                "price": 0.40,
                "hours": [16, 17, 18, 19, 20],
            },
            "off_peak": {
                "price": 0.20,
                "hours": [h for h in range(24) if h not in range(16, 21)],
            },
        },
    },
    "winter": {
        "tiers": {
            "on_peak": {
                "price": 0.35,
                "hours": [16, 17, 18, 19, 20],
            },
            "off_peak": {
                "price": 0.18,
                "hours": [h for h in range(24) if h not in range(16, 21)],
            },
        },
    },
}

# Illustrative TOU — aggressive peak pricing, 4-9 PM on-peak.
_ILLUSTRATIVE_TOU_ON = [16, 17, 18, 19, 20]
_ILLUSTRATIVE_TOU_OFF = [h for h in range(24) if h not in _ILLUSTRATIVE_TOU_ON]

TOU_ILLUSTRATIVE_TOU: RateSchedule = {
    "name": "Illustrative TOU",
    "customer_charge": 10.00,
    "summer": {
        "tiers": {
            "on_peak": {"price": 0.731, "hours": _ILLUSTRATIVE_TOU_ON},
            "off_peak": {"price": 0.285, "hours": _ILLUSTRATIVE_TOU_OFF},
        },
    },
    "winter": {
        "tiers": {
            "on_peak": {"price": 0.478, "hours": _ILLUSTRATIVE_TOU_ON},
            "off_peak": {"price": 0.286, "hours": _ILLUSTRATIVE_TOU_OFF},
        },
    },
}

RATE_PRESETS: dict[str, RateSchedule] = {
    "Eversource R-1": FLAT_RATE,
    "Eversource R-1HP": EVERSOURCE_R1HP,
    "Illustrative TOU": TOU_ILLUSTRATIVE_TOU,
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def get_season(
    month: int | npt.NDArray[np.integer],
) -> str | npt.NDArray[np.str_]:
    """Map month number(s) to season label(s).

    Args:
        month: Month number (1-12) or array of month numbers.

    Returns:
        'summer' for months 5-10 (May-Oct),
        'winter' for months 1-4 and 11-12 (Nov-Apr).
        If input is an array, returns an array of the same shape.

    Raises:
        ValueError: If any month is not in 1-12.
    """
    if isinstance(month, (int, np.integer)):
        if not 1 <= int(month) <= 12:
            raise ValueError(f"month must be 1-12, got {month}")
        return "summer" if int(month) in SUMMER_MONTHS else "winter"

    month_arr = np.asarray(month)
    if np.any((month_arr < 1) | (month_arr > 12)):
        raise ValueError("All months must be in range 1-12")
    return np.where(
        (month_arr >= 5) & (month_arr <= 10), "summer", "winter"
    )


def _build_hour_rate_map(
    season_schedule: SeasonSchedule,
) -> npt.NDArray[np.float64]:
    """Build a (24,) lookup array mapping hour-of-day to $/kWh for one season.

    Validates that hours 0-23 are each assigned to exactly one tier.

    Raises:
        ValueError: If any hour is outside 0-23, assigned to multiple tiers,
            or not assigned to any tier.
    """
    rate_map = np.full(24, np.nan, dtype=np.float64)
    for tier_name, tier_def in season_schedule["tiers"].items():
        for h in tier_def["hours"]:
            if not 0 <= h <= 23:
                raise ValueError(
                    f"Tier '{tier_name}': hour {h} not in 0-23"
                )
            if not np.isnan(rate_map[h]):
                raise ValueError(
                    f"Hour {h} assigned to multiple tiers in season"
                )
            rate_map[h] = tier_def["price"]
    if np.any(np.isnan(rate_map)):
        missing = np.where(np.isnan(rate_map))[0].tolist()
        raise ValueError(f"Hours not assigned to any tier: {missing}")
    return rate_map


def tou_lookup(
    months: npt.NDArray[np.integer],
    hours_of_day: npt.NDArray[np.integer],
    schedule: RateSchedule,
) -> npt.NDArray[np.float64]:
    """Map every hour of the year to a $/kWh electricity rate.

    Uses the schedule's seasonal tier assignments to look up the rate
    for each hour based on its month (determines season) and hour-of-day
    (determines tier within the season).

    All operations are vectorized NumPy — no Python loops over hours.

    Args:
        months: (N,) array of month numbers (1-12), one per hour.
        hours_of_day: (N,) array of hour-of-day (0-23), one per hour.
        schedule: A RateSchedule dict defining seasonal tier assignments.

    Returns:
        (N,) array of $/kWh rates, dtype float64.

    Raises:
        ValueError: If months and hours_of_day have different shapes.
    """
    months = np.asarray(months)
    hours_of_day = np.asarray(hours_of_day)
    if months.shape != hours_of_day.shape:
        raise ValueError(
            f"Shape mismatch: months {months.shape} "
            f"vs hours_of_day {hours_of_day.shape}"
        )

    summer_rates = _build_hour_rate_map(schedule["summer"])  # (24,)
    winter_rates = _build_hour_rate_map(schedule["winter"])  # (24,)

    is_summer = (months >= 5) & (months <= 10)

    return np.where(
        is_summer,
        summer_rates[hours_of_day],
        winter_rates[hours_of_day],
    )


def gas_rate_lookup(
    months: npt.NDArray[np.integer],
    summer_rate: float,
    winter_rate: float,
) -> npt.NDArray[np.float64]:
    """Map every hour to a $/therm gas rate based on season.

    Uses the same seasonal definition as electricity rates:
    summer = May-Oct (months 5-10), winter = Nov-Apr (months 11,12,1-4).

    Args:
        months: (N,) array of month numbers (1-12), one per hour.
        summer_rate: Gas rate in $/therm for summer months.
        winter_rate: Gas rate in $/therm for winter months.

    Returns:
        (N,) array of $/therm rates, dtype float64.
    """
    months = np.asarray(months)
    is_summer = np.isin(months, list(SUMMER_MONTHS))
    return np.where(is_summer, summer_rate, winter_rate).astype(np.float64)


# ---------------------------------------------------------------------------
# Schedule introspection and builder helpers
# ---------------------------------------------------------------------------


def schedule_type(schedule: RateSchedule) -> str:
    """Return 'flat' if all seasons have exactly 1 tier, else 'tou'."""
    for season_key in ("summer", "winter"):
        if len(schedule[season_key]["tiers"]) > 1:
            return "tou"
    return "flat"


def extract_on_peak_hours(schedule: RateSchedule) -> list[int]:
    """Extract on-peak hours from a TOU schedule (uses summer season).

    Returns the hours of the tier named 'on_peak', or the highest-price
    tier if no tier is named 'on_peak'.
    """
    tiers = schedule["summer"]["tiers"]
    if "on_peak" in tiers:
        return sorted(tiers["on_peak"]["hours"])
    max_tier = max(tiers.values(), key=lambda t: t["price"])
    return sorted(max_tier["hours"])


def extract_tou_prices(
    schedule: RateSchedule,
    season: str,
) -> tuple[float, float]:
    """Extract (on_peak_price, off_peak_price) from a season.

    Identifies on-peak as the tier named 'on_peak' or the highest-price
    tier.  Off-peak is 'off_peak' or the lowest-price tier.
    """
    tiers = schedule[season]["tiers"]
    if "on_peak" in tiers and "off_peak" in tiers:
        return tiers["on_peak"]["price"], tiers["off_peak"]["price"]
    prices = [(name, t["price"]) for name, t in tiers.items()]
    prices.sort(key=lambda x: x[1])
    return prices[-1][1], prices[0][1]


def build_flat_schedule(
    rate: float,
    customer_charge: float,
    name: str = "Custom Flat",
) -> RateSchedule:
    """Build a flat RateSchedule with a single rate for all hours."""
    season: SeasonSchedule = {
        "tiers": {"all": {"price": rate, "hours": _ALL_HOURS}},
    }
    return {
        "name": name,
        "customer_charge": customer_charge,
        "summer": season,
        "winter": season,
    }


def build_seasonal_flat_schedule(
    summer_rate: float,
    winter_rate: float,
    customer_charge: float,
    name: str = "Custom Seasonal",
) -> RateSchedule:
    """Build a seasonal flat schedule (one rate per season)."""
    return {
        "name": name,
        "customer_charge": customer_charge,
        "summer": {"tiers": {"all": {"price": summer_rate, "hours": _ALL_HOURS}}},
        "winter": {"tiers": {"all": {"price": winter_rate, "hours": _ALL_HOURS}}},
    }


def build_tou_schedule(
    peak_start: int,
    peak_end: int,
    summer_on_peak: float,
    summer_off_peak: float,
    winter_on_peak: float,
    winter_off_peak: float,
    customer_charge: float,
    name: str = "Custom TOU",
) -> RateSchedule:
    """Build a TOU RateSchedule from peak hour range and rate values.

    On-peak hours are *peak_start* through *peak_end* (inclusive).
    All other hours are off-peak.
    """
    if peak_start > peak_end:
        raise ValueError(
            f"On-peak start ({peak_start}) must be <= end ({peak_end})"
        )
    on_peak_hours = list(range(peak_start, peak_end + 1))
    off_peak_hours = [h for h in range(24) if h not in on_peak_hours]

    def _make_season(on_price: float, off_price: float) -> SeasonSchedule:
        tiers: dict[str, TierDef] = {
            "on_peak": {"price": on_price, "hours": on_peak_hours},
        }
        if off_peak_hours:
            tiers["off_peak"] = {"price": off_price, "hours": off_peak_hours}
        return {"tiers": tiers}

    sched: RateSchedule = {
        "name": name,
        "customer_charge": customer_charge,
        "summer": _make_season(summer_on_peak, summer_off_peak),
        "winter": _make_season(winter_on_peak, winter_off_peak),
    }
    # Validate completeness
    _build_hour_rate_map(sched["summer"])
    _build_hour_rate_map(sched["winter"])
    return sched
