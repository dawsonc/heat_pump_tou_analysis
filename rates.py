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
- FLAT_RATE, EVERSOURCE_R1HP -- preset RateSchedule dicts
- RATE_PRESETS -- name -> RateSchedule lookup
- CUSTOM_TOU_TEMPLATE -- example multi-tier schedule
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

# Approximate MA all-in residential rate (~$0.30/kWh).
# Customer charge matches Eversource R-1 ($10.00/month).
FLAT_RATE: RateSchedule = {
    "name": "Flat Rate",
    "customer_charge": 10.00,
    "summer": {
        "tiers": {
            "all": {"price": 0.30, "hours": _ALL_HOURS},
        },
    },
    "winter": {
        "tiers": {
            "all": {"price": 0.30, "hours": _ALL_HOURS},
        },
    },
}

# Eversource R-1HP seasonal heat pump rate (Eastern MA).
# Summer: standard R-1 residential all-in rate (~$0.30/kWh).
# Winter: ~$0.07/kWh delivery savings -> ~$0.23/kWh all-in.
#   Transmission: $0.04545 -> $0.01492/kWh (saves ~$0.031).
#   Distribution also reduced (additional ~$0.04 savings).
# Customer charge: $10.00/month (same as standard R-1).
# Source: Mass.gov, Eversource tariff filings.
# Note: values are approximate all-in rates; users should verify
# against their actual bills.
EVERSOURCE_R1HP: RateSchedule = {
    "name": "Eversource R-1HP",
    "customer_charge": 10.00,
    "summer": {
        "tiers": {
            "standard": {"price": 0.30, "hours": _ALL_HOURS},
        },
    },
    "winter": {
        "tiers": {
            "heat_pump": {"price": 0.23, "hours": _ALL_HOURS},
        },
    },
}

# Example multi-tier TOU schedule for the custom schedule editor.
CUSTOM_TOU_TEMPLATE: RateSchedule = {
    "name": "Custom TOU",
    "customer_charge": 10.00,
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

RATE_PRESETS: dict[str, RateSchedule] = {
    "Flat Rate": FLAT_RATE,
    "Eversource R-1HP": EVERSOURCE_R1HP,
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
