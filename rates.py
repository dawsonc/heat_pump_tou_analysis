"""TOU (Time-of-Use) electricity rate schedule definition and lookup.

Maps every hour of the year to a $/kWh rate based on:
- Rate tiers (on-peak, off-peak, etc.) each with a price.
- Time-of-day assignments per tier.
- Seasonal variation: summer (May-Oct) vs. winter (Nov-Apr).

Provides preset schedules (Flat Rate, Eversource R-1HP) and
supports user-defined custom TOU schedules.

Key functions (to be implemented):
- tou_lookup: given month and hour_of_day arrays, return (8760,) rate array
- get_season: map month to 'summer' or 'winter'
"""
