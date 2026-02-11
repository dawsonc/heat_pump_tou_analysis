"""Shared pytest fixtures for the Heat Pump TOU Calculator test suite.

Fixtures:
- sample_weather_df: minimal 24-row weather DataFrame for unit tests
- boston_weather_df: real Boston TMY3 data (skips if CSV not present)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def months_8760() -> np.ndarray:
    """(8760,) array of month numbers (1-12) for a standard non-leap year.

    January has 744 hours (31 days * 24), February 672, etc.
    """
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return np.concatenate([
        np.full(d * 24, m, dtype=np.int32)
        for m, d in enumerate(days_per_month, start=1)
    ])


@pytest.fixture
def hours_of_day_8760() -> np.ndarray:
    """(8760,) array of hour-of-day (0-23) repeating for 365 days."""
    return np.tile(np.arange(24, dtype=np.int32), 365)

@pytest.fixture
def sample_weather_df():
    """A minimal 24-row weather DataFrame for unit testing."""
    return pd.DataFrame({
        "hour": list(range(24)),
        "month": [1] * 24,
        "day": [15] * 24,
        "hour_of_day": list(range(24)),
        "T_drybulb_F": [20.0 + i * 0.5 for i in range(24)],
    })


@pytest.fixture
def boston_weather_df():
    """Load real Boston TMY3 weather data (integration test fixture).

    Skips if the data file is not present.
    """
    csv_path = Path(__file__).parent.parent / "data" / "boston_tmy3.csv"
    if not csv_path.exists():
        pytest.skip("Boston TMY3 data not available (run scripts/fetch_tmy3.py)")
    from weather import load_weather
    return load_weather()
