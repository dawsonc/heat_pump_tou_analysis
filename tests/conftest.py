"""Shared pytest fixtures for the Heat Pump TOU Calculator test suite."""

import numpy as np
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
