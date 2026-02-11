"""Tests for weather.py: TMY3 data loading and validation.

Tests cover:
- load_weather() returns correct shape and columns
- Validation catches malformed data
- Boston TMY3 temperature ranges are plausible
- No NaN or missing values
"""

import numpy as np
import pandas as pd
import pytest

from weather import HOURS_PER_YEAR, load_weather


class TestLoadWeatherStructure:
    """Test that load_weather returns correctly structured data."""

    def test_returns_dataframe(self, boston_weather_df):
        assert isinstance(boston_weather_df, pd.DataFrame)

    def test_row_count(self, boston_weather_df):
        assert len(boston_weather_df) == HOURS_PER_YEAR

    def test_columns_present(self, boston_weather_df):
        expected = {"hour", "month", "day", "hour_of_day", "T_drybulb_F"}
        assert expected == set(boston_weather_df.columns)

    def test_hour_index_sequential(self, boston_weather_df):
        np.testing.assert_array_equal(
            boston_weather_df["hour"].values,
            np.arange(HOURS_PER_YEAR),
        )

    def test_no_nan_values(self, boston_weather_df):
        assert boston_weather_df.isna().sum().sum() == 0

    def test_month_range(self, boston_weather_df):
        assert boston_weather_df["month"].min() == 1
        assert boston_weather_df["month"].max() == 12

    def test_hour_of_day_range(self, boston_weather_df):
        assert boston_weather_df["hour_of_day"].min() == 0
        assert boston_weather_df["hour_of_day"].max() == 23

    def test_dtypes(self, boston_weather_df):
        assert boston_weather_df["hour"].dtype in (np.int64, np.int32)
        assert boston_weather_df["month"].dtype in (np.int64, np.int32)
        assert boston_weather_df["day"].dtype in (np.int64, np.int32)
        assert boston_weather_df["hour_of_day"].dtype in (np.int64, np.int32)
        assert boston_weather_df["T_drybulb_F"].dtype == np.float64


class TestBostonTemperatures:
    """Validate Boston TMY3 temperatures against published climate normals.

    Boston Logan 1991-2020 normals (NOAA):
    - January average: ~30 F (range roughly 15-40 F)
    - July average: ~74 F (range roughly 60-95 F)
    - Annual min: typically around -5 to 10 F
    - Annual max: typically around 90-100 F
    """

    def test_january_temp_range(self, boston_weather_df):
        jan = boston_weather_df[boston_weather_df["month"] == 1]["T_drybulb_F"]
        assert jan.min() >= -10.0, f"Jan min {jan.min()} unreasonably low"
        assert jan.max() <= 65.0, f"Jan max {jan.max()} unreasonably high"
        assert 20.0 <= jan.mean() <= 40.0, f"Jan mean {jan.mean()} outside expected"

    def test_july_temp_range(self, boston_weather_df):
        jul = boston_weather_df[boston_weather_df["month"] == 7]["T_drybulb_F"]
        assert jul.min() >= 45.0, f"Jul min {jul.min()} unreasonably low"
        assert jul.max() <= 105.0, f"Jul max {jul.max()} unreasonably high"
        assert 65.0 <= jul.mean() <= 85.0, f"Jul mean {jul.mean()} outside expected"

    def test_annual_temp_range(self, boston_weather_df):
        temps = boston_weather_df["T_drybulb_F"]
        assert temps.min() >= -20.0, "Annual min below -20 F is implausible"
        assert temps.max() <= 110.0, "Annual max above 110 F is implausible"

    def test_each_month_has_data(self, boston_weather_df):
        for m in range(1, 13):
            count = (boston_weather_df["month"] == m).sum()
            assert count > 0, f"Month {m} has no data"
            # Each month: between 28*24=672 and 31*24=744 hours
            assert 672 <= count <= 744, f"Month {m} has {count} hours"


class TestLoadWeatherErrors:
    """Test error handling for missing or malformed data."""

    def test_missing_file_raises_file_not_found(self, tmp_path):
        fake_path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError, match="Weather data file not found"):
            load_weather(csv_path=fake_path)

    def test_wrong_row_count_raises_value_error(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        df = pd.DataFrame({
            "hour": range(100),
            "month": [1] * 100,
            "day": [1] * 100,
            "hour_of_day": [0] * 100,
            "T_drybulb_F": [32.0] * 100,
        })
        df.to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="8760"):
            load_weather(csv_path=bad_csv)

    def test_missing_column_raises_value_error(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        df = pd.DataFrame({
            "hour": range(8760),
            "month": [1] * 8760,
            "day": [1] * 8760,
            "hour_of_day": [0] * 8760,
        })
        df.to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_weather(csv_path=bad_csv)
