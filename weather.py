"""TMY3 weather data loading and parsing for Boston Logan (USAF 725090).

Loads pre-processed CSV from data/boston_tmy3.csv containing 8,760 hourly
dry-bulb temperatures. Architecture supports additional locations.

Data source: NREL NSRDB TMY3 archive
https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/tmy3/

Expected CSV columns: hour, month, day, hour_of_day, T_drybulb_F

Key functions:
- load_weather: read CSV and return structured data as a DataFrame
"""

from pathlib import Path

import pandas as pd

HOURS_PER_YEAR = 8760

_DATA_DIR = Path(__file__).parent / "data"
_BOSTON_CSV = _DATA_DIR / "boston_tmy3.csv"

_REQUIRED_COLUMNS = ["hour", "month", "day", "hour_of_day", "T_drybulb_F"]
_DTYPES = {
    "hour": int,
    "month": int,
    "day": int,
    "hour_of_day": int,
    "T_drybulb_F": float,
}


def load_weather(csv_path: Path | None = None) -> pd.DataFrame:
    """Load pre-processed TMY3 weather data from CSV.

    Parameters
    ----------
    csv_path : Path or None, optional
        Path to the weather CSV file. Defaults to data/boston_tmy3.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hour, month, day, hour_of_day, T_drybulb_F

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the data fails validation.
    """
    if csv_path is None:
        csv_path = _BOSTON_CSV

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Weather data file not found: {csv_path}\n"
            f"Run 'python scripts/fetch_tmy3.py' to download and process "
            f"the TMY3 data."
        )

    df = pd.read_csv(csv_path, dtype=_DTYPES)
    _validate(df, csv_path)
    return df


def _validate(df: pd.DataFrame, csv_path: Path) -> None:
    """Validate weather DataFrame structure and values."""
    missing = set(_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Weather file {csv_path} missing columns: {missing}. "
            f"Expected: {_REQUIRED_COLUMNS}"
        )

    if len(df) != HOURS_PER_YEAR:
        raise ValueError(
            f"Weather file {csv_path} has {len(df)} rows, "
            f"expected {HOURS_PER_YEAR}."
        )

    nan_counts = df[_REQUIRED_COLUMNS].isna().sum()
    if nan_counts.any():
        bad = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(
            f"Weather file {csv_path} contains NaN values: {bad}"
        )

    if not df["month"].between(1, 12).all():
        raise ValueError("month values must be in range 1-12")
    if not df["hour_of_day"].between(0, 23).all():
        raise ValueError("hour_of_day values must be in range 0-23")
    if not df["hour"].between(0, HOURS_PER_YEAR - 1).all():
        raise ValueError(f"hour values must be in range 0-{HOURS_PER_YEAR - 1}")
