#!/usr/bin/env python3
"""Download and convert NREL TMY3 data for Boston Logan (USAF 725090).

Fetches the raw TMY3 CSV from the gridlab-d GitHub mirror of NREL's NSRDB
archive and produces a simplified CSV with hourly dry-bulb temperatures in
Fahrenheit, ready for use by weather.load_weather().

Data provenance:
    Original source: NREL NSRDB TMY3 archive (station 725090)
    https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/tmy3/
    Mirror: https://github.com/gridlab-d/data/tree/master/US/tmy3

Usage:
    python scripts/fetch_tmy3.py
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

# Mirror of NREL TMY3 data hosted on GitHub (original NREL site returns 403)
_MA_ZIP_URL = "https://raw.githubusercontent.com/gridlab-d/data/master/US/tmy3/MA.zip"
_STATION_ID = "725090"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_CSV = _PROJECT_ROOT / "data" / "boston_tmy3.csv"


def download_ma_zip() -> bytes:
    """Download the Massachusetts TMY3 zip archive."""
    print(f"Downloading {_MA_ZIP_URL} ...")
    with urllib.request.urlopen(_MA_ZIP_URL, timeout=60) as resp:
        data = resp.read()
    print(f"Downloaded {len(data):,} bytes.")
    return data


def extract_station_csv(zip_data: bytes) -> str:
    """Extract the Boston Logan TMY3 file from the MA zip archive.

    The gridlab-d archive names files by city (e.g. MA-Boston_Logan_Intl_Arpt.tmy3)
    rather than by USAF station ID. We search for 'Boston' in the filename.
    """
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        names = zf.namelist()
        # Try station ID first, then city name
        matches = [n for n in names if _STATION_ID in n]
        if not matches:
            matches = [n for n in names if "boston" in n.lower()]
        if not matches:
            raise FileNotFoundError(
                f"Boston station not found in archive. "
                f"Available files: {names}"
            )
        chosen = matches[0]
        print(f"Extracting {chosen} from archive ...")
        return zf.read(chosen).decode("latin-1")


def parse_tmy3(csv_text: str) -> pd.DataFrame:
    """Parse raw TMY3 CSV text into a simplified weather DataFrame.

    TMY3 format:
        Row 1: location metadata (station ID, name, lat, lon, etc.)
        Row 2: column headers
        Rows 3-8762: hourly data (8,760 rows)

    Key columns used:
        Date (MM/DD/YYYY), Time (HH:MM), Dry-bulb (C)

    TMY3 hour convention: Time runs 01:00-24:00 (not 00:00-23:00).
        01:00 = midnight-to-1AM (hour_of_day 0)
        24:00 = 11PM-to-midnight (hour_of_day 23)
    """
    raw = pd.read_csv(io.StringIO(csv_text), skiprows=1)

    # Strip whitespace from column names (some TMY3 files have trailing spaces)
    raw.columns = raw.columns.str.strip()

    # Find the dry-bulb temperature column (case-insensitive search)
    drybulb_col = None
    for col in raw.columns:
        if "dry-bulb" in col.lower() or "dry bulb" in col.lower():
            drybulb_col = col
            break
    if drybulb_col is None:
        raise ValueError(
            f"Could not find dry-bulb temperature column. "
            f"Available columns: {list(raw.columns)}"
        )

    # Find date/time columns
    date_col = [c for c in raw.columns if "date" in c.lower()][0]
    time_col = [c for c in raw.columns if "time" in c.lower()][0]

    # Parse month and day from date string "MM/DD/YYYY"
    date_parts = raw[date_col].astype(str).str.split("/")
    months = date_parts.str[0].astype(int)
    days = date_parts.str[1].astype(int)

    # Parse hour_of_day from time string "HH:MM"
    # TMY3 uses 1-24; convert to 0-23
    time_hours = raw[time_col].astype(str).str.split(":").str[0].astype(int)
    hour_of_day = time_hours - 1  # 1->0, 2->1, ..., 24->23

    # Convert dry-bulb from Celsius to Fahrenheit
    drybulb_c = raw[drybulb_col].astype(float)
    drybulb_f = (drybulb_c * 9.0 / 5.0 + 32.0).round(1)

    output = pd.DataFrame({
        "hour": range(len(raw)),
        "month": months.values,
        "day": days.values,
        "hour_of_day": hour_of_day.values,
        "T_drybulb_F": drybulb_f.values,
    })

    return output


def verify(df: pd.DataFrame) -> None:
    """Print verification summary and check basic invariants."""
    print(f"\n--- Verification ---")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"NaN count: {df.isna().sum().sum()}")
    print(f"Annual temp range: {df['T_drybulb_F'].min():.1f} F to "
          f"{df['T_drybulb_F'].max():.1f} F")
    print()

    for m in range(1, 13):
        mdf = df[df["month"] == m]
        print(f"  Month {m:2d}: mean={mdf['T_drybulb_F'].mean():5.1f} F, "
              f"min={mdf['T_drybulb_F'].min():5.1f} F, "
              f"max={mdf['T_drybulb_F'].max():5.1f} F, "
              f"hours={len(mdf)}")

    # Basic sanity checks
    assert len(df) == 8760, f"Expected 8760 rows, got {len(df)}"
    assert df.isna().sum().sum() == 0, "Found NaN values"
    jan_mean = df[df["month"] == 1]["T_drybulb_F"].mean()
    jul_mean = df[df["month"] == 7]["T_drybulb_F"].mean()
    assert 15 <= jan_mean <= 45, f"Jan mean {jan_mean:.1f} F outside plausible range"
    assert 60 <= jul_mean <= 85, f"Jul mean {jul_mean:.1f} F outside plausible range"
    print("\nAll checks passed.")


def main() -> None:
    zip_data = download_ma_zip()
    csv_text = extract_station_csv(zip_data)
    df = parse_tmy3(csv_text)
    verify(df)

    _OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUTPUT_CSV, index=False)
    print(f"\nWrote {_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
