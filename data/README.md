# Weather Data

This directory contains pre-processed TMY3 weather data.

## Expected File

**`boston_tmy3.csv`** — 8,760 rows of hourly weather data for Boston Logan
International Airport (USAF station 725090).

### Columns

| Column | Type | Description |
|---|---|---|
| hour | int | Hour index 0-8759 |
| month | int | Month 1-12 |
| day | int | Day of month 1-31 |
| hour_of_day | int | Hour 0-23 |
| T_drybulb_F | float | Dry-bulb temperature in °F |

### Source

NREL NSRDB TMY3 archive (station 725090):
https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/tmy3/

The CSV is produced by downloading the real TMY3 file for Boston Logan and
extracting the hourly dry-bulb temperature column. See Phase 1 of
[`docs/roadmap.md`](../docs/roadmap.md) for details.
