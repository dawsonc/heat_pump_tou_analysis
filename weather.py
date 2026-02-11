"""TMY3 weather data loading and parsing for Boston Logan (USAF 725090).

Loads pre-processed CSV from data/boston_tmy3.csv containing 8,760 hourly
dry-bulb temperatures. Architecture supports additional locations.

Data source: NREL NSRDB TMY3 archive
https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/tmy3/

Expected CSV columns: hour, month, day, hour_of_day, T_drybulb_F

Key functions (to be implemented):
- load_weather: read CSV and return structured data (DataFrame or dict of arrays)
"""
