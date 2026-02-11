"""Daily drill-down data extraction and diagnostic plot generation.

Supports the Advanced Diagnostics tab by extracting 24-hour slices
from the full 8,760-hour computation results and generating Plotly
figures for temperature, load, COP, energy, and cost.

Key functions (to be implemented):
- extract_day_data: slice hourly arrays for a given month/day
- plot_daily_temperature: outdoor temp with setpoint lines
- plot_daily_load: heating and cooling load area plots
- plot_daily_cop: COP with lockout threshold
- plot_daily_energy: HP kWh + backup kWh stacked, gas therms
- plot_daily_cost: HP vs gas cost with TOU tier background
- plot_cop_curve: COP vs outdoor temp with TMY histogram
- plot_load_duration: sorted heating load with HP capacity overlay
"""
