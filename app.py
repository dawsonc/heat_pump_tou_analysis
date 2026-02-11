"""Streamlit UI for the Heat Pump TOU Electricity Cost Calculator.

Provides two tabs:
- Results: summary metrics, monthly cost comparison, electricity breakdown, cost heatmap.
- Advanced Diagnostics: daily drill-down with hour-by-hour plots.

Sidebar controls allow selection of rate schedule, building parameters,
heat pump preset, and gas furnace comparison settings.
"""

import streamlit as st

st.set_page_config(page_title="Heat Pump TOU Calculator", layout="wide")
st.title("Heat Pump TOU Electricity Cost Calculator")
st.info("Application under construction. See docs/spec.md for the full specification.")
