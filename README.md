# Heat Pump TOU Electricity Cost Calculator

An interactive Streamlit app that lets users define custom time-of-use (TOU)
electricity rate schedules with seasonal variation and estimate the annual
electricity cost of operating a cold-climate air-source heat pump for heating
and cooling in Massachusetts. Compares heat pump heating costs against a natural
gas furnace baseline.

See [`docs/spec.md`](docs/spec.md) for the full application specification and
[`docs/roadmap.md`](docs/roadmap.md) for the implementation plan.

## Quick Start

### Prerequisites

- Python 3.10+

### Install dependencies

```bash
pip install -r requirements.txt
```

This installs Streamlit, NumPy, Pandas, Plotly, and pytest.

### Run the app

```bash
streamlit run app.py
```

The app opens in your browser (default `http://localhost:8501`). Use the sidebar
to select a rate schedule, building parameters, heat pump preset, and gas
furnace comparison settings. Results update instantly as you change inputs.

### App overview

**Tab 1 — Results**: Summary metric cards (annual HP cost, gas cost, savings,
kWh, backup share), monthly heating cost comparison bar chart, monthly
electricity breakdown stacked bar chart, and an hourly cost heatmap.

**Tab 2 — Advanced Diagnostics**: Pick any day of the year to see hour-by-hour
temperature, thermal load, COP, energy consumption, and cost plots. Download
the hourly data as CSV. View the COP curve with Boston TMY temperature
histogram and the load duration curve with HP capacity overlay.

## Running Tests

```bash
pytest
```

All 154 tests cover the thermal model, rate schedules, gas calculations, cost
integration, and weather data loading.
