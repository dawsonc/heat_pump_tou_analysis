# Heat Pump TOU Electricity Cost Calculator

An interactive Streamlit app that lets users define custom time-of-use (TOU)
electricity rate schedules with seasonal variation and estimate the annual
electricity cost of operating a cold-climate air-source heat pump for heating
and cooling in Massachusetts. Compares heat pump heating costs against a natural
gas furnace baseline.

See [`docs/spec.md`](docs/spec.md) for the full application specification and
[`docs/roadmap.md`](docs/roadmap.md) for the implementation plan.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Running Tests

```bash
pytest
```
