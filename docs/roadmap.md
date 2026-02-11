# Implementation Roadmap

## Dependency Diagram

```
Phase 0 (Setup)
    │
    ├──→ Phase 1 (Weather) ──┐
    │                         │
    ├──→ Phase 2 (Presets) ──┼──→ Phase 3 (Model) ──┐
    │                                                 │
    ├──→ Phase 4 (Rates) ────────────────────────────┼──→ Phase 5 (Cost Integration)
                                                      │
                                                      └──→ Phase 6 (Results UI)
                                                              │
                                                              └──→ Phase 7 (Diagnostics)
                                                                      │
                                                                      └──→ Phase 8 (Polish)
```

---

## Data Sourcing Principle

**All data must come from real external sources. No placeholder, synthetic, or generated data.**

- **Weather**: actual TMY3 data for Boston Logan (USAF 725090) from the NREL NSRDB archive.
- **COP reference points**: from NEEP ccASHP Product List, PNNL-37127 field data, and NREL mini-split test data.
- **Electricity rates**: actual Eversource R-1HP tariff; flat rate based on real MA residential average.
- **Gas prices**: real MA residential gas rates from EIA / Mass.gov.
- **Building UA values**: derived from published heat-loss-per-square-foot data (sourced in spec).

---

## Phase 0: Project Setup

Scaffolding, configuration, and module stubs.

**Deliverables**:
- `.gitignore`, `requirements.txt`, `pyproject.toml`
- `README.md`
- Module stubs: `app.py`, `model.py`, `rates.py`, `weather.py`, `presets.py`, `diagnostics.py`
- Test stubs: `tests/conftest.py`, `tests/test_model.py`, `tests/test_rates.py`, `tests/test_gas.py`
- `data/README.md` documenting expected weather file format and source

**Depends on**: nothing

**Verification**:
- `pip install -r requirements.txt` succeeds
- `pytest` runs (0 tests collected)
- All modules importable
- `streamlit run app.py` launches with placeholder page

---

## Phase 1: Weather Data ✅

Download and parse real TMY3 data for Boston.

**Deliverables**:
- `data/boston_tmy3.csv` — 8,760 rows of real hourly dry-bulb temperatures from NREL TMY3 archive (station 725090, https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/tmy3/)
- `weather.py` — `load_weather()` function returning structured arrays
- `scripts/fetch_tmy3.py` — reproducible download/conversion script (fetches from gridlab-d mirror of NREL archive)
- `tests/test_weather.py` — 15 tests (structure, temperature plausibility, error handling)

**Depends on**: Phase 0

**Verification** (all passing):
- `load_weather()` returns exactly 8,760 rows
- January mean 26.6°F, July mean 74.1°F — consistent with NOAA 1991–2020 normals
- Annual range −4.0°F to 99.0°F — plausible for Boston
- No NaN or missing values
- 15/15 pytest tests pass

---

## Phase 2: Presets ✅

Define all preset data structures from spec tables and cited sources.

**Deliverables**:
- `presets.py`:
  - `HP_PRESETS` — COP reference points at 47°F / 17°F / 5°F, lockout temps, capacity curves (from NEEP data)
  - `BUILDING_PRESETS` — size × insulation UA matrix (from spec Table)
  - `GAS_FURNACE_PRESETS` — AFUE values (96%, 92%, 80%)

**Depends on**: Phase 0

**Verification**:
- Preset values match spec tables exactly
- COP values traceable to cited NEEP / PNNL / NREL sources
- All presets importable and well-structured

---

## Phase 3: Core Thermal Model

Implement the computation pipeline from the spec.

**Deliverables**:
- `model.py`:
  - `compute_heating_load(T_out, UA, T_set_heat)` → `(8760,)` array
  - `compute_cooling_load(T_out, UA, T_set_cool)` → `(8760,)` array
  - `compute_cop(T_out, cop_curve, lockout_temp)` — piecewise-linear interpolation
  - `compute_capacity(T_out, capacity_curve)` — heating capacity derating
  - `compute_hp_energy(heat_load, cool_load, cop, cop_cool, capacity, lockout_mask)` → kWh arrays
  - `compute_gas_energy(heat_load, afue)` → therms array
- `tests/test_model.py` — edge cases: sub-zero temps, lockout, capacity derating, zero-load, AFUE = 100%

**Depends on**: Phase 1 (weather for integration tests), Phase 2 (presets for defaults)

**Verification**:
- All `test_model.py` tests pass
- Vectorized NumPy only — no Python for-loops over hours
- Each formula has a comment citing the spec Computation Pipeline section

---

## Phase 4: Rate Schedules

Implement TOU rate lookup with seasonal variation.

**Deliverables**:
- `rates.py`:
  - Rate schedule data structure: tiers with $/kWh, hour-of-day assignments, seasonal mappings
  - `get_season(month)` — summer (May–Oct) vs. winter (Nov–Apr)
  - `tou_lookup(months, hours_of_day, schedule)` → `(8760,)` rate array
  - Preset schedules: Flat Rate ($0.30/kWh), Eversource R-1HP (from actual tariff)
  - Custom TOU structure support
- `tests/test_rates.py` — edge cases: season boundary (Apr 30 → May 1), midnight transitions, flat rate uniformity

**Depends on**: Phase 0

**Verification**:
- All `test_rates.py` tests pass
- Eversource R-1HP rates match published tariff values
- `tou_lookup` returns correct rates for known month/hour combos

---

## Phase 5: Cost Integration

Wire model and rate modules together into end-to-end cost calculation.

**Deliverables**:
- Cost calculation: `cost_elec[h] = kwh_total[h] × rate[h]`
- Gas cost calculation: `cost_gas[h] = therms[h] × gas_rate`
- Aggregation functions: monthly totals (cost, kWh, therms), annual totals
- `tests/test_gas.py` — gas edge cases: zero load → zero therms, AFUE sanity checks

**Depends on**: Phase 3, Phase 4

**Verification**:
- All tests pass (test_model, test_rates, test_gas)
- End-to-end on TMY data produces plausible annual costs for a medium/average home
- HP heating cost and gas heating cost are in reasonable ranges

---

## Phase 6: Results UI (Tab 1)

Build the main Streamlit interface.

**Deliverables**:
- `app.py` Tab 1:
  - Sidebar: rate schedule selector, building size/insulation, HP preset, gas furnace inputs
  - Summary metric cards: annual HP cost, gas cost, savings, total kWh, backup share
  - Monthly heating cost comparison bar chart (Plotly)
  - Monthly electricity breakdown stacked bar chart
  - Hourly cost heatmap (24h × 365d)

**Depends on**: Phase 5

**Verification**:
- App runs, sidebar controls update charts
- Changing building size/insulation produces proportional cost changes
- Changing HP preset changes backup resistance share
- Charts render correctly with Plotly

---

## Phase 7: Advanced Diagnostics (Tab 2)

Daily drill-down for model verification.

**Deliverables**:
- `diagnostics.py` — data extraction and Plotly figure generation
- `app.py` Tab 2:
  - Date picker (month/day)
  - Daily plots: temperature, thermal load, COP, energy consumption, hourly cost
  - Downloadable hourly data table (CSV)
  - COP curve plot with TMY hour histogram
  - Load duration curve with HP capacity overlay

**Depends on**: Phase 6

**Verification**:
- Selecting a cold winter day shows high heating load, low COP, potential backup use
- Selecting a mild day shows zero or low load
- Selecting a hot summer day shows cooling load
- CSV download works
- COP curve matches preset reference points

---

## Phase 8: Polish and Documentation

Final quality pass.

**Deliverables**:
- Function-level docstrings citing spec sections
- Inline formula comments referencing Computation Pipeline
- Assumptions documented in app UI (from spec Key Assumptions section)
- Finalize `README.md` with usage instructions
- Full test suite green

**Depends on**: Phase 7

**Verification**:
- `pytest` — all tests pass
- App runs end-to-end without errors
- All spec assumptions visible in the UI

---

## Stretch Goals (Out of Scope for v1)

These are deferred but the architecture should accommodate them. See spec § Stretch Goals:

1. Load-shifting simulator (thermal mass pre-heat/pre-cool)
2. Rate plan optimizer (breakeven TOU differential)
3. Export (CSV hourly results, PDF summary)
4. Comparison mode (two HP presets or two rate schedules side-by-side)
5. Additional weather locations (Worcester, Springfield, Cape Cod)
6. Demand charges and weekend/holiday overrides
7. Gas block rates and fixed charges
8. Oil furnace comparison
