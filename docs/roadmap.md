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

## Phase 1: Weather Data

Download and parse real TMY3 data for Boston.

**Deliverables**:
- `data/boston_tmy3.csv` — 8,760 rows of real hourly dry-bulb temperatures from NREL TMY3 archive (station 725090, https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/tmy3/)
- `weather.py` — `load_weather()` function returning structured arrays

**Depends on**: Phase 0

**Verification**:
- `load_weather()` returns exactly 8,760 rows
- January temps plausible for Boston (~15–40°F range), July temps (~60–95°F)
- No NaN or missing values
- Spot-check against published Boston climate normals

---

## Phase 2: Presets

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

## Phase 4: Rate Schedules ✅

Implement TOU rate lookup with seasonal variation.

**Deliverables**:
- `rates.py`:
  - Rate schedule data structure: TypedDicts (`RateSchedule`, `SeasonSchedule`, `TierDef`) with $/kWh, hour-of-day assignments, seasonal mappings, and monthly customer charge ($/month)
  - `get_season(month)` — summer (May–Oct) vs. winter (Nov–Apr), scalar and vectorized
  - `tou_lookup(months, hours_of_day, schedule)` → `(N,)` rate array, fully vectorized NumPy
  - `_build_hour_rate_map(season_schedule)` — validates tier completeness (no gaps/overlaps)
  - Preset schedules: Flat Rate ($0.30/kWh), Eversource R-1HP ($0.30 summer / $0.23 winter), both with $10/month customer charge
  - `CUSTOM_TOU_TEMPLATE` — example multi-tier on/off-peak schedule
  - `RATE_PRESETS` dict for UI dropdown population
- `tests/test_rates.py` — 45 tests (7 classes): season boundary (Apr 30 → May 1), midnight tier transitions, flat rate uniformity, multi-tier TOU, preset integrity, customer charge validation
- `tests/conftest.py` — shared fixtures (`months_8760`, `hours_of_day_8760`) reusable by future phases

**Depends on**: Phase 0

**Verification** (all passing):
- 45 `test_rates.py` tests pass (`pytest tests/test_rates.py -v`)
- Eversource R-1HP rates based on published tariff: ~$0.07/kWh winter delivery savings (transmission $0.04545→$0.01492 + distribution reduction)
- `tou_lookup` returns correct rates for known month/hour combos, verified at season boundaries

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
