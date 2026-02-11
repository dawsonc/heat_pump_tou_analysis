# Heat Pump TOU Electricity Cost Calculator — App Spec

## Purpose

Build an interactive Streamlit app that lets a user define custom time-of-use (TOU) electricity rate schedules — with seasonal variation — and estimate the annual electricity cost of operating a cold-climate air-source heat pump for heating (winter) and cooling (summer) in Massachusetts. The tool compares heat pump heating costs against a natural gas furnace baseline, and helps users evaluate TOU rate structures.

---

## Design Principles

1. **Calculations must be transparent and reviewable.** Every intermediate quantity (thermal load, COP, kWh, cost) should be computable from clearly defined inputs and formulae. No hidden corrections or fudge factors.
2. **An Advanced tab provides drill-down diagnostics.** Users should be able to select any day of the year and see hour-by-hour plots of temperature, heating/cooling load, COP, electricity and gas consumption, and cost — to sanity-check the model against intuition.
3. **Cold-climate heat pumps are the default.** The app targets Massachusetts; the default heat pump preset must reflect a NEEP-listed cold-climate unit (COP ≥ 1.75 at 5°F).

---

## Core Concepts

### Time-of-Use Rate Schedule

A TOU schedule maps every hour of the year to a $/kWh rate. The user defines:

- **Rate tiers** (e.g. on-peak, off-peak, super-off-peak), each with a name and $/kWh price.
- **Time-of-day assignments**: for each tier, the hours it applies (e.g. on-peak = 16:00–21:00).
- **Seasonal variation** (must-have): separate hour→tier mappings for summer (May–Oct) and winter (Nov–Apr). This is essential because Massachusetts utilities already define seasonal rate structures (e.g. Eversource's R-1HP heat pump rate lowers distribution and transmission charges Nov–Apr only).

**Out of scope for v1**: demand charges ($/kW), weekend/holiday overrides. The data model should accommodate these as future extensions (e.g. a `day_type` field in the rate lookup that defaults to "all").

Provide preset rate schedules:

- **Flat rate**: single price all hours, all seasons. Default to ~$0.30/kWh (approximate MA all-in residential rate).
- **Eversource R-1HP (seasonal heat pump rate)**: lower delivery rate Nov–Apr, standard rate May–Oct. See reference below.
- **Custom TOU**: user-defined tiers and hour blocks per season.

**Rate references**:
- Eversource Heat Pump Rate page: https://www.eversource.com/residential/account-billing/manage-bill/about-your-bill/rates-tariffs/heat-pump-rate
- Mass.gov overview of seasonal heat pump rates: https://www.mass.gov/info-details/residential-electric-seasonal-heat-pump-rates
- Eversource delivery rate tables: https://www.eversource.com/residential/account-billing/manage-bill/about-your-bill/rates-tariffs/electric-delivery-rates/ema

### Natural Gas Furnace Model (Heating Cost Comparison)

To give users a meaningful cost comparison, model a natural gas furnace alongside the heat pump. This comparison applies **to heating costs only** — cooling is electric-only in both scenarios.

**Inputs**:
- **Gas rate ($/therm)**: user-configurable. Default: $2.50/therm (approximate MA all-in residential rate as of winter 2024–25, including supply ~$0.85–0.95/therm and delivery ~$1.50–1.60/therm). This is a single blended rate; users who know their exact supply + delivery components can sum them.
- **Furnace AFUE (%)**: annual fuel utilisation efficiency. Provide presets:
  - High-efficiency condensing: 96% AFUE
  - Standard high-efficiency: 92% AFUE
  - Older furnace: 80% AFUE (default — represents the most common installed base)
- **Monthly customer/fixed charge** for gas service: user-configurable, default $0 (to keep the comparison focused on variable heating cost; note this simplification in the UI).

**Calculation** (per hour, heating mode only):

```
gas_therms = heating_load_btu / (AFUE × 100,000)
gas_cost   = gas_therms × gas_rate_per_therm
```

Where 100,000 BTU = 1 therm. This runs on the same hourly heating load as the heat pump model, making the comparison apples-to-apples on the same building and weather.

**What this comparison shows**: annual and monthly heating cost for heat pump electricity vs. gas furnace, using the same building load profile. It does not include equipment cost, maintenance, or non-heating energy use.

**Gas rate references**:
- Mass.gov gas supply and delivery charges: https://www.mass.gov/info-details/information-on-gas-supply-and-delivery-charges
- EIA Massachusetts residential gas price history: https://www.eia.gov/dnav/ng/hist/n3010ma3m.htm
- National Grid MA gas service rates: https://www.nationalgridus.com/MA-Gas-Home/Service-Rates/

### Heat Pump Model

Use a simplified but physically grounded model:

- **Inputs**: rated heating capacity (BTU/h), rated cooling capacity (BTU/h), rated COP at reference outdoor temps, auxiliary/backup heat type (electric resistance or none).
- **COP degradation curve**: COP drops as outdoor temp diverges from rated conditions. Use a piecewise-linear approximation based on published test data. Below a user-configurable **lockout temperature**, the heat pump shuts off and backup electric resistance (COP = 1.0) covers the full load.
- **Cooling mode**: COP also degrades at very high outdoor temps but less dramatically; a simpler linear model is acceptable.
- **Capacity derating**: heating capacity also falls at low temps. Below the rated capacity at a given outdoor temp, any unmet load is covered by backup resistance heat.

**The default preset must be a cold-climate heat pump.** Provide presets:

| Preset | Heating COP @47°F | Heating COP @17°F | Heating COP @5°F | Lockout | Notes |
|---|---|---|---|---|---|
| **Cold-climate (Hyper-Heat class)** ★ default | 3.5 | 2.5 | 1.75 | −15°F | EVI compressor, NEEP-listed |
| Standard cold-climate | 3.3 | 2.2 | 1.5 | 0°F | Typical inverter-driven |
| Baseline (non-cold-climate) | 3.0 | 1.8 | — | 15°F | Switches to resistance below lockout |

Intermediate COP values should be linearly interpolated between these reference points. Below the lowest reference point, hold COP constant until lockout.

**COP data references** (use for validation and to populate presets):

- **NEEP Cold Climate ASHP Product List** — searchable database with COP at 47°F, 17°F, and 5°F for 40,000+ systems: https://ashp.neep.org/
- **NEEP ccASHP Specification v4.0** — defines cold-climate minimum: COP ≥ 1.75 at 5°F, capacity ≥ 70% of 47°F rating: https://neep.org/heating-electrification/ccashp-specification-product-list
- **DOE Cold Climate Heat Pump Technology Challenge field results** (PNNL-37127) — measured COP by outdoor air temp bin: https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-37127.pdf
- **NREL: Empirical COP model for ductless mini-splits** — analytical function COP(T_outdoor, load fraction) from cold-chamber tests: https://docs.nrel.gov/docs/fy23osti/85081.pdf
- **ACEEE 2024: Cold Climate Heat Pumps in the Field** — COP ranges 1.0–2.5 at 5°F to −22°F across ten field studies: https://www.aceee.org/sites/default/files/proceedings/ssb24/pdfs/Rising%20up%20to%20the%20Challenge%20-%20Cold%20Climate%20Heat%20Pumps%20in%20the%20Field.pdf
- **LearnMetrics COP vs. Temperature table** — typical COP at 5°F increments, 0°F–60°F: https://learnmetrics.com/heat-pump-efficiency-vs-temperature-graph/
- **PickHVAC COP curves guide** — standard vs. cold-climate vs. inverter-driven COP profiles: https://www.pickhvac.com/heat-pump-efficiency-temperature-cop-curves-smart-cold/

### Building Thermal Load

Use a **degree-day / balance-point method** for hourly load estimation:

- **Heating load (BTU/h)** = UA × max(0, T_setpoint_heat − T_outdoor)
- **Cooling load (BTU/h)** = UA × max(0, T_outdoor − T_setpoint_cool)
- **UA (BTU/h·°F)**: building heat loss coefficient.

**Building presets** — two-axis selector (size × insulation quality):

| | Small (~1,200 sq ft) | Medium (~2,000 sq ft) | Large (~3,000 sq ft) |
|---|---|---|---|
| **Excellent** (new construction, well-sealed) | UA ≈ 200 | UA ≈ 330 | UA ≈ 500 |
| **Good** (upgraded insulation, some air sealing) | UA ≈ 280 | UA ≈ 470 | UA ≈ 700 |
| **Average** (typical 1970s–1990s construction) | UA ≈ 380 | UA ≈ 630 | UA ≈ 950 |
| **Poor** (old, uninsulated, leaky) | UA ≈ 520 | UA ≈ 870 | UA ≈ 1,300 |

UA in BTU/(h·°F). Derived from typical heat-loss-per-square-foot assumptions (Excellent ≈ 5–6, Poor ≈ 13–15 BTU/hr/sq ft per °F ΔT). Document derivation in code comments. Allow direct UA override via "Advanced" toggle.

**Default**: Medium / Average (UA ≈ 630), representing a typical ~2,000 sq ft Massachusetts home.

**Thermostat setpoints**: user-configurable (default: 68°F heating, 75°F cooling).

Internal gains and solar gains are ignored (note in UI).

### Weather Data

- **TMY3** hourly data for Boston Logan International Airport (USAF 725090).
- Bundle a pre-processed CSV of 8,760 hourly dry-bulb temperatures. No runtime API call.
- Single location for v1 (Boston). Architecture should support additional locations.

**Weather data sources**:
- NREL NSRDB TMY3 archive (station 725090): https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/tmy3/
- TMY3 Users Manual: https://docs.nrel.gov/docs/fy08osti/43156.pdf
- NSRDB TMY (newer PSM-based): https://nsrdb.nrel.gov/data-sets/tmy
- Climate.OneBuilding.org (TMY3/TMYx in EPW format): https://climate.onebuilding.org/
- EnergyPlus weather data: https://energyplus.net/weather
- MA EEAC TMY study (TMY3 vs. TMYx comparison): https://ma-eeac.org/wp-content/uploads/MA22C04-B-TMY-Final_Report.pdf

---

## Computation Pipeline

All calculations operate on NumPy arrays of shape `(8760,)`. No Python for-loops over hours.

### Shared: Building Load

For each hour `h`:

```
T_out[h]       = weather dry-bulb temperature (°F)
heat_load[h]   = UA × max(0, T_set_heat − T_out[h])     # BTU/h
cool_load[h]   = UA × max(0, T_out[h] − T_set_cool)     # BTU/h
```

### Path A: Heat Pump Electricity Cost

```
cop[h]         = piecewise_linear_interp(T_out[h], cop_curve)  # dimensionless
                 where cop[h] = 1.0 if T_out[h] < lockout_temp (resistance backup)
                 and cop[h] = 0 is not possible (minimum = 1.0)

cap_hp[h]      = piecewise_linear_interp(T_out[h], capacity_curve)  # BTU/h
load_hp[h]     = min(heat_load[h], cap_hp[h])              # BTU/h served by HP
load_backup[h] = heat_load[h] − load_hp[h]                 # BTU/h served by resistance
                 (if T_out[h] < lockout_temp, load_hp=0, load_backup=heat_load)

kwh_heat[h]    = load_hp[h] / (cop[h] × 3412) + load_backup[h] / 3412
kwh_cool[h]    = cool_load[h] / (cop_cool[h] × 3412)
kwh_total[h]   = kwh_heat[h] + kwh_cool[h]

rate[h]        = tou_lookup(month[h], hour_of_day[h], season[h])  # $/kWh
cost_elec[h]   = kwh_total[h] × rate[h]                          # $
```

### Path B: Gas Furnace Heating Cost

```
therms[h]      = heat_load[h] / (AFUE × 100_000)     # therms
cost_gas[h]    = therms[h] × gas_rate                 # $
```

Cooling cost is identical in both paths (electric AC). The comparison is **heating cost only**: `sum(cost_elec_heat)` vs. `sum(cost_gas)`.

### Aggregation

- Monthly totals: cost, kWh, therms, peak demand (kW).
- Annual totals: same.
- Heating cost comparison: HP electricity (heating only) vs. gas furnace.

---

## UI Layout

### Tab 1: Results (Main View)

**Sidebar inputs** (persistent across tabs):

1. **Rate Schedule**: preset dropdown + custom editor with seasonal hour→tier grid.
2. **Building**: Size dropdown (Small/Medium/Large) × Insulation dropdown (Poor/Average/Good/Excellent). Read-only UA display. Advanced expander for UA override and setpoints.
3. **Heat Pump**: preset dropdown (default: Cold-climate Hyper-Heat). Custom fields when "Custom" selected. Backup heat toggle.
4. **Gas Furnace** (for comparison): gas rate ($/therm), AFUE preset dropdown.

**Main panel**:

1. **Summary cards** (top row):
   - Annual HP electricity cost ($)
   - Annual gas furnace heating cost ($)
   - Annual heating cost savings: HP vs. gas ($, %)
   - Annual total kWh (HP)
   - Backup resistance share (% of heating kWh)

2. **Monthly heating cost comparison** (bar chart): side-by-side bars for each month showing HP heating electricity cost vs. gas furnace cost. This is the hero chart — it immediately answers "is the heat pump cheaper?"

3. **Monthly electricity breakdown** (stacked bar): heating kWh, cooling kWh, backup resistance kWh by month, coloured by TOU rate tier where applicable.

4. **Hourly cost heatmap**: 24h × 365d heatmap of electricity cost, coloured by rate tier. Shows TOU impact visually.

### Tab 2: Advanced Diagnostics

Purpose: let the user drill into individual days to verify the model produces sensible results.

**Controls**:
- Date picker (any day in the TMY year, presented as month/day since TMY has no specific year).
- Toggle: show heat pump path, gas path, or both.

**Plots for the selected day** (all sharing a common x-axis of hour 0–23):

1. **Temperature**: outdoor dry-bulb (°F) with heating and cooling setpoint lines.
2. **Thermal load**: heating load and cooling load (BTU/h) as area plots.
3. **COP**: heat pump COP at each hour, with lockout threshold marked. If the HP is in lockout for any hours, highlight those in red.
4. **Energy consumption**:
   - HP path: kWh from heat pump, kWh from backup resistance (stacked).
   - Gas path: therms consumed (on secondary y-axis or separate subplot).
5. **Hourly cost**: HP electricity cost and gas cost as overlaid lines, with TOU rate tier shown as background colour bands.
6. **Data table**: expandable table showing all hourly values for the selected day (T_out, heat_load, cool_load, COP, kWh_hp, kWh_backup, therms_gas, rate_tier, cost_elec, cost_gas). This table should be downloadable as CSV.

Additionally, show (not day-specific):

7. **COP curve plot**: the active heat pump COP profile (COP vs. outdoor temp, −20°F to 60°F) with a histogram of Boston TMY hours overlaid on the x-axis, so users can see where the system operates most.

8. **Load duration curve**: sorted hourly heating load (descending) with HP capacity overlay, showing how many hours require backup resistance.

---

## Technical Requirements

- **Framework**: Streamlit (Python).
- **Charting**: Plotly for all interactive charts.
- **Performance**: full 8,760-hour calculation in <1 second. Use NumPy vectorised operations exclusively. The `model.py` module should expose a single function that takes arrays and returns arrays — no Streamlit dependency.
- **State management**: `st.session_state` for rate edits and selections.
- **File structure**:
  ```
  app.py             — Streamlit UI, tab layout, sidebar controls
  model.py           — pure functions: thermal load, COP interpolation,
                       HP energy calc, gas energy calc, cost calc
                       (all operate on numpy arrays, no Streamlit imports)
  rates.py           — TOU schedule definition, hour→rate lookup,
                       season logic, presets
  weather.py         — TMY data loading and parsing
  presets.py         — HP presets, building presets, gas furnace presets
  diagnostics.py     — daily drill-down data extraction and plot generation
  data/
    boston_tmy3.csv   — 8,760 rows: hour, month, day, hour_of_day, T_drybulb_F
  tests/
    test_model.py    — thermal load, COP interp, energy calcs, edge cases
    test_rates.py    — rate lookup, season boundaries, tier assignment
    test_gas.py      — gas consumption calc, AFUE edge cases
  README.md
  requirements.txt
  ```
- **Testing**: pytest. Key edge cases:
  - Sub-zero temps (COP at/below lockout)
  - Capacity derating and backup resistance handoff
  - Season boundary (Apr 30 → May 1)
  - Midnight tier transitions
  - Zero-load hours (between setpoints)
  - Gas calc at zero load (should be zero therms)
  - AFUE = 100% (sanity check: therms = load / 100,000)
- **Documentation**: module-level and function-level docstrings. Cite data sources where assumptions are made. Each formula in `model.py` should have a comment referencing the corresponding line in this spec's Computation Pipeline section.

---

## Scope Boundaries (Out of Scope for v1)

- Demand charges ($/kW on peak monthly demand).
- Weekend/holiday rate overrides.
- Real-time utility API integration or bill parsing.
- Multiple weather locations (Boston only; architecture supports adding more).
- Multi-zone / multi-unit building modelling.
- Financial modelling (payback period, equipment cost, incentives).
- Detailed ductwork / refrigerant-level simulation.
- Gas cooling cost or gas equipment sizing.
- Gas fixed/customer charges (noted as simplification in UI).

---

## Key Assumptions to Document in the App

- Steady-state hourly load model; no thermal mass or setback recovery dynamics.
- No internal or solar heat gains.
- COP degradation is piecewise-linear interpolation from 3 reference test points, not full manufacturer maps.
- TMY3 represents a "typical" year, not any specific year.
- Backup heat is electric resistance only (COP = 1.0).
- Gas comparison uses a single blended $/therm rate; actual gas bills have tiered block rates and fixed charges that may differ.
- Rate presets are illustrative; users should verify against actual bills.
- The gas vs. heat pump comparison covers **heating cost only** — it does not reflect total energy bills, equipment costs, or maintenance.

---

## Stretch Goals

Out of scope for v1. Design data structures and module boundaries so these are additive, not requiring rewrites:

1. **Load-shifting simulator**: pre-heat/pre-cool during off-peak using a thermal mass parameter (BTU/°F). Show cost savings from shifting load.
2. **Rate plan optimizer**: compute breakeven TOU differential that makes TOU worthwhile vs. flat rate.
3. **Export**: download hourly results as CSV; PDF summary report.
4. **Comparison mode**: side-by-side two HP presets or two rate schedules on the same building.
5. **Additional weather locations**: Worcester, Springfield, Cape Cod with bundled TMY CSVs.
6. **Demand charges and weekend/holiday overrides**: extend rate model with `day_type` and monthly peak tracker.
7. **Gas block rates and fixed charges**: model tiered gas pricing and monthly customer charges for more accurate comparison.
8. **Oil furnace comparison**: add heating oil model ($/gallon, 138,500 BTU/gal, efficiency).
