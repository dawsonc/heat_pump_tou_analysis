"""Preset configurations for heat pumps, buildings, and gas furnaces.

Heat pump presets define COP reference points, capacity curves, and lockout temps.
Building presets define UA values on a size x insulation quality grid.
Gas furnace presets define AFUE values.

All preset values are sourced from docs/spec.md and cited references:
- NEEP Cold Climate ASHP Product List: https://ashp.neep.org/
- NEEP ccASHP Specification v4.0: https://neep.org/heating-electrification/ccashp-specification-product-list
- DOE Cold Climate HP field results (PNNL-37127)
- NREL ductless mini-split COP model (NREL 85081)
"""

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
# [1] NEEP Cold Climate ASHP Product List — COP at 47/17/5 degF for 40,000+
#     systems: https://ashp.neep.org/
# [2] NEEP ccASHP Specification v4.0 — cold-climate minimum: COP >= 1.75 at
#     5 degF, capacity >= 70% of 47 degF rating:
#     https://neep.org/heating-electrification/ccashp-specification-product-list
# [3] DOE Cold Climate Heat Pump Technology Challenge (PNNL-37127) — measured
#     COP by outdoor air temp bin:
#     https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-37127.pdf
# [4] NREL Empirical COP Model for Ductless Mini-Splits (NREL 85081) —
#     COP(T_outdoor, load_fraction) from cold-chamber tests:
#     https://docs.nrel.gov/docs/fy23osti/85081.pdf
# [5] ACEEE 2024: Cold Climate Heat Pumps in the Field — COP ranges 1.0-2.5
#     at 5 degF to -22 degF across ten field studies
# [6] LearnMetrics COP vs. Temperature table — typical COP at 5 degF
#     increments: https://learnmetrics.com/heat-pump-efficiency-vs-temperature-graph/
# [7] PickHVAC COP curves guide — standard vs. cold-climate vs. inverter-driven
#     profiles: https://www.pickhvac.com/heat-pump-efficiency-temperature-cop-curves-smart-cold/

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

# Thermostat setpoints (degF) — spec section "Building Thermal Load"
DEFAULT_T_SET_HEAT_F = 68
DEFAULT_T_SET_COOL_F = 75

# Gas rates — approximate MA all-in residential rates.
# Winter rate reflects higher supply costs during heating season.
# Component breakdown (winter): supply $0.9477, delivery $0.9319,
#   efficiency $0.0691, revenue decoupling -$0.0099, LDAC $0.0,
#   energy efficiency $0.4170, RPS $0.0833, clean energy $0.0558.
# Summer rates are typically 20-30% lower due to reduced supply costs.
DEFAULT_GAS_WINTER_RATE_PER_THERM = (
    0.9477 + 0.9319 + 0.0691 - 0.0099 + 0.0 + 0.4170 + 0.0833 + 0.0558
)
DEFAULT_GAS_SUMMER_RATE_PER_THERM = (
    1.8287
)

# Backward compatibility alias: flat rate equals the winter rate.
DEFAULT_GAS_RATE_PER_THERM = DEFAULT_GAS_WINTER_RATE_PER_THERM

DEFAULT_GAS_MONTHLY_CHARGE = 9.00

# ---------------------------------------------------------------------------
# Heat pump presets
# ---------------------------------------------------------------------------
# Each preset contains:
#   cop_curve           - list of (temp_F, COP) tuples, sorted ascending by
#                         temp. Used by piecewise_linear_interp(). Below the
#                         lowest reference point, COP is held constant until
#                         lockout. Above the highest, COP is held constant.
#   capacity_curve      - list of (temp_F, fraction_of_rated) tuples, sorted
#                         ascending by temp. Fraction of rated_capacity_btu_h
#                         at each outdoor temp. Same interpolation rules.
#   lockout_temp_f      - below this temp, HP shuts off. With backup heat,
#                         electric resistance (COP=1.0) covers full load.
#                         Without backup, load becomes unmet demand.
#   rated_capacity_btu_h - default rated heating capacity at 47 degF (BTU/h).
#                         User can override in the UI.
#   cop_cooling         - single cooling COP value. Spec allows "a simpler
#                         linear model" for cooling; constant is simplest.
#   has_backup_heat     - whether electric resistance backup is available.
#                         True = resistance covers shortfall (COP=1.0).
#                         False = unmet demand when HP capacity is exceeded
#                         or below lockout.
#   notes               - description for UI display / tooltips.

HP_PRESETS = {
    # ★ Default preset. EVI compressor, NEEP-listed cold-climate unit.
    # COP values from spec table (line 81), sourced from NEEP product list [1]
    # and NEEP ccASHP Spec v4.0 [2]. COP >= 1.75 at 5 degF meets the ccASHP
    # specification minimum.
    # Capacity: NEEP ccASHP Spec v4.0 [2] requires >= 70% of 47 degF rated
    # capacity at 5 degF. 85% at 17 degF is consistent with PNNL-37127 [3]
    # field data for EVI-equipped units (typically 80-90% at 17 degF).
    "Cold-climate (Hyper-Heat)": {
        "cop_curve": [
            (5, 1.75),
            (17, 2.5),
            (47, 3.5),
        ],
        "capacity_curve": [
            (5, 0.70),
            (17, 0.85),
            (47, 1.00),
        ],
        "lockout_temp_f": -15,
        "rated_capacity_btu_h": 36_000,
        "cop_cooling": 3.8,
        "has_backup_heat": True,
        "notes": "EVI compressor, NEEP-listed. COP >= 1.75 at 5 deg F.",
    },

    # Typical inverter-driven cold-climate unit. COP values from spec table
    # (line 82). Capacity derating is steeper than Hyper-Heat but still
    # performs well in cold weather. Values consistent with PNNL-37127 [3]
    # field data and ACEEE 2024 [5] field studies for mid-tier inverter units.
    "Standard cold-climate": {
        "cop_curve": [
            (5, 1.5),
            (17, 2.2),
            (47, 3.3),
        ],
        "capacity_curve": [
            (5, 0.65),
            (17, 0.80),
            (47, 1.00),
        ],
        "lockout_temp_f": 0,
        "rated_capacity_btu_h": 36_000,
        "cop_cooling": 3.5,
        "has_backup_heat": True,
        "notes": "Typical inverter-driven cold-climate unit.",
    },

    # Non-cold-climate unit. COP values from spec table (line 83). No COP
    # reference point at 5 degF because the unit locks out at 15 degF.
    # Capacity derating from NREL 85081 [4] and PickHVAC [7]: standard ASHPs
    # lose ~40% capacity by 17 degF. Between lockout (15 degF) and the lowest
    # COP reference (17 degF), COP is held at 1.8 per spec interpolation rule.
    "Baseline (non-cold-climate)": {
        "cop_curve": [
            (17, 1.8),
            (47, 3.0),
        ],
        "capacity_curve": [
            (17, 0.60),
            (47, 1.00),
        ],
        "lockout_temp_f": 15,
        "rated_capacity_btu_h": 36_000,
        "cop_cooling": 3.2,
        "has_backup_heat": True,
        "notes": (
            "Non-cold-climate unit. Switches to resistance below "
            "15 deg F lockout."
        ),
    },
}

DEFAULT_HP_PRESET = "Cold-climate (Hyper-Heat)"

# ---------------------------------------------------------------------------
# Building presets — UA values in BTU/(h*degF)
# ---------------------------------------------------------------------------
# Two-axis selector: insulation quality x building size.
# UA values from spec table (lines 107-112).
#
# Derivation: UA = heat_loss_per_sqft * floor_area, where heat_loss_per_sqft
# is in BTU/(h*sqft*degF). Ranges by insulation quality:
#   Excellent (new construction, well-sealed):     ~5-6 BTU/(h*sqft*degF)
#   Good (upgraded insulation, some air sealing):  ~7-8
#   Average (typical 1970s-1990s):                 ~9-11
#   Poor (old, uninsulated, leaky):                ~13-15
#
# Floor areas: Small ~1,200 sqft, Medium ~2,000 sqft, Large ~3,000 sqft.
# Default: Medium / Average (UA = 630), representing a typical ~2,000 sqft
# Massachusetts home.

BUILDING_PRESETS = {
    "Excellent": {"Small": 200, "Medium": 330, "Large": 500},
    "Good":      {"Small": 280, "Medium": 470, "Large": 700},
    "Average":   {"Small": 380, "Medium": 630, "Large": 950},
    "Poor":      {"Small": 520, "Medium": 870, "Large": 1300},
}

# Ordered lists for UI dropdowns.
BUILDING_SIZES = ["Small", "Medium", "Large"]
INSULATION_LEVELS = ["Excellent", "Good", "Average", "Poor"]

DEFAULT_BUILDING_SIZE = "Medium"
DEFAULT_BUILDING_INSULATION = "Average"

# ---------------------------------------------------------------------------
# Gas furnace presets
# ---------------------------------------------------------------------------
# AFUE values from spec table (lines 46-49). Stored as fractions (not %).
# Default is "Older furnace" (80% AFUE) — represents the most common
# installed base per spec.

GAS_FURNACE_PRESETS = {
    "High-efficiency condensing": {"afue": 0.96},
    "Standard high-efficiency":   {"afue": 0.92},
    "Older furnace":              {"afue": 0.80},
}

DEFAULT_GAS_FURNACE_PRESET = "Older furnace"
