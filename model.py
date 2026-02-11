"""Pure computation functions for thermal load, COP, energy, and cost calculations.

All functions operate on NumPy arrays of shape (8760,). No Streamlit dependency.
Each formula references the Computation Pipeline section of docs/spec.md.

Key functions (to be implemented):
- compute_heating_load: UA * max(0, T_set_heat - T_out)
- compute_cooling_load: UA * max(0, T_out - T_set_cool)
- compute_cop: piecewise-linear interpolation from reference COP points
- compute_capacity: heating capacity derating at low outdoor temps
- compute_hp_energy: HP kWh + backup resistance kWh
- compute_gas_energy: gas therms from heating load and AFUE
- compute_cost: hourly cost from kWh and rate arrays
"""
