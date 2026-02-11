"""Preset configurations for heat pumps, buildings, and gas furnaces.

Heat pump presets define COP reference points, capacity curves, and lockout temps.
Building presets define UA values on a size x insulation quality grid.
Gas furnace presets define AFUE values.

All preset values are sourced from docs/spec.md and cited references:
- NEEP Cold Climate ASHP Product List: https://ashp.neep.org/
- NEEP ccASHP Specification v4.0: https://neep.org/heating-electrification/ccashp-specification-product-list
- DOE Cold Climate HP field results (PNNL-37127)
- NREL ductless mini-split COP model (NREL 85081)

Key data (to be implemented):
- HP_PRESETS: dict of heat pump configurations
- BUILDING_PRESETS: size x insulation -> UA mapping
- GAS_FURNACE_PRESETS: dict of AFUE configurations
"""
