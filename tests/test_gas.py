"""Tests for gas furnace cost calculations.

Edge cases from spec:
- Gas calc at zero load (should be zero therms)
- AFUE = 100% (therms = load / 100,000)
- Various AFUE presets produce expected relative costs
"""
