"""Tests for model.py: thermal load, COP interpolation, energy calculations.

Edge cases from spec:
- Sub-zero temps (COP at/below lockout)
- Capacity derating and backup resistance handoff
- Zero-load hours (between heating and cooling setpoints)
"""
