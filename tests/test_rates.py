"""Tests for rates.py: rate lookup, season boundaries, tier assignment.

Edge cases from spec:
- Season boundary (Apr 30 -> May 1)
- Midnight tier transitions
- Flat rate returns same value for all hours
"""
