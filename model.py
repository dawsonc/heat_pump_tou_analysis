"""Pure computation functions for thermal load, COP, energy, and cost calculations.

All functions operate on NumPy arrays of shape (8760,). No Streamlit dependency.
Each formula references the Computation Pipeline section of docs/spec.md.

Key functions:
- compute_heating_load: UA * max(0, T_set_heat - T_out)
- compute_cooling_load: UA * max(0, T_out - T_set_cool)
- compute_cop: piecewise-linear interpolation from reference COP points
- compute_capacity: heating capacity derating at low outdoor temps
- compute_hp_energy: HP kWh + backup resistance kWh
- compute_gas_energy: gas therms from heating load and AFUE
- compute_electric_cost: hourly electricity cost from kWh and rate arrays
- compute_gas_cost: hourly gas cost from therms and gas rate
- aggregate_monthly: sum hourly values into (12,) monthly totals
- aggregate_annual: sum hourly values into scalar annual total
- compute_peak_demand_monthly: peak hourly kW per month
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Conversion constants — spec Computation Pipeline (lines 164-165, 175)
# ---------------------------------------------------------------------------

BTU_PER_KWH = 3412  # 1 kWh = 3,412 BTU
BTU_PER_THERM = 100_000  # 1 therm = 100,000 BTU


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


class HPEnergy(NamedTuple):
    """Hourly heat pump electricity consumption and load split.

    All arrays have the same shape as the input load arrays.
    """

    kwh_heat: npt.NDArray[np.float64]  # kWh for heating (HP + backup)
    kwh_cool: npt.NDArray[np.float64]  # kWh for cooling
    kwh_total: npt.NDArray[np.float64]  # kwh_heat + kwh_cool
    load_hp: npt.NDArray[np.float64]  # BTU/h served by heat pump
    load_backup: npt.NDArray[np.float64]  # BTU/h served by resistance backup


# ---------------------------------------------------------------------------
# Building thermal load
# ---------------------------------------------------------------------------


def compute_heating_load(
    T_out: npt.NDArray[np.float64],
    UA: float,
    T_set_heat: float,
) -> npt.NDArray[np.float64]:
    """Compute hourly heating load in BTU/h.

    heat_load[h] = UA * max(0, T_set_heat - T_out[h])
    — spec Computation Pipeline, "Shared: Building Load"

    Parameters
    ----------
    T_out : (N,) array
        Outdoor dry-bulb temperature in deg F, one per hour.
    UA : float
        Building heat-loss coefficient in BTU/(h*degF).
    T_set_heat : float
        Heating thermostat setpoint in deg F.

    Returns
    -------
    (N,) array of float64
        Heating load in BTU/h. Zero when T_out >= T_set_heat.
    """
    # spec: heat_load[h] = UA * max(0, T_set_heat - T_out[h])
    return UA * np.maximum(0.0, T_set_heat - T_out)


def compute_cooling_load(
    T_out: npt.NDArray[np.float64],
    UA: float,
    T_set_cool: float,
) -> npt.NDArray[np.float64]:
    """Compute hourly cooling load in BTU/h.

    cool_load[h] = UA * max(0, T_out[h] - T_set_cool)
    — spec Computation Pipeline, "Shared: Building Load"

    Parameters
    ----------
    T_out : (N,) array
        Outdoor dry-bulb temperature in deg F, one per hour.
    UA : float
        Building heat-loss coefficient in BTU/(h*degF).
    T_set_cool : float
        Cooling thermostat setpoint in deg F.

    Returns
    -------
    (N,) array of float64
        Cooling load in BTU/h. Zero when T_out <= T_set_cool.
    """
    # spec: cool_load[h] = UA * max(0, T_out[h] - T_set_cool)
    return UA * np.maximum(0.0, T_out - T_set_cool)


# ---------------------------------------------------------------------------
# Heat pump COP and capacity
# ---------------------------------------------------------------------------


def compute_cop(
    T_out: npt.NDArray[np.float64],
    cop_curve: list[tuple[float, float]],
    lockout_temp: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Compute hourly heating COP via piecewise-linear interpolation.

    cop[h] = piecewise_linear_interp(T_out[h], cop_curve)
    where cop[h] = 1.0 if T_out[h] < lockout_temp (resistance backup)
    — spec Computation Pipeline, "Path A: Heat Pump Electricity Cost"

    Uses np.interp for piecewise-linear interpolation with constant
    extrapolation beyond endpoints (holds COP constant above highest
    and below lowest reference temp per spec).

    Parameters
    ----------
    T_out : (N,) array
        Outdoor dry-bulb temperature in deg F.
    cop_curve : list of (temp_F, COP) tuples
        Reference COP points sorted ascending by temperature.
        Example: [(5, 1.75), (17, 2.5), (47, 3.5)]
    lockout_temp : float
        Temperature in deg F below which the HP shuts off.
        Below lockout, COP is set to 1.0 (resistance backup).

    Returns
    -------
    cop : (N,) array of float64
        COP values. Minimum 1.0 (resistance backup below lockout).
    lockout_mask : (N,) array of bool
        True where T_out < lockout_temp (HP is locked out).
    """
    # Unzip curve into parallel arrays for np.interp
    temps = np.array([pt[0] for pt in cop_curve], dtype=np.float64)
    cops = np.array([pt[1] for pt in cop_curve], dtype=np.float64)

    # spec: piecewise-linear interpolation; np.interp clamps beyond endpoints
    cop = np.interp(T_out, temps, cops)

    # spec: cop[h] = 1.0 if T_out[h] < lockout_temp (resistance backup)
    lockout_mask = T_out < lockout_temp
    cop = np.where(lockout_mask, 1.0, cop)

    # spec: cop[h] = 0 is not possible (minimum = 1.0)
    cop = np.maximum(cop, 1.0)

    return cop, lockout_mask


def compute_cooling_cop(
    T_out: npt.NDArray[np.float64],
    cop_cooling_curve: list[tuple[float, float]],
) -> npt.NDArray[np.float64]:
    """Compute hourly cooling COP via piecewise-linear interpolation.

    cop_cool[h] = piecewise_linear_interp(T_out[h], cop_cooling_curve)
    — spec Computation Pipeline line 165: cop_cool[h] notation confirms
      per-hour COP.
    — spec line 74: "COP also degrades at very high outdoor temps but
      less dramatically; a simpler linear model is acceptable."

    Uses np.interp for piecewise-linear interpolation with constant
    extrapolation beyond endpoints (holds best COP below lowest
    reference temp, worst COP above highest reference temp).

    Parameters
    ----------
    T_out : (N,) array
        Outdoor dry-bulb temperature in deg F.
    cop_cooling_curve : list of (temp_F, COP) tuples
        Reference COP points sorted ascending by temperature.
        COP values typically decrease with increasing temperature.
        Example: [(82, 4.2), (95, 3.5), (115, 2.6)]

    Returns
    -------
    (N,) array of float64
        Cooling COP at each hour. Minimum 1.0.
    """
    temps = np.array([pt[0] for pt in cop_cooling_curve], dtype=np.float64)
    cops = np.array([pt[1] for pt in cop_cooling_curve], dtype=np.float64)

    # spec: piecewise-linear interpolation; np.interp clamps beyond endpoints
    cop = np.interp(T_out, temps, cops)

    # Floor at 1.0 (same safety constraint as heating COP)
    cop = np.maximum(cop, 1.0)

    return cop


def compute_capacity(
    T_out: npt.NDArray[np.float64],
    capacity_curve: list[tuple[float, float]],
    rated_capacity_btu_h: float,
) -> npt.NDArray[np.float64]:
    """Compute hourly heat pump heating capacity in BTU/h.

    cap_hp[h] = piecewise_linear_interp(T_out[h], capacity_curve) * rated_capacity
    — spec Computation Pipeline, "Path A: Heat Pump Electricity Cost"

    Uses np.interp for piecewise-linear interpolation with constant
    extrapolation (capacity fraction clamped at endpoint values).

    Parameters
    ----------
    T_out : (N,) array
        Outdoor dry-bulb temperature in deg F.
    capacity_curve : list of (temp_F, fraction_of_rated) tuples
        Capacity derating points sorted ascending by temperature.
        Example: [(5, 0.70), (17, 0.85), (47, 1.00)]
    rated_capacity_btu_h : float
        Rated heating capacity at 47 deg F in BTU/h.

    Returns
    -------
    (N,) array of float64
        Available heating capacity in BTU/h at each hour.
    """
    temps = np.array([pt[0] for pt in capacity_curve], dtype=np.float64)
    fractions = np.array([pt[1] for pt in capacity_curve], dtype=np.float64)

    # spec: cap_hp[h] = piecewise_linear_interp(T_out[h], capacity_curve)
    fraction = np.interp(T_out, temps, fractions)

    return fraction * rated_capacity_btu_h


# ---------------------------------------------------------------------------
# Energy calculations
# ---------------------------------------------------------------------------


def compute_hp_energy(
    heat_load: npt.NDArray[np.float64],
    cool_load: npt.NDArray[np.float64],
    cop: npt.NDArray[np.float64],
    cop_cool: float | npt.NDArray[np.float64],
    capacity: npt.NDArray[np.float64],
    lockout_mask: npt.NDArray[np.bool_],
    has_backup: bool = True,
) -> HPEnergy:
    """Compute hourly heat pump electricity consumption in kWh.

    load_hp[h]     = min(heat_load[h], cap_hp[h])  (0 during lockout)
    load_backup[h] = heat_load[h] - load_hp[h]
    kwh_heat[h]    = load_hp[h] / (cop[h] * 3412) + load_backup[h] / 3412
    kwh_cool[h]    = cool_load[h] / (cop_cool[h] * 3412)
    kwh_total[h]   = kwh_heat[h] + kwh_cool[h]
    — spec Computation Pipeline, "Path A: Heat Pump Electricity Cost"

    Parameters
    ----------
    heat_load : (N,) array
        Heating load in BTU/h.
    cool_load : (N,) array
        Cooling load in BTU/h.
    cop : (N,) array
        Heating COP at each hour (already lockout-adjusted to 1.0).
    cop_cool : float or (N,) array of float64
        Cooling COP. A scalar applies a constant COP to all hours.
        An array (from compute_cooling_cop) applies per-hour COP that
        degrades at higher outdoor temperatures per spec line 74.
    capacity : (N,) array
        Available HP heating capacity in BTU/h.
    lockout_mask : (N,) array of bool
        True where HP is locked out (all load goes to backup).
    has_backup : bool
        If True (default), electric resistance backup covers unserved
        load and its kWh is included in kwh_heat.  If False, load_backup
        is still computed (as unmet demand) but not converted to kWh.

    Returns
    -------
    HPEnergy
        Named tuple of (kwh_heat, kwh_cool, kwh_total, load_hp, load_backup).
    """
    # spec: load_hp[h] = min(heat_load[h], cap_hp[h])
    # spec: if T_out[h] < lockout_temp, load_hp=0, load_backup=heat_load
    load_hp = np.where(lockout_mask, 0.0, np.minimum(heat_load, capacity))

    # spec: load_backup[h] = heat_load[h] - load_hp[h]
    load_backup = heat_load - load_hp

    # spec: kwh_heat[h] = load_hp[h] / (cop[h] * 3412) + load_backup[h] / 3412
    backup_elec = load_backup / BTU_PER_KWH if has_backup else np.zeros_like(load_backup)
    kwh_heat = load_hp / (cop * BTU_PER_KWH) + backup_elec

    # spec: kwh_cool[h] = cool_load[h] / (cop_cool[h] * 3412)
    kwh_cool = cool_load / (cop_cool * BTU_PER_KWH)

    # spec: kwh_total[h] = kwh_heat[h] + kwh_cool[h]
    kwh_total = kwh_heat + kwh_cool

    return HPEnergy(
        kwh_heat=kwh_heat,
        kwh_cool=kwh_cool,
        kwh_total=kwh_total,
        load_hp=load_hp,
        load_backup=load_backup,
    )


def compute_gas_energy(
    heat_load: npt.NDArray[np.float64],
    afue: float,
) -> npt.NDArray[np.float64]:
    """Compute hourly gas consumption in therms.

    therms[h] = heat_load[h] / (AFUE * 100,000)
    — spec Computation Pipeline, "Path B: Gas Furnace Heating Cost"

    Parameters
    ----------
    heat_load : (N,) array
        Heating load in BTU/h.
    afue : float
        Annual Fuel Utilization Efficiency as a fraction (e.g., 0.80).

    Returns
    -------
    (N,) array of float64
        Gas consumption in therms per hour.
    """
    # spec: therms[h] = heat_load[h] / (AFUE * 100_000)
    return heat_load / (afue * BTU_PER_THERM)


# ---------------------------------------------------------------------------
# Cost calculations
# ---------------------------------------------------------------------------


def compute_electric_cost(
    kwh: npt.NDArray[np.float64],
    rates: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute hourly electricity cost.

    cost[h] = kwh[h] * rates[h]
    — spec Computation Pipeline, Path A: "cost_elec[h] = kwh_total[h] × rate[h]"

    Generic: pass kwh_total for total cost, or kwh_heat for heating-only
    comparison against gas.

    Parameters
    ----------
    kwh : (N,) array
        Electricity consumption in kWh per hour.
    rates : (N,) array
        Electricity rate in $/kWh per hour (from tou_lookup).

    Returns
    -------
    (N,) array of float64
        Electricity cost in dollars per hour.

    Raises
    ------
    ValueError
        If kwh and rates have different shapes.
    """
    kwh = np.asarray(kwh, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    if kwh.shape != rates.shape:
        raise ValueError(
            f"Shape mismatch: kwh {kwh.shape} vs rates {rates.shape}"
        )
    # spec: cost_elec[h] = kwh_total[h] * rate[h]
    return kwh * rates


def compute_gas_cost(
    therms: npt.NDArray[np.float64],
    gas_rate: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute hourly gas cost.

    cost_gas[h] = therms[h] * gas_rate[h]
    — spec Computation Pipeline, Path B: "cost_gas[h] = therms[h] × gas_rate"

    Parameters
    ----------
    therms : (N,) array
        Gas consumption in therms per hour (from compute_gas_energy).
    gas_rate : float or (N,) array of float64
        Gas rate in $/therm.  A scalar applies a flat rate to all hours.
        An array (from gas_rate_lookup) applies seasonal rates per hour.

    Returns
    -------
    (N,) array of float64
        Gas cost in dollars per hour.
    """
    # spec: cost_gas[h] = therms[h] * gas_rate
    return np.asarray(therms, dtype=np.float64) * gas_rate


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_monthly(
    values: npt.NDArray[np.float64],
    months: npt.NDArray[np.integer],
) -> npt.NDArray[np.float64]:
    """Sum hourly values into monthly totals.

    Parameters
    ----------
    values : (N,) array
        Hourly values to aggregate (cost, kWh, therms, etc.).
    months : (N,) array of int
        Month number (1-12) for each hour.

    Returns
    -------
    (12,) array of float64
        Monthly totals. Index 0 = January, index 11 = December.
    """
    values = np.asarray(values, dtype=np.float64)
    months = np.asarray(months)
    # months are 1-based; shift to 0-based for bincount
    totals = np.bincount(months - 1, weights=values, minlength=12)
    return totals[:12].astype(np.float64)


def aggregate_annual(
    values: npt.NDArray[np.float64],
) -> float:
    """Sum hourly values into a scalar annual total.

    Parameters
    ----------
    values : (N,) array
        Hourly values to sum.

    Returns
    -------
    float
        Annual total.
    """
    return float(np.sum(values))


def compute_peak_demand_monthly(
    kwh: npt.NDArray[np.float64],
    months: npt.NDArray[np.integer],
) -> npt.NDArray[np.float64]:
    """Find peak hourly demand (kW) in each month.

    Since each time step is 1 hour, kWh/h = kW average power.

    Parameters
    ----------
    kwh : (N,) array
        Hourly electricity consumption in kWh (= kW for 1-hour intervals).
    months : (N,) array of int
        Month number (1-12) for each hour.

    Returns
    -------
    (12,) array of float64
        Peak hourly demand in kW for each month. Index 0 = January.
    """
    kwh = np.asarray(kwh, dtype=np.float64)
    months = np.asarray(months)
    peak = np.zeros(12, dtype=np.float64)
    for m in range(1, 13):
        mask = months == m
        if np.any(mask):
            peak[m - 1] = np.max(kwh[mask])
    return peak
