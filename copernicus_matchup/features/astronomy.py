from __future__ import annotations

import numpy as np


def photoperiod_hours(lat_deg, day_of_year):
    """FAO-56/Allen et al. daylight-hours formula.

    Vectorized over numpy arrays or xarray DataArrays; a function of latitude and
    day-of-year only (independent of longitude and time-of-day). Polar day/night is
    handled by clamping cos(hour angle) to [-1, 1] rather than branching -- this is
    mathematically identical to the usual branched formulation, since arccos(-1) = pi
    radians -> 24h and arccos(1) = 0 -> 0h.
    """
    lat_rad = np.radians(lat_deg)
    declination = np.radians(23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365))
    cos_omega = np.clip(-np.tan(lat_rad) * np.tan(declination), -1.0, 1.0)
    omega_s = np.arccos(cos_omega)
    return (2 * np.degrees(omega_s)) / 15.0
