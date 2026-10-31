"""International Standard Atmosphere (ISA / ICAO Doc 7488) through the lower stratosphere.

Geopotential altitude h [m]. Returns SI: T [K], p [Pa], rho [kg/m^3], a [m/s].
Layer model: troposphere 0..11 km (lapse 6.5 K/km), isothermal 11..20 km.
"""
from __future__ import annotations
import numpy as np
from .units import STANDARD_GRAVITY

# ICAO / ISO sea-level and layer constants (exact by definition of the tables).
T0: float = 288.15          # K
P0: float = 101_325.0       # Pa
RHO0: float = 1.225         # kg/m^3
LAPSE: float = 0.0065       # K/m (positive magnitude; T decreases)
H_TROPOPAUSE: float = 11_000.0  # m
H_STRATO_TOP: float = 20_000.0  # m
GAMMA_AIR: float = 1.4
# Specific gas constant implied by sea-level state: R = p/(rho T)
R_AIR: float = P0 / (RHO0 * T0)  # ≈ 287.05287 J/(kg K)
# Hydrostatic exponent g/(R L) for the troposphere power law.
_EXP: float = STANDARD_GRAVITY / (R_AIR * LAPSE)


def temperature(h: float) -> float:
    """Static temperature [K] at geopotential altitude h [m]."""
    h = float(h)
    if h < H_TROPOPAUSE:
        return T0 - LAPSE * h
    if h <= H_STRATO_TOP:
        return T0 - LAPSE * H_TROPOPAUSE  # 216.65 K
    raise ValueError(f"ISA model limited to h <= {H_STRATO_TOP} m; got {h}")


def pressure(h: float) -> float:
    """Static pressure [Pa] at geopotential altitude h [m]."""
    h = float(h)
    T = temperature(h)
    if h < H_TROPOPAUSE:
        return P0 * (T / T0) ** _EXP
    # Isothermal layer above tropopause.
    T11 = temperature(H_TROPOPAUSE)
    p11 = P0 * (T11 / T0) ** _EXP
    return p11 * np.exp(-STANDARD_GRAVITY * (h - H_TROPOPAUSE) / (R_AIR * T11))


def density(h: float) -> float:
    """Mass density [kg/m^3] via ideal gas law rho = p / (R T)."""
    return pressure(h) / (R_AIR * temperature(h))


def speed_of_sound(h: float) -> float:
    """Isentropic speed of sound a = sqrt(gamma R T) [m/s]."""
    return float(np.sqrt(GAMMA_AIR * R_AIR * temperature(h)))


def state(h: float) -> dict[str, float]:
    """Convenience: temperature, pressure, density, speed_of_sound at h [m]."""
    return {
        "temperature": temperature(h),
        "pressure": pressure(h),
        "density": density(h),
        "speed_of_sound": speed_of_sound(h),
    }
