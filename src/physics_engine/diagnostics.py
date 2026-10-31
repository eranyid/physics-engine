"""Conserved-quantity diagnostics (SI)."""
from __future__ import annotations
import numpy as np


def linear_momentum(world, s=None) -> np.ndarray:
    s = s or world.state
    return (world.mass[:, None] * s.v).sum(axis=0)


def angular_momentum(world, s=None, origin=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Total angular momentum about origin: sum r x m v + spin L [kg m^2/s]."""
    s = s or world.state
    r = s.x - np.asarray(origin, float)
    return (np.cross(r, world.mass[:, None] * s.v) + s.L).sum(axis=0)


def kinetic_energy(world, s=None) -> float:
    """Translational + rotational kinetic energy [J]."""
    s = s or world.state
    w = world.omega(s)
    return float(0.5 * np.sum(world.mass * np.sum(s.v**2, axis=1)) + 0.5 * np.sum(w * s.L))


def potential_energy(world, s=None) -> float:
    s = s or world.state
    return float(sum(g.potential(world, s) for g in world.generators if hasattr(g, "potential")))


def total_energy(world, s=None) -> float:
    return kinetic_energy(world, s) + potential_energy(world, s)


def body_angular_velocity(world, s=None) -> np.ndarray:
    """Angular velocity in body frame, (N,3)."""
    from . import quaternion as quat
    s = s or world.state
    R = quat.to_matrix(s.q)
    return np.einsum("nji,nj->ni", R, world.omega(s))
