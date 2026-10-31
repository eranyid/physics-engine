"""Rigid body description and inertia helpers (SI)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from . import quaternion as quat


def inertia_solid_sphere(m: float, r: float) -> np.ndarray:
    return np.eye(3) * (2.0 / 5.0) * m * r * r


def inertia_solid_box(m: float, a: float, b: float, c: float) -> np.ndarray:
    """Box with full edge lengths a (x), b (y), c (z)."""
    return np.diag([m * (b * b + c * c), m * (a * a + c * c), m * (a * a + b * b)]) / 12.0


def inertia_solid_cylinder(m: float, r: float, h: float) -> np.ndarray:
    """Cylinder with axis along body z."""
    i = m * (3 * r * r + h * h) / 12.0
    return np.diag([i, i, 0.5 * m * r * r])


@dataclass
class RigidBody:
    """Initial description of a body. Positions/velocities are world frame, at the centre of mass.

    mass [kg], inertia_body [kg m^2] (body frame, about COM), position [m], velocity [m/s],
    orientation (unit quaternion, body->world), angular_velocity [rad/s] (world frame).
    """
    mass: float
    inertia_body: np.ndarray = field(default_factory=lambda: np.eye(3))
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=quat.identity)
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    name: str = ""

    def __post_init__(self):
        if not self.mass > 0:
            raise ValueError("mass must be positive")
        self.inertia_body = np.asarray(self.inertia_body, float)
        if not np.allclose(self.inertia_body, self.inertia_body.T):
            raise ValueError("inertia tensor must be symmetric")
        if np.any(np.linalg.eigvalsh(self.inertia_body) <= 0):
            raise ValueError("inertia tensor must be positive definite")
        for k in ("position", "velocity", "angular_velocity"):
            setattr(self, k, np.asarray(getattr(self, k), float).reshape(3))
        self.orientation = quat.normalize(np.asarray(self.orientation, float).reshape(4))
