"""Collision shapes (body frame, centred on the centre of mass) and mass properties (SI)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Sphere:
    radius: float  # m

    def volume(self) -> float:
        return 4.0 / 3.0 * np.pi * self.radius**3

    def inertia(self, m: float) -> np.ndarray:
        return np.eye(3) * 0.4 * m * self.radius**2

    def aabb(self, x, R):
        r = np.full(3, self.radius)
        return x - r, x + r


@dataclass(frozen=True)
class Box:
    half_extents: tuple  # m, (hx, hy, hz)

    @property
    def h(self) -> np.ndarray:
        return np.asarray(self.half_extents, float)

    def volume(self) -> float:
        return float(8 * np.prod(self.h))

    def inertia(self, m: float) -> np.ndarray:
        a, b, c = 2 * self.h
        return np.diag([b * b + c * c, a * a + c * c, a * a + b * b]) * m / 12.0

    def aabb(self, x, R):
        e = np.abs(R) @ self.h
        return x - e, x + e

    def vertices(self, x, R) -> np.ndarray:
        s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        return x + (s * self.h) @ R.T


@dataclass(frozen=True)
class Capsule:
    """Segment of half-length `half_height` along body z, swept by `radius`."""
    radius: float
    half_height: float

    def volume(self) -> float:
        r, h = self.radius, 2 * self.half_height
        return np.pi * r * r * h + 4.0 / 3.0 * np.pi * r**3

    def inertia(self, m: float) -> np.ndarray:
        r, h = self.radius, 2 * self.half_height
        vc, vs = np.pi * r * r * h, 4.0 / 3.0 * np.pi * r**3
        mc, ms = m * vc / (vc + vs), m * vs / (vc + vs)
        iz = mc * r * r / 2 + ms * 2 * r * r / 5
        ix = mc * (h * h / 12 + r * r / 4) + ms * (2 * r * r / 5 + h * h / 4 + 3 * h * r / 8)
        return np.diag([ix, ix, iz])

    def segment(self, x, R):
        a = R[:, 2] * self.half_height
        return x - a, x + a

    def aabb(self, x, R):
        p, q = self.segment(x, R)
        return np.minimum(p, q) - self.radius, np.maximum(p, q) + self.radius


@dataclass(frozen=True)
class Plane:
    """Static half-space {p : normal·p <= offset}; normal points out of the solid (world frame)."""
    normal: tuple
    offset: float = 0.0

    @property
    def n(self) -> np.ndarray:
        v = np.asarray(self.normal, float)
        return v / np.linalg.norm(v)

    def aabb(self, x, R):
        return np.full(3, -np.inf), np.full(3, np.inf)


@dataclass(frozen=True)
class Material:
    restitution: float = 0.0  # coefficient of restitution e (combined with max)
    friction: float = 0.5     # Coulomb mu (combined with sqrt(mu_a mu_b))
