"""Force generators. Each has apply(world, state, t, F, T) adding into F [N] and T [N m],
and optionally potential(world, state) -> J for conservative forces."""
from __future__ import annotations
import numpy as np
from .units import G as G_SI, STANDARD_GRAVITY


class UniformGravity:
    """F = m g. g [m/s^2] vector, default (0, 0, -g0)."""

    def __init__(self, g=(0.0, 0.0, -STANDARD_GRAVITY)):
        self.g = np.asarray(g, float)

    def apply(self, world, s, t, F, T):
        F += world.mass[:, None] * self.g

    def potential(self, world, s) -> float:
        return float(-np.sum(world.mass * (s.x @ self.g)))


class NewtonianGravity:
    """Pairwise point-mass gravity F_ij = G m_i m_j r_ij / (|r_ij|^2 + eps^2)^(3/2).

    Pairwise antisymmetric construction -> exact Newton's third law (momentum conserved).
    """

    def __init__(self, G: float = G_SI, softening: float = 0.0):
        self.G = G
        self.eps2 = softening**2

    def _pairs(self, world, s):
        d = s.x[None, :, :] - s.x[:, None, :]          # d[i,j] = x_j - x_i
        r2 = np.einsum("ijk,ijk->ij", d, d) + self.eps2
        np.fill_diagonal(r2, 1.0)
        return d, r2

    def apply(self, world, s, t, F, T):
        d, r2 = self._pairs(world, s)
        mm = self.G * np.outer(world.mass, world.mass)
        np.fill_diagonal(mm, 0.0)
        F += np.einsum("ij,ijk->ik", mm / (r2 * np.sqrt(r2)), d)

    def potential(self, world, s) -> float:
        d, r2 = self._pairs(world, s)
        mm = self.G * np.outer(world.mass, world.mass)
        iu = np.triu_indices(world.n, 1)
        return float(-np.sum(mm[iu] / np.sqrt(r2[iu])))


class LinearDrag:
    """Stokes drag F = -b v, b [kg/s] (scalar or per body). Example: b = 6 pi mu r."""

    def __init__(self, b, wind=(0.0, 0.0, 0.0)):
        self.b = np.asarray(b, float)
        self.wind = np.asarray(wind, float)

    def apply(self, world, s, t, F, T):
        F -= np.reshape(self.b, (-1, 1)) * (s.v - self.wind)


class QuadraticDrag:
    """Newton drag F = -k |v| v, k [kg/m] (= 0.5 rho C_d A)."""

    def __init__(self, k, wind=(0.0, 0.0, 0.0)):
        self.k = np.asarray(k, float)
        self.wind = np.asarray(wind, float)

    @staticmethod
    def coefficient(rho: float, cd: float, area: float) -> float:
        return 0.5 * rho * cd * area

    def apply(self, world, s, t, F, T):
        u = s.v - self.wind
        F -= np.reshape(self.k, (-1, 1)) * np.linalg.norm(u, axis=1, keepdims=True) * u


class Spring:
    """Hookean spring of stiffness k [N/m] pulling body i's COM to a fixed anchor (rest length 0)."""

    def __init__(self, body: int, anchor, k: float):
        self.i, self.anchor, self.k = body, np.asarray(anchor, float), k

    def apply(self, world, s, t, F, T):
        F[self.i] -= self.k * (s.x[self.i] - self.anchor)

    def potential(self, world, s) -> float:
        d = s.x[self.i] - self.anchor
        return 0.5 * self.k * float(d @ d)


class ConstantTorque:
    """Constant world-frame torque [N m] on body i."""

    def __init__(self, body: int, torque):
        self.i, self.tau = body, np.asarray(torque, float)

    def apply(self, world, s, t, F, T):
        T[self.i] += self.tau
