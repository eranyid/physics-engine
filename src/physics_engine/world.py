"""World: structure-of-arrays state for N rigid bodies plus force generators."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import quaternion as quat
from .body import RigidBody


@dataclass
class State:
    """Full dynamic state. x (N,3) m, q (N,4), v (N,3) m/s, L (N,3) kg m^2/s (world frame)."""
    x: np.ndarray
    q: np.ndarray
    v: np.ndarray
    L: np.ndarray

    def copy(self) -> "State":
        return State(self.x.copy(), self.q.copy(), self.v.copy(), self.L.copy())

    def pack(self) -> np.ndarray:
        return np.concatenate([self.x.ravel(), self.q.ravel(), self.v.ravel(), self.L.ravel()])

    @staticmethod
    def unpack(y: np.ndarray, n: int) -> "State":
        a, b, c = 3 * n, 7 * n, 10 * n
        return State(y[:a].reshape(n, 3), y[a:b].reshape(n, 4), y[b:c].reshape(n, 3), y[c:].reshape(n, 3))


class World:
    def __init__(self, bodies: list[RigidBody], generators=()):
        self.bodies = list(bodies)
        self.generators = list(generators)
        self.n = len(self.bodies)
        self.mass = np.array([b.mass for b in self.bodies], float)
        self.inertia_body = np.array([b.inertia_body for b in self.bodies], float)
        self.inv_inertia_body = np.linalg.inv(self.inertia_body)
        q = np.array([b.orientation for b in self.bodies], float)
        R = quat.to_matrix(q)
        I_w = R @ self.inertia_body @ np.swapaxes(R, -1, -2)
        w = np.array([b.angular_velocity for b in self.bodies], float)
        self.state = State(
            x=np.array([b.position for b in self.bodies], float),
            q=q,
            v=np.array([b.velocity for b in self.bodies], float),
            L=np.einsum("nij,nj->ni", I_w, w),
        )
        self.t = 0.0

    # --- derived quantities -------------------------------------------------
    def inv_inertia_world(self, q: np.ndarray) -> np.ndarray:
        R = quat.to_matrix(q)
        return R @ self.inv_inertia_body @ np.swapaxes(R, -1, -2)

    def omega(self, s: State) -> np.ndarray:
        """World-frame angular velocity [rad/s]."""
        return np.einsum("nij,nj->ni", self.inv_inertia_world(s.q), s.L)

    def forces(self, s: State, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Accumulate force [N] and torque about COM [N m] from all generators."""
        F = np.zeros((self.n, 3))
        T = np.zeros((self.n, 3))
        for g in self.generators:
            g.apply(self, s, t, F, T)
        return F, T

    def derivative(self, s: State, t: float) -> State:
        F, T = self.forces(s, t)
        w = self.omega(s)
        wq = np.concatenate([np.zeros((self.n, 1)), w], axis=1)
        return State(x=s.v.copy(), q=0.5 * quat.multiply(wq, s.q), v=F / self.mass[:, None], L=T)

    # --- stepping -----------------------------------------------------------
    def step(self, dt: float, integrator) -> None:
        self.state = integrator(self, self.state, self.t, dt)
        self.state.q = quat.normalize(self.state.q)
        self.t += dt

    def run(self, t_end: float, dt: float, integrator, record=None):
        """Advance with fixed dt until t_end (last step shortened to land exactly)."""
        out = []
        while self.t < t_end - 1e-12 * max(1.0, abs(t_end)):
            h = min(dt, t_end - self.t)
            self.step(h, integrator)
            if record is not None:
                out.append(record(self))
        return out
