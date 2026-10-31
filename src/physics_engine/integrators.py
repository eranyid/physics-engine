"""Fixed-step integrators: f(world, state, t, dt) -> new state.

Translational DOF use (x, v); rotational DOF use (q, L) with L the world angular momentum.
"""
from __future__ import annotations
import numpy as np
from . import quaternion as quat
from .world import State


def _axpy(s: State, d: State, h: float) -> State:
    return State(s.x + h * d.x, s.q + h * d.q, s.v + h * d.v, s.L + h * d.L)


def explicit_euler(world, s: State, t: float, dt: float) -> State:
    """Forward Euler, order 1, not symplectic (energy grows on oscillators)."""
    return _axpy(s, world.derivative(s, t), dt)


def _drift_rotation(world, q, L, h, iters: int = 4):
    """Advance q with L frozen: q+ = exp(h w(q_mid)) q, w evaluated at the midpoint attitude.

    Implicit-midpoint (time-reversible, 2nd order) solved by fixed-point iteration; this keeps
    torque-free kinetic energy bounded instead of drifting as with w(q_n)."""
    def omega(qq):
        return np.einsum("nij,nj->ni", world.inv_inertia_world(qq), L)
    w = omega(q)
    for _ in range(iters):
        w = omega(quat.multiply(quat.exp_rotvec(0.5 * h * w), q))
    return quat.normalize(quat.multiply(quat.exp_rotvec(w * h), q))


def symplectic_euler(world, s: State, t: float, dt: float) -> State:
    """Semi-implicit Euler: kick velocities with forces at x_n, then drift with new velocities."""
    F, T = world.forces(s, t)
    v = s.v + dt * F / world.mass[:, None]
    L = s.L + dt * T
    return State(s.x + dt * v, _drift_rotation(world, s.q, L, dt), v, L)


def velocity_verlet(world, s: State, t: float, dt: float) -> State:
    """Kick-drift-kick leapfrog, order 2 and symplectic for position-dependent forces.
    The closing kick evaluates velocity-dependent forces at the half-step velocity."""
    h = 0.5 * dt
    F, T = world.forces(s, t)
    vh = s.v + h * F / world.mass[:, None]
    Lh = s.L + h * T
    mid = State(s.x + dt * vh, _drift_rotation(world, s.q, Lh, dt), vh, Lh)
    F2, T2 = world.forces(mid, t + dt)
    return State(mid.x, mid.q, vh + h * F2 / world.mass[:, None], Lh + h * T2)


def rk4(world, s: State, t: float, dt: float) -> State:
    """Classical 4th-order Runge-Kutta on the full state (q renormalised by World.step)."""
    k1 = world.derivative(s, t)
    k2 = world.derivative(_axpy(s, k1, dt / 2), t + dt / 2)
    k3 = world.derivative(_axpy(s, k2, dt / 2), t + dt / 2)
    k4 = world.derivative(_axpy(s, k3, dt), t + dt)
    return State(*(a + dt / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(
        (s.x, s.q, s.v, s.L), (k1.x, k1.q, k1.v, k1.L), (k2.x, k2.q, k2.v, k2.L),
        (k3.x, k3.q, k3.v, k3.L), (k4.x, k4.q, k4.v, k4.L))))


ALL = {"explicit_euler": explicit_euler, "symplectic_euler": symplectic_euler,
       "velocity_verlet": velocity_verlet, "rk4": rk4}
