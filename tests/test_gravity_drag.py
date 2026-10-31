"""Translational motion vs closed-form solutions (SI units: m, s, kg)."""
import numpy as np
import pytest
from physics_engine import RigidBody, World, forces, integrators
from physics_engine.units import STANDARD_GRAVITY as g0
from conftest import ALL_INTEGRATORS


@pytest.mark.parametrize("name,integ", ALL_INTEGRATORS)
def test_projectile_closed_form(name, integ):
    x0, v0 = np.array([0.0, 0.0, 2.0]), np.array([12.0, -3.0, 15.0])  # m, m/s
    w = World([RigidBody(mass=2.5, position=x0, velocity=v0)], [forces.UniformGravity()])
    t = 2.0  # s
    w.run(t, 1e-3, integ)
    gv = np.array([0, 0, -g0])
    x_exact = x0 + v0 * t + 0.5 * gv * t**2
    v_exact = v0 + gv * t
    # Euler-type methods carry O(dt) error g*t*dt/2 ~ 1e-2 m; Verlet/RK4 exact for constant accel
    tol = 1.5e-2 if "euler" in name else 1e-9
    np.testing.assert_allclose(w.state.x[0], x_exact, atol=tol)
    np.testing.assert_allclose(w.state.v[0], v_exact, atol=1e-9)


def test_free_fall_time_to_ground():
    h = 100.0  # m
    w = World([RigidBody(mass=1.0, position=[0, 0, h])], [forces.UniformGravity()])
    w.run(np.sqrt(2 * h / g0), 1e-3, integrators.velocity_verlet)
    assert w.state.x[0, 2] == pytest.approx(0.0, abs=1e-9)
    assert w.state.v[0, 2] == pytest.approx(-np.sqrt(2 * g0 * h), rel=1e-12)


def test_linear_drag_exponential_approach():
    m, b = 0.2, 0.05  # kg, kg/s ; tau = m/b = 4 s, v_t = m g / b
    w = World([RigidBody(mass=m)], [forces.UniformGravity(), forces.LinearDrag(b)])
    vt, tau = m * g0 / b, m / b
    for t in (1.0, 4.0, 12.0, 40.0):
        w.run(t, 1e-2, integrators.rk4)
        v_exact = -vt * (1 - np.exp(-t / tau))
        z_exact = -(vt * t - vt * tau * (1 - np.exp(-t / tau)))
        assert w.state.v[0, 2] == pytest.approx(v_exact, rel=1e-9)
        assert w.state.x[0, 2] == pytest.approx(z_exact, rel=1e-9)
    assert -w.state.v[0, 2] == pytest.approx(vt, rel=1e-4)  # 10 tau -> within e^-10


def test_quadratic_drag_tanh_solution_and_terminal_velocity():
    # skydiver-like: m = 80 kg, rho = 1.225, Cd = 1.0, A = 0.7 m^2
    m = 80.0
    k = forces.QuadraticDrag.coefficient(1.225, 1.0, 0.7)  # kg/m
    vt = np.sqrt(m * g0 / k)
    w = World([RigidBody(mass=m, position=[0, 0, 4000.0])], [forces.UniformGravity(), forces.QuadraticDrag(k)])
    for t in (2.0, 5.0, 10.0, 30.0):
        w.run(t, 1e-2, integrators.rk4)
        v_exact = -vt * np.tanh(g0 * t / vt)
        drop = vt**2 / g0 * np.log(np.cosh(g0 * t / vt))
        assert w.state.v[0, 2] == pytest.approx(v_exact, rel=1e-8)
        assert w.state.x[0, 2] == pytest.approx(4000.0 - drop, rel=1e-8)
    assert -w.state.v[0, 2] == pytest.approx(vt, rel=1e-5)
    assert vt == pytest.approx(42.7, abs=0.1)  # m/s, sanity for skydiver


def test_drag_opposes_relative_wind():
    w = World([RigidBody(mass=1.0)], [forces.QuadraticDrag(0.5, wind=[10.0, 0, 0])])
    F, _ = w.forces(w.state, 0.0)
    np.testing.assert_allclose(F[0], [50.0, 0, 0])  # k |u| u with u = -10 -> +50 N
