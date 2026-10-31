"""Torque-free rigid-body rotation (Euler's equations)."""
import numpy as np
import pytest
from physics_engine import RigidBody, World, forces, integrators, diagnostics, quaternion as quat
from physics_engine.body import inertia_solid_cylinder


def spinning(I, w_body, q=None):
    q = quat.identity() if q is None else q
    w_world = quat.rotate(q, np.asarray(w_body, float))
    return World([RigidBody(mass=1.0, inertia_body=np.diag(I), orientation=q, angular_velocity=w_world)])


@pytest.mark.parametrize("name", ["rk4", "velocity_verlet", "symplectic_euler"])
def test_torque_free_conserves_L_and_T(name):
    q0 = quat.from_axis_angle([1, 1, 0], 0.3)
    w = spinning([1.0, 2.0, 3.5], [0.4, 1.0, -2.0], q0)   # kg m^2, rad/s
    L0, T0 = diagnostics.angular_momentum(w).copy(), diagnostics.kinetic_energy(w)
    norms = w.run(20.0, 1e-3, integrators.ALL[name], record=lambda w: np.linalg.norm(w.state.q[0]))
    # L is the integrated state and torque = 0 -> conserved to round-off for every scheme
    np.testing.assert_allclose(diagnostics.angular_momentum(w), L0, rtol=1e-14, atol=1e-14)
    assert np.max(np.abs(np.array(norms) - 1)) < 1e-14
    tol = {"rk4": 1e-9, "velocity_verlet": 1e-6, "symplectic_euler": 1e-6}[name]
    print(name, "rel kinetic-energy error after 20 s:", diagnostics.kinetic_energy(w) / T0 - 1)
    assert diagnostics.kinetic_energy(w) == pytest.approx(T0, rel=tol)


def test_symmetric_top_body_frame_precession():
    I1, I3 = 2.0, 1.0                   # oblate/prolate doesn't matter: Omega = (I3-I1) w3 / I1
    wp, w3 = 0.5, 3.0
    w = spinning([I1, I1, I3], [wp, 0.0, w3])
    Om = (I3 - I1) * w3 / I1
    for t in (1.0, 5.0, 10.0):
        w.run(t, 1e-3, integrators.rk4)
        wb = diagnostics.body_angular_velocity(w)[0]
        np.testing.assert_allclose(wb, [wp * np.cos(Om * t), wp * np.sin(Om * t), w3], atol=1e-9)


def test_symmetric_top_space_frame_precession_rate():
    """Body symmetry axis precesses about L at rate |L| / I1 (cone of constant angle)."""
    I1, I3 = 2.0, 1.0
    w = spinning([I1, I1, I3], [0.5, 0.0, 3.0])
    L = diagnostics.angular_momentum(w)
    Lhat = L / np.linalg.norm(L)
    rate = np.linalg.norm(L) / I1
    T_prec = 2 * np.pi / rate
    z0 = quat.rotate(w.state.q[0], np.array([0, 0, 1.0]))
    w.run(T_prec, 1e-3, integrators.rk4)
    z1 = quat.rotate(w.state.q[0], np.array([0, 0, 1.0]))
    np.testing.assert_allclose(z1, z0, atol=1e-9)  # one full precession period
    w.run(1.5 * T_prec, 1e-3, integrators.rk4)
    assert np.dot(quat.rotate(w.state.q[0], np.array([0, 0, 1.0])), Lhat) == pytest.approx(np.dot(z0, Lhat), abs=1e-9)


def test_intermediate_axis_instability():
    """Spin about the intermediate axis flips (tennis-racket); about min/max axes it is stable."""
    I = [1.0, 2.0, 3.0]
    eps = 1e-3
    def max_dev(axis):
        wb = np.full(3, eps)
        wb[axis] = 5.0
        w = spinning(I, wb)
        out = w.run(20.0, 2e-3, integrators.rk4, record=lambda w: diagnostics.body_angular_velocity(w)[0][axis])
        return np.min(out)
    assert max_dev(0) > 4.99 and max_dev(2) > 4.99   # stays near +5 rad/s
    assert max_dev(1) < -4.9                          # flips to ~ -5 rad/s


def test_constant_torque_spins_up_sphere():
    I = 0.4  # kg m^2
    w = World([RigidBody(mass=1.0, inertia_body=np.eye(3) * I)], [forces.ConstantTorque(0, [0, 0, 2.0])])
    w.run(3.0, 1e-3, integrators.rk4)
    np.testing.assert_allclose(w.omega(w.state)[0], [0, 0, 2.0 * 3.0 / I], atol=1e-12)
    # angle = 1/2 alpha t^2 about z
    ang = 2 * np.arctan2(w.state.q[0, 3], w.state.q[0, 0])
    assert np.mod(ang, 2 * np.pi) == pytest.approx(np.mod(0.5 * 5.0 * 9.0, 2 * np.pi), abs=1e-6)


def test_cylinder_inertia_formula():
    np.testing.assert_allclose(inertia_solid_cylinder(2.0, 0.5, 1.0), np.diag([0.29166666666666663] * 2 + [0.25]))
