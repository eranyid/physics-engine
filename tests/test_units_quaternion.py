import numpy as np
import pytest
from physics_engine import quaternion as quat, units as u
from physics_engine.body import RigidBody, inertia_solid_box


def test_dimensions_of_key_formulas():
    # terminal velocity sqrt(m g / k), k in kg/m
    k = u.MASS / u.LENGTH
    assert ((u.MASS * u.ACCEL) / k) ** 0.5 == u.VELOCITY
    # Stokes b in kg/s: m g / b is a velocity
    assert (u.MASS * u.ACCEL) / (u.MASS / u.TIME) == u.VELOCITY
    # Kepler period 2 pi sqrt(a^3 / (G M)) is a time
    assert (u.LENGTH**3 / (u.GRAV_CONST * u.MASS)) ** 0.5 == u.TIME
    # G M m / r is an energy
    assert u.GRAV_CONST * u.MASS * u.MASS / u.LENGTH == u.ENERGY


def test_constants_are_codata():
    assert u.G == pytest.approx(6.67430e-11, rel=1e-6)
    assert u.STANDARD_GRAVITY == 9.80665


def test_quaternion_rotation_convention():
    q = quat.from_axis_angle([0, 0, 1], np.pi / 2)  # +90 deg about z, active
    np.testing.assert_allclose(quat.rotate(q, np.array([1.0, 0, 0])), [0, 1, 0], atol=1e-15)
    R = quat.to_matrix(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_quaternion_composition_matches_matrices():
    rng = np.random.default_rng(0)
    a, b = quat.normalize(rng.normal(size=4)), quat.normalize(rng.normal(size=4))
    np.testing.assert_allclose(quat.to_matrix(quat.multiply(a, b)),
                               quat.to_matrix(a) @ quat.to_matrix(b), atol=1e-14)


def test_exp_rotvec_matches_axis_angle_and_small_angle():
    axis = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
    np.testing.assert_allclose(quat.exp_rotvec(0.7 * axis), quat.from_axis_angle(axis, 0.7), atol=1e-15)
    np.testing.assert_allclose(quat.exp_rotvec(np.zeros(3)), [1, 0, 0, 0])


def test_body_validation_and_box_inertia():
    with pytest.raises(ValueError):
        RigidBody(mass=0.0)
    with pytest.raises(ValueError):
        RigidBody(mass=1.0, inertia_body=np.diag([1.0, -1.0, 1.0]))
    # cube: I = m a^2 / 6
    np.testing.assert_allclose(inertia_solid_box(6.0, 1.0, 1.0, 1.0), np.eye(3))
