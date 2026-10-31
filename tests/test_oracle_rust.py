"""Oracle tests: identical scenarios on Python reference vs Rust wheel.

Tolerances (see docs/DETERMINISM.md):
- Same operation order within Rust is bit-identical (libm, no FMA).
- Cross-impl (NumPy vectorized vs scalar Rust) may differ at ~1e-12..1e-15
  relative due to reduction order; drag/Kepler use looser relative bounds
  documented per test. Not a silent loosen — each bound is scenario-specific.
"""
from __future__ import annotations

import numpy as np
import pytest

from physics_engine import RigidBody, World, forces, integrators, diagnostics
from physics_engine.units import STANDARD_GRAVITY as g0, G

pe = pytest.importorskip("physics_engine_rs")


def _py_projectile():
    x0, v0 = np.array([0.0, 0.0, 2.0]), np.array([12.0, -3.0, 15.0])
    w = World([RigidBody(mass=2.5, position=x0, velocity=v0)], [forces.UniformGravity()])
    w.run(2.0, 1e-3, integrators.rk4)
    return w


def _rs_projectile():
    w = pe.World(
        [pe.RigidBody(2.5, position=[0.0, 0.0, 2.0], velocity=[12.0, -3.0, 15.0])],
        [pe.UniformGravity()],
    )
    w.run(2.0, 1e-3, "rk4")
    return w


def test_oracle_projectile_rk4_near_bit_close():
    """Constant accel: RK4 is exact for polynomials → expect ~machine agreement."""
    py, rs = _py_projectile(), _rs_projectile()
    np.testing.assert_allclose(rs.x[0], py.state.x[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(rs.v[0], py.state.v[0], rtol=0, atol=1e-14)
    # Documented: often bit-identical on this path; allow 1 ULP-scale if libm≠NumPy.
    assert np.max(np.abs(rs.x[0] - py.state.x[0])) < 1e-14


def test_oracle_free_fall():
    h = 100.0
    t = np.sqrt(2 * h / g0)
    py = World([RigidBody(mass=1.0, position=[0, 0, h])], [forces.UniformGravity()])
    py.run(t, 1e-3, integrators.velocity_verlet)
    rs = pe.World([pe.RigidBody(1.0, position=[0.0, 0.0, h])], [pe.UniformGravity()])
    rs.run(t, 1e-3, "velocity_verlet")
    np.testing.assert_allclose(rs.x[0], py.state.x[0], atol=1e-12)
    np.testing.assert_allclose(rs.v[0], py.state.v[0], atol=1e-12)


def test_oracle_linear_drag():
    m, b = 0.2, 0.05
    py = World([RigidBody(mass=m)], [forces.UniformGravity(), forces.LinearDrag(b)])
    rs = pe.World([pe.RigidBody(m)], [pe.UniformGravity(), pe.LinearDrag(b)])
    for t in (1.0, 4.0, 12.0):
        py.run(t, 1e-2, integrators.rk4)
        rs.run(t, 1e-2, "rk4")
        # Velocity-dependent RK4: NumPy vs scalar libm — expect ~1e-12 rel.
        np.testing.assert_allclose(rs.v[0], py.state.v[0], rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(rs.x[0], py.state.x[0], rtol=1e-11, atol=1e-14)


def test_oracle_quadratic_drag():
    m = 80.0
    k = forces.QuadraticDrag.coefficient(1.225, 1.0, 0.7)
    assert pe.QuadraticDrag.coefficient(1.225, 1.0, 0.7) == pytest.approx(k)
    py = World(
        [RigidBody(mass=m, position=[0, 0, 4000.0])],
        [forces.UniformGravity(), forces.QuadraticDrag(k)],
    )
    rs = pe.World(
        [pe.RigidBody(m, position=[0.0, 0.0, 4000.0])],
        [pe.UniformGravity(), pe.QuadraticDrag(k)],
    )
    py.run(10.0, 1e-2, integrators.rk4)
    rs.run(10.0, 1e-2, "rk4")
    np.testing.assert_allclose(rs.v[0], py.state.v[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(rs.x[0], py.state.x[0], rtol=1e-10, atol=1e-9)


def test_oracle_harmonic_oscillator():
    m, k = 2.0, 8.0
    py = World([RigidBody(mass=m, position=[1.0, 0, 0])], [forces.Spring(0, [0, 0, 0], k)])
    rs = pe.World([pe.RigidBody(m, position=[1.0, 0.0, 0.0])], [pe.Spring(0, [0.0, 0.0, 0.0], k)])
    py.run(2.0, 1e-3, integrators.rk4)
    rs.run(2.0, 1e-3, "rk4")
    np.testing.assert_allclose(rs.x[0], py.state.x[0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(rs.v[0], py.state.v[0], rtol=1e-12, atol=1e-14)


@pytest.mark.slow
def test_oracle_kepler_half_period():
    """Eccentric orbit to apoapsis — slow due to 10k RK4 steps."""
    M_SUN, M_EARTH, AU = 1.98847e30, 5.9722e24, 1.495978707e11
    a, e = AU, 0.5
    M = M_SUN + M_EARTH
    rp = a * (1 - e)
    vp = np.sqrt(G * M * (1 + e) / rp)
    f1, f2 = M_EARTH / M, M_SUN / M
    bodies_py = [
        RigidBody(mass=M_SUN, position=[-f1 * rp, 0, 0], velocity=[0, -f1 * vp, 0]),
        RigidBody(mass=M_EARTH, position=[f2 * rp, 0, 0], velocity=[0, f2 * vp, 0]),
    ]
    bodies_rs = [
        pe.RigidBody(M_SUN, position=[-f1 * rp, 0.0, 0.0], velocity=[0.0, -f1 * vp, 0.0]),
        pe.RigidBody(M_EARTH, position=[f2 * rp, 0.0, 0.0], velocity=[0.0, f2 * vp, 0.0]),
    ]
    P = 2 * np.pi * np.sqrt(a**3 / (G * M))
    py = World(bodies_py, [forces.NewtonianGravity()])
    rs = pe.World(bodies_rs, [pe.NewtonianGravity()])
    py.run(P / 2, P / 20000, integrators.rk4)
    rs.run(P / 2, P / 20000, "rk4")
    # Pairwise einsum vs scalar i<j loop: document ~1e-9 relative on AU scale.
    r_py = np.linalg.norm(py.state.x[1] - py.state.x[0])
    r_rs = np.linalg.norm(rs.x[1] - rs.x[0])
    assert r_rs == pytest.approx(r_py, rel=1e-9)
    assert abs(rs.total_energy() - diagnostics.total_energy(py)) / abs(
        diagnostics.total_energy(py)
    ) < 1e-9


def test_oracle_torque_free_energy():
    q0 = [np.cos(0.15), 0.0, 0.0, np.sin(0.15)]  # approx from_axis_angle z
    # Use identical axis-angle via python quaternion then feed both
    from physics_engine import quaternion as quat

    q = quat.from_axis_angle([1, 1, 0], 0.3)
    I = [1.0, 2.0, 3.5]
    w_body = np.array([0.4, 1.0, -2.0])
    w_world = quat.rotate(q, w_body)
    py = World(
        [
            RigidBody(
                mass=1.0,
                inertia_body=np.diag(I),
                orientation=q,
                angular_velocity=w_world,
            )
        ]
    )
    rs = pe.World(
        [
            pe.RigidBody(
                1.0,
                inertia_body=I,
                orientation=list(q),
                angular_velocity=list(w_world),
            )
        ]
    )
    py.run(5.0, 1e-3, integrators.rk4)
    rs.run(5.0, 1e-3, "rk4")
    np.testing.assert_allclose(rs.L[0], py.state.L[0], rtol=0, atol=1e-14)
    assert rs.kinetic_energy() == pytest.approx(diagnostics.kinetic_energy(py), rel=1e-12)


def test_rust_determinism_repeat():
    def once():
        w = pe.World(
            [pe.RigidBody(1.0, position=[0.0, 0.0, 10.0], velocity=[1.0, -0.5, 0.2])],
            [pe.UniformGravity(), pe.LinearDrag(0.05), pe.Spring(0, [0.0, 0.0, 0.0], 2.0)],
        )
        w.run(3.0, 1e-3, "rk4")
        return w.pack(), w.state_hash()

    a, ha = once()
    b, hb = once()
    assert np.array_equal(a, b)
    assert ha == hb


def test_canonical_cross_os_hash_matches_rust_fixture():
    """Locked hash from `cargo run -p pe-core --bin canonical_hash --release`."""
    expected = 0xCA172FE3E3E57C79
    assert pe.canonical_determinism_hash() == expected
