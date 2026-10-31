"""Oracle tests: Python ContactWorld vs Rust physics_engine_rs.ContactWorld.

Tolerances (see docs/DETERMINISM.md):
- Aim for near bit-exact on the same algorithm/ordering.
- Measured max |Rust−Python| is scenario-specific; each bound below is documented.
- Within Rust, packed state is bit-identical on repeat (libm, sorted pairs, no HashMap).
"""
from __future__ import annotations

import numpy as np
import pytest

from physics_engine import (
    RigidBody,
    World,
    ContactWorld,
    StaticCollider,
    Sphere,
    Box,
    Plane,
    Material,
    forces,
    diagnostics as dg,
    quaternion as quat,
)
from physics_engine.units import STANDARD_GRAVITY as g0

pe = pytest.importorskip("physics_engine_rs")


def _py_make(bodies, shapes, statics=(), mats=None, gravity=True, **kw):
    w = World(bodies, [forces.UniformGravity()] if gravity else [])
    return ContactWorld(w, shapes, statics, mats, **kw)


def _rs_make(bodies_rs, shapes_rs, statics=(), mats=None, gravity=True, **kw):
    gens = [pe.UniformGravity()] if gravity else []
    w = pe.World(bodies_rs, gens)
    return pe.ContactWorld(w, shapes_rs, list(statics), mats, **kw)


def _diag_inertia(i3):
    return np.diag(i3).tolist()


def test_oracle_head_on_elastic_equal_mass():
    """Equal-mass e=1 head-on: velocity exchange; expect ~machine agreement."""
    e = 1.0
    py = _py_make(
        [
            RigidBody(1.0, Sphere(0.5).inertia(1.0), position=[-1, 0, 0], velocity=[2, 0, 0]),
            RigidBody(1.0, Sphere(0.5).inertia(1.0), position=[1, 0, 0], velocity=[-2, 0, 0]),
        ],
        [Sphere(0.5)] * 2,
        mats=[Material(e, 0.0)] * 2,
        gravity=False,
        restitution_threshold=0.0,
    )
    rs = _rs_make(
        [
            pe.RigidBody(1.0, position=[-1.0, 0.0, 0.0], velocity=[2.0, 0.0, 0.0],
                         inertia_body=_diag_inertia(Sphere(0.5).inertia(1.0))),
            pe.RigidBody(1.0, position=[1.0, 0.0, 0.0], velocity=[-2.0, 0.0, 0.0],
                         inertia_body=_diag_inertia(Sphere(0.5).inertia(1.0))),
        ],
        [pe.Sphere(0.5)] * 2,
        mats=[pe.Material(e, 0.0)] * 2,
        gravity=False,
        restitution_threshold=0.0,
    )
    py.run(1.0, 1e-3)
    rs.run(1.0, 1e-3)
    np.testing.assert_allclose(rs.v[0], py.world.state.v[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(rs.v[1], py.world.state.v[1], rtol=0, atol=1e-12)
    assert np.max(np.abs(rs.pack() - py.world.state.pack())) < 1e-12


def test_oracle_block_on_incline_slide():
    theta, mu, T = np.radians(30), 0.3, 1.0
    n = np.array([np.sin(theta), 0, np.cos(theta)])
    tdown = np.array([np.cos(theta), 0, -np.sin(theta)])
    hb = (0.2, 0.2, 0.05)
    q = quat.from_axis_angle([0, 1, 0], theta)
    py = _py_make(
        [RigidBody(1.0, Box(hb).inertia(1.0), position=n * (hb[2] - 1e-6), orientation=q)],
        [Box(hb)],
        [StaticCollider(Plane(tuple(n)), material=Material(0, mu))],
        [Material(0, mu)],
    )
    rs = _rs_make(
        [pe.RigidBody(1.0, position=list(n * (hb[2] - 1e-6)), orientation=list(q),
                      inertia_body=_diag_inertia(Box(hb).inertia(1.0)))],
        [pe.Box(hb)],
        [pe.StaticCollider(pe.Plane(list(n)), material=pe.Material(0.0, mu))],
        [pe.Material(0.0, mu)],
    )
    py.run(T, 1 / 480)
    rs.run(T, 1 / 480)
    a = g0 * (np.sin(theta) - mu * np.cos(theta))
    assert rs.v[0] @ tdown == pytest.approx(a * T, rel=1e-9)
    assert py.world.state.v[0] @ tdown == pytest.approx(a * T, rel=1e-9)
    # Cross-impl: same discrete contact path → expect very tight agreement.
    assert abs((rs.v[0] - py.world.state.v[0]) @ tdown) < 1e-10


def test_oracle_block_on_incline_static():
    theta, mu = np.radians(30), 0.6
    assert np.tan(theta) < mu
    n = np.array([np.sin(theta), 0, np.cos(theta)])
    hb = (0.2, 0.2, 0.05)
    q = quat.from_axis_angle([0, 1, 0], theta)

    def run_rs(T):
        cw = _rs_make(
            [pe.RigidBody(1.0, position=list(n * (hb[2] - 1e-6)), orientation=list(q),
                          inertia_body=_diag_inertia(Box(hb).inertia(1.0)))],
            [pe.Box(hb)],
            [pe.StaticCollider(pe.Plane(list(n)), material=pe.Material(0.0, mu))],
            [pe.Material(0.0, mu)],
        )
        cw.run(T, 1 / 240)
        return cw

    s1, s2 = run_rs(0.5), run_rs(2.0)
    tdown = np.array([np.cos(theta), 0, -np.sin(theta)])
    assert abs(s1.x[0] @ tdown) < 1e-4
    assert abs((s2.x[0] - s1.x[0]) @ tdown) < 1e-9


def test_oracle_rolling_sphere_five_sevenths():
    v0, mu, r = 3.0, 0.3, 0.1
    py = _py_make(
        [RigidBody(1.0, Sphere(r).inertia(1.0), position=[0, 0, r - 1e-6], velocity=[v0, 0, 0])],
        [Sphere(r)],
        [StaticCollider(Plane((0, 0, 1)), material=Material(0, mu))],
        [Material(0, mu)],
    )
    rs = _rs_make(
        [pe.RigidBody(1.0, position=[0.0, 0.0, r - 1e-6], velocity=[v0, 0.0, 0.0],
                      inertia_body=_diag_inertia(Sphere(r).inertia(1.0)))],
        [pe.Sphere(r)],
        [pe.StaticCollider(pe.Plane([0.0, 0.0, 1.0]), material=pe.Material(0.0, mu))],
        [pe.Material(0.0, mu)],
    )
    py.run(1.0, 1e-3)
    rs.run(1.0, 1e-3)
    assert rs.v[0, 0] == pytest.approx(5 / 7 * v0, rel=1e-5)
    assert abs(rs.v[0, 0] - py.world.state.v[0, 0]) < 1e-9


def test_oracle_sphere_box_bounce_restitution():
    """Drop sphere onto static box; apex ratio ≈ e² (detection lag ≤ dt)."""
    e, r, h, dt = 0.8, 0.1, 1.0, 1e-4
    ground = pe.StaticCollider(pe.Plane([0.0, 0.0, 1.0]), material=pe.Material(e, 0.0))
    rs = _rs_make(
        [pe.RigidBody(1.0, position=[0.0, 0.0, r + h],
                      inertia_body=_diag_inertia(Sphere(r).inertia(1.0)))],
        [pe.Sphere(r)],
        [ground],
        [pe.Material(e, 0.0)],
    )
    zs, vs = [], []
    t = 0.0
    while t < 2.5 - 1e-12:
        rs.step(dt)
        t = rs.t
        zs.append(rs.x[0, 2])
        vs.append(rs.v[0, 2])
    zs, vs = map(np.array, (zs, vs))
    apex = np.where((vs[1:] <= 0) & (vs[:-1] > 0))[0] + 1
    assert len(apex) >= 4
    ratios = (zs[apex[1:4]] - r) / (zs[apex[0:3]] - r)
    np.testing.assert_allclose(ratios, e * e, rtol=3e-3)


def test_oracle_three_box_stack_short():
    """Short stack compare (0.5 s) — full 5 s is marked slow in Python suite."""
    hb = (0.5, 0.5, 0.5)
    bodies_py = [RigidBody(1.0, Box(hb).inertia(1.0), position=[0.0, 0.0, 0.5 + i]) for i in range(3)]
    py = _py_make(bodies_py, [Box(hb)] * 3,
                  [StaticCollider(Plane((0, 0, 1)), material=Material(0, 1.0))],
                  [Material(0, 0.6)] * 3, iterations=30)
    bodies_rs = [
        pe.RigidBody(1.0, position=[0.0, 0.0, 0.5 + i],
                     inertia_body=_diag_inertia(Box(hb).inertia(1.0)))
        for i in range(3)
    ]
    rs = _rs_make(
        bodies_rs,
        [pe.Box(hb)] * 3,
        [pe.StaticCollider(pe.Plane([0.0, 0.0, 1.0]), material=pe.Material(0.0, 1.0))],
        [pe.Material(0.0, 0.6)] * 3,
        iterations=30,
    )
    py.run(0.5, 1 / 240)
    rs.run(0.5, 1 / 240)
    # Stack settling: allow micron-scale drift between libm paths.
    np.testing.assert_allclose(rs.x, py.world.state.x, atol=1e-6)
    assert np.max(np.abs(rs.pack() - py.world.state.pack())) < 1e-5


def test_rust_contact_determinism_repeat():
    def once():
        cw = _rs_make(
            [pe.RigidBody(1.0, position=[0.0, 0.0, 0.5],
                          inertia_body=_diag_inertia(Box((0.2, 0.2, 0.2)).inertia(1.0)))],
            [pe.Box((0.2, 0.2, 0.2))],
            [pe.StaticCollider(pe.Plane([0.0, 0.0, 1.0]), material=pe.Material(0.0, 1.0))],
            [pe.Material(0.3, 0.5)],
        )
        cw.run(0.5, 1 / 240)
        return cw.pack(), cw.state_hash()

    a, ha = once()
    b, hb = once()
    assert np.array_equal(a, b)
    assert ha == hb


def test_canonical_collision_hash_matches_fixture():
    expected = 0x50D52B2193AA528C
    assert pe.canonical_collision_hash() == expected
