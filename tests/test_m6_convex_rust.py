"""M6 convex / GJK oracle checks against physics_engine_rs."""

from __future__ import annotations

import pytest

pe = pytest.importorskip("physics_engine_rs")


def test_canonical_convex_hash_locked():
    h = pe.canonical_convex_hash()
    assert h == pe.canonical_convex_hash()
    assert h == 0x4CA980096C036ADE


def test_legacy_collision_hash_unchanged():
    assert pe.canonical_collision_hash() == 0x50D52B2193AA528C


def test_convex_hull_box_euler():
    h = pe.ConvexHull.from_box(0.5, 0.5, 0.5)
    h.validate()
    assert h.n_vertices == 8
    assert h.n_faces >= 4
    assert abs(h.volume() - 1.0) < 1e-9


def test_capsule_box_contact_world_settles():
    body = pe.RigidBody(
        1.0,
        position=[0.0, 0.0, 0.8],
        inertia_body=pe.inertia_solid_sphere(1.0, 0.15),
    )
    w = pe.World([body], [pe.UniformGravity()])
    cw = pe.ContactWorld(
        w,
        [pe.Capsule(0.15, 0.2)],
        statics=[pe.StaticCollider(pe.Box([2.0, 2.0, 0.1]), position=[0.0, 0.0, -0.1])],
        materials=[pe.Material(0.0, 0.5)],
    )
    for _ in range(240):
        cw.step(1.0 / 240.0)
    z = float(cw.x[0, 2])
    assert z > 0.0
    assert z < 0.6


def test_rounded_shape_constructs():
    r = pe.Rounded(pe.Box([0.2, 0.2, 0.2]), 0.05)
    assert r.radius == 0.05
