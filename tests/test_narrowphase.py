import numpy as np
import pytest
from physics_engine import Sphere, Box, Capsule, Plane, quaternion as quat
from physics_engine.narrowphase import collide

I = np.eye(3)


def rot(axis, ang):
    return quat.to_matrix(quat.from_axis_angle(axis, ang))


def test_sphere_sphere_depth_and_normal():
    c = collide(Sphere(1.0), np.zeros(3), I, Sphere(0.5), np.array([1.2, 0, 0]), I)
    assert len(c) == 1 and c[0].depth == pytest.approx(0.3)
    np.testing.assert_allclose(c[0].normal, [1, 0, 0])
    assert collide(Sphere(1.0), np.zeros(3), I, Sphere(0.5), np.array([1.51, 0, 0]), I) == []


def test_sphere_plane_and_symmetry():
    pl = Plane((0, 0, 1), 0.0)
    c = collide(Sphere(0.5), np.array([0, 0, 0.4]), I, pl, None, None)
    assert c[0].depth == pytest.approx(0.1)
    np.testing.assert_allclose(c[0].normal, [0, 0, -1])
    c2 = collide(pl, None, None, Sphere(0.5), np.array([0, 0, 0.4]), I)
    np.testing.assert_allclose(c2[0].normal, [0, 0, 1])


def test_box_plane_four_corners_and_tilted():
    pl = Plane((0, 0, 1), 0.0)
    c = collide(Box((1, 1, 1)), np.array([0, 0, 0.99]), I, pl, None, None)
    assert len(c) == 4 and all(k.depth == pytest.approx(0.01) for k in c)
    R = rot([0, 1, 0], np.pi / 4)   # edge down: lowest point at z = x - sqrt(2)
    c = collide(Box((1, 1, 1)), np.array([0, 0, 1.4]), R, pl, None, None)
    assert len(c) == 2
    assert c[0].depth == pytest.approx(np.sqrt(2) - 1.4)


def test_sphere_box_outside_edge_and_inside():
    b = Box((1, 1, 1))
    x = np.array([1.3, 1.3, 0])                    # near the z-edge
    c = collide(Sphere(0.5), x, I, b, np.zeros(3), I)
    assert c[0].depth == pytest.approx(0.5 - 0.3 * np.sqrt(2))
    np.testing.assert_allclose(c[0].normal, -np.array([1, 1, 0]) / np.sqrt(2))
    c = collide(Sphere(0.2), np.array([0.9, 0, 0]), I, b, np.zeros(3), I)
    assert c[0].depth == pytest.approx(0.3)
    np.testing.assert_allclose(c[0].normal, [-1, 0, 0])
    assert collide(Sphere(0.2), np.array([1.3, 0, 0]), I, b, np.zeros(3), I) == []


def test_box_box_face_manifold():
    A, B = Box((1, 1, 1)), Box((0.5, 0.5, 0.5))
    c = collide(A, np.zeros(3), I, B, np.array([0.2, 0.1, 1.45]), rot([0, 0, 1], 0.3))
    assert len(c) == 4
    for k in c:
        np.testing.assert_allclose(k.normal, [0, 0, 1], atol=1e-12)
        assert k.depth == pytest.approx(0.05)
    # contact points lie inside the top face of A
    assert all(abs(k.point[0]) <= 1 and abs(k.point[1]) <= 1 for k in c)


def test_box_box_offset_manifold_is_clipped():
    A = Box((1, 1, 1))
    c = collide(A, np.zeros(3), I, A, np.array([1.5, 0, 1.9]), I)  # overlap region x in [0.5, 1]
    xs = sorted(k.point[0] for k in c)
    assert len(c) == 4 and xs[0] == pytest.approx(0.5) and xs[-1] == pytest.approx(1.0)
    assert all(k.depth == pytest.approx(0.1) for k in c)


def test_box_box_edge_edge():
    A = Box((1, 1, 1))
    Ra = rot([1, 0, 0], np.pi / 4)               # edge along x on top at z = sqrt2
    Rb = rot([0, 1, 0], np.pi / 4)               # edge along y at bottom
    h = 2 * np.sqrt(2) - 0.1
    c = collide(A, np.zeros(3), Ra, A, np.array([0, 0, h]), Rb)
    assert len(c) == 1
    assert c[0].depth == pytest.approx(0.1, rel=1e-9)
    np.testing.assert_allclose(c[0].normal, [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(c[0].point, [0, 0, h / 2], atol=1e-9)


def test_box_box_separated_by_edge_axis():
    A = Box((1, 1, 1))
    Ra, Rb = rot([1, 0, 0], np.pi / 4), rot([0, 1, 0], np.pi / 4)
    # face axes overlap (AABBs overlap) but the edge-edge axis separates
    assert collide(A, np.zeros(3), Ra, A, np.array([0, 0, 2 * np.sqrt(2) + 0.01]), Rb) == []


def test_capsule_contacts():
    cap = Capsule(0.2, 0.5)
    R = rot([0, 1, 0], np.pi / 2)   # lying along x
    c = collide(cap, np.array([0, 0, 0.19]), R, Plane((0, 0, 1)), None, None)
    assert len(c) == 2 and all(k.depth == pytest.approx(0.01) for k in c)
    c = collide(cap, np.zeros(3), R, Sphere(0.3), np.array([0.3, 0, 0.45]), I)
    assert c[0].depth == pytest.approx(0.05)
    c = collide(cap, np.zeros(3), R, cap, np.array([0.1, 0, 0.85]), I)  # crossed, gap 0.35
    assert c[0].depth == pytest.approx(0.05)
    np.testing.assert_allclose(c[0].normal, [0, 0, 1], atol=1e-12)
