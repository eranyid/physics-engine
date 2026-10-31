import numpy as np
import pytest
from physics_engine import Sphere, Box, Capsule, quaternion as quat
from physics_engine.broadphase import sweep_and_prune, brute_force_pairs
from physics_engine.body import inertia_solid_box


def test_box_and_sphere_mass_properties():
    b = Box((0.5, 1.0, 1.5))
    np.testing.assert_allclose(b.inertia(3.0), inertia_solid_box(3.0, 1, 2, 3))
    assert b.volume() == pytest.approx(6.0)
    assert Sphere(2.0).volume() == pytest.approx(32 / 3 * np.pi)


def test_capsule_inertia_vs_monte_carlo():
    cap, m = Capsule(0.3, 0.5), 2.0
    rng = np.random.default_rng(0)
    p = rng.uniform([-0.3, -0.3, -0.8], [0.3, 0.3, 0.8], size=(1_000_000, 3))
    z = np.clip(p[:, 2], -0.5, 0.5)
    inside = p[:, 0] ** 2 + p[:, 1] ** 2 + (p[:, 2] - z) ** 2 <= 0.09
    q = p[inside]
    assert inside.mean() * 0.6 * 0.6 * 1.6 == pytest.approx(cap.volume(), rel=3e-3)
    I_mc = m * np.diag([np.mean(q[:, 1] ** 2 + q[:, 2] ** 2), np.mean(q[:, 0] ** 2 + q[:, 2] ** 2),
                        np.mean(q[:, 0] ** 2 + q[:, 1] ** 2)])
    np.testing.assert_allclose(np.diag(cap.inertia(m)), np.diag(I_mc), rtol=5e-3)


def test_rotated_box_aabb_is_tight():
    b = Box((1.0, 0.5, 0.25))
    R = quat.to_matrix(quat.from_axis_angle([0, 0, 1], np.pi / 4))
    lo, hi = b.aabb(np.zeros(3), R)
    v = b.vertices(np.zeros(3), R)
    np.testing.assert_allclose(lo, v.min(0), atol=1e-12)
    np.testing.assert_allclose(hi, v.max(0), atol=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_sweep_and_prune_equals_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 200
    c = rng.uniform(0, 10, (n, 3))
    e = rng.uniform(0.05, 0.8, (n, 3))
    lo, hi = c - e, c + e
    lo[:5, 0] = hi[5:10, 0]   # exact touching edge cases
    hi[:5, 0] = np.maximum(hi[:5, 0], lo[:5, 0])
    skip = rng.random(n) < 0.2
    expect = brute_force_pairs(lo, hi, skip)
    assert len(expect) > 50
    assert sweep_and_prune(lo, hi, skip) == expect
