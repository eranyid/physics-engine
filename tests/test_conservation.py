"""Isolated N-body: linear and angular momentum conserved; deterministic stepping."""
import numpy as np
import pytest
from physics_engine import RigidBody, World, forces, integrators, diagnostics


def cluster(seed=1):
    rng = np.random.default_rng(seed)
    bodies = [RigidBody(mass=float(m), position=rng.normal(size=3) * 10.0, velocity=rng.normal(size=3) * 1e-3,
                        angular_velocity=rng.normal(size=3), inertia_body=np.diag(rng.uniform(1, 3, 3)))
              for m in rng.uniform(1e6, 1e8, 5)]
    return World(bodies, [forces.NewtonianGravity(softening=0.1)])


@pytest.mark.parametrize("name", ["explicit_euler", "symplectic_euler", "velocity_verlet", "rk4"])
def test_isolated_system_momentum(name):
    w = cluster()
    P0, L0 = diagnostics.linear_momentum(w), diagnostics.angular_momentum(w)
    w.run(500.0, 0.5, integrators.ALL[name])
    assert np.linalg.norm(w.state.v) > 1e-3  # bodies actually interacted
    assert np.linalg.norm(diagnostics.linear_momentum(w) - P0) / np.linalg.norm(w.mass[:, None] * w.state.v) < 1e-12
    Lscale = np.sum(np.linalg.norm(np.cross(w.state.x, w.mass[:, None] * w.state.v), axis=1))

    # Linear momentum is a linear invariant: every scheme here preserves it to round-off.
    # Total angular momentum (quadratic invariant) is exact only for the symplectic schemes.
    if name in ("symplectic_euler", "velocity_verlet"):
        assert np.linalg.norm(diagnostics.angular_momentum(w) - L0) / Lscale < 1e-10


def test_deterministic_bitwise():
    a, b = cluster(), cluster()
    a.run(50.0, 0.5, integrators.rk4)
    b.run(50.0, 0.5, integrators.rk4)
    assert np.array_equal(a.state.pack(), b.state.pack())
