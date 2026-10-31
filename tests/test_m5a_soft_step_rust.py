"""Python tests for M5a: soft-step solver choice and sleep state via physics_engine_rs."""
from __future__ import annotations

import numpy as np
import pytest

from physics_engine import Box

pe = pytest.importorskip("physics_engine_rs")


def _diag_inertia(i3):
    return np.diag(i3).tolist()


def _pyramid(rows: int, solver: str = "pgs", allow_sleep: bool = False):
    h = 0.5
    hb = (h, h, h)
    bodies = []
    shapes = []
    mats = []
    for row in range(rows):
        n = row + 1
        x0 = -(n - 1) * h
        for i in range(n):
            x = x0 + 2 * h * i
            z = h + 2 * h * row
            bodies.append(
                pe.RigidBody(
                    1.0,
                    position=[x, 0.0, z],
                    inertia_body=_diag_inertia(Box(hb).inertia(1.0)),
                )
            )
            shapes.append(pe.Box(hb))
            mats.append(pe.Material(0.0, 0.5))
    w = pe.World(bodies, [pe.UniformGravity()])
    ground = pe.StaticCollider(
        pe.Plane([0.0, 0.0, 1.0], 0.0), material=pe.Material(0.0, 1.0)
    )
    return pe.ContactWorld(
        w,
        shapes,
        [ground],
        mats,
        solver=solver,
        allow_sleep=allow_sleep,
        broad_phase="tree",
    )


def test_solver_mode_roundtrip():
    cw = _pyramid(2, solver="pgs")
    assert cw.solver == "pgs"
    cw.solver = "soft_step"
    assert cw.solver == "soft_step"
    with pytest.raises(ValueError):
        cw.solver = "nope"


def test_soft_step_pyramid_runs_and_hashes_stable():
    cw = _pyramid(3, solver="soft_step")
    cw.run(0.25, 1.0 / 120.0)
    assert np.all(np.isfinite(cw.pack()))
    h1 = pe.canonical_soft_step_collision_hash()
    h2 = pe.canonical_soft_step_collision_hash()
    assert h1 == h2
    assert h1 == 0x676D67A223120E34


def test_sleep_and_wake_exposed():
    cw = _pyramid(1, solver="soft_step", allow_sleep=True)
    assert cw.allow_sleep is True
    cw.run(2.0, 1.0 / 120.0)
    awake = np.asarray(cw.awake)
    assert awake.shape == (1,)
    assert awake[0] == False
    cw.wake_body(0)
    assert bool(cw.awake[0]) is True


def test_pgs_default_unchanged_for_oracle():
    """Oracle-compatible default remains PGS."""
    cw = _pyramid(2)
    assert cw.solver == "pgs"
    assert cw.allow_sleep is False
