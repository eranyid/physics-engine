"""M5b joints / CCD oracle checks against physics_engine_rs."""

from __future__ import annotations

import math

import numpy as np
import pytest

pe = pytest.importorskip("physics_engine_rs")

G = pe.STANDARD_GRAVITY


def test_canonical_joints_hash_locked():
    h = pe.canonical_joints_hash()
    assert h == pe.canonical_joints_hash()
    assert h == 0x62156ADA3FA770AD


def test_pendulum_small_angle_period():
    length = 1.0
    body = pe.RigidBody(
        1.0,
        position=[0.05, 0.0, -length],
        inertia_body=pe.inertia_solid_sphere(1.0, 0.05),
    )
    w = pe.World([body], [pe.UniformGravity()])
    cw = pe.ContactWorld(w, [pe.Sphere(0.05)], solver="soft_step")
    cw.add_distance_joint(0, None, [0, 0, 0], [0, 0, 0], length, hertz=120.0, zeta=0.7)
    dt = 1.0 / 240.0
    xs = []
    ts = []
    for _ in range(2400):
        cw.step(dt)
        xs.append(float(cw.x[0, 0]))
        ts.append(cw.t)
    crossings = []
    for i in range(1, len(xs)):
        if xs[i - 1] <= 0.0 < xs[i]:
            frac = -xs[i - 1] / (xs[i] - xs[i - 1])
            crossings.append(ts[i - 1] + frac * (ts[i] - ts[i - 1]))
    assert len(crossings) >= 2
    t_meas = float(np.mean(np.diff(crossings)))
    t_theory = 2.0 * math.pi * math.sqrt(length / G)
    assert abs(t_meas - t_theory) / t_theory < 0.08


def test_breakable_joint_breaks():
    body = pe.RigidBody(
        1.0,
        position=[0.0, 0.0, -1.0],
        inertia_body=pe.inertia_solid_sphere(1.0, 0.05),
    )
    w = pe.World([body], [pe.UniformGravity()])
    cw = pe.ContactWorld(w, [pe.Sphere(0.05)], solver="soft_step")
    jid = cw.add_distance_joint(
        0, None, [0, 0, 0], [0, 0, 0], 1.0, hertz=60.0, zeta=1.0, break_force=5.0
    )
    for _ in range(240):
        cw.step(1.0 / 240.0)
    assert cw.joint_broken(jid)


def test_ccd_bullet_no_tunnel():
    body = pe.RigidBody(
        1.0,
        position=[0.0, 0.0, 1.0],
        velocity=[0.0, 0.0, -1000.0],
        inertia_body=pe.inertia_solid_sphere(1.0, 0.05),
    )
    w = pe.World([body], [])
    ground = pe.StaticCollider(pe.Plane([0, 0, 1], 0.0))
    cw = pe.ContactWorld(
        w,
        [pe.Sphere(0.05)],
        [ground],
        [pe.Material(0.0, 0.0)],
        solver="pgs",
    )
    cw.set_ccd(speculative=True, toi=True, speculative_distance=2.0)
    cw.set_bullet(0, True)
    cw.step(1.0 / 60.0)
    assert float(cw.x[0, 2]) >= 0.05 - 0.02
