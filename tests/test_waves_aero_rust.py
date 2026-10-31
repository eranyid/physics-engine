"""Python tests for Rust waves + compressibility (physics_engine_rs)."""
from __future__ import annotations

import math

import numpy as np
import pytest

pe = pytest.importorskip("physics_engine_rs")
G = pe.STANDARD_GRAVITY


def test_deep_water_wavelength_and_speeds():
    lam = pe.wave_deep_water_wavelength(10.0)
    assert lam == pytest.approx(G * 100.0 / (2.0 * math.pi), rel=0, abs=1e-12)
    assert lam == pytest.approx(156.13, abs=0.06)
    omega = 2.0 * math.pi / 10.0
    k = pe.wave_solve_k(omega, 1e6)
    c = pe.wave_phase_speed(omega, k)
    assert c == pytest.approx(15.61, abs=0.01)
    cg = pe.wave_group_speed(omega, k, 1e6)
    assert cg == pytest.approx(0.5 * c, rel=1e-6)


def test_shallow_phase_speed():
    h = 1.0
    omega = 0.05
    k = pe.wave_solve_k(omega, h)
    c = pe.wave_phase_speed(omega, k)
    assert c == pytest.approx(math.sqrt(G * h), rel=1e-4)


def test_airy_height_crest():
    assert pe.airy_height(1.0, 10.0, 1e5, 0.0, 0.0, 0.0) == pytest.approx(1.0, abs=1e-12)


def test_pm_hs():
    u = 10.0
    hs_a = pe.pm_hs_approx(u)
    assert hs_a == pytest.approx(0.21 * u * u / G, abs=1e-12)
    hs_d = pe.pm_hs_discrete(u, 200.0, n_omega=48, n_theta=12, seed=12345)
    assert abs(hs_d - hs_a) / hs_a < 0.08


def test_heave_period_added_mass():
    t = pe.heave_period_added_mass(24000.0, 1000.0, 1000.0, 48.0)
    expect = 2.0 * math.pi * math.sqrt((24000.0 + 1000.0) / (1000.0 * G * 48.0))
    assert t == pytest.approx(expect, abs=1e-12)


def test_wave_buoyancy_heave():
    hull = pe.BoxHull(12.0, 4.0, 2.0)
    mass = 24000.0
    draft = hull.upright_draft_for_mass(mass, pe.RHO_FRESHWATER)
    z = -draft + 0.5 * 2.0
    body = pe.RigidBody(mass, position=[0.0, 0.0, z], inertia_body=pe.inertia_solid_box(mass, 12, 4, 2))
    b = pe.Buoyancy(
        0,
        hull,
        wave_amplitude=0.3,
        wave_period=6.0,
        wave_depth=40.0,
        froude_krylov=True,
    )
    w = pe.World([body], [pe.UniformGravity(), b])
    z0 = float(w.x[0, 2])
    w.run(6.0, 1e-2, "rk4")
    assert abs(float(w.x[0, 2]) - z0) > 1e-3


def test_normal_shock_and_isentropic():
    s = pe.gas_normal_shock(2.0, 1.4)
    assert s["M2"] == pytest.approx(0.5774, abs=5e-4)
    assert s["p2_p1"] == pytest.approx(4.5, abs=1e-4)
    assert s["rho2_rho1"] == pytest.approx(2.6667, abs=5e-4)
    assert s["T2_T1"] == pytest.approx(1.6875, abs=5e-4)
    assert s["p02_p01"] == pytest.approx(0.7209, abs=5e-4)
    iso = pe.gas_isentropic(1.0, 1.4)
    assert iso["p_p0"] == pytest.approx(0.5283, abs=5e-4)
    assert iso["T_T0"] == pytest.approx(0.8333, abs=5e-4)


def test_compressibility_helpers():
    f = pe.compressibility_prandtl_glauert(1.0, 0.5)
    assert f == pytest.approx(1.0 / math.sqrt(0.75), rel=1e-12)
    assert pe.compressibility_ackeret_cl(0.1, 2.0) == pytest.approx(
        4.0 * 0.1 / math.sqrt(3.0), rel=1e-12
    )
    assert pe.compressibility_ackeret_cl(0.1, 0.5) is None
    cp = pe.compressibility_karman_tsien(-0.5, 0.6)
    assert cp < -0.5


def test_wave_body_hash_stable():
    a = pe.canonical_wave_body_hash()
    b = pe.canonical_wave_body_hash()
    assert a == b
    assert isinstance(a, int)
