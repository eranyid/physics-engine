"""Python oracles for pe-aero 6-DOF FDM (measured tolerances — do not loosen)."""

from __future__ import annotations

import math

import pytest

pe = pytest.importorskip("physics_engine_rs")


def test_canonical_fdm_hashes_locked():
    assert f"{pe.canonical_fdm_hash():016x}" == "ba6d68e43c69d315"
    assert f"{pe.canonical_dryden_gust_hash():016x}" == "2895b75a14f621c5"
    assert f"{pe.canonical_fdm_gust_hash():016x}" == "58f491fb7f3b34a6"
    assert pe.canonical_fdm_hash() == pe.canonical_fdm_hash()
    assert pe.canonical_fdm_gust_hash() == pe.canonical_fdm_gust_hash()


def test_phugoid_approximation_lanchester():
    v = 50.0
    cl, cd = 0.41, 0.05
    w, z = pe.fdm_phugoid_approximation(v, cl, cd)
    g = pe.STANDARD_GRAVITY
    assert w == pytest.approx(g * math.sqrt(2.0) / v, rel=1e-15)
    assert z == pytest.approx(cd / (math.sqrt(2.0) * cl), rel=1e-15)


def test_coordinated_turn_within_half_percent():
    for deg in (15.0, 30.0, 45.0, 60.0):
        phi = math.radians(deg)
        n = pe.fdm_coordinated_turn_n(phi)
        assert n == pytest.approx(1.0 / math.cos(phi), rel=1e-15)
        rate = pe.fdm_coordinated_turn_rate(phi, 50.0)
        assert rate == pytest.approx(pe.STANDARD_GRAVITY * math.tan(phi) / 50.0, rel=1e-15)
        assert abs(n / (1.0 / math.cos(phi)) - 1.0) < 0.005
        assert abs(rate / (pe.STANDARD_GRAVITY * math.tan(phi) / 50.0) - 1.0) < 0.005


def test_navion_longitudinal_vs_published():
    m = pe.fdm_navion_longitudinal_modes()
    assert m["verified"] is True
    # Published: phugoid −0.0164±0.2135i (ζ=0.0766); SP −2.444±2.538i (ζ=0.6937).
    w_eig = m["phugoid_omega_n"]
    z_eig = m["phugoid_zeta"]
    assert abs(w_eig - 0.2135064) / 0.2135064 < 0.005
    assert abs(z_eig - 0.076589) < 0.012
    w_sp = m["short_period_omega_n"]
    z_sp = m["short_period_zeta"]
    w_sp_pub = math.sqrt(2.444170**2 + 2.537795**2)
    assert abs(w_sp - w_sp_pub) / w_sp_pub < 0.03
    assert abs(z_sp - 0.6936957) < 0.012
    # α̇ raises short-period damping.
    assert m["short_period_zeta"] > m["short_period_zeta_no_alphadot"] + 0.05
    # Lanchester sanity at its true gap (~20% in ω).
    w_approx = m["phugoid_omega_approx"]
    assert abs(w_eig - w_approx) / w_approx < 0.25
    assert m["trim_residual"] < 1e-3


def test_navion_lateral_vs_published():
    m = pe.fdm_navion_lateral_modes()
    assert m["verified"] is True
    # Published: roll −8.255, spiral −0.0088, Dutch −0.4743±2.3109i.
    assert abs(m["roll"] - (-8.2550)) / 8.2550 < 0.08
    assert abs(m["spiral"] - (-0.0088)) < 0.002
    assert abs(m["dutch_roll_omega_n"] - 2.3591) / 2.3591 < 0.12
    assert abs(m["dutch_roll_zeta"] - 0.2010) < 0.02


def test_ground_effect_formula():
    for h_b in (0.05, 0.1, 0.25, 0.5, 1.0):
        r = 16.0 * h_b
        sigma = (r * r) / (1.0 + r * r)
        assert pe.fdm_wieselsberger_sigma(h_b) == pytest.approx(sigma, rel=1e-12)
        assert pe.fdm_mccormick_lift_factor(h_b) == pytest.approx(2.0 - sigma, rel=1e-12)
