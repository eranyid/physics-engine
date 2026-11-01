"""Python oracles for pe-aero 6-DOF FDM (measured tolerances — do not loosen)."""

from __future__ import annotations

import math

import pytest

pe = pytest.importorskip("physics_engine_rs")


def test_canonical_fdm_hashes_locked():
    assert f"{pe.canonical_fdm_hash():016x}" == "fb8c25d5fe2184fd"
    assert f"{pe.canonical_dryden_gust_hash():016x}" == "233148553b199f90"
    assert pe.canonical_fdm_hash() == pe.canonical_fdm_hash()


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


def test_navion_modes_vs_textbook():
    m = pe.fdm_navion_longitudinal_modes()
    assert m["verified"] is True
    # Measured: phugoid vs Lanchester / Nelson root (docs/AERO.md).
    w_eig = m["phugoid_omega_n"]
    z_eig = m["phugoid_zeta"]
    w_approx = m["phugoid_omega_approx"]
    z_approx = m["phugoid_zeta_approx"]
    assert abs(w_eig - w_approx) / w_approx < 0.20
    assert abs(z_eig - z_approx) < 0.05
    assert abs(w_eig - 0.2135) / 0.2135 < 0.05
    assert abs(z_eig - 0.0766) < 0.02
    assert m["short_period_omega_n"] > w_eig
    assert 0.2 < m["short_period_zeta"] < 1.5
    assert m["trim_residual"] < 1e-3
