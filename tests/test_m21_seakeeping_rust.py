"""Python oracles for M21 seakeeping / ITTC-57 / Nomoto (physics_engine_rs)."""

import math

import pytest

pe = pytest.importorskip("physics_engine_rs")


def test_cummins6_hash_locked():
    a = pe.canonical_cummins6_hash()
    b = pe.canonical_cummins6_hash()
    assert a == b
    assert a == 0xB72A60E18EFD983D


def test_ittc57_cf_at_re_1e9():
    cf = pe.ittc57_friction_cf(1e9)
    expect = 0.075 / ((9.0 - 2.0) ** 2)
    assert cf == pytest.approx(expect, rel=1e-14)


def test_nomoto_step_matches_analytic():
    k, t_const = 0.15, 8.0
    delta = math.radians(10.0)
    time = 40.0
    r = pe.nomoto_r_step(k, t_const, delta, time)
    psi = pe.nomoto_psi_step(k, t_const, delta, time)
    r_an = k * delta * (1.0 - math.exp(-time / t_const))
    psi_an = k * delta * (time - t_const * (1.0 - math.exp(-time / t_const)))
    assert r == pytest.approx(r_an, rel=1e-14)
    assert psi == pytest.approx(psi_an, rel=1e-14)


def test_encounter_frequency_head_seas():
    omega, u = 0.8, 5.0
    g = 9.80665
    we = pe.encounter_frequency_py(omega, u, 0.0)
    k = omega * omega / g
    assert we == pytest.approx(omega - u * k, rel=1e-12)
