"""Python oracles for M34 underwater physics (physics_engine_rs)."""

import math

import pytest

pe = pytest.importorskip("physics_engine_rs")

G = 9.80665
ALG = 1e-12


def test_o1_hydrostatic_pressure():
    assert pe.hydrostatic_pressure(10.0, pe.RHO_FRESHWATER, G, 0.0) == pytest.approx(
        98066.5, rel=ALG
    )
    assert pe.hydrostatic_pressure(10.0, pe.RHO_SEAWATER, G, 0.0) == pytest.approx(
        100518.1625, rel=ALG
    )
    assert pe.hydrostatic_pressure(10.0, pe.RHO_FRESHWATER, G, 101325.0) == pytest.approx(
        199391.5, rel=ALG
    )
    assert pe.hydrostatic_pressure(10.0, pe.RHO_SEAWATER, G, 101325.0) == pytest.approx(
        201843.1625, rel=ALG
    )
    assert pe.hydrostatic_pressure(30.0, 1028.0, G, 0.0) == pytest.approx(
        302437.086, rel=ALG
    )


def test_o2_buoyancy_1m3():
    assert pe.RHO_FRESHWATER * G == pytest.approx(9806.65, rel=ALG)
    assert pe.RHO_SEAWATER * G == pytest.approx(10051.81625, rel=ALG)


def test_o4_sphere_added_mass():
    a = pe.sphere_added_mass(0.1, pe.RHO_FRESHWATER)
    assert a == pytest.approx(2.0943951023931957, rel=ALG)
    vol = 4.0 / 3.0 * math.pi * 0.1**3
    m_n = pe.RHO_FRESHWATER * vol
    assert 1.0 / (m_n + a) == pytest.approx(0.15915494309189532, rel=ALG)
    m_d = 2.0 * pe.RHO_FRESHWATER * vol
    a0 = (m_d - pe.RHO_FRESHWATER * vol) * G / (m_d + a)
    assert a0 == pytest.approx(3.92266, abs=5e-6)


def test_o6_dispersion():
    lam0 = pe.wave_deep_water_wavelength(8.0, G)
    assert lam0 == pytest.approx(99.88971665101667, rel=ALG)
    omega = 2.0 * math.pi / 8.0
    assert pe.wave_solve_k(omega, 50.0, G) == pytest.approx(0.06312967213742765, rel=ALG)
    assert pe.wave_solve_k(omega, 20.0, G) == pytest.approx(0.07078053498187947, rel=ALG)
    assert pe.wave_solve_k(omega, 10.0, G) == pytest.approx(0.08864112882243125, rel=ALG)


def test_o10_spheroid_lamb_k():
    k1, k2, kp = pe.spheroid_lamb_k(1.0, 0.5)
    assert k1 == pytest.approx(0.21001504897664153, rel=ALG)
    assert k2 == pytest.approx(0.704210425850353, rel=ALG)
    assert kp == pytest.approx(0.23942389319515428, rel=ALG)
    k1b, k2b, kpb = pe.spheroid_lamb_k(0.9, 0.11)
    assert k1b == pytest.approx(0.028261311929324996, rel=ALG)
    assert k2b == pytest.approx(0.9465012650157768, rel=ALG)
    assert kpb == pytest.approx(0.844410854231194, rel=ALG)
    a6 = pe.spheroid_added_mass6(0.9, 0.11, pe.RHO_SEAWATER)
    assert len(a6) == 6
    assert a6[3] == 0.0


def test_canonical_underwater_hash_locked():
    a = pe.canonical_underwater_hash()
    b = pe.canonical_underwater_hash()
    assert a == b
    assert a == 0xBA1847A1A9BD042C
