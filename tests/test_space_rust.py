"""Python bindings tests for pe-space (M7 + IAS15/WHFast)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pe = pytest.importorskip("physics_engine_rs")


def test_earth_constants_exposed():
    assert pe.EARTH_MU == pytest.approx(3.986004418e14)
    assert pe.EARTH_RADIUS == pytest.approx(6_378_137.0)
    assert pe.EARTH_J2 == pytest.approx(1.08262668e-3)


def test_leo_period_formula():
    r = 7_000_000.0
    t = pe.kepler_period(pe.EARTH_MU, r)
    assert t == pytest.approx(5828.5, rel=2e-4)


def test_geo_period_formula():
    r = 42_164.17e3
    t = pe.kepler_period(pe.EARTH_MU, r)
    assert t == pytest.approx(86_164.09, rel=2e-6)


def test_elements_roundtrip():
    r, v = pe.kepler_elements_to_state(
        pe.EARTH_MU, 8e6, 0.1, 0.4, 0.2, 0.3, 0.5
    )
    a, e, i, raan, arg_p, nu = pe.kepler_state_to_elements(pe.EARTH_MU, r, v)
    assert a == pytest.approx(8e6, rel=1e-9)
    assert e == pytest.approx(0.1, rel=1e-9)


def test_kepler_propagate_one_period():
    r0 = np.array([7e6, 0.0, 0.0])
    v0 = np.array([0.0, math.sqrt(pe.EARTH_MU / 7e6), 0.0])
    t = pe.kepler_period(pe.EARTH_MU, 7e6)
    rf, vf = pe.kepler_propagate(pe.EARTH_MU, r0, v0, t)
    assert np.linalg.norm(rf - r0) < 1e-4
    assert np.linalg.norm(vf - v0) < 1e-8


def test_j2_nodal_rate_sso():
    a = pe.EARTH_RADIUS + 800_000.0
    want = 0.9856 * math.pi / 180.0 / 86400.0
    i = math.radians(98.6)
    rate = pe.j2_nodal_regression_rate(a, 0.0, i)
    assert rate == pytest.approx(want, rel=5e-3)


def test_figure_eight_period_error_leapfrog():
    err = pe.figure_eight_period_error(4000, integrator="leapfrog")
    assert err < 5e-3


def test_figure_eight_period_error_ias15():
    err = pe.figure_eight_period_error(200, integrator="ias15")
    assert err < 1e-8


def test_nbody_central_leo_radius():
    r = 7e6
    v = math.sqrt(pe.EARTH_MU / r)
    t = pe.kepler_period(pe.EARTH_MU, r)
    flat = pe.nbody_integrate(
        [1.0],
        [[r, 0.0, 0.0]],
        [[0.0, v, 0.0]],
        t_end=t,
        dt=t / 5000.0,
        integrator="leapfrog",
        mu=pe.EARTH_MU,
    )
    pos = np.asarray(flat).reshape(-1, 3)
    assert abs(np.linalg.norm(pos[0]) - r) < 1e-3


def test_nbody_ias15_leo_radius():
    r = 7e6
    v = math.sqrt(pe.EARTH_MU / r)
    t = pe.kepler_period(pe.EARTH_MU, r)
    flat = pe.nbody_integrate(
        [1.0],
        [[r, 0.0, 0.0]],
        [[0.0, v, 0.0]],
        t_end=t,
        dt=t / 100.0,
        integrator="ias15",
        mu=pe.EARTH_MU,
    )
    pos = np.asarray(flat).reshape(-1, 3)
    assert abs(np.linalg.norm(pos[0]) - r) < 1e-3


def test_nbody_whfast_two_body():
    # G=1, m0=1, m1=1e-3, circular a=1
    m0, m1, a = 1.0, 1e-3, 1.0
    v = math.sqrt((m0 + m1) / a)
    # COM frame
    x0 = -a * m1 / (m0 + m1)
    x1 = a * m0 / (m0 + m1)
    v0 = -v * m1 / (m0 + m1)
    v1 = v * m0 / (m0 + m1)
    flat = pe.nbody_integrate(
        [m0, m1],
        [[x0, 0.0, 0.0], [x1, 0.0, 0.0]],
        [[0.0, v0, 0.0], [0.0, v1, 0.0]],
        t_end=2.0 * math.pi,
        dt=0.01,
        integrator="whfast",
        geometric=True,
        g_const=1.0,
    )
    pos = np.asarray(flat).reshape(-1, 3)
    sep = np.linalg.norm(pos[1] - pos[0])
    assert abs(sep - a) < 1e-4


def test_canonical_space_hash_stable():
    h1 = pe.canonical_space_hash()
    h2 = pe.canonical_space_hash()
    assert h1 == h2
    assert isinstance(h1, int)


def test_canonical_ias15_hash_stable():
    h1 = pe.canonical_ias15_hash()
    h2 = pe.canonical_ias15_hash()
    assert h1 == h2
    assert h1 == 0xB918884CA264B70D
