"""Python bindings tests for pe-space (M7)."""

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
    # Solve for i from +0.9856 deg/day
    want = 0.9856 * math.pi / 180.0 / 86400.0
    # Bracket: inclination near 98.6 deg
    i = math.radians(98.6)
    rate = pe.j2_nodal_regression_rate(a, 0.0, i)
    assert rate == pytest.approx(want, rel=5e-3)


def test_figure_eight_period_error():
    err = pe.figure_eight_period_error(4000)
    assert err < 5e-3


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


def test_canonical_space_hash_stable():
    h1 = pe.canonical_space_hash()
    h2 = pe.canonical_space_hash()
    assert h1 == h2
    assert isinstance(h1, int)
