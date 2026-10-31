"""Python bindings tests for pe-blast and pe-ballistics."""

from __future__ import annotations

import json
import math

import pytest

pe = pytest.importorskip("physics_engine_rs")


def test_kinney_peak_matches_formula():
    p0 = 101_325.0
    # Z=1 → ≈1008.79 kPa
    ps = pe.blast_peak_overpressure(1.0, p0)
    assert ps / 1e3 == pytest.approx(1008.7895037932, rel=1e-10)


def test_friedlander_impulse_triangle():
    i = pe.blast_friedlander_impulse(1e5, 0.01, 0.0)
    assert i == pytest.approx(500.0)


def test_scaled_distance():
    z = pe.blast_scaled_distance(2.0, 8.0, 1.0)
    assert z == pytest.approx(1.0)


def test_sedov_xi0():
    xi = pe.blast_sedov_xi0()
    assert abs(xi - 1.033) < 0.05


def test_vacuum_range():
    g = pe.STANDARD_GRAVITY
    r = pe.ballistics_vacuum_range(100.0, math.pi / 4, g)
    assert r == pytest.approx(100.0**2 / g)


def test_ammo_bundled_and_energy():
    names = pe.ammo_bundled_names()
    assert "generic_pistol" in names
    e = pe.ammo_muzzle_energy("generic_pistol")
    data = json.loads(pe.ammo_profile_json("generic_pistol"))
    expect = 0.5 * data["mass_kg"] * data["muzzle_velocity_m_s"] ** 2
    assert e == pytest.approx(expect)


def test_demo_volley_advances():
    xs = pe.ballistics_demo_volley("generic_intermediate_rifle", n=5, steps=30)
    assert len(xs) == 5
    assert all(x > 10.0 for x in xs)


def test_hashes_stable():
    assert pe.canonical_blast_hash() == pe.canonical_blast_hash()
    assert pe.canonical_ballistics_hash() == pe.canonical_ballistics_hash()
