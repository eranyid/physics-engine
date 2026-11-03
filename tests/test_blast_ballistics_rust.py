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
    assert "5.56_55gr_fmjbt_federal_ae193" in names
    assert "308_168gr_sierra_matchking" in names
    e = pe.ammo_muzzle_energy("generic_pistol")
    data = json.loads(pe.ammo_profile_json("generic_pistol"))
    expect = 0.5 * data["mass_kg"] * data["muzzle_velocity_m_s"] ** 2
    assert e == pytest.approx(expect)
    fed = json.loads(pe.ammo_profile_json("5.56_55gr_fmjbt_federal_ae193"))
    assert fed["provenance"]["verified"] is True
    assert len(fed["provenance"]["note"].strip()) > 0
    assert fed["bc"]["provenance"]["verified"] is True
    assert fed["bc"]["drag_model"] == "G1"


def test_demo_volley_advances():
    xs = pe.ballistics_demo_volley("generic_intermediate_rifle", n=5, steps=30)
    assert len(xs) == 5
    assert all(x > 10.0 for x in xs)


def test_hashes_stable():
    assert pe.canonical_blast_hash() == pe.canonical_blast_hash()
    assert pe.canonical_blast_hash() == 0xB51E11E89D4DD055
    assert pe.canonical_blast_external_hash() == 0x693B60BE33F9B8F2
    assert pe.canonical_blast_fragments_hash() == 0x36D3F61D77C14859
    assert pe.canonical_blast_destruction_hash() == 0x38F55BAC9AB3D16F
    assert pe.canonical_ballistics_hash() == pe.canonical_ballistics_hash()


def test_gurney_cylindrical_cm_half():
    v = pe.blast_gurney_velocity(2440.0, 0.5, "cylindrical")
    assert v == pytest.approx(2440.0 * math.sqrt(0.4), rel=1e-12)


def test_mott_mean_closed_form():
    assert pe.blast_mott_mean_mass(0.01) == pytest.approx(0.02)
    samples = pe.blast_mott_sample_many(7, 0.01, 8)
    assert len(samples) == 8
    assert samples == pe.blast_mott_sample_many(7, 0.01, 8)


def test_rigid_wall_reflection_limits():
    assert pe.blast_rigid_wall_reflection(1e-8) == pytest.approx(2.0, rel=1e-5)
    assert pe.blast_rigid_wall_reflection(1e10) == pytest.approx(8.0, rel=1e-6)
