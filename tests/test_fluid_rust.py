"""Python checks for pe-fluid via physics_engine_rs (skip if wheel absent)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pe = pytest.importorskip("physics_engine_rs")


def test_canonical_fluid_hash_stable():
    a = pe.canonical_fluid_hash()
    b = pe.canonical_fluid_hash()
    assert a == b
    assert isinstance(a, int)


def test_canonical_dfsph_hash_stable():
    a = pe.canonical_dfsph_hash()
    b = pe.canonical_dfsph_hash()
    assert a == b
    assert isinstance(a, int)
    # Locked cross-OS value (docs/DETERMINISM.md); do not loosen silently.
    assert a == 0xF1F2825B5EEAAB0D


def test_canonical_coupling_hash_stable():
    a = pe.canonical_coupling_hash()
    b = pe.canonical_coupling_hash()
    assert a == b
    assert isinstance(a, int)
    # Locked cross-OS value (docs/DETERMINISM.md); do not loosen silently.
    assert a == 0xE9A6A84FEF4CE9D5


def test_archimedes_and_heave_oracles():
    d = pe.fluid_archimedes_box_draft(500.0, 1000.0, 0.16)
    assert d == pytest.approx(0.08, rel=0, abs=1e-15)
    t = pe.fluid_heave_period_analytic(19.2, 1000.0, 0.24, pe.STANDARD_GRAVITY)
    assert t == pytest.approx(
        2.0 * math.pi * math.sqrt(19.2 / (1000.0 * pe.STANDARD_GRAVITY * 0.24)),
        rel=0,
        abs=1e-12,
    )


def test_ritter_front_matches_formula():
    g, h0, t = 9.81, 1.0, 0.5
    xf = pe.fluid_ritter_front(g, h0, t)
    assert xf == pytest.approx(2.0 * math.sqrt(g * h0) * t, rel=0, abs=1e-12)
    h = pe.fluid_ritter_height(g, h0, 0.0, t)
    assert h == pytest.approx((4.0 / 9.0) * h0, rel=0, abs=1e-9)


def test_hydrostatic_pressure():
    p = pe.fluid_hydrostatic_pressure(1000.0, pe.STANDARD_GRAVITY, 2.0)
    assert p == pytest.approx(1000.0 * pe.STANDARD_GRAVITY * 2.0)


def test_pbf_tank_demo_numpy_shapes():
    pos, dens, err = pe.fluid_pbf_tank_demo(n_side=4, steps=20, dt=1.0 / 120.0, dim="3")
    assert isinstance(pos, np.ndarray)
    assert pos.ndim == 2 and pos.shape[1] == 3
    assert dens.shape == (pos.shape[0],)
    assert dens.dtype == np.float64
    assert err >= 0.0
    # Settling may not reach 1% in 20 steps; just sanity-bound
    assert err < 0.5


def test_dfsph_tank_demo_numpy_shapes():
    pos, dens, err = pe.fluid_dfsph_tank_demo(n_side=4, steps=20, dt=1.0 / 120.0, dim="3")
    assert isinstance(pos, np.ndarray)
    assert pos.ndim == 2 and pos.shape[1] == 3
    assert dens.shape == (pos.shape[0],)
    assert dens.dtype == np.float64
    assert err >= 0.0
    assert err < 0.5


def test_poly6_w0_positive():
    assert pe.fluid_poly6_w0(0.1) > 0.0


def test_canonical_mass_positive():
    assert pe.fluid_canonical_mass() > 0.0
    ke = pe.fluid_canonical_ke_after_hash_steps()
    assert ke >= 0.0
