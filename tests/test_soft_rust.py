"""Pytest checks for pe-soft Python bindings (NumPy state + materials)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pe = pytest.importorskip("physics_engine_rs")


def test_soft_material_table_unverified():
    names = pe.soft_material_names()
    assert "jelly" in names
    assert "rubber_soft" in names
    d = pe.soft_material_dict("jelly")
    assert d["verified"] == "false"
    assert float(d["density"]) > 0.0


def test_soft_lame_and_compliance():
    lam, mu = pe.soft_lame_parameters(2.0e11, 0.3)
    assert mu is not None and lam is not None
    assert abs(mu - 2.0e11 / (2.0 * 1.3)) < 1e-3
    alpha = pe.soft_stretch_compliance(1.0e6, 1.0, 0.01)
    assert alpha is not None
    assert abs(alpha * (1.0e6 * 0.01 / 1.0) - 1.0) < 1e-12


def test_soft_rope_numpy_and_step():
    w = pe.SoftWorld.rope([-0.5, 0.0, 1.0], [0.5, 0.0, 1.0], segments=8, mass=0.4)
    w.set_params(substeps=15, damping=1.0)
    w.add_ground_plane([0.0, 0.0, 1.0], 0.0)
    x0 = np.array(w.x(), copy=True)
    assert x0.shape == (9, 3)
    w.run(0.5, 1.0 / 60.0)
    x1 = np.array(w.x(), copy=True)
    assert x1.shape == x0.shape
    # Pins stay put
    np.testing.assert_allclose(x1[0], x0[0], atol=1e-12)
    np.testing.assert_allclose(x1[-1], x0[-1], atol=1e-12)
    pack = np.array(w.pack(), copy=True)
    assert pack.shape == (9 * 6,)
    assert math.isfinite(w.kinetic_energy())


def test_soft_cloth_hangs():
    w = pe.SoftWorld.cloth([0.0, 0.0, 1.0], nx=4, ny=4, mass=0.2, pin_corners=True)
    w.set_params(substeps=20, damping=1.5)
    w.run(1.0, 1.0 / 60.0)
    z = np.array(w.x())[:, 2]
    assert np.all(np.isfinite(z))
    assert z.min() > -0.05


def test_soft_cube_builds():
    w = pe.SoftWorld.soft_cube([0.0, 0.0, 0.5], size=0.25, divisions=2, material="jelly")
    assert w.n > 8
    w.set_params(substeps=10, damping=1.0)
    w.add_ground_plane([0.0, 0.0, 1.0], 0.0)
    w.step(1.0 / 60.0)
    assert np.isfinite(w.total_energy())


def test_canonical_soft_hash_stable():
    a = pe.canonical_soft_hash()
    b = pe.canonical_soft_hash()
    assert a == b
    assert isinstance(a, int)


def test_canonical_materials_hash_stable():
    a = pe.canonical_materials_hash()
    b = pe.canonical_materials_hash()
    assert a == b
    assert a == 0x19498AB91F6FEBDC


def test_j2_uniaxial_and_expansion_bindings():
    e = 2.0e11
    sy = 2.5e8
    h = 2.0e9
    eps_y = sy / e
    stress, eps_p, alpha, plastic = pe.soft_j2_uniaxial_step(
        e, 0.3, sy, h, eps_y + 1e-3, 0.0, 0.0
    )
    assert plastic
    assert stress > sy
    assert eps_p > 0.0 and alpha > 0.0
    dl = pe.soft_free_expansion_delta_l(1.0, 1e-5, 100.0)
    assert abs(dl - 1e-3) < 1e-15
    t = pe.soft_rod_fourier_mode1(0.5, 0.0, 1.0, 10.0, 1e-4)
    assert abs(t - 10.0) < 1e-12


def test_material_table_has_copper_and_fields():
    names = pe.soft_material_names()
    assert "copper_nominal" in names
    assert "polycarbonate_nominal" in names
    d = pe.soft_material_dict("copper_nominal")
    assert d["verified"] == "false"
    assert float(d["thermal_expansion"]) > 0.0
    assert float(d["hardening_modulus"]) >= 0.0
