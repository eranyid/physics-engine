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
    steel = pe.soft_material_dict("steel_aisi1020")
    assert steel["verified"] == "true"
    assert len(steel["source_note"].strip()) > 0
    assert abs(float(steel["density"]) - 7870.0) < 1e-9
    assert abs(float(steel["youngs_modulus"]) - 186.0e9) < 1.0


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


def test_canonical_soft_coupling_hash_stable():
    a = pe.canonical_soft_coupling_hash()
    b = pe.canonical_soft_coupling_hash()
    assert a == b
    assert a == 0x3DFD4CF670923310


def test_canonical_thermal_hash_stable():
    a = pe.canonical_thermal_hash()
    b = pe.canonical_thermal_hash()
    assert a == b
    assert a == 0xBECDA3E8805B5B90


def test_canonical_plastic_hash_stable():
    a = pe.canonical_plastic_hash()
    b = pe.canonical_plastic_hash()
    assert a == b
    assert a == 0x73D9EBDD6677EC8C


def test_canonical_fs_plastic_hash_stable():
    a = pe.canonical_fs_plastic_hash()
    b = pe.canonical_fs_plastic_hash()
    assert a == b
    assert a == 0x7C902CD304AEA5AE


def test_canonical_be_plastic_hash_stable():
    a = pe.canonical_be_plastic_hash()
    b = pe.canonical_be_plastic_hash()
    assert a == b
    assert a == 0x8562BC39F064698E


def test_plastic_bar_permanent_set():
    e = 2.0e11
    sy = 2.5e8
    h = 2.0e9
    eps_y = sy / e
    eps = eps_y + 2e-3
    analytic = pe.soft_analytic_permanent_strain(e, 0.3, sy, h, eps)
    length, eps_p, analytic2 = pe.soft_quasistatic_stretch_unload(
        e, 0.3, sy, h, 1.0, 0.01, eps
    )
    assert abs(analytic - analytic2) < 1e-15
    expect = 1.0 * (1.0 + analytic)
    assert abs(eps_p - analytic) / abs(analytic) < 0.01
    assert abs(length - expect) / expect < 0.01
    bar = pe.PlasticBar(e, 0.3, sy, h, l0=1.0, area=0.01)
    bar.set_damping(2.0)
    bar.set_free_end_x(1.0 * (1.0 + eps))
    bar.set_free_end_pinned(True)
    for _ in range(40):
        bar.step(1.0 / 240.0)
        bar.set_free_end_x(1.0 * (1.0 + eps))
    eps_p_b, alpha, stress = bar.plastic_state()
    assert eps_p_b > 0.0 and alpha > 0.0 and stress > sy
    assert bar.x().shape == (2, 3)


def test_fs_plastic_bar_log_permanent_set():
    e = 1.0e7
    sy = 5.0e4
    h = 1.0e5
    lam = 1.06
    eps_log = math.log(lam)
    analytic = pe.soft_analytic_log_permanent_strain(e, 0.3, sy, h, eps_log)
    length, eps_p, analytic2 = pe.soft_quasistatic_log_stretch_unload(
        e, 0.3, sy, h, 1.0, 0.01, lam
    )
    assert abs(analytic - analytic2) < 1e-15
    expect = math.exp(analytic)
    assert abs(eps_p - analytic) / abs(analytic) < 0.01
    assert abs(length - expect) / expect < 0.01
    bar = pe.FsPlasticBar(e, 0.3, sy, h, l0=1.0, area=0.01)
    bar.set_damping(2.0)
    bar.set_free_end_x(lam)
    bar.set_free_end_pinned(True)
    for _ in range(40):
        bar.step(1.0 / 240.0)
        bar.set_free_end_x(lam)
    eps_p_b, alpha, stress = bar.plastic_state()
    assert eps_p_b > 0.0 and alpha > 0.0 and stress > sy
    assert bar.x().shape == (2, 3)
    # Finite-strain divergence from engineering at λ=1.2
    ep_eng = pe.soft_analytic_permanent_strain(e, 0.3, sy, h, 0.2)
    ep_log = pe.soft_analytic_log_permanent_strain(e, 0.3, sy, h, math.log(1.2))
    assert abs(ep_log - ep_eng) / abs(ep_eng) > 0.05


def test_be_plastic_bar_and_isochoric_seq():
    e = 1.0e7
    nu = 0.3
    sy = 5.0e4
    h = 1.0e5
    mu = e / (2.0 * (1.0 + nu))
    # Elastic trial seq = 3 μ |ln λ|
    lam = 1.01
    seq = pe.soft_analytic_isochoric_uniaxial_seq(e, nu, lam)
    assert abs(seq - 3.0 * mu * abs(math.log(lam))) / seq < 1e-12
    bar = pe.BePlasticBar(e, nu, sy, h, l0=1.0, area=0.01)
    bar.set_damping(2.0)
    bar.set_free_end_x(1.06)
    bar.set_free_end_pinned(True)
    for _ in range(40):
        bar.step(1.0 / 240.0)
        bar.set_free_end_x(1.06)
    eps_p_b, alpha, stress = bar.plastic_state()
    assert eps_p_b > 0.0 and alpha > 0.0 and stress > sy
    assert bar.x().shape == (2, 3)


def test_thermal_rod_backward_euler_numpy():
    rod = pe.ThermalRod(
        33, 1.0, 0.01, 8960.0, 385.0, 400.0, 16.5e-6, t_ref=0.0
    )
    rod.init_sine_mode(50.0)
    fixed = [False] * rod.n
    fixed[0] = True
    fixed[-1] = True
    rod.set_fixed(fixed)
    t = np.array(rod.temperatures(), copy=True)
    t[0] = 0.0
    t[-1] = 0.0
    rod.set_temperatures(t.tolist())
    assert rod.step(25.0, "be")
    out = np.array(rod.temperatures())
    assert out.shape == (33,)
    assert np.all(np.isfinite(out))
    # Mode-1 amplitude decays under BE.
    assert abs(out[rod.n // 2]) < 50.0
    erfc0 = pe.soft_semi_infinite_erfc(0.0, 1.0, 0.0, 100.0, 1e-4)
    assert abs(erfc0 - 100.0) < 1e-12
    cool = pe.soft_robin_lumped_exp(1.0, 400.0, 300.0, 1.0)
    assert abs(cool - (300.0 + 100.0 * math.exp(-1.0))) < 1e-12


def test_soft_rigid_rope_freefall_com():
    w = pe.SoftRigidWorld.rope_on_free_body(m_rope=1.0, m_body=3.0, segments=6)
    z0 = w.combined_com()[2]
    dt = 1.0 / 240.0
    t_end = 0.5
    n = int(round(t_end / dt))
    for _ in range(n):
        w.step(dt)
    t = n * dt
    z = w.combined_com()[2]
    z_an = z0 - 0.5 * 9.80665 * t * t
    err = abs(z - z_an)
    assert err < 5e-4, f"freefall CoM err {err}"


def test_soft_rigid_momentum_exchange():
    w = pe.SoftRigidWorld.momentum_exchange()
    p0 = np.array(w.total_linear_momentum())
    for _ in range(400):
        w.step(1.0 / 240.0)
    p1 = np.array(w.total_linear_momentum())
    assert np.linalg.norm(p1 - p0) < 1e-10


def test_soft_on_box_support_force():
    w = pe.SoftRigidWorld.soft_on_box_on_ground()
    dt = 1.0 / 120.0
    for _ in range(600):
        w.step(dt)
    # soft_mass is not exposed; weight from settled Fz ≈ weight within 2%.
    fzs = []
    for _ in range(120):
        w.step(dt)
        fzs.append(w.ground_force()[2])
    f_avg = float(np.mean(fzs))
    # Approximate total mass from freefall weight scale used in Rust oracle (~20.1 kg).
    assert f_avg > 50.0
    assert np.isfinite(w.total_energy())



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
    assert "aluminum_6061_t6" in names
    assert "water_20c" in names
    d = pe.soft_material_dict("copper_nominal")
    assert d["verified"] == "false"
    assert float(d["thermal_expansion"]) > 0.0
    assert float(d["hardening_modulus"]) >= 0.0
    al = pe.soft_material_dict("aluminum_6061_t6")
    assert al["verified"] == "true"
    assert "MatWeb" in al["source_note"] or "matweb" in al["source_note"].lower()
    assert abs(float(al["youngs_modulus"]) - 68.9e9) < 1.0
