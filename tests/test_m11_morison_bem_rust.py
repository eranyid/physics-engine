"""Python tests for M11 Morison / seakeeping / BEM (physics_engine_rs)."""
from __future__ import annotations

import math

import pytest

pe = pytest.importorskip("physics_engine_rs")
G = pe.STANDARD_GRAVITY


def test_hover_vi_and_fm():
    t, a = 10000.0, math.pi * 25.0
    vh = pe.rotor_hover_vi(t, pe.RHO0, a)
    assert vh == pytest.approx(math.sqrt(t / (2.0 * pe.RHO0 * a)), abs=1e-12)
    p = pe.rotor_ideal_hover_power(t, pe.RHO0, a)
    assert pe.rotor_figure_of_merit(t, p, pe.RHO0, a) == pytest.approx(1.0, abs=1e-12)


def test_bem_hover_converges():
    r = pe.bem_evaluate(
        2, 4.0, 0.3, 0.2, math.radians(10), math.radians(4), 16, 35.0, v_inf=0.0
    )
    assert r["converged"]
    assert r["thrust"] > 0.0
    assert r["power"] > 0.0
    a = math.pi * 16.0
    fm = pe.rotor_figure_of_merit(r["thrust"], r["power"], pe.RHO0, a)
    assert 0.3 < fm < 1.05


def test_prop_thrust_vs_advance():
    t0 = pe.bem_evaluate(3, 1.0, 0.15, 0.08, math.radians(25), math.radians(15), 12, 200.0, 0.0)[
        "thrust"
    ]
    t1 = pe.bem_evaluate(3, 1.0, 0.15, 0.08, math.radians(25), math.radians(15), 12, 200.0, 20.0)[
        "thrust"
    ]
    assert t0 > t1


def test_morison_inertia_vs_closed_form():
    kwargs = dict(
        amplitude=0.4,
        period=10.0,
        depth=25.0,
        diameter=0.3,
        z_bottom=-12.0,
        z_top=0.0,
        t=1.3,
        cm=pe.NOMINAL_CM_CIRCULAR,
        cd=0.0,
        n_segments=128,
    )
    f_num = pe.morison_vertical_force(**kwargs)
    f_cf = pe.morison_airy_inertia_closed_form(
        kwargs["amplitude"],
        kwargs["period"],
        kwargs["depth"],
        kwargs["diameter"],
        kwargs["z_bottom"],
        kwargs["z_top"],
        kwargs["t"],
        cm=kwargs["cm"],
    )
    denom = max(abs(f_cf[0]), 1.0)
    assert abs(f_num[0] - f_cf[0]) / denom < 5e-3


def test_long_wave_heave_rao_near_one():
    length, beam, draft = 80.0, 12.0, 3.0
    mass = pe.RHO_FRESHWATER * length * beam * draft
    iyy = mass * (length * length + draft * draft) / 12.0
    omega = 2.0 * math.pi / 60.0
    rao = pe.seakeeping_heave_rao(length, beam, draft, mass, iyy, omega, 200.0)
    assert abs(rao - 1.0) < 0.10


def test_box_heave_omega_positive():
    w = pe.seakeeping_box_heave_omega(40.0, 8.0, 2.0, pe.RHO_FRESHWATER * 40 * 8 * 2)
    assert 0.3 < w < 5.0


def test_roll_period_formula():
    i44, rho, vol, gm = 1.0e6, 1000.0, 500.0, 1.2
    t = pe.seakeeping_roll_period(i44, rho, vol, gm)
    expect = 2.0 * math.pi * math.sqrt(i44 / (rho * G * vol * gm))
    assert t == pytest.approx(expect, abs=1e-12)


def test_m11_hash_stable_and_locked():
    a = pe.canonical_m11_hash()
    b = pe.canonical_m11_hash()
    assert a == b
    assert a == 0x83AEBE96F5CB8E4C


def test_prandtl_tip_loss_bounds():
    f = pe.rotor_prandtl_tip_loss(3, 5.0, 2.5, 0.2)
    assert 0.0 < f <= 1.0
