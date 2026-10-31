"""Oracle tests: Python stage-3 aero/hydro vs Rust `physics_engine_rs` (M4).

Tolerances (see docs/DETERMINISM.md): pure algebra is near bit-exact; paths that
call `libm` (atan2/sin/sqrt/exp) vs NumPy may differ at ~1e-12..1e-15.
"""
from __future__ import annotations

import numpy as np
import pytest

from physics_engine import aero, atmosphere as atm, hydro, quaternion as quat
from physics_engine import RigidBody, World, forces
from physics_engine.units import STANDARD_GRAVITY as g0
from physics_engine.body import inertia_solid_box

pe = pytest.importorskip("physics_engine_rs")


def _light_polar_py():
    return aero.WingPolar(
        S=16.0, c=1.6, b=10.0, CD0=0.025, e=0.8,
        CL0=0.2, CL_alpha=5.7, alpha_stall=np.deg2rad(14.0), CL_fall=1.2,
        Cm0=0.0, Cm_alpha=-0.8, Cm_q=-12.0,
    )


def _light_polar_rs():
    return pe.WingPolar(
        16.0, 1.6, 10.0, CD0=0.025, e=0.8,
        CL0=0.2, CL_alpha=5.7, alpha_stall=np.deg2rad(14.0), CL_fall=1.2,
        Cm0=0.0, Cm_alpha=-0.8, Cm_q=-12.0,
    )


@pytest.mark.parametrize("h", [0.0, 1000.0, 5000.0, 11000.0, 20000.0])
def test_oracle_isa_state(h):
    assert pe.isa_temperature(h) == pytest.approx(atm.temperature(h), rel=0, abs=0.0)
    assert pe.isa_pressure(h) == pytest.approx(atm.pressure(h), rel=1e-14, abs=1e-9)
    assert pe.isa_density(h) == pytest.approx(atm.density(h), rel=1e-14, abs=1e-15)
    assert pe.isa_speed_of_sound(h) == pytest.approx(atm.speed_of_sound(h), rel=1e-14, abs=1e-12)


def test_oracle_polar_and_stall():
    py, rs = _light_polar_py(), _light_polar_rs()
    for a in (-0.2, 0.0, 0.1, py.alpha_stall, py.alpha_stall + 0.05):
        assert rs.CL(a) == pytest.approx(py.CL(a), rel=0, abs=0.0)
        cl = py.CL(a)
        assert rs.CD(cl) == pytest.approx(py.CD(cl), rel=0, abs=0.0)


def test_oracle_level_flight_trim():
    mass = 1000.0
    py_p, rs_p = _light_polar_py(), _light_polar_rs()
    py_t = aero.level_flight_trim(mass, py_p, atm.RHO0)
    rs_t = pe.level_flight_trim(mass, rs_p, pe.RHO0)
    for k in ("V", "alpha", "thrust", "CL", "CD", "L_over_D"):
        assert rs_t[k] == pytest.approx(py_t[k], rel=1e-14, abs=1e-14)


def test_oracle_aero_forces_at_alpha_zero():
    polar_py = aero.WingPolar(
        S=10.0, c=1.0, b=10.0, CD0=0.02, e=1.0, CL0=0.5, CL_alpha=5.0,
        alpha_stall=0.5, Cm0=0.0, Cm_alpha=0.0, Cm_q=0.0,
    )
    polar_rs = pe.WingPolar(
        10.0, 1.0, 10.0, CD0=0.02, e=1.0, CL0=0.5, CL_alpha=5.0,
        alpha_stall=0.5, Cm0=0.0, Cm_alpha=0.0, Cm_q=0.0,
    )
    rho, V = 1.225, 50.0
    py = World(
        [RigidBody(mass=500.0, inertia_body=np.eye(3) * 100.0, velocity=[V, 0, 0])],
        [aero.AircraftAero(0, polar_py, atmosphere=lambda h: rho)],
    )
    rs = pe.World(
        [pe.RigidBody(500.0, inertia_body=[100.0, 100.0, 100.0], velocity=[V, 0, 0])],
        [pe.AircraftAero(0, polar_rs, atmosphere=rho)],
    )
    Fp, _ = py.forces(py.state, 0.0)
    Fr, _ = rs.forces()
    np.testing.assert_allclose(Fr[0], Fp[0], rtol=1e-12, atol=1e-12)


def test_oracle_trim_net_force():
    mass = 1000.0
    py_p, rs_p = _light_polar_py(), _light_polar_rs()
    trim = aero.level_flight_trim(mass, py_p, atm.RHO0)
    q_att = quat.from_axis_angle([0, 1, 0], -trim["alpha"])
    e_v = np.array([np.cos(trim["alpha"]), 0.0, -np.sin(trim["alpha"])])
    I = inertia_solid_box(mass, 8.0, 10.0, 2.0)
    py = World(
        [RigidBody(mass=mass, inertia_body=I, velocity=[trim["V"], 0, 0], orientation=q_att)],
        [
            forces.UniformGravity(),
            aero.AircraftAero(0, py_p, atmosphere=lambda h: atm.RHO0),
            aero.FixedThrust(0, trim["thrust"], axis_body=e_v),
        ],
    )
    rs = pe.World(
        [pe.RigidBody(mass, inertia_body=I, velocity=[trim["V"], 0, 0], orientation=list(q_att))],
        [
            pe.UniformGravity(),
            pe.AircraftAero(0, rs_p, atmosphere=pe.RHO0),
            pe.FixedThrust(0, trim["thrust"], axis_body=list(e_v)),
        ],
    )
    Fp, _ = py.forces(py.state, 0.0)
    Fr, _ = rs.forces()
    np.testing.assert_allclose(Fr[0], Fp[0], atol=1e-5)


def test_oracle_buoyancy_equilibrium():
    hull_py = hydro.BoxHull(10.0, 4.0, 2.0)
    hull_rs = pe.BoxHull(10.0, 4.0, 2.0)
    mass = 20_000.0
    T = hull_py.upright_draft_for_mass(mass, hydro.RHO_FRESHWATER)
    z_c = -T + 0.5 * hull_py.depth
    I = inertia_solid_box(mass, 10, 4, 2)
    py = World(
        [RigidBody(mass=mass, inertia_body=I, position=[0, 0, z_c])],
        [forces.UniformGravity(), hydro.Buoyancy(0, hull_py, rho=hydro.RHO_FRESHWATER)],
    )
    rs = pe.World(
        [pe.RigidBody(mass, inertia_body=I, position=[0, 0, z_c])],
        [pe.UniformGravity(), pe.Buoyancy(0, hull_rs, rho=pe.RHO_FRESHWATER)],
    )
    Fp, _ = py.forces(py.state, 0.0)
    Fr, _ = rs.forces()
    np.testing.assert_allclose(Fr[0], Fp[0], atol=1e-8)


def test_oracle_gm_and_gz_curve():
    hull_py = hydro.BoxHull(30.0, 8.0, 4.0)
    hull_rs = pe.BoxHull(30.0, 8.0, 4.0)
    draft, KG = 2.0, 1.5
    assert hull_rs.upright_KB(draft) == pytest.approx(hull_py.upright_KB(draft))
    assert hull_rs.upright_BM_roll(draft) == pytest.approx(hull_py.upright_BM_roll(draft))
    assert hull_rs.upright_GM(draft, KG, True) == pytest.approx(hull_py.upright_GM(draft, KG, True))
    phis = [0.01, 0.05, np.deg2rad(5.0)]
    gz_py = hydro.box_righting_arm_curve(hull_py, draft, KG, phis)
    gz_rs = pe.box_righting_arm_curve(hull_rs, draft, KG, phis)
    np.testing.assert_allclose(gz_rs, gz_py, rtol=1e-9, atol=1e-12)


def test_oracle_roll_period():
    hull = hydro.BoxHull(30.0, 8.0, 4.0)
    draft, KG = 2.0, 1.5
    GM = hull.upright_GM(draft, KG)
    k = np.sqrt((hull.beam ** 2 + hull.depth ** 2) / 12.0)
    assert pe.roll_period(k, GM) == pytest.approx(hydro.roll_period(k, GM), rel=1e-15)


def test_oracle_thrust_models():
    jet_py = aero.JetThrust(0, T_sl=10_000.0, throttle=0.5, n=1.0, atmosphere=lambda h: atm.RHO0)
    jet_rs = pe.JetThrust(0, 10_000.0, throttle=0.5, n=1.0, atmosphere=pe.RHO0)
    assert jet_rs.thrust_magnitude(pe.RHO0) == pytest.approx(jet_py.thrust_magnitude(atm.RHO0))
    prop_py = aero.PropellerThrust(
        0, T_static=2000.0, power=100_000.0, eta=0.8, atmosphere=lambda h: atm.RHO0
    )
    prop_rs = pe.PropellerThrust(0, 2000.0, 100_000.0, eta=0.8, atmosphere=pe.RHO0)
    assert prop_rs.thrust_magnitude(80.0, pe.RHO0) == pytest.approx(
        prop_py.thrust_magnitude(80.0, atm.RHO0)
    )


def test_imo_is_code_exposed():
    phis = np.linspace(0.0, np.deg2rad(50.0), 81)
    gz = 0.45 * np.sin(phis)
    r = pe.check_imo_is_code(list(phis), list(gz), 0.5)
    assert r["passed"] is True
    gz_bad = 0.05 * np.sin(phis)
    r2 = pe.check_imo_is_code(list(phis), list(gz_bad), 0.05)
    assert r2["passed"] is False


def test_canonical_aero_hydro_hash_stable():
    h1 = pe.canonical_aero_hydro_hash()
    h2 = pe.canonical_aero_hydro_hash()
    assert h1 == h2
    assert h1 == 0x25ecc6e6a8911d8c
