"""Aircraft aerodynamics vs closed-form trim / polar / stall results (SI)."""
import numpy as np
import pytest
from physics_engine import RigidBody, World, forces, integrators, aero, atmosphere as atm, quaternion as quat
from physics_engine.units import STANDARD_GRAVITY as g0
from physics_engine.body import inertia_solid_box


def _light_aircraft(mass=1000.0):
    """Generic GA-like polar: S=16 m^2, b=10 m, c=1.6 m, CD0=0.025, e=0.8."""
    polar = aero.WingPolar(
        S=16.0, c=1.6, b=10.0, CD0=0.025, e=0.8,
        CL0=0.2, CL_alpha=5.7, alpha_stall=np.deg2rad(14.0), CL_fall=1.2,
        Cm0=0.0, Cm_alpha=-0.8, Cm_q=-12.0,
    )
    return mass, polar


def test_drag_polar_and_induced_drag():
    _, polar = _light_aircraft()
    CL = 0.8
    CD = polar.CD(CL)
    CD_ind = CL ** 2 / (np.pi * polar.e * polar.aspect_ratio)
    assert polar.aspect_ratio == pytest.approx(10.0 ** 2 / 16.0)
    assert CD == pytest.approx(polar.CD0 + CD_ind, rel=1e-15)


def test_lift_curve_linear_and_stall():
    _, polar = _light_aircraft()
    a = 0.1
    assert polar.CL(a) == pytest.approx(polar.CL0 + polar.CL_alpha * a)
    a_s = polar.alpha_stall
    cl_peak = polar.CL(a_s)
    cl_post = polar.CL(a_s + 0.05)
    assert cl_peak == pytest.approx(polar.CL0 + polar.CL_alpha * a_s)
    assert cl_post < cl_peak  # stall break
    assert polar.CL(-(a_s + 0.05)) > polar.CL(-a_s)  # symmetric fall on negative side (less negative)


def test_best_glide_ratio_matches_polar():
    """Max L/D for a parabolic polar is at CL = sqrt(π e AR CD0); (L/D)_max = 1/(2 sqrt(CD0 K))."""
    _, polar = _light_aircraft()
    K = 1.0 / (np.pi * polar.e * polar.aspect_ratio)
    CL_opt = np.sqrt(polar.CD0 / K)
    LD_max = CL_opt / (polar.CD0 + K * CL_opt ** 2)
    assert LD_max == pytest.approx(1.0 / (2.0 * np.sqrt(polar.CD0 * K)), rel=1e-12)
    alpha = (CL_opt - polar.CL0) / polar.CL_alpha
    assert polar.L_over_D(alpha) == pytest.approx(LD_max, rel=1e-12)


def test_level_flight_trim_lift_equals_weight_thrust_equals_drag():
    mass, polar = _light_aircraft()
    rho = atm.RHO0
    trim = aero.level_flight_trim(mass, polar, rho)
    W = mass * g0
    q = 0.5 * rho * trim["V"] ** 2
    L = q * polar.S * trim["CL"]
    D = q * polar.S * trim["CD"]
    assert L == pytest.approx(W, rel=1e-12)
    assert trim["thrust"] == pytest.approx(D, rel=1e-12)
    assert trim["L_over_D"] == pytest.approx(L / D, rel=1e-12)
    # Force generator at trim: net aero+thrust+weight ≈ 0 in steady wings-level flight.
    # Nose-up by α about −y (z-up world: +y pitch is nose-down); velocity stays horizontal.
    I = inertia_solid_box(mass, 8.0, 10.0, 2.0)
    q_att = quat.from_axis_angle([0, 1, 0], -trim["alpha"])
    body = RigidBody(
        mass=mass, inertia_body=I,
        position=[0, 0, 0.0],
        velocity=[trim["V"], 0.0, 0.0],
        orientation=q_att,
    )
    # Thrust along the freestream (body e_v) cancels drag; lift alone cancels weight.
    e_v_body = np.array([np.cos(trim["alpha"]), 0.0, -np.sin(trim["alpha"])])
    gens = [
        forces.UniformGravity(),
        aero.AircraftAero(0, polar, atmosphere=lambda h: rho),
        aero.FixedThrust(0, trim["thrust"], axis_body=e_v_body),
    ]
    w = World([body], gens)
    ad = w.generators[1].air_data(w, w.state)
    assert ad["alpha"] == pytest.approx(trim["alpha"], rel=1e-9)
    assert ad["lift"] == pytest.approx(mass * g0, rel=1e-9)
    assert ad["drag"] == pytest.approx(trim["thrust"], rel=1e-9)
    F, _ = w.forces(w.state, 0.0)
    np.testing.assert_allclose(F[0], [0, 0, 0], atol=1e-5)


def test_aero_forces_in_body_frame_at_known_alpha():
    """At α=0, V along +x_body: lift +z, drag −x."""
    polar = aero.WingPolar(S=10.0, c=1.0, b=10.0, CD0=0.02, e=1.0, CL0=0.5, CL_alpha=5.0,
                           alpha_stall=0.5, Cm0=0.0, Cm_alpha=0.0, Cm_q=0.0)
    rho = 1.225
    V = 50.0
    body = RigidBody(mass=500.0, inertia_body=np.eye(3) * 100.0, velocity=[V, 0, 0])
    gen = aero.AircraftAero(0, polar, atmosphere=lambda h: rho)
    w = World([body], [gen])
    F, T = w.forces(w.state, 0.0)
    qbar = 0.5 * rho * V * V
    CL, CD = polar.CL(0.0), polar.CD(polar.CL(0.0))
    np.testing.assert_allclose(F[0], [-qbar * polar.S * CD, 0, qbar * polar.S * CL], rtol=1e-12)
    np.testing.assert_allclose(T[0], [0, 0, 0], atol=1e-12)


def test_pitch_stability_derivative_produces_restoring_moment():
    polar = aero.WingPolar(S=16.0, c=1.6, b=10.0, CD0=0.025, e=0.8,
                           CL0=0.0, CL_alpha=5.0, alpha_stall=0.4,
                           Cm0=0.0, Cm_alpha=-1.0, Cm_q=0.0)
    rho, V, alpha = 1.225, 40.0, 0.1
    q_att = quat.from_axis_angle([0, 1, 0], -alpha)  # nose-up
    body = RigidBody(mass=800.0, inertia_body=np.diag([500.0, 800.0, 900.0]),
                     velocity=[V, 0.0, 0.0], orientation=q_att)
    w = World([body], [aero.AircraftAero(0, polar, atmosphere=lambda h: rho)])
    ad = w.generators[0].air_data(w, w.state)
    assert ad["alpha"] == pytest.approx(alpha, rel=1e-12)
    _, T = w.forces(w.state, 0.0)
    # Body-frame pitching moment = qbar S c Cm_α α < 0 (nose-down restoring).
    R = quat.to_matrix(q_att)
    m_b = R.T @ T[0]
    qbar = 0.5 * rho * V * V
    assert m_b[1] == pytest.approx(qbar * polar.S * polar.c * polar.Cm_alpha * alpha, rel=1e-9)
    assert m_b[1] < 0


def test_terminal_velocity_vertical_dive():
    mass, polar = _light_aircraft()
    # Neutral moment polar so the dive attitude stays nose-down.
    polar = aero.WingPolar(
        S=polar.S, c=polar.c, b=polar.b, CD0=polar.CD0, e=polar.e,
        CL0=0.0, CL_alpha=polar.CL_alpha, alpha_stall=polar.alpha_stall,
        Cm0=0.0, Cm_alpha=0.0, Cm_q=0.0,
    )
    rho = atm.RHO0
    Vt = aero.terminal_velocity_vertical(mass, polar, rho, alpha=0.0)
    CD = polar.CD(polar.CL(0.0))
    assert Vt == pytest.approx(np.sqrt(2 * mass * g0 / (rho * polar.S * CD)), rel=1e-15)
    # +y rotation by +π/2 points body +x along world −z (nose down).
    q_att = quat.from_axis_angle([0, 1, 0], np.pi / 2)
    body = RigidBody(
        mass=mass, inertia_body=inertia_solid_box(mass, 8, 10, 2),
        position=[0, 0, 8000.0], velocity=[0, 0, 0], orientation=q_att,
    )
    w = World([body], [
        forces.UniformGravity(),
        aero.AircraftAero(0, polar, atmosphere=lambda h: rho),
    ])
    w.run(90.0, 1e-2, integrators.rk4)
    assert -w.state.v[0, 2] == pytest.approx(Vt, rel=5e-3)


def test_fixed_and_jet_and_propeller_thrust():
    body = RigidBody(mass=10.0, inertia_body=np.eye(3), position=[0, 0, 0])
    w = World([body], [aero.FixedThrust(0, 100.0)])
    F, _ = w.forces(w.state, 0.0)
    np.testing.assert_allclose(F[0], [100.0, 0, 0])

    jet = aero.JetThrust(0, T_sl=10_000.0, throttle=0.5, n=1.0, atmosphere=lambda h: atm.RHO0)
    assert jet.thrust_magnitude(atm.RHO0) == pytest.approx(5000.0)
    assert jet.thrust_magnitude(0.5 * atm.RHO0) == pytest.approx(2500.0)

    prop = aero.PropellerThrust(0, T_static=2000.0, power=100_000.0, eta=0.8, throttle=1.0,
                                V_min=1.0, atmosphere=lambda h: atm.RHO0)
    # At high speed, power-limited: ηP/V = 80000/V
    assert prop.thrust_magnitude(80.0, atm.RHO0) == pytest.approx(1000.0)
    # At low speed, static-thrust limited
    assert prop.thrust_magnitude(1.0, atm.RHO0) == pytest.approx(2000.0)


def test_stall_reduces_lift_below_linear_extrapolation():
    _, polar = _light_aircraft()
    a = polar.alpha_stall + np.deg2rad(5.0)
    cl_lin = polar.CL0 + polar.CL_alpha * a
    assert polar.CL(a) < cl_lin
    assert polar.CL(a) < polar.CL(polar.alpha_stall)


@pytest.mark.slow
def test_trimmed_level_flight_holds_altitude():
    mass, polar = _light_aircraft()
    rho = atm.RHO0
    trim = aero.level_flight_trim(mass, polar, rho)
    q_att = quat.from_axis_angle([0, 1, 0], -trim["alpha"])
    body = RigidBody(
        mass=mass, inertia_body=inertia_solid_box(mass, 8, 10, 2),
        position=[0, 0, 1000.0],
        velocity=[trim["V"], 0.0, 0.0],
        orientation=q_att,
    )
    # Freeze attitude: zero pitch stiffness derivatives so we test force balance only.
    polar_rigid = aero.WingPolar(
        S=polar.S, c=polar.c, b=polar.b, CD0=polar.CD0, e=polar.e,
        CL0=polar.CL0, CL_alpha=polar.CL_alpha, alpha_stall=polar.alpha_stall,
        Cm0=0.0, Cm_alpha=0.0, Cm_q=0.0,
    )
    e_v_body = np.array([np.cos(trim["alpha"]), 0.0, -np.sin(trim["alpha"])])
    w = World([body], [
        forces.UniformGravity(),
        aero.AircraftAero(0, polar_rigid, atmosphere=lambda h: rho),
        aero.FixedThrust(0, trim["thrust"], axis_body=e_v_body),
    ])
    z0 = w.state.x[0, 2]
    w.run(5.0, 1e-3, integrators.rk4)
    assert w.state.x[0, 2] == pytest.approx(z0, abs=0.05)
    assert np.linalg.norm(w.state.v[0]) == pytest.approx(trim["V"], rel=1e-3)
