//! Stage-3 aero analytic / published-value tests (parity with Python suite).

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_aero::atmosphere::{self, DensityModel, RHO0};
use pe_aero::{
    dynamic_pressure, level_flight_trim_earth, terminal_velocity_vertical, AircraftAero,
    FixedThrust, JetThrust, PropellerThrust, WingPolar,
};
use pe_core::forces::UniformGravity;
use pe_core::integrators::rk4;
use pe_core::{inertia_solid_box, RigidBody, World};
use pe_math::libm_ops;
use pe_math::{Quat, Vec3, STANDARD_GRAVITY};

#[test]
fn isa_table_vs_published() {
    // Torenbeek / ICAO Doc 7488 table (same fixtures as Python).
    let table = [
        (0.0, 288.15, 101_325.0, 1.2250, 340.29),
        (1000.0, 281.65, 89_874.0, 1.1117, 336.43),
        (5000.0, 255.65, 54_020.0, 0.7361, 320.53),
        (11_000.0, 216.65, 22_632.0, 0.3639, 295.07),
        (20_000.0, 216.65, 5475.0, 0.0880, 295.07),
    ];
    for (h, t, p, rho, a) in table {
        assert_abs_diff_eq!(atmosphere::temperature(h).unwrap(), t, epsilon = 0.01);
        let p_c = atmosphere::pressure(h).unwrap();
        assert!(
            (p_c - p).abs() <= (2e-4 * p).max(2.0),
            "p at {h}: {p_c} vs {p}"
        );
        let rho_c = atmosphere::density(h).unwrap();
        assert!(
            (rho_c - rho).abs() <= (2e-4 * rho).max(2e-4),
            "rho at {h}: {rho_c} vs {rho}"
        );
        let a_c = atmosphere::speed_of_sound(h).unwrap();
        assert!(
            (a_c - a).abs() <= (2e-4 * a).max(0.05),
            "a at {h}: {a_c} vs {a}"
        );
    }
}

#[test]
fn best_glide_ratio_matches_polar() {
    let polar = WingPolar::light_aircraft();
    let k = 1.0 / (core::f64::consts::PI * polar.e * polar.aspect_ratio);
    let cl_opt = libm_ops::sqrt(polar.cd0 / k);
    let ld_max = cl_opt / (polar.cd0 + k * cl_opt * cl_opt);
    assert_relative_eq!(
        ld_max,
        1.0 / (2.0 * libm_ops::sqrt(polar.cd0 * k)),
        max_relative = 1e-12
    );
    let alpha = (cl_opt - polar.cl0) / polar.cl_alpha;
    assert_relative_eq!(polar.l_over_d(alpha), ld_max, max_relative = 1e-12);
}

#[test]
fn level_flight_trim_net_force_near_zero() {
    let mass = 1000.0;
    let polar = WingPolar::light_aircraft();
    let trim = level_flight_trim_earth(mass, &polar, RHO0);
    let w_weight = mass * STANDARD_GRAVITY;
    let q = 0.5 * RHO0 * trim.v * trim.v;
    assert_relative_eq!(q * polar.s * trim.cl, w_weight, max_relative = 1e-12);
    assert_relative_eq!(trim.thrust, q * polar.s * trim.cd, max_relative = 1e-12);

    let q_att = Quat::from_axis_angle(Vec3::unit_y(), -trim.alpha);
    let e_v_body = Vec3::new(libm_ops::cos(trim.alpha), 0.0, -libm_ops::sin(trim.alpha));
    let body = RigidBody::new(mass)
        .with_inertia(inertia_solid_box(mass, 8.0, 10.0, 2.0))
        .with_velocity(Vec3::new(trim.v, 0.0, 0.0))
        .with_orientation(q_att);
    let aero = AircraftAero::new(0, polar.clone(), DensityModel::Constant(RHO0), Vec3::zero());
    let w = World::new(
        vec![body],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(aero),
            Box::new(FixedThrust::new(0, trim.thrust, e_v_body)),
        ],
    );
    let (f, _) = w.forces(&w.state, 0.0);
    assert_abs_diff_eq!(f[0].x, 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(f[0].y, 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(f[0].z, 0.0, epsilon = 1e-5);
}

#[test]
fn aero_forces_at_alpha_zero() {
    let polar = WingPolar::new(
        10.0, 1.0, 10.0, 0.02, 1.0, 0.5, 5.0, 0.5, 1.0, None, 0.0, 0.0, 0.0,
    )
    .unwrap();
    let rho = 1.225;
    let v = 50.0;
    let body = RigidBody::new(500.0)
        .with_inertia_diag(100.0, 100.0, 100.0)
        .with_velocity(Vec3::new(v, 0.0, 0.0));
    let w = World::new(
        vec![body],
        vec![Box::new(AircraftAero::new(
            0,
            polar.clone(),
            DensityModel::Constant(rho),
            Vec3::zero(),
        ))],
    );
    let (f, t) = w.forces(&w.state, 0.0);
    let qbar = dynamic_pressure(rho, v);
    let cl = polar.cl(0.0);
    let cd = polar.cd(cl);
    assert_relative_eq!(f[0].x, -qbar * polar.s * cd, max_relative = 1e-12);
    assert_abs_diff_eq!(f[0].y, 0.0, epsilon = 1e-12);
    assert_relative_eq!(f[0].z, qbar * polar.s * cl, max_relative = 1e-12);
    assert_abs_diff_eq!(t[0].norm(), 0.0, epsilon = 1e-12);
}

#[test]
fn pitch_restoring_moment() {
    let polar = WingPolar::new(
        16.0, 1.6, 10.0, 0.025, 0.8, 0.0, 5.0, 0.4, 1.0, None, 0.0, -1.0, 0.0,
    )
    .unwrap();
    let rho = 1.225;
    let v = 40.0;
    let alpha = 0.1;
    let q_att = Quat::from_axis_angle(Vec3::unit_y(), -alpha);
    let body = RigidBody::new(800.0)
        .with_inertia_diag(500.0, 800.0, 900.0)
        .with_velocity(Vec3::new(v, 0.0, 0.0))
        .with_orientation(q_att);
    let aero = AircraftAero::new(0, polar.clone(), DensityModel::Constant(rho), Vec3::zero());
    let w = World::new(vec![body], vec![Box::new(aero.clone())]);
    let ad = aero.air_data(&w, &w.state);
    assert_relative_eq!(ad.alpha, alpha, max_relative = 1e-12);
    let (_, tau) = w.forces(&w.state, 0.0);
    let m_b = q_att.inverse_rotate(tau[0]);
    let qbar = 0.5 * rho * v * v;
    assert_relative_eq!(
        m_b.y,
        qbar * polar.s * polar.c * polar.cm_alpha * alpha,
        max_relative = 1e-9
    );
    assert!(m_b.y < 0.0);
}

#[test]
fn terminal_dive_approaches_closed_form() {
    let base = WingPolar::light_aircraft();
    let polar = WingPolar::new(
        base.s,
        base.c,
        base.b,
        base.cd0,
        base.e,
        0.0,
        base.cl_alpha,
        base.alpha_stall,
        base.cl_fall,
        None,
        0.0,
        0.0,
        0.0,
    )
    .unwrap();
    let mass = 1000.0;
    let vt = terminal_velocity_vertical(mass, &polar, RHO0, 0.0, STANDARD_GRAVITY);
    let q_att = Quat::from_axis_angle(Vec3::unit_y(), core::f64::consts::FRAC_PI_2);
    let body = RigidBody::new(mass)
        .with_inertia(inertia_solid_box(mass, 8.0, 10.0, 2.0))
        .with_position(Vec3::new(0.0, 0.0, 8000.0))
        .with_orientation(q_att);
    let mut w = World::new(
        vec![body],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(AircraftAero::new(
                0,
                polar,
                DensityModel::Constant(RHO0),
                Vec3::zero(),
            )),
        ],
    );
    w.run_for(90.0, 1e-2, rk4);
    assert_relative_eq!(-w.state.v[0].z, vt, max_relative = 5e-3);
}

#[test]
fn thrust_models() {
    let body = RigidBody::new(10.0);
    let w = World::new(
        vec![body],
        vec![Box::new(FixedThrust::new(0, 100.0, Vec3::unit_x()))],
    );
    let (f, _) = w.forces(&w.state, 0.0);
    assert_abs_diff_eq!(f[0].x, 100.0, epsilon = 1e-12);

    let jet = JetThrust::new(
        0,
        10_000.0,
        0.5,
        1.0,
        RHO0,
        DensityModel::Constant(RHO0),
        Vec3::unit_x(),
    );
    assert_abs_diff_eq!(jet.thrust_magnitude(RHO0), 5000.0, epsilon = 1e-12);

    let prop = PropellerThrust::new(
        0,
        2000.0,
        100_000.0,
        0.8,
        1.0,
        1.0,
        RHO0,
        DensityModel::Constant(RHO0),
        Vec3::unit_x(),
        Vec3::zero(),
    );
    assert_abs_diff_eq!(prop.thrust_magnitude(80.0, RHO0), 1000.0, epsilon = 1e-12);
    assert_abs_diff_eq!(prop.thrust_magnitude(1.0, RHO0), 2000.0, epsilon = 1e-12);
}
