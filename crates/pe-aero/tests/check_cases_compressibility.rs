//! NASA TM flat-Earth check cases + compressibility / Dryden oracles.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_aero::check_cases::{
    angular_momentum_about_origin, dragless_ballistic, mechanical_energy,
    quadratic_drag_terminal_speed, quat_norm, run_check_case, CheckCaseConfig, NasaCheckCase,
};
use pe_aero::compressibility::{
    ackeret_cl, isentropic, karman_tsien, normal_shock, prandtl_glauert, prandtl_glauert_factor,
};
use pe_aero::turbulence::{DrydenParams, DrydenTurbulence, SteadyWind};
use pe_math::libm_ops;
use pe_math::{Vec3, STANDARD_GRAVITY};

#[test]
fn case1_dragless_sphere_matches_ballistic() {
    let cfg = CheckCaseConfig::default();
    let t_end = 5.0;
    let dt = 1e-3;
    let w0 = pe_aero::build_check_case(NasaCheckCase::DraglessSphere, &cfg);
    let e0 = mechanical_energy(&w0, &w0.state, cfg.g);
    let w = run_check_case(NasaCheckCase::DraglessSphere, &cfg, t_end, dt);
    let (x_a, v_a) = dragless_ballistic(cfg.x0, cfg.v0, cfg.g, t_end);
    assert_relative_eq!(w.state.x[0].x, x_a.x, max_relative = 1e-9);
    assert_relative_eq!(w.state.x[0].y, x_a.y, max_relative = 1e-9);
    assert_relative_eq!(w.state.x[0].z, x_a.z, max_relative = 1e-9);
    assert_relative_eq!(w.state.v[0].x, v_a.x, max_relative = 1e-9);
    let e1 = mechanical_energy(&w, &w.state, cfg.g);
    assert_relative_eq!(e1, e0, max_relative = 1e-8);
}

#[test]
fn case2_tumbling_brick_conserves_l_and_rot_ke() {
    let cfg = CheckCaseConfig::default();
    let mut w = pe_aero::build_check_case(NasaCheckCase::TumblingBrick, &cfg);
    // Angular momentum about COM is s.l[0]; torque-free → L conserved.
    let l0 = w.state.l[0];
    let ke_r0 = {
        let om = w.omega_i(&w.state, 0);
        0.5 * om.dot(w.state.l[0])
    };
    w.run_for(2.0, 1e-3, pe_core::integrators::rk4);
    assert_abs_diff_eq!(w.state.l[0].x, l0.x, epsilon = 1e-10);
    assert_abs_diff_eq!(w.state.l[0].y, l0.y, epsilon = 1e-10);
    assert_abs_diff_eq!(w.state.l[0].z, l0.z, epsilon = 1e-10);
    let om = w.omega_i(&w.state, 0);
    let ke_r1 = 0.5 * om.dot(w.state.l[0]);
    assert_relative_eq!(ke_r1, ke_r0, max_relative = 1e-8);
    assert_abs_diff_eq!(quat_norm(w.state.q[0]), 1.0, epsilon = 1e-12);
    // Translation still follows ballistic COM motion.
    let (x_a, _) = dragless_ballistic(cfg.x0, cfg.v0, cfg.g, 2.0);
    assert_relative_eq!(w.state.x[0].x, x_a.x, max_relative = 1e-9);
}

#[test]
fn case3_sphere_drag_terminal_and_energy_loss() {
    let cfg = CheckCaseConfig {
        v0: Vec3::new(0.0, 0.0, 0.0),
        x0: Vec3::new(0.0, 0.0, 5000.0),
        ..Default::default()
    };
    let v_term = quadratic_drag_terminal_speed(cfg.mass, cfg.drag_k, cfg.g);
    assert!(v_term > 0.0);
    let mut w = pe_aero::build_check_case(NasaCheckCase::SphereWithDrag, &cfg);
    let e0 = mechanical_energy(&w, &w.state, cfg.g);
    // Fall long enough to approach terminal speed.
    w.run_for(60.0, 1e-2, pe_core::integrators::rk4);
    let speed = w.state.v[0].norm();
    assert_relative_eq!(speed, v_term, max_relative = 0.02);
    let e1 = mechanical_energy(&w, &w.state, cfg.g);
    assert!(e1 < e0, "drag must dissipate energy");
}

#[test]
fn normal_shock_m1_2_gamma_14() {
    let s = normal_shock(2.0, 1.4).unwrap();
    assert_abs_diff_eq!(s.m2, 0.5774, epsilon = 5e-4);
    assert_abs_diff_eq!(s.p_ratio, 4.5, epsilon = 1e-4);
    assert_abs_diff_eq!(s.rho_ratio, 2.6667, epsilon = 5e-4);
    assert_abs_diff_eq!(s.t_ratio, 1.6875, epsilon = 5e-4);
    assert_abs_diff_eq!(s.p0_ratio, 0.7209, epsilon = 5e-4);
}

#[test]
fn isentropic_sonic() {
    let s = isentropic(1.0, 1.4);
    assert_abs_diff_eq!(s.p_over_p0, 0.5283, epsilon = 5e-4);
    assert_abs_diff_eq!(s.t_over_t0, 0.8333, epsilon = 5e-4);
}

#[test]
fn prandtl_glauert_and_clamp() {
    let f = prandtl_glauert_factor(0.5);
    assert_relative_eq!(f, 1.0 / libm_ops::sqrt(1.0 - 0.25), max_relative = 1e-12);
    let cl = prandtl_glauert(1.0, 0.5);
    assert_relative_eq!(cl, f, max_relative = 1e-12);
    // Near M=1 the factor is finite (clamped).
    assert!(prandtl_glauert_factor(0.999).is_finite());
    assert!(prandtl_glauert_factor(1.0).is_finite());
}

#[test]
fn karman_tsien_and_ackeret() {
    let cp = karman_tsien(-0.5, 0.6);
    assert!(cp < -0.5); // more negative than incompressible for suction
    let cl = ackeret_cl(0.1, 2.0).unwrap();
    assert_relative_eq!(cl, 4.0 * 0.1 / libm_ops::sqrt(3.0), max_relative = 1e-12);
    assert!(ackeret_cl(0.1, 0.9).is_none());
}

#[test]
fn dryden_seeded_deterministic() {
    let p = DrydenParams::light_low_altitude();
    let mut a = DrydenTurbulence::new(p, 99);
    let mut b = DrydenTurbulence::new(p, 99);
    for _ in 0..20 {
        let ga = a.step(50.0, 0.01);
        let gb = b.step(50.0, 0.01);
        assert_eq!(ga.x.to_bits(), gb.x.to_bits());
        assert_eq!(ga.y.to_bits(), gb.y.to_bits());
        assert_eq!(ga.z.to_bits(), gb.z.to_bits());
    }
    let mut c = DrydenTurbulence::new(p, 100);
    assert_ne!(
        a.step(50.0, 0.01).x.to_bits(),
        c.step(50.0, 0.01).x.to_bits()
    );
}

#[test]
fn steady_wind_met() {
    let w = SteadyWind::from_met(10.0, 0.0); // from north → blows south (−y)
    assert_abs_diff_eq!(w.velocity.x, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(w.velocity.y, -10.0, epsilon = 1e-12);
}

#[test]
fn case2_angular_momentum_about_origin_finite() {
    let cfg = CheckCaseConfig::default();
    let w = pe_aero::build_check_case(NasaCheckCase::TumblingBrick, &cfg);
    let l = angular_momentum_about_origin(&w, &w.state);
    assert!(l.norm().is_finite());
    let _ = STANDARD_GRAVITY;
}
