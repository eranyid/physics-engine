//! Analytic Morison oracles: closed-form Airy inertia / drag vs segment integration.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_hydro::hull::RHO_SEAWATER;
use pe_hydro::morison::{
    airy_drag_closed_form_no_reversal, airy_inertia_closed_form, integrate_morison_fixed,
    MorisonMember, NOMINAL_CD_CIRCULAR, NOMINAL_CM_CIRCULAR,
};
use pe_hydro::waves::{AiryWave, Wave};
use pe_math::{Vec3, STANDARD_GRAVITY};

#[test]
fn airy_acceleration_matches_velocity_fd() {
    let w = AiryWave::earth(0.5, 8.0, 40.0, 0.0);
    let (x, y, z, t) = (1.0, 0.0, -2.0, 0.7);
    let a = w.acceleration(x, y, z, t);
    let dt = 1e-5;
    let fd = (w.velocity(x, y, z, t + dt) - w.velocity(x, y, z, t - dt)) * (0.5 / dt);
    assert_relative_eq!(a.x, fd.x, max_relative = 1e-6);
    assert_relative_eq!(a.z, fd.z, max_relative = 1e-6);
}

#[test]
fn inertia_force_converges_to_closed_form_finite_depth() {
    let wave = AiryWave::from_period(0.4, 10.0, 25.0, 0.0, 0.0, STANDARD_GRAVITY, 0.0);
    let d = 0.3;
    let z_b = -12.0;
    let z_t = 0.0;
    let t = 1.3; // general phase
    let (f_cf, m_cf) = airy_inertia_closed_form(
        &wave,
        0.0,
        0.0,
        z_b,
        z_t,
        d,
        NOMINAL_CM_CIRCULAR,
        RHO_SEAWATER,
        t,
    );
    let mut prev_err = f64::INFINITY;
    for &n in &[4usize, 8, 16, 32, 64, 128] {
        let member = MorisonMember::vertical_cylinder(z_b, z_t, d, NOMINAL_CM_CIRCULAR, 0.0, n);
        let load =
            integrate_morison_fixed(&member, &wave, RHO_SEAWATER, t, Vec3::new(0.0, 0.0, z_b));
        let err_f = (load.force - f_cf).norm() / f_cf.norm().max(1.0);
        let err_m = (load.moment - m_cf).norm() / m_cf.norm().max(1.0);
        assert!(
            err_f < prev_err + 1e-12,
            "force error should not increase: n={n} err={err_f} prev={prev_err}"
        );
        prev_err = err_f;
        if n >= 64 {
            // Midpoint rule on smooth integrand: well under 0.5% at 64 segments.
            assert!(err_f < 5e-3, "force rel err {err_f} at n={n} (tol 0.5%)");
            assert!(err_m < 5e-3, "moment rel err {err_m} at n={n} (tol 0.5%)");
        }
    }
}

#[test]
fn inertia_force_converges_deep_water() {
    // Deep: h = 500 m, T = 8 s ⇒ kh ≫ 1.
    let wave = AiryWave::earth(0.5, 8.0, 500.0, 0.0);
    let d = 0.4;
    let z_b = -20.0;
    let z_t = 0.0;
    let t = 0.5;
    let (f_cf, m_cf) = airy_inertia_closed_form(
        &wave,
        0.0,
        0.0,
        z_b,
        z_t,
        d,
        NOMINAL_CM_CIRCULAR,
        RHO_SEAWATER,
        t,
    );
    let member = MorisonMember::vertical_cylinder(z_b, z_t, d, NOMINAL_CM_CIRCULAR, 0.0, 128);
    let load = integrate_morison_fixed(&member, &wave, RHO_SEAWATER, t, Vec3::new(0.0, 0.0, z_b));
    assert_relative_eq!(load.force.x, f_cf.x, max_relative = 1e-3);
    assert_relative_eq!(load.moment.y, m_cf.y, max_relative = 1e-3);
}

#[test]
fn drag_force_at_crest_matches_closed_form() {
    // χ = 0 at origin/t=0 ⇒ cos χ = 1, no reversal.
    let wave = AiryWave::earth(0.3, 12.0, 40.0, 0.0);
    let d = 0.25;
    let z_b = -8.0;
    let z_t = 0.0;
    let t = 0.0;
    let f_cf = airy_drag_closed_form_no_reversal(
        &wave,
        0.0,
        0.0,
        z_b,
        z_t,
        d,
        NOMINAL_CD_CIRCULAR,
        RHO_SEAWATER,
        t,
    );
    let member = MorisonMember::vertical_cylinder(z_b, z_t, d, 0.0, NOMINAL_CD_CIRCULAR, 128);
    let load = integrate_morison_fixed(&member, &wave, RHO_SEAWATER, t, Vec3::zero());
    assert_relative_eq!(load.force.x, f_cf.x, max_relative = 2e-3);
    assert_abs_diff_eq!(load.force.y, 0.0, epsilon = 1e-9);
}

#[test]
fn segment_refinement_monotonic_for_full_morison() {
    let wave = AiryWave::earth(0.5, 9.0, 35.0, 0.2);
    let d = 0.35;
    let z_b = -15.0;
    let z_t = 0.0;
    let t = 2.1;
    let ref_member = MorisonMember::vertical_cylinder(
        z_b,
        z_t,
        d,
        NOMINAL_CM_CIRCULAR,
        NOMINAL_CD_CIRCULAR,
        256,
    );
    let reference = integrate_morison_fixed(&ref_member, &wave, RHO_SEAWATER, t, Vec3::zero());
    let mut prev = f64::INFINITY;
    for &n in &[8usize, 16, 32, 64] {
        let m = MorisonMember::vertical_cylinder(
            z_b,
            z_t,
            d,
            NOMINAL_CM_CIRCULAR,
            NOMINAL_CD_CIRCULAR,
            n,
        );
        let load = integrate_morison_fixed(&m, &wave, RHO_SEAWATER, t, Vec3::zero());
        let err = (load.force - reference.force).norm() / reference.force.norm().max(1.0);
        assert!(err <= prev + 1e-14, "n={n} err={err} prev={prev}");
        prev = err;
    }
    // 64 segments within 1% of 256-segment reference.
    assert!(prev < 0.01);
}
