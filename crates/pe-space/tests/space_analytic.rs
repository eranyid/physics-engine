//! Analytic / published-constant validation for `pe-space` (M7 + IAS15/WHFast).
//!
//! Only values derivable from formulas or published constants. Anything not
//! independently verified is labelled in comments / assertion messages.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_math::libm_ops;
use pe_math::Vec3;
use pe_space::constants::{
    deg_per_day_to_rad_per_s, EARTH_J2, EARTH_MU, EARTH_MU_KM3_S2, EARTH_RADIUS, EARTH_RADIUS_KM,
    TWO_PI,
};
use pe_space::diagnostics::{angular_momentum, total_energy};
use pe_space::field::{CentralMu, GravityField, NBodyPointMass};
use pe_space::integrator::{AdaptiveStep, Integrator};
use pe_space::integrators::{
    compute_radau_nodes, Ias15, Leapfrog, Rk4, Rkf78, Whfast, WhfastCoords, Yoshida4, RADAU_H,
};
use pe_space::kepler::{
    elements_to_state, mean_motion, orbital_period, propagate_universal, state_to_elements,
    KeplerElements,
};
use pe_space::perturbations::{accel_j2, j2_nodal_rate, J2Model};
use pe_space::scenarios::{
    figure_eight_hash, figure_eight_state, ias15_hash, FIGURE_EIGHT_PERIOD_RETURN,
};
use pe_space::state::NBodyState;

fn circular_leo_state() -> (NBodyState, f64, f64) {
    let r_km = 7000.0;
    let r = r_km * 1e3;
    let mu = EARTH_MU;
    let v = libm_ops::sqrt(mu / r);
    let t = orbital_period(mu, r);
    let state = NBodyState::new(
        vec![1.0],
        vec![Vec3::new(r, 0.0, 0.0)],
        vec![Vec3::new(0.0, v, 0.0)],
    );
    (state, t, v)
}

#[test]
fn leo_period_and_speed_match_formula() {
    let r_km = 7000.0;
    let mu_km = EARTH_MU_KM3_S2;
    let t = TWO_PI * libm_ops::sqrt(r_km * r_km * r_km / mu_km);
    let v = libm_ops::sqrt(mu_km / r_km);
    assert_relative_eq!(t, 5828.5, max_relative = 2e-4);
    assert_relative_eq!(v, 7.5460, max_relative = 2e-4);
}

#[test]
fn geo_radius_period_match_formula() {
    let r_km = 42_164.17;
    let mu_km = EARTH_MU_KM3_S2;
    let t = TWO_PI * libm_ops::sqrt(r_km * r_km * r_km / mu_km);
    assert_relative_eq!(t, 86_164.09, max_relative = 2e-6);
}

#[test]
fn two_body_vs_analytic_kepler_position_error_per_orbit() {
    let (mut state, period, _) = circular_leo_state();
    let field = CentralMu::new(EARTH_MU);
    let r0 = state.x[0].norm();
    let steps = 20_000;
    let dt = period / steps as f64;
    let mut integ = Leapfrog::default();
    let mut t = 0.0;
    for _ in 0..steps {
        let res = integ.step(&field, &mut state, t, dt);
        t = res.t;
    }
    let radius_err = (state.x[0].norm() - r0).abs();
    assert!(
        radius_err < 1e-6,
        "LEO radius error after 1 period: {radius_err} m"
    );

    let (mut s2, period, _) = circular_leo_state();
    let (ra, _) = propagate_universal(EARTH_MU, s2.x[0], s2.v[0], period).unwrap();
    let mut integ = Leapfrog::default();
    let dt = period / steps as f64;
    let mut t = 0.0;
    for _ in 0..steps {
        let res = integ.step(&field, &mut s2, t, dt);
        t = res.t;
    }
    let err_vs_analytic = (s2.x[0] - ra).norm();
    assert!(
        err_vs_analytic < 5.0,
        "vs analytic Kepler position error per orbit: {err_vs_analytic} m (leapfrog {steps} steps)"
    );
}

#[test]
fn eccentric_kepler_energy_symplectic_vs_rk4() {
    let mu = EARTH_MU;
    let a = 10_000_000.0;
    let e = 0.9;
    let el = KeplerElements {
        a,
        e,
        i: 0.3,
        raan: 0.1,
        arg_p: 0.2,
        true_anomaly: 0.0,
    };
    let (r, v) = elements_to_state(mu, el).unwrap();
    let mass = vec![1.0];
    let field = CentralMu::new(mu);
    let period = orbital_period(mu, a);
    let n_orbits = 1000;
    let steps_sym = 32_768;
    let dt_sym = period / steps_sym as f64;

    let mut sym = NBodyState::new(mass.clone(), vec![r], vec![v]);
    let e0 = total_energy(&field, &sym);
    let mut lf = Yoshida4::default();
    let mut t = 0.0;
    let mut max_rel = 0.0_f64;
    for _orbit in 0..n_orbits {
        for _ in 0..steps_sym {
            let res = lf.step(&field, &mut sym, t, dt_sym);
            t = res.t;
        }
        let e_sym = total_energy(&field, &sym);
        let rel = libm_ops::fabs((e_sym - e0) / e0);
        if rel > max_rel {
            max_rel = rel;
        }
    }
    assert!(
        max_rel < 1e-10,
        "Yoshida4 relative energy error over 1000 orbits (per-orbit samples): {max_rel}"
    );

    let steps_rk = 2048;
    let dt_rk = period / steps_rk as f64;
    let mut rk = NBodyState::new(mass, vec![r], vec![v]);
    let e0r = total_energy(&field, &rk);
    let mut integ = Rk4;
    let mut t = 0.0;
    let mut err_at_10 = 0.0;
    let mut err_at_50 = 0.0;
    for step in 0..(50 * steps_rk) {
        let res = integ.step(&field, &mut rk, t, dt_rk);
        t = res.t;
        if step + 1 == 10 * steps_rk {
            let e_rk = total_energy(&field, &rk);
            err_at_10 = libm_ops::fabs((e_rk - e0r) / e0r);
        }
        if step + 1 == 50 * steps_rk {
            let e_rk = total_energy(&field, &rk);
            err_at_50 = libm_ops::fabs((e_rk - e0r) / e0r);
        }
    }
    assert!(
        err_at_50 > err_at_10,
        "expected RK4 energy drift to grow with time: 10orb={err_at_10} 50orb={err_at_50}"
    );
    assert!(
        err_at_50 > max_rel * 1e3,
        "RK4 drift should dwarf symplectic bound; rk50={err_at_50} sym={max_rel}"
    );
}

/// Orbit-averaged mean nodal rate from ascending-node crossings.
///
/// Osculating Ω carries J₂ short-period oscillations of relative size
/// ∼ J₂ (R/a)². Sampling Ω at ascending-node crossings (`z` crosses 0 with
/// `v_z > 0`) and fitting a linear trend recovers the secular rate in
/// `Ω̇ = −1.5 n J₂ (R/p)² cos i` to well under 0.5 %.
#[test]
fn j2_nodal_regression_orbit_averaged_vs_formula() {
    let model = J2Model::earth();
    let a = 6_778_000.0;
    let i = 51.6_f64.to_radians();
    let e = 0.0;
    let rate_analytic = j2_nodal_rate(&model, a, e, i);

    let el = KeplerElements {
        a,
        e,
        i,
        raan: 0.0,
        arg_p: 0.0,
        true_anomaly: 0.0,
    };
    let (r, v) = elements_to_state(EARTH_MU, el).unwrap();
    let mut state = NBodyState::new(vec![1.0], vec![r], vec![v]);

    struct J2Only(J2Model);
    impl GravityField for J2Only {
        fn accelerations(&self, state: &NBodyState, _t: f64, out: &mut [Vec3]) {
            for a in out.iter_mut() {
                *a = Vec3::zero();
            }
            if state.n() > 0 {
                out[0] = accel_j2(&self.0, state.x[0]);
            }
        }
        fn potential(&self, state: &NBodyState) -> Option<f64> {
            CentralMu::new(self.0.mu).potential(state)
        }
    }

    let field = J2Only(model);
    let mut integ = Leapfrog::default();
    let period = orbital_period(EARTH_MU, a);
    let n_orbits = 400;
    let steps_per = 8000;
    let dt = period / steps_per as f64;
    let mut t = 0.0;
    let mut samples: Vec<(f64, f64)> = Vec::with_capacity(n_orbits);
    let mut prev_z = state.x[0].z;
    let mut prev_raan: Option<f64> = None;
    let mut unwrapped = 0.0_f64;
    let total_steps = steps_per * n_orbits;
    for _ in 0..total_steps {
        let res = integ.step(&field, &mut state, t, dt);
        t = res.t;
        let z = state.x[0].z;
        let vz = state.v[0].z;
        if prev_z < 0.0 && z >= 0.0 && vz > 0.0 {
            let el = state_to_elements(EARTH_MU, state.x[0], state.v[0]).unwrap();
            match prev_raan {
                None => {
                    unwrapped = el.raan;
                    prev_raan = Some(el.raan);
                }
                Some(prev) => {
                    let mut d = el.raan - prev;
                    if d > core::f64::consts::PI {
                        d -= TWO_PI;
                    }
                    if d < -core::f64::consts::PI {
                        d += TWO_PI;
                    }
                    unwrapped += d;
                    prev_raan = Some(el.raan);
                }
            }
            samples.push((t, unwrapped));
        }
        prev_z = z;
    }
    assert!(
        samples.len() > 50,
        "expected many ascending-node samples, got {}",
        samples.len()
    );
    // Drop a few early nodes (transient), then least-squares Ω̇
    let skip = 5.min(samples.len().saturating_sub(10));
    let samples = &samples[skip..];
    let n = samples.len() as f64;
    let mut sum_t = 0.0;
    let mut sum_o = 0.0;
    let mut sum_tt = 0.0;
    let mut sum_to = 0.0;
    for &(ti, oi) in samples {
        sum_t += ti;
        sum_o += oi;
        sum_tt += ti * ti;
        sum_to += ti * oi;
    }
    let denom = n * sum_tt - sum_t * sum_t;
    let rate_num = (n * sum_to - sum_t * sum_o) / denom;
    let rel = libm_ops::fabs((rate_num - rate_analytic) / rate_analytic);
    assert!(
        rel < 0.005,
        "J2 orbit-averaged nodal rate rel err {rel} (num={rate_num}, analytic={rate_analytic})"
    );
}

#[test]
fn sun_synchronous_inclination_near_98_6() {
    let model = J2Model::earth();
    let a = EARTH_RADIUS + 800_000.0;
    let want = deg_per_day_to_rad_per_s(0.9856);
    let n = mean_motion(model.mu, a);
    let factor = -1.5 * n * model.j2 * (model.radius / a) * (model.radius / a);
    let cosi = want / factor;
    let i = libm_ops::acos(cosi);
    assert_relative_eq!(i.to_degrees(), 98.6, max_relative = 5e-3);
    assert_eq!(EARTH_J2, 1.082_626_68e-3);
    assert_eq!(EARTH_RADIUS_KM, 6_378.137);
}

#[test]
fn figure_eight_returns_after_one_period_ias15() {
    let mut state = figure_eight_state();
    state.enable_compensated_positions();
    let x0: Vec<_> = state.x.clone();
    let v0: Vec<_> = state.v.clone();
    let field = NBodyPointMass::geometric();
    // Refined period for the 8-digit IC (see FIGURE_EIGHT_PERIOD_RETURN).
    let t_end = FIGURE_EIGHT_PERIOD_RETURN;
    let mut integ = Ias15::new(AdaptiveStep {
        dt: t_end / 200.0,
        dt_min: 1e-10,
        dt_max: t_end / 20.0,
        ..AdaptiveStep::default()
    });
    integ.epsilon_b = 1e-9;
    integ.integrate_to(&field, &mut state, 0.0, t_end, t_end / 200.0);
    let mut max_dx = 0.0_f64;
    for i in 0..3 {
        let dx = (state.x[i] - x0[i]).norm();
        let dv = (state.v[i] - v0[i]).norm();
        if dx > max_dx {
            max_dx = dx;
        }
        if dv > max_dx {
            max_dx = dv;
        }
    }
    assert!(
        max_dx < 1e-8,
        "figure-eight IAS15 return error {max_dx} (tol 1e-8)"
    );
}

#[test]
fn pythagorean_three_body_ias15_energy() {
    // Burrau Pythagorean problem: masses 3,4,5 at (1,3), (−2,−1), (1,−1); v=0.
    let mut state = NBodyState::new(
        vec![3.0, 4.0, 5.0],
        vec![
            Vec3::new(1.0, 3.0, 0.0),
            Vec3::new(-2.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
        ],
        vec![Vec3::zero(); 3],
    );
    state.enable_compensated_positions();
    let field = NBodyPointMass::geometric();
    let e0 = total_energy(&field, &state);
    let mut integ = Ias15::new(AdaptiveStep {
        atol: 1e-12,
        rtol: 1e-12,
        dt: 1e-3,
        dt_min: 1e-14,
        dt_max: 0.05,
    });
    integ.epsilon_b = 1e-9;
    let t_end = 10.0; // through close encounters
    integ.integrate_to(&field, &mut state, 0.0, t_end, 1e-3);
    let e1 = total_energy(&field, &state);
    let rel = libm_ops::fabs((e1 - e0) / e0);
    assert!(rel < 1e-10, "Pythagorean IAS15 energy relative drift {rel}");
}

#[test]
fn radau_table_matches_computed_nodes() {
    let h = compute_radau_nodes();
    for i in 0..h.len() {
        assert_abs_diff_eq!(h[i], RADAU_H[i], epsilon = 1e-15);
    }
}

/// Synthetic hierarchical 5-body (G=1): central mass 1 + four planets on
/// circular coplanar orbits. Verifiable from two-body circular velocity
/// `v = √(M/a)` with light planets (no published ephemeris constants).
fn synthetic_outer_system() -> NBodyState {
    let m0 = 1.0;
    let masses = [1.0e-3, 5.0e-4, 2.0e-4, 1.0e-4];
    let semimajor = [1.0, 2.0, 4.0, 8.0];
    let mut m = vec![m0];
    let mut x = vec![Vec3::zero()];
    let mut v = vec![Vec3::zero()];
    for (&mi, &a) in masses.iter().zip(semimajor.iter()) {
        let vcirc = libm_ops::sqrt((m0 + mi) / a);
        m.push(mi);
        x.push(Vec3::new(a, 0.0, 0.0));
        v.push(Vec3::new(0.0, vcirc, 0.0));
    }
    // Shift to COM frame
    let mut state = NBodyState::new(m, x, v);
    state.rebase_to_com();
    state
}

#[test]
fn whfast_synthetic_5body_energy_bounded() {
    let mut state = synthetic_outer_system();
    let field = NBodyPointMass::geometric();
    let e0 = total_energy(&field, &state);
    let mut integ = Whfast::new(1.0, WhfastCoords::Jacobi);
    let dt = 0.05; // ~1/125 of innermost period (2π)
    let t_end = 1.0e6;
    let mut t = 0.0;
    let mut max_rel = 0.0_f64;
    let sample_every = 20_000usize;
    let mut step = 0usize;
    while t < t_end {
        let res = integ.step(&field, &mut state, t, dt);
        t = res.t;
        step += 1;
        if step % sample_every == 0 {
            let e = total_energy(&field, &state);
            let rel = libm_ops::fabs((e - e0) / e0);
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    let e_final = total_energy(&field, &state);
    let rel_final = libm_ops::fabs((e_final - e0) / e0);
    max_rel = max_rel.max(rel_final);
    assert!(
        max_rel < 1e-6,
        "WHFast 5-body energy bound over 1e6 time: max rel {max_rel}"
    );
}

#[test]
fn whfast_convergence_order_2() {
    let field = NBodyPointMass::geometric();
    // Short integration; compare energy error vs analytic two-body for the
    // innermost planet alone would be order 2 in the interaction — use the
    // full 5-body energy drift magnitude at final time as the error measure
    // against a fine reference.
    let t_end: f64 = 20.0;
    let dts: [f64; 3] = [0.2, 0.1, 0.05];
    let mut errs = [0.0_f64; 3];
    // Reference with very small dt
    let mut ref_state = synthetic_outer_system();
    let mut integ_ref = Whfast::new(1.0, WhfastCoords::Jacobi);
    let mut t: f64 = 0.0;
    let dt_ref: f64 = 0.005;
    while t < t_end {
        let res = integ_ref.step(&field, &mut ref_state, t, dt_ref);
        t = res.t;
    }
    for (k, &dt) in dts.iter().enumerate() {
        let mut state = synthetic_outer_system();
        let mut integ = Whfast::new(1.0, WhfastCoords::Jacobi);
        let mut t: f64 = 0.0;
        while t < t_end - 1e-14 {
            let h = dt.min(t_end - t);
            let res = integ.step(&field, &mut state, t, h);
            t = res.t;
        }
        let mut err = 0.0;
        for i in 0..state.n() {
            err += (state.x[i] - ref_state.x[i]).norm_squared();
        }
        errs[k] = libm_ops::sqrt(err);
    }
    // Order 2: err(2h)/err(h) ≈ 4
    let r01 = errs[0] / errs[1];
    let r12 = errs[1] / errs[2];
    assert!(
        r01 > 3.0 && r01 < 5.5,
        "WHFast order-2 ratio err(0.2)/err(0.1)={r01} (errs={errs:?})"
    );
    assert!(
        r12 > 3.0 && r12 < 5.5,
        "WHFast order-2 ratio err(0.1)/err(0.05)={r12} (errs={errs:?})"
    );
}

#[test]
fn whfast_reversibility() {
    let field = NBodyPointMass::geometric();
    let mut state = synthetic_outer_system();
    let x0: Vec<_> = state.x.clone();
    let v0: Vec<_> = state.v.clone();
    let mut integ = Whfast::new(1.0, WhfastCoords::Jacobi);
    let dt = 0.05;
    let nsteps = 200;
    let mut t = 0.0;
    for _ in 0..nsteps {
        let res = integ.step(&field, &mut state, t, dt);
        t = res.t;
    }
    // Reverse velocities and integrate backward
    for i in 0..state.n() {
        state.v[i] *= -1.0;
    }
    for _ in 0..nsteps {
        let res = integ.step(&field, &mut state, t, dt);
        t = res.t;
    }
    for i in 0..state.n() {
        state.v[i] *= -1.0;
    }
    let mut max_err = 0.0_f64;
    for i in 0..state.n() {
        max_err = max_err.max((state.x[i] - x0[i]).norm());
        max_err = max_err.max((state.v[i] - v0[i]).norm());
    }
    assert!(max_err < 1e-10, "WHFast reversibility error {max_err}");
}

#[test]
fn figure_eight_hash_bit_stable_on_repeat() {
    let h1 = figure_eight_hash();
    let h2 = figure_eight_hash();
    assert_eq!(h1, h2);
}

#[test]
fn ias15_hash_bit_stable_on_repeat() {
    let h1 = ias15_hash();
    let h2 = ias15_hash();
    assert_eq!(h1, h2);
}

#[test]
fn elements_roundtrip() {
    let el = KeplerElements {
        a: 8_000_000.0,
        e: 0.2,
        i: 0.5,
        raan: 0.4,
        arg_p: 0.3,
        true_anomaly: 0.7,
    };
    let (r, v) = elements_to_state(EARTH_MU, el).unwrap();
    let el2 = state_to_elements(EARTH_MU, r, v).unwrap();
    assert_relative_eq!(el2.a, el.a, max_relative = 1e-9);
    assert_relative_eq!(el2.e, el.e, max_relative = 1e-9);
    assert_relative_eq!(el2.i, el.i, max_relative = 1e-9);
}

#[test]
fn angular_momentum_conserved_nbody_leapfrog() {
    let mut state = figure_eight_state();
    let field = NBodyPointMass::geometric();
    let l0 = angular_momentum(&state, Vec3::zero());
    let mut integ = Leapfrog::default();
    let mut t = 0.0;
    for _ in 0..500 {
        let res = integ.step(&field, &mut state, t, 1e-3);
        t = res.t;
    }
    let l1 = angular_momentum(&state, Vec3::zero());
    assert_abs_diff_eq!(l1.x, l0.x, epsilon = 1e-10);
    assert_abs_diff_eq!(l1.y, l0.y, epsilon = 1e-10);
    assert_abs_diff_eq!(l1.z, l0.z, epsilon = 1e-10);
}

#[test]
fn rkf78_still_available_as_dop853_standin() {
    // DOP853 tableau was not reproduced from Hairer/Norsett/Wanner in this PR;
    // RKF78 remains the adaptive high-order stand-in (`Dop853` type alias).
    let mut state = figure_eight_state();
    let field = NBodyPointMass::geometric();
    let mut integ = Rkf78 {
        ctrl: AdaptiveStep {
            atol: 1e-10,
            rtol: 1e-10,
            dt: 1e-3,
            dt_min: 1e-12,
            dt_max: 0.05,
        },
    };
    let res = integ.step(&field, &mut state, 0.0, 1e-3);
    assert!(res.dt_taken > 0.0);
}
