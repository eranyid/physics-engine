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
    butcher_consistency_residual, compute_radau_nodes, ode_step, ode_step_fixed, Dop853, Ias15,
    Leapfrog, Rk4, Rkf78, Whfast, WhfastCoords, WhfastCorrector, Yoshida4, RADAU_H,
};
use pe_space::kepler::{
    elements_to_state, mean_motion, orbital_period, propagate_universal, state_to_elements,
    KeplerElements,
};
use pe_space::perturbations::{accel_j2, j2_nodal_rate, J2Model};
use pe_space::scenarios::{
    dop853_hash, figure_eight_hash, figure_eight_state, ias15_hash, FIGURE_EIGHT_PERIOD_RETURN,
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
fn rkf78_still_available() {
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

#[test]
fn dop853_butcher_consistency() {
    let (row, bsum) = butcher_consistency_residual();
    assert!(row < 1e-14, "Σ A_ij = c_i residual {row}");
    assert!(bsum < 1e-14, "Σ B_i = 1 residual {bsum}");
}

#[test]
fn dop853_convergence_order_8_smooth_ode() {
    // y' = y, y(0)=1 → y(T)=e^T. Coarse h∈{1, 1/2, 1/4} sits below the pure
    // asymptotic h^8 regime (leading constant + lower-order pollution), so the
    // measured slope is ~7.86 (ratios ~208/232 vs ideal 256). Finer steps below
    // approach order 8 until floating-point floors.
    let f = |_t: f64, y: &[f64]| vec![y[0]];
    let t_end: f64 = 2.0;
    let exact = libm_ops::exp(t_end);
    let hs = [1.0_f64, 0.5, 0.25];
    let mut errs = [0.0_f64; 3];
    for (k, &h) in hs.iter().enumerate() {
        let n = (t_end / h + 0.5) as usize;
        assert!(((n as f64) * h - t_end).abs() < 1e-15);
        let mut y = vec![1.0];
        let mut t = 0.0;
        for _ in 0..n {
            y = ode_step_fixed(&f, t, &y, h);
            t += h;
        }
        errs[k] = libm_ops::fabs(y[0] - exact);
    }
    let r01 = errs[0] / errs[1];
    let r12 = errs[1] / errs[2];
    let slope = libm_ops::log(r12) / libm_ops::log(2.0);
    eprintln!(
        "DOP853 coarse order-8 errs={errs:?} ratios r01={r01:.2} r12={r12:.2} slope={slope:.2}"
    );
    assert!(
        r01 > 64.0 && r12 > 64.0,
        "expected ~256 ratios for order 8; got {r01}, {r12} (errs={errs:?})"
    );
    assert!(
        slope > 7.0 && slope < 9.5,
        "convergence slope {slope} (want ~8)"
    );
}

#[test]
fn dop853_fixed_step_order_near_8_nonlinear_ode() {
    // y' = −½ y³, y(0)=1 → y(t)=(1+t)^{−1/2}. Larger leading truncation than
    // y'=y, so a moderate fixed-step ladder stays above the f64 floor and the
    // observed order approaches 8 (the coarse y'=y ladder measures ~7.86).
    let f = |_t: f64, y: &[f64]| vec![-0.5 * y[0] * y[0] * y[0]];
    let t_end: f64 = 1.0;
    let exact = 1.0 / libm_ops::sqrt(1.0 + t_end);
    // h = 1/2, 1/4, 1/8 → n = 2, 4, 8
    let hs = [0.5_f64, 0.25, 0.125];
    let mut errs = [0.0_f64; 3];
    for (k, &h) in hs.iter().enumerate() {
        let n = (t_end / h + 0.5) as usize;
        assert!(((n as f64) * h - t_end).abs() < 1e-15);
        let mut y = vec![1.0];
        let mut t = 0.0;
        for _ in 0..n {
            y = ode_step_fixed(&f, t, &y, h);
            t += h;
        }
        errs[k] = libm_ops::fabs(y[0] - exact);
    }
    let r01 = errs[0] / errs[1];
    let r12 = errs[1] / errs[2];
    let slope = libm_ops::log(r12) / libm_ops::log(2.0);
    eprintln!(
        "DOP853 nonlinear order-8 errs={errs:?} ratios r01={r01:.2} r12={r12:.2} slope={slope:.2}"
    );
    // Ideal ratio 256; require ≳128 and slope near 8.
    assert!(
        r01 > 128.0 && r12 > 128.0,
        "nonlinear-ODE ratios too low for order ~8: {r01}, {r12} (errs={errs:?})"
    );
    assert!(
        slope > 7.5 && slope < 9.0,
        "nonlinear-ODE slope {slope} (want ~8)"
    );
}

#[test]
fn dop853_kepler_energy_and_angular_momentum() {
    let mu = EARTH_MU;
    let a = 1.0e7;
    let el = KeplerElements {
        a,
        e: 0.5,
        i: 0.0,
        raan: 0.0,
        arg_p: 0.0,
        true_anomaly: 0.0,
    };
    let (r, v) = elements_to_state(mu, el).unwrap();
    let mut state = NBodyState::new(vec![1.0], vec![r], vec![v]);
    let field = CentralMu::new(mu);
    let e0 = total_energy(&field, &state);
    let l0 = angular_momentum(&state, Vec3::zero());
    let period = orbital_period(mu, a);
    let mut integ = Dop853::new(AdaptiveStep {
        dt: period / 50.0,
        dt_min: 1e-4,
        dt_max: period / 10.0,
        atol: 1e-12,
        rtol: 1e-12,
    });
    let mut t = 0.0;
    while t < period {
        let res = integ.step(&field, &mut state, t, integ.ctrl.dt.min(period - t));
        t = res.t;
    }
    let e1 = total_energy(&field, &state);
    let l1 = angular_momentum(&state, Vec3::zero());
    let de = libm_ops::fabs((e1 - e0) / e0);
    let dl = (l1 - l0).norm() / l0.norm();
    eprintln!("DOP853 Kepler: |ΔE/E|={de:.3e} |ΔL|/|L|={dl:.3e}");
    assert!(de < 1e-10, "energy drift {de}");
    assert!(dl < 1e-10, "angular-momentum drift {dl}");
}

#[test]
fn dop853_arenstorf_period_closure() {
    // Hairer Arenstorf orbit (restricted 3-body, rotating frame).
    let mu = 0.012_277_471;
    let mup = 1.0 - mu;
    let period = 17.065_216_560_157_964;
    // Hairer Arenstorf IC (digits beyond f64 truncated for clippy).
    let y0 = [0.994, 0.0, 0.0, -2.001_585_106_379_082_4];
    let f = move |_t: f64, y: &[f64]| -> Vec<f64> {
        let (y1, y2, y3, y4) = (y[0], y[1], y[2], y[3]);
        let d1 = libm_ops::pow((y1 + mu) * (y1 + mu) + y2 * y2, 1.5);
        let d2 = libm_ops::pow((y1 - mup) * (y1 - mup) + y2 * y2, 1.5);
        vec![
            y3,
            y4,
            y1 + 2.0 * y4 - mup * (y1 + mu) / d1 - mu * (y1 - mup) / d2,
            y2 - 2.0 * y3 - mup * y2 / d1 - mu * y2 / d2,
        ]
    };
    let mut ctrl = AdaptiveStep {
        dt: 1e-2,
        dt_min: 1e-8,
        dt_max: 0.5,
        atol: 1e-10,
        rtol: 1e-10,
    };
    let mut y = y0.to_vec();
    let mut t = 0.0;
    while t < period - 1e-14 {
        let h = ctrl.dt.min(period - t);
        let (yn, tn, _dt, _err) = ode_step(&f, t, &y, h, &mut ctrl);
        y = yn;
        t = tn;
    }
    let mut err = 0.0;
    for i in 0..4 {
        err += (y[i] - y0[i]) * (y[i] - y0[i]);
    }
    err = libm_ops::sqrt(err);
    eprintln!("Arenstorf period closure err={err:.3e}");
    assert!(err < 1e-6, "Arenstorf closure {err} (tol 1e-6)");
}

#[test]
fn dop853_hash_bit_stable_on_repeat() {
    let h1 = dop853_hash();
    let h2 = dop853_hash();
    assert_eq!(h1, h2);
}

#[test]
fn whfast_corrector_improves_energy_vs_dkd() {
    // Large dt so O(ε dt²) dominates plain DKD; correctors remove that term
    // (Rein & Tamayo 2015 / Wisdom 2006). Measured |ΔE/E| must drop.
    let field = NBodyPointMass::geometric();
    let t_end: f64 = 500.0;
    let dt = 1.0; // ~1/6 of innermost period 2π
    let n_steps: usize = (t_end / dt).round() as usize;

    let mut plain = synthetic_outer_system();
    let e0 = total_energy(&field, &plain);
    let mut integ_plain = Whfast::new(1.0, WhfastCoords::Jacobi);
    integ_plain.integrate_corrected(&field, &mut plain, 0.0, dt, n_steps);
    let err_plain = libm_ops::fabs((total_energy(&field, &plain) - e0) / e0);

    let mut corr = synthetic_outer_system();
    let e0c = total_energy(&field, &corr);
    let mut integ_c =
        Whfast::new(1.0, WhfastCoords::Jacobi).with_corrector(WhfastCorrector::Order11);
    integ_c.integrate_corrected(&field, &mut corr, 0.0, dt, n_steps);
    let err_corr = libm_ops::fabs((total_energy(&field, &corr) - e0c) / e0c);

    eprintln!("WHFast energy |ΔE/E| dt=1: plain={err_plain:.3e} c11={err_corr:.3e}");
    assert!(
        err_corr < err_plain * 0.5,
        "corrector should cut energy error: plain={err_plain} c11={err_corr}"
    );
}

#[test]
fn whfast_corrector_order_scaling() {
    // Position error vs fine reference: plain DKD ~ O(dt²); with order-5
    // corrector the leading O(ε dt²) map error is removed so accuracy improves
    // at fixed large dt (Rein & Tamayo). Energy still scales ~dt² from O(ε² dt²).
    let field = NBodyPointMass::geometric();
    let t_end: f64 = 20.0;
    // Fine reference (plain, small dt)
    let mut ref_state = synthetic_outer_system();
    let mut integ_ref = Whfast::new(1.0, WhfastCoords::Jacobi);
    let dt_ref = 0.01;
    let n_ref: usize = (t_end / dt_ref).round() as usize;
    integ_ref.integrate_corrected(&field, &mut ref_state, 0.0, dt_ref, n_ref);

    let dt = 0.5_f64;
    let n: usize = (t_end / dt).round() as usize;
    let mut plain = synthetic_outer_system();
    let mut integ_p = Whfast::new(1.0, WhfastCoords::Jacobi);
    integ_p.integrate_corrected(&field, &mut plain, 0.0, dt, n);
    let mut err_p = 0.0;
    for i in 0..plain.n() {
        err_p += (plain.x[i] - ref_state.x[i]).norm_squared();
    }
    err_p = libm_ops::sqrt(err_p);

    let mut corr = synthetic_outer_system();
    let mut integ_c =
        Whfast::new(1.0, WhfastCoords::Jacobi).with_corrector(WhfastCorrector::Order5);
    integ_c.integrate_corrected(&field, &mut corr, 0.0, dt, n);
    let mut err_c = 0.0;
    for i in 0..corr.n() {
        err_c += (corr.x[i] - ref_state.x[i]).norm_squared();
    }
    err_c = libm_ops::sqrt(err_c);

    eprintln!("WHFast pos err vs fine ref dt=0.5: plain={err_p:.3e} c5={err_c:.3e}");
    assert!(
        err_c < err_p * 0.8,
        "order-5 corrector should improve position accuracy: plain={err_p} c5={err_c}"
    );

    // Energy scaling of corrected map remains at least order 2.
    let dts = [1.0_f64, 0.5];
    let mut errs_e = [0.0_f64; 2];
    for (k, &dti) in dts.iter().enumerate() {
        let n = (t_end / dti).round() as usize;
        let mut s = synthetic_outer_system();
        let e0 = total_energy(&field, &s);
        let mut integ =
            Whfast::new(1.0, WhfastCoords::Jacobi).with_corrector(WhfastCorrector::Order5);
        integ.integrate_corrected(&field, &mut s, 0.0, dti, n);
        errs_e[k] = libm_ops::fabs((total_energy(&field, &s) - e0) / e0);
    }
    let r = errs_e[0] / errs_e[1];
    eprintln!("WHFast c5 energy errs={errs_e:?} ratio={r:.2} (expect ~4 for dt²)");
    assert!(
        r > 2.5,
        "corrected energy error should scale ~dt², ratio={r}"
    );
}

#[test]
fn whfast_corrector_roundtrip() {
    let mut state = synthetic_outer_system();
    let x0: Vec<_> = state.x.clone();
    let v0: Vec<_> = state.v.clone();
    let mut integ = Whfast::new(1.0, WhfastCoords::Jacobi).with_corrector(WhfastCorrector::Order3);
    let dt = 1.0;
    integ.to_mapping(&mut state, dt);
    integ.to_physical(&mut state);
    let mut err = 0.0;
    for i in 0..state.n() {
        err += (state.x[i] - x0[i]).norm_squared() + (state.v[i] - v0[i]).norm_squared();
    }
    err = libm_ops::sqrt(err);
    eprintln!("corrector roundtrip err={err:.3e}");
    assert!(err < 1e-10, "roundtrip err {err}");
}

#[test]
fn whfast_corrector_alone_changes_state() {
    let mut state = synthetic_outer_system();
    let x0: Vec<_> = state.x.clone();
    let mut integ = Whfast::new(1.0, WhfastCoords::Jacobi).with_corrector(WhfastCorrector::Order3);
    integ.to_mapping(&mut state, 1.0);
    let mut err = 0.0;
    for (xi, x0i) in state.x.iter().zip(x0.iter()) {
        err += (*xi - *x0i).norm_squared();
    }
    err = libm_ops::sqrt(err);
    eprintln!("mapping displacement={err:.3e}");
    assert!(
        err > 1e-8,
        "corrector should move state into mapping coords"
    );
}
