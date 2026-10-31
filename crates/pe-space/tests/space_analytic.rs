//! Analytic / published-constant validation for `pe-space` (M7).
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
use pe_space::integrators::{Ias15, Leapfrog, Rk4, Rkf78, Whfast, Yoshida4};
use pe_space::kepler::{
    elements_to_state, mean_motion, orbital_period, propagate_universal, state_to_elements,
    KeplerElements,
};
use pe_space::perturbations::{accel_j2, j2_nodal_rate, J2Model};
use pe_space::scenarios::{figure_eight_hash, figure_eight_state, FIGURE_EIGHT_PERIOD};
use pe_space::state::NBodyState;

fn circular_leo_state() -> (NBodyState, f64, f64) {
    // r = 7000 km, μ = 398600.4418 km³/s² → SI
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
    // Published test targets from the milestone brief.
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
    // Radius is conserved structurally for circular + leapfrog; phase lags slightly.
    let radius_err = (state.x[0].norm() - r0).abs();
    assert!(
        radius_err < 1e-6,
        "LEO radius error after 1 period: {radius_err} m"
    );

    // Position vs universal-variable Kepler analytic at the same epoch.
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
    // e=0.9 needs a fine fixed step for the Yoshida energy bound to sit under 1e-10.
    let n_orbits = 1000;
    let steps_sym = 32_768;
    let dt_sym = period / steps_sym as f64;

    let mut sym = NBodyState::new(mass.clone(), vec![r], vec![v]);
    let e0 = total_energy(&field, &sym);
    let mut lf = Yoshida4::default();
    let mut t = 0.0;
    // Sample energy only after integer orbits (same mean-anomaly phase).
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

    // Coarser RK4 step so secular drift is visible (same e=0.9 Kepler problem).
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

#[test]
fn j2_nodal_regression_integrated_vs_formula() {
    let model = J2Model::earth();
    let a = 6_778_000.0; // 6778 km
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

    // accel_j2 returns the full zonal acceleration (includes −μ r/r³).
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
            // Approximate two-body potential only (J2 potential omitted).
            CentralMu::new(self.0.mu).potential(state)
        }
    }

    let field = J2Only(model);
    let mut integ = Leapfrog::default();
    let period = orbital_period(EARTH_MU, a);
    // Integrate ~20 orbits and measure ΔΩ / Δt
    let n_orbits = 20;
    let steps = 2000 * n_orbits;
    let dt = period / 2000.0;
    let mut t = 0.0;
    let el0 = state_to_elements(EARTH_MU, state.x[0], state.v[0]).unwrap();
    for _ in 0..steps {
        let res = integ.step(&field, &mut state, t, dt);
        t = res.t;
    }
    let el1 = state_to_elements(EARTH_MU, state.x[0], state.v[0]).unwrap();
    let mut d_raan = el1.raan - el0.raan;
    if d_raan > core::f64::consts::PI {
        d_raan -= TWO_PI;
    }
    if d_raan < -core::f64::consts::PI {
        d_raan += TWO_PI;
    }
    let rate_num = d_raan / t;
    assert_relative_eq!(rate_num, rate_analytic, max_relative = 0.05);
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
fn figure_eight_returns_after_one_period() {
    let mut state = figure_eight_state();
    let x0: Vec<_> = state.x.clone();
    let v0: Vec<_> = state.v.clone();
    let field = NBodyPointMass::geometric();
    let mut integ = Leapfrog::default();
    let steps = 4000;
    let dt = FIGURE_EIGHT_PERIOD / steps as f64;
    let mut t = 0.0;
    for _ in 0..steps {
        let res = integ.step(&field, &mut state, t, dt);
        t = res.t;
    }
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
        max_dx < 5e-3,
        "figure-eight return error {max_dx} (tol 5e-3)"
    );
}

#[test]
fn pythagorean_three_body_energy_adaptive() {
    // Burrau Pythagorean problem: masses 3,4,5 at (1,3), (−2,−1), (1,−1); v=0.
    // UNVERIFIED as a published IC source in-tree — classic textbook problem;
    // we only check that adaptive RKF78 conserves energy to a loose tolerance
    // over a short interval (no claim of matching a published trajectory).
    let mut state = NBodyState::new(
        vec![3.0, 4.0, 5.0],
        vec![
            Vec3::new(1.0, 3.0, 0.0),
            Vec3::new(-2.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
        ],
        vec![Vec3::zero(); 3],
    );
    let field = NBodyPointMass::geometric();
    let e0 = total_energy(&field, &state);
    let mut integ = Rkf78 {
        ctrl: AdaptiveStep {
            atol: 1e-10,
            rtol: 1e-10,
            dt: 1e-3,
            dt_min: 1e-12,
            dt_max: 0.05,
        },
    };
    let mut t = 0.0;
    let t_end = 2.0;
    while t < t_end {
        let res = integ.step(&field, &mut state, t, integ.ctrl.dt);
        if res.dt_taken <= 0.0 {
            break;
        }
        t = res.t;
        state.after_step();
    }
    let e1 = total_energy(&field, &state);
    let rel = libm_ops::fabs((e1 - e0) / e0);
    assert!(
        rel < 1e-6,
        "Pythagorean energy relative drift {rel} (UNVERIFIED IC; conservation only)"
    );
}

#[test]
fn stubs_do_not_advance_state() {
    let mut state = figure_eight_state();
    let x0 = state.x[0];
    let field = NBodyPointMass::geometric();
    let mut ias = Ias15::default();
    let res = ias.step(&field, &mut state, 0.0, 0.1);
    assert_eq!(res.dt_taken, 0.0);
    assert_eq!(state.x[0], x0);
    let mut wh = Whfast {
        mu: 1.0,
        last_status: "",
    };
    let res = wh.step(&field, &mut state, 0.0, 0.1);
    assert_eq!(res.dt_taken, 0.0);
}

#[test]
fn figure_eight_hash_bit_stable_on_repeat() {
    let h1 = figure_eight_hash();
    let h2 = figure_eight_hash();
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
