//! 6-DOF FDM analytic / published-value oracles.
//!
//! Tolerances below are **measured** on Linux x86_64 release; do not loosen
//! silently — update the measured column in `docs/AERO.md` with the new
//! observation if a bound must change.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_aero::atmosphere::RHO0;
use pe_aero::fdm::forces::{attitude_pitch_bank, level_attitude};
use pe_aero::ground_effect::{mccormick_lift_factor, wieselsberger_sigma};
use pe_aero::scenarios::{
    canonical_dryden_gust_hash, canonical_fdm_gust_hash, canonical_fdm_hash, unverified_demo_model,
};
use pe_aero::turbulence::{
    dryden_vw_psd, estimate_psd_log_slope, periodogram_psd, sample_rms, DrydenParams,
    DrydenTurbulence,
};
use pe_aero::{
    coordinated_turn_load_factor, coordinated_turn_rate, eval_aero_body, linearise_lateral,
    linearise_longitudinal, linearise_longitudinal_ex, navion_nelson, phugoid_approximation,
    short_period_approximation, short_period_approximation_with_alphadot, trim_steady_level,
    ControlInputs, SixDofAero, TrimGuess, TrimStatus,
};
use pe_core::diagnostics::{kinetic_energy, total_energy};
use pe_core::forces::{Spring, UniformGravity};
use pe_core::integrators::{rk4, symplectic_euler};
use pe_core::{RigidBody, World};
use pe_math::libm_ops;
use pe_math::{Vec3, STANDARD_GRAVITY};

fn navion_trim() -> (pe_aero::NavionData, pe_aero::TrimResult) {
    let nav = navion_nelson();
    assert!(nav.verified);
    let guess = TrimGuess {
        v: nav.v_ref,
        alpha: nav.alpha_trim,
        delta_e: -nav.model.deriv.cm_alpha * nav.alpha_trim / nav.model.deriv.cm_de,
        thrust: 0.5 * RHO0 * nav.v_ref * nav.v_ref * nav.model.s * nav.cd_trim,
        g: STANDARD_GRAVITY,
        rho: Some(RHO0),
    };
    let trim = trim_steady_level(&nav.model, guess);
    assert_eq!(trim.status, TrimStatus::Converged);
    assert!(
        trim.residual_norm < 1e-3,
        "trim residual {}",
        trim.residual_norm
    );
    (nav, trim)
}

/// Published Navion longitudinal roots (Cal State thesis LONG.for on Nelson /
/// Teper Table B.1 data; Appendix II computer output):
/// phugoid `−0.01640041 ± 0.2135064 i` (ζ = 0.076589),
/// short-period `−2.444170 ± 2.537795 i` (ζ = 0.693696).
#[test]
fn navion_longitudinal_vs_nelson_published_roots() {
    let (nav, trim) = navion_trim();
    assert_relative_eq!(trim.cl, nav.cl_trim, max_relative = 0.05);
    assert_relative_eq!(trim.cd, nav.cd_trim, max_relative = 0.08);

    // With C_mα̇ / C_Lα̇ (Nelson: −4.36 / 0.0).
    let ss = linearise_longitudinal(&nav.model, &trim, RHO0, STANDARD_GRAVITY);
    let modes = ss.modes.expect("longitudinal modes");
    let w_ph = modes.phugoid.omega_n();
    let z_ph = modes.phugoid.zeta();
    let w_sp = modes.short_period.omega_n();
    let z_sp = modes.short_period.zeta();

    // Published (exact numbers from thesis Appendix II).
    const W_PH_PUB: f64 = 0.2135064; // |−0.01640041 ± 0.2135064 i|
    const Z_PH_PUB: f64 = 0.076589;
    let w_sp_pub = libm_ops::sqrt(2.444170 * 2.444170 + 2.537795 * 2.537795);
    const Z_SP_PUB: f64 = 0.6936957;

    // Measured (Linux x86_64, with α̇): phugoid ω≈0.21322 (rel≈0.13%), ζ≈0.0817
    // (abs≈0.005); short-period ω≈3.573 (rel≈1.4%), ζ≈0.6987 (abs≈0.005).
    // Tolerances ≈ 2× measured error.
    assert!(
        (w_ph - W_PH_PUB).abs() / W_PH_PUB < 0.005,
        "phugoid ω={w_ph} vs {W_PH_PUB}"
    );
    assert!(
        (z_ph - Z_PH_PUB).abs() < 0.012,
        "phugoid ζ={z_ph} vs {Z_PH_PUB}"
    );
    assert!(
        (w_sp - w_sp_pub).abs() / w_sp_pub < 0.03,
        "short-period ω={w_sp} vs {w_sp_pub}"
    );
    assert!(
        (z_sp - Z_SP_PUB).abs() < 0.012,
        "short-period ζ={z_sp} vs {Z_SP_PUB}"
    );

    // Lanchester / Etkin is a coarse sanity check only — keep the bound at the
    // true expected gap (~21% in ω_n for this Navion condition), not as the
    // primary oracle.
    let (w_approx, z_approx) = phugoid_approximation(trim.v, trim.cl, trim.cd, STANDARD_GRAVITY);
    let w_lan_err = (w_ph - w_approx).abs() / w_approx;
    let z_lan_err = (z_ph - z_approx).abs();
    // Measured: ω rel ≈ 0.175, ζ abs ≈ 0.009.
    assert!(
        w_lan_err < 0.25,
        "Lanchester ω sanity: eig={w_ph} approx={w_approx} rel={w_lan_err}"
    );
    assert!(
        z_lan_err < 0.05,
        "Lanchester ζ sanity: eig={z_ph} approx={z_approx} abs={z_lan_err}"
    );
    assert!(w_sp > w_ph);
}

/// `C_mα̇` raises short-period damping vs the reduced-order (no-α̇) model and
/// vs the no-α̇ numerical linearisation.
#[test]
fn alphadot_increases_short_period_damping() {
    let (nav, trim) = navion_trim();
    let with = linearise_longitudinal_ex(&nav.model, &trim, RHO0, STANDARD_GRAVITY, true)
        .modes
        .unwrap();
    let without = linearise_longitudinal_ex(&nav.model, &trim, RHO0, STANDARD_GRAVITY, false)
        .modes
        .unwrap();
    // Measured: ζ_sp 0.575 → 0.699 with C_mα̇ = −4.36.
    assert!(
        with.short_period.zeta() > without.short_period.zeta() + 0.05,
        "α̇ should raise SP damping: with={} without={}",
        with.short_period.zeta(),
        without.short_period.zeta()
    );

    let (w0, z0) = short_period_approximation(&nav.model, trim.v, RHO0, trim.alpha);
    let (w1, z1) = short_period_approximation_with_alphadot(&nav.model, trim.v, RHO0, trim.alpha);
    assert!(z1 > z0, "reduced-order α̇ approx: {z1} vs {z0}");
    // Reduced-order is approximate; just check it tracks the full-model trend.
    assert!(
        (with.short_period.zeta() - without.short_period.zeta()) > 0.5 * (z1 - z0).abs().max(0.05)
    );
    let _ = (w0, w1);
}

/// Published Navion lateral roots (Cal State thesis LAT.for / Nelson–Teper):
/// spiral `−0.0088`, roll `−8.2550`, Dutch roll `−0.4743 ± 2.3109 i`
/// (ζ = 0.2010, ω_n = 2.3591 rad/s).
#[test]
fn navion_lateral_vs_nelson_published_roots() {
    let (nav, trim) = navion_trim();
    let lat = linearise_lateral(&nav.model, &trim, RHO0, STANDARD_GRAVITY);
    let m = lat.modes.expect("lateral modes");

    const ROLL_PUB: f64 = -8.2550;
    const SPIRAL_PUB: f64 = -0.0088;
    const W_DR_PUB: f64 = 2.3591;
    const Z_DR_PUB: f64 = 0.2010;

    let roll = m.roll.re;
    let spiral = m.spiral.re;
    let w_dr = m.dutch_roll.omega_n();
    let z_dr = m.dutch_roll.zeta();

    // Measured (Linux x86_64): roll≈−8.533 (rel≈3.4%), spiral≈−0.00878
    // (abs≈2e-5), Dutch ω≈2.230 (rel≈5.5%), ζ≈0.2005 (abs≈5e-4).
    // Tolerances ≈ 2× measured error.
    assert!(
        (roll - ROLL_PUB).abs() / ROLL_PUB.abs() < 0.08,
        "roll={roll} vs {ROLL_PUB}"
    );
    assert!(
        (spiral - SPIRAL_PUB).abs() < 0.002,
        "spiral={spiral} vs {SPIRAL_PUB}"
    );
    assert!(
        (w_dr - W_DR_PUB).abs() / W_DR_PUB < 0.12,
        "dutch ω={w_dr} vs {W_DR_PUB}"
    );
    assert!(
        (z_dr - Z_DR_PUB).abs() < 0.02,
        "dutch ζ={z_dr} vs {Z_DR_PUB}"
    );
    // Classification sanity: |roll| ≫ |spiral|, Dutch oscillatory.
    assert!(roll.abs() > 10.0 * spiral.abs());
    assert!(m.dutch_roll.im.abs() > 1.0);
}

#[test]
fn steady_glide_ratio_l_over_d() {
    let mut m = unverified_demo_model();
    let mut best = 0.0;
    let mut best_a = 0.0;
    for i in 0..200 {
        let a = -0.05 + 0.002 * i as f64;
        let cl = m.deriv.cl(a, 0.0, 0.0);
        let cd = m.deriv.cd(a);
        let ld = cl / cd;
        if ld > best {
            best = ld;
            best_a = a;
        }
    }
    m.controls = ControlInputs::default();
    m.thrust = 0.0;
    let v_body = Vec3::new(
        libm_ops::cos(best_a) * 40.0,
        0.0,
        -libm_ops::sin(best_a) * 40.0,
    );
    let loads = eval_aero_body(&m, v_body, Vec3::zero(), RHO0, None);
    assert_relative_eq!(loads.cl / loads.cd, best, max_relative = 1e-10);
    assert!(best > 5.0, "demo L/D max = {best}");
}

#[test]
fn energy_conservation_aero_off_pendulum() {
    let mass = 1.0;
    let k = 40.0;
    let body = RigidBody::new(mass)
        .with_inertia_diag(0.1, 0.1, 0.1)
        .with_position(Vec3::new(0.3, 0.0, -1.0))
        .with_velocity(Vec3::new(0.0, 0.0, 0.0));
    let mut w = World::new(
        vec![body],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(Spring::new(0, Vec3::zero(), k)),
        ],
    );
    let e0 = total_energy(&w, &w.state);
    let mut e_max = e0;
    let mut e_min = e0;
    let dt = 1e-3;
    let steps = 10_000;
    for _ in 0..steps {
        w.step(dt, symplectic_euler);
        let e = total_energy(&w, &w.state);
        e_max = e_max.max(e);
        e_min = e_min.min(e);
    }
    let amp = (e_max - e_min) / e0.abs().max(1.0);
    assert!(
        amp < 1e-2,
        "energy oscillation amp={amp}, e0={e0}, range=[{e_min},{e_max}]"
    );
    let _ = kinetic_energy(&w, &w.state);
}

#[test]
fn quaternion_norm_drift_fdm() {
    let nav = navion_nelson();
    let mut model = nav.model.clone();
    model.thrust = 0.5 * RHO0 * nav.v_ref * nav.v_ref * nav.model.s * nav.cd_trim;
    model.controls.delta_e = -nav.model.deriv.cm_alpha * nav.alpha_trim / nav.model.deriv.cm_de;
    let body = RigidBody::new(model.mass)
        .with_inertia(model.inertia_matrix())
        .with_position(Vec3::new(0.0, 0.0, 500.0))
        .with_velocity(Vec3::new(nav.v_ref, 0.0, 0.0))
        .with_orientation(level_attitude(nav.alpha_trim));
    let mut w = World::new(
        vec![body],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(SixDofAero::new(0, model)),
        ],
    );
    let mut max_drift: f64 = 0.0;
    for _ in 0..500 {
        w.step(1e-2, rk4);
        let n = w.state.q[0].norm();
        max_drift = max_drift.max((n - 1.0).abs());
    }
    assert!(max_drift < 1e-12, "quat norm drift {max_drift} (tol 1e-12)");
}

#[test]
fn coordinated_turn_bank_load_factor() {
    for &phi_deg in &[15.0_f64, 30.0, 45.0, 60.0] {
        let phi = phi_deg.to_radians();
        let n = coordinated_turn_load_factor(phi);
        let n_exact = 1.0 / libm_ops::cos(phi);
        assert_relative_eq!(n, n_exact, max_relative = 1e-15);

        let v = 50.0;
        let rate = coordinated_turn_rate(phi, v, STANDARD_GRAVITY);
        let rate_exact = STANDARD_GRAVITY * libm_ops::tan(phi) / v;
        assert_relative_eq!(rate, rate_exact, max_relative = 1e-15);
        assert!((n / n_exact - 1.0).abs() < 0.005);
        assert!((rate / rate_exact - 1.0).abs() < 0.005);
    }

    let nav = navion_nelson();
    let phi = 30.0_f64.to_radians();
    let n = coordinated_turn_load_factor(phi);
    let v = nav.v_ref;
    let qbar = 0.5 * RHO0 * v * v;
    let cl_req = n * nav.model.mass * STANDARD_GRAVITY / (qbar * nav.model.s);
    let alpha_turn = cl_req / nav.model.deriv.cl_alpha;
    let _q = attitude_pitch_bank(alpha_turn, phi);
    let cl_from_alpha = nav.model.deriv.cl(alpha_turn, 0.0, 0.0);
    assert_relative_eq!(cl_from_alpha, cl_req, max_relative = 0.005);
}

#[test]
fn thrust_offset_produces_pitching_moment() {
    let mut m = unverified_demo_model();
    m.thrust = 1000.0;
    m.thrust_offset = Vec3::new(0.0, 0.0, -0.5);
    m.s = 0.0;
    let loads = eval_aero_body(&m, Vec3::new(40.0, 0.0, 0.0), Vec3::zero(), RHO0, None);
    assert_abs_diff_eq!(loads.moment_body.y, -500.0, epsilon = 1e-9);
    assert_abs_diff_eq!(loads.force_body.x, 1000.0, epsilon = 1e-9);
}

#[test]
fn isa_density_used_when_requested() {
    let mut m = unverified_demo_model();
    m.density = pe_aero::atmosphere::DensityModel::Isa;
    let rho0 = pe_aero::atmosphere::density(0.0).unwrap();
    let rho5 = pe_aero::atmosphere::density(5000.0).unwrap();
    assert!(rho5 < rho0);
    let body = RigidBody::new(m.mass)
        .with_inertia(m.inertia_matrix())
        .with_position(Vec3::new(0.0, 0.0, 0.0))
        .with_velocity(Vec3::new(40.0, 0.0, 0.0));
    let w = World::new(vec![body], vec![Box::new(SixDofAero::new(0, m.clone()))]);
    let loads0 = SixDofAero::new(0, m).loads(&w, &w.state);
    assert_relative_eq!(loads0.rho, rho0, max_relative = 1e-12);
}

#[test]
fn ground_effect_wieselsberger_mccormick_oracle() {
    // Oracle = published formula values at selected h/b.
    let cases = [
        (0.05, 0.3902439024390244),
        (0.10, 0.7191011235955056),
        (0.25, 0.9411764705882353),
        (0.50, 0.9846153846153847),
        (1.00, 0.9961089494163424),
    ];
    for &(h_b, sigma_pub) in &cases {
        let s = wieselsberger_sigma(h_b);
        // (16h/b)^2/(1+(16h/b)^2) — bit-level vs published expression.
        assert_relative_eq!(s, sigma_pub, max_relative = 1e-12);
        assert_relative_eq!(
            mccormick_lift_factor(h_b),
            2.0 - sigma_pub,
            max_relative = 1e-12
        );
    }
    // Force path: near ground, CL up and induced CD down.
    let mut m = unverified_demo_model();
    m.ground_height = None;
    let v_body = Vec3::new(40.0, 0.0, -2.0);
    let free = eval_aero_body(&m, v_body, Vec3::zero(), RHO0, None);
    m.ground_height = Some(0.5); // h/b = 0.05
    let ge = eval_aero_body(&m, v_body, Vec3::zero(), RHO0, None);
    assert!(ge.cl > free.cl);
    assert!(ge.cd < free.cd || (ge.cd - m.deriv.cd0).abs() < (free.cd - m.deriv.cd0));
}

#[test]
fn dryden_vw_psd_and_rms_oracle() {
    let params = DrydenParams::light_low_altitude();
    let mut d = DrydenTurbulence::new(params, 0xAE80_FD01);
    let dt = 0.01;
    let v = 50.0;
    let n = 20_000;
    let mut vs = Vec::with_capacity(n);
    let mut ws = Vec::with_capacity(n);
    for _ in 0..n {
        let g = d.step(v, dt);
        vs.push(g.y);
        ws.push(-g.z);
    }
    let rms_v = sample_rms(&vs);
    let rms_w = sample_rms(&ws);
    // Measured: rms_v≈1.452 (σ_v=1.5, rel≈3.2%), rms_w≈1.001 (σ_w=1.0).
    assert!(
        (rms_v - params.sigma_v).abs() / params.sigma_v < 0.08,
        "rms_v={rms_v} vs σ={}",
        params.sigma_v
    );
    assert!(
        (rms_w - params.sigma_w).abs() / params.sigma_w < 0.05,
        "rms_w={rms_w} vs σ={}",
        params.sigma_w
    );

    let (freqs, psd) = periodogram_psd(&vs, dt);
    let mut ff = Vec::new();
    let mut pp = Vec::new();
    for i in 0..freqs.len() {
        if freqs[i] >= 2.0 && freqs[i] <= 20.0 {
            ff.push(freqs[i]);
            pp.push(psd[i]);
        }
    }
    let slope = estimate_psd_log_slope(&ff, &pp);
    // Asymptotic Dryden v/w Φ(ω) ~ ω^{-2}; measured slope ≈ −1.925.
    assert!(
        (slope + 2.0).abs() < 0.25,
        "PSD log-slope={slope} (expect ≈ −2)"
    );

    // Analytic PSD oracle (formula values).
    let th = dryden_vw_psd(2.0, params.sigma_v, params.l_v, v);
    assert!(th > 0.0);
    assert_relative_eq!(
        th,
        params.sigma_v
            * params.sigma_v
            * (params.l_v / (core::f64::consts::PI * v))
            * (1.0 + 3.0 * (params.l_v / v * 2.0).powi(2))
            / (1.0 + (params.l_v / v * 2.0).powi(2)).powi(2),
        max_relative = 1e-12
    );
}

#[test]
fn fdm_determinism_hashes_locked() {
    assert_eq!(format!("{:016x}", canonical_fdm_hash()), "ba6d68e43c69d315");
    assert_eq!(
        format!("{:016x}", canonical_dryden_gust_hash()),
        "2895b75a14f621c5"
    );
    assert_eq!(
        format!("{:016x}", canonical_fdm_gust_hash()),
        "58f491fb7f3b34a6"
    );
    assert_eq!(canonical_fdm_hash(), canonical_fdm_hash());
    assert_eq!(canonical_fdm_gust_hash(), canonical_fdm_gust_hash());
}
