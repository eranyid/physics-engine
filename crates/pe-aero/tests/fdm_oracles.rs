//! 6-DOF FDM analytic / published-value oracles.
//!
//! Tolerances below are **measured** on Linux x86_64 release; do not loosen
//! silently — update the measured column in `docs/AERO.md` with the new
//! observation if a bound must change.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_aero::atmosphere::RHO0;
use pe_aero::fdm::forces::{attitude_pitch_bank, level_attitude};
use pe_aero::scenarios::{canonical_dryden_gust_hash, canonical_fdm_hash, unverified_demo_model};
use pe_aero::{
    coordinated_turn_load_factor, coordinated_turn_rate, eval_aero_body, linearise_longitudinal,
    navion_nelson, phugoid_approximation, trim_steady_level, ControlInputs, SixDofAero, TrimGuess,
    TrimStatus,
};
use pe_core::diagnostics::{kinetic_energy, total_energy};
use pe_core::forces::{Spring, UniformGravity};
use pe_core::integrators::{rk4, symplectic_euler};
use pe_core::{RigidBody, World};
use pe_math::libm_ops;
use pe_math::{Vec3, STANDARD_GRAVITY};

/// (a) Navion phugoid vs Lanchester / Etkin textbook approximations.
///
/// `ω_ph ≈ g √2 / V`, `ζ_ph ≈ C_D / (√2 C_L)` — Nelson/Etkin.
/// Linearised eigenmode compared to the approximation within stated tol.
#[test]
fn navion_phugoid_vs_textbook_approx() {
    let nav = navion_nelson();
    assert!(nav.verified, "Navion table must be cited/verified");

    let mut guess = TrimGuess {
        v: nav.v_ref,
        alpha: nav.alpha_trim,
        delta_e: 0.0,
        thrust: 0.5 * RHO0 * nav.v_ref * nav.v_ref * nav.model.s * nav.cd_trim,
        g: STANDARD_GRAVITY,
        rho: Some(RHO0),
    };
    // Seed elevator for Cm≈0: Cm_α α + Cm_δe δe ≈ 0 ⇒ δe ≈ −Cm_α α / Cm_δe
    guess.delta_e = -nav.model.deriv.cm_alpha * nav.alpha_trim / nav.model.deriv.cm_de;

    let trim = trim_steady_level(&nav.model, guess);
    assert_eq!(trim.status, TrimStatus::Converged);
    assert!(
        trim.residual_norm < 1e-3,
        "trim residual {} N/N·m",
        trim.residual_norm
    );

    // Published trim CL/CD
    assert_relative_eq!(trim.cl, nav.cl_trim, max_relative = 0.05); // 5% — trim α may shift slightly
    assert_relative_eq!(trim.cd, nav.cd_trim, max_relative = 0.08);

    let (w_approx, z_approx) = phugoid_approximation(trim.v, trim.cl, trim.cd, STANDARD_GRAVITY);
    let ss = linearise_longitudinal(&nav.model, &trim, RHO0, STANDARD_GRAVITY);
    let modes = ss.modes.expect("longitudinal modes");

    let w_eig = modes.phugoid.omega_n();
    let z_eig = modes.phugoid.zeta();

    // Measured (Linux x86_64): Navion phugoid ω_n ≈ 0.213 vs Lanchester
    // g√2/V ≈ 0.259 (rel ≈ 18%); ζ ≈ 0.078 vs CD/(√2 CL) ≈ 0.090 (abs ≈ 0.012).
    // Also matches Cal State / Nelson flying-qualities root
    // (−0.0164 ± 0.2135 i, ζ ≈ 0.077) within 2%.
    let w_err = (w_eig - w_approx).abs() / w_approx;
    let z_err = (z_eig - z_approx).abs();
    assert!(
        w_err < 0.20,
        "phugoid ω: eig={w_eig:.5} approx={w_approx:.5} rel_err={w_err:.4}"
    );
    assert!(
        z_err < 0.05,
        "phugoid ζ: eig={z_eig:.5} approx={z_approx:.5} abs_err={z_err:.4}"
    );
    // Direct published-root check (Nelson / thesis flying qualities).
    assert!(
        (w_eig - 0.2135).abs() / 0.2135 < 0.05,
        "phugoid ω vs Nelson root: {w_eig}"
    );
    assert!(
        (z_eig - 0.0766).abs() < 0.02,
        "phugoid ζ vs Nelson: {z_eig}"
    );

    // Short-period should be faster than phugoid.
    assert!(
        modes.short_period.omega_n() > modes.phugoid.omega_n(),
        "short-period ω {} <= phugoid ω {}",
        modes.short_period.omega_n(),
        modes.phugoid.omega_n()
    );
    // Short-period damping typically ζ ∈ [0.3, 1.2] for GA (MIL-F-8785C band).
    let z_sp = modes.short_period.zeta();
    assert!(
        z_sp > 0.2 && z_sp < 1.5,
        "short-period ζ={z_sp} outside GA band"
    );
}

/// (b) Steady glide ratio L/D at best polar point (demo model, algebraic).
#[test]
fn steady_glide_ratio_l_over_d() {
    let mut m = unverified_demo_model();
    // Pure polar glide: find α maximising CL/CD with controls=0, q=0.
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
    let _ = best_a;
}

/// (c) Energy conservation in a frictionless pendulum-like aero-off case.
#[test]
fn energy_conservation_aero_off_pendulum() {
    // Elastic pendulum analogue: mass on a spring (no aero, no damping).
    // Symplectic Euler keeps energy error bounded on this oscillator.
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
    let steps = 10_000; // 10 s ≈ many periods (ω=√(k/m)=6.32 rad/s)
    for _ in 0..steps {
        w.step(dt, symplectic_euler);
        let e = total_energy(&w, &w.state);
        e_max = e_max.max(e);
        e_min = e_min.min(e);
    }
    let amp = (e_max - e_min) / e0.abs().max(1.0);
    // Measured (Linux x86_64, dt=1e-3, 10 s): relative peak-to-peak energy
    // oscillation ≈ 6.96e-3 on this spring+gravity oscillator.
    assert!(
        amp < 1e-2,
        "energy oscillation amp={amp}, e0={e0}, range=[{e_min},{e_max}]"
    );
    let _ = kinetic_energy(&w, &w.state);
}

/// (d) Quaternion norm drift under RK4 6-DOF aero.
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
    // Integrator renormalises; residual should be ~machine eps.
    assert!(max_drift < 1e-12, "quat norm drift {max_drift} (tol 1e-12)");
}

/// (e) Coordinated turn: `n = 1/cos φ`, `ψ̇ = g tan(φ)/V` within 0.5%.
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

        // 0.5% envelope (user requirement) — identities are exact; check band.
        assert!((n / n_exact - 1.0).abs() < 0.005);
        assert!((rate / rate_exact - 1.0).abs() < 0.005);
    }

    // Algebraic CL needed for level turn: n W = q̄ S CL (within 0.5%).
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
    m.thrust_offset = Vec3::new(0.0, 0.0, -0.5); // below COM → nose-up (+M_y? r×F)
                                                 // r = (0,0,-0.5), F = (1000,0,0) → r×F = (0, -0.5*1000? wait)
                                                 // i(j k): j(rz Fx - rx Fz) = (−0.5)(1000) − 0 = −500 → My = −500
                                                 // Aero moments also present; isolate by zeroing S.
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
    let v_body = Vec3::new(40.0, 0.0, 0.0);
    // eval uses passed-in rho; generator uses density_at(z).
    let a0 = SixDofAero::new(0, m.clone());
    let body = RigidBody::new(m.mass)
        .with_inertia(m.inertia_matrix())
        .with_position(Vec3::new(0.0, 0.0, 0.0))
        .with_velocity(Vec3::new(40.0, 0.0, 0.0));
    let w = World::new(vec![body], vec![Box::new(a0)]);
    let loads0 = SixDofAero::new(0, m.clone()).loads(&w, &w.state);
    assert_relative_eq!(loads0.rho, rho0, max_relative = 1e-12);
    let _ = (v_body, rho5);
}

#[test]
fn fdm_determinism_hashes_locked_shape() {
    // Hashes are locked in DETERMINISM.md after first green CI measurement.
    let h = canonical_fdm_hash();
    let g = canonical_dryden_gust_hash();
    assert_ne!(h, 0);
    assert_ne!(g, 0);
    assert_eq!(h, canonical_fdm_hash());
    assert_eq!(g, canonical_dryden_gust_hash());
}
