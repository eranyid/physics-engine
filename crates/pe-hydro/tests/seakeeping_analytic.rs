//! Strip-theory seakeeping oracles (heave RAO long-wave, roll/heave naturals, free decay).

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_hydro::helpers::{bm_from_inertia, metacentric_height};
use pe_hydro::hull::{BoxHull, RHO_FRESHWATER};
use pe_hydro::seakeeping::{
    assemble_strip_matrices, box_barge_stations, box_heave_natural_omega, free_decay_heave,
    heave_pitch_rao, roll_natural_period, sectional_a33_inf, HeavePitchTd,
};
use pe_math::STANDARD_GRAVITY;

#[test]
fn heave_rao_approaches_one_in_long_waves() {
    // Box barge 80×12×draft~3 m; long wave T=60 s ⇒ λ ≫ L ⇒ ship follows wave.
    let length = 80.0;
    let beam = 12.0;
    let draft = 3.0;
    let rho = RHO_FRESHWATER;
    let mass = rho * length * beam * draft;
    let iyy = mass * (length * length + draft * draft) / 12.0;
    let stations = box_barge_stations(length, beam, draft, 20);
    let omega = 2.0 * std::f64::consts::PI / 60.0;
    let rao = heave_pitch_rao(&stations, mass, iyy, rho, omega, 200.0, STANDARD_GRAVITY);
    // Long-wave heave RAO → 1. Engineering strip + FK/diffraction: within 10%.
    assert!(
        (rao.heave - 1.0).abs() < 0.10,
        "long-wave heave RAO = {} (tol 10% of 1)",
        rao.heave
    );
}

#[test]
fn box_heave_natural_matches_analytic() {
    let length = 40.0;
    let beam = 8.0;
    let draft = 2.0;
    let rho = RHO_FRESHWATER;
    let mass = rho * length * beam * draft;
    let w_n = box_heave_natural_omega(length, beam, draft, mass, rho, STANDARD_GRAVITY, 16);
    let stations = box_barge_stations(length, beam, draft, 16);
    let mats = assemble_strip_matrices(&stations, rho, 1e3, STANDARD_GRAVITY);
    let a_wp = length * beam;
    let expect = (rho * STANDARD_GRAVITY * a_wp / (mass + mats.a[0][0])).sqrt();
    assert_relative_eq!(w_n, expect, max_relative = 1e-12);
    // Sanity: period of order several seconds for this barge.
    let t = 2.0 * std::f64::consts::PI / w_n;
    assert!(t > 1.0 && t < 20.0, "T_heave={t}");
}

#[test]
fn roll_natural_period_formula() {
    let hull = BoxHull::new(30.0, 8.0, 4.0);
    let draft = 2.0;
    let rho = RHO_FRESHWATER;
    let volume = hull.length * hull.beam * draft;
    let mass = rho * volume;
    // BM = I_wp / ∇, I_wp = L B³/12, KB = draft/2, KG = 1.5 m from keel → GM.
    let i_wp = hull.length * hull.beam.powi(3) / 12.0;
    let bm = bm_from_inertia(i_wp, volume);
    let kb = 0.5 * draft;
    let kg = 1.5;
    let gm = metacentric_height(kb, bm, kg);
    assert!(gm > 0.0);
    // I44 about COM: box about x through COM at half-depth from keel...
    // COM at z = -draft + kg from MWL; use geometric box inertia about longitudinal axis.
    let i44 = mass * (hull.beam * hull.beam + hull.depth * hull.depth) / 12.0;
    // A44 = 0 (not predicted here) — documented.
    let t = roll_natural_period(i44, rho, volume, gm, STANDARD_GRAVITY);
    let expect = 2.0 * std::f64::consts::PI * (i44 / (rho * STANDARD_GRAVITY * volume * gm)).sqrt();
    assert_abs_diff_eq!(t, expect, epsilon = 1e-12);
}

#[test]
fn free_decay_rate_matches_damping_coefficient() {
    let mass = 24_000.0;
    let a33 = sectional_a33_inf(RHO_FRESHWATER, 4.0, 1.0) * 12.0; // rough strip sum
    let a_wp = 12.0 * 4.0;
    let c33 = RHO_FRESHWATER * STANDARD_GRAVITY * a_wp;
    let b33 = 2.0 * 0.05 * ((mass + a33) * c33).sqrt(); // ζ ≈ 0.05
    let (_traj, beta_a, beta_fit) = free_decay_heave(mass, a33, b33, c33, 0.2, 1e-3, 20_000);
    // Analytic β = B/(2m_tot); fit from peaks within 15% (discrete + peak sampling).
    assert_relative_eq!(beta_a, b33 / (2.0 * (mass + a33)), max_relative = 1e-12);
    assert!(
        (beta_fit - beta_a).abs() / beta_a < 0.15,
        "beta_fit={beta_fit} beta_a={beta_a}"
    );
}

#[test]
fn time_domain_heave_bounded_in_regular_waves() {
    let length = 50.0;
    let beam = 10.0;
    let draft = 2.5;
    let rho = RHO_FRESHWATER;
    let mass = rho * length * beam * draft;
    let iyy = mass * (length * length) / 12.0;
    let omega = 2.0 * std::f64::consts::PI / 8.0;
    let mut td = HeavePitchTd::box_barge(
        length,
        beam,
        draft,
        mass,
        iyy,
        rho,
        0.3,
        omega,
        100.0,
        STANDARD_GRAVITY,
        16,
    );
    let mut z_max = 0.0_f64;
    for _ in 0..4000 {
        td.step(0.01);
        z_max = z_max.max(td.xi[0].abs());
    }
    // Should not diverge; amplitude order of wave amp × RAO (O(1 m)).
    assert!(z_max < 5.0, "z_max={z_max}");
    assert!(z_max > 0.01, "should respond to waves");
}

#[test]
fn sectional_a33_inf_circle_formula() {
    let rho = 1000.0;
    let beam = 4.0;
    let a = sectional_a33_inf(rho, beam, 1.0);
    let r = 2.0;
    assert_abs_diff_eq!(a, rho * std::f64::consts::PI * r * r, epsilon = 1e-12);
}
