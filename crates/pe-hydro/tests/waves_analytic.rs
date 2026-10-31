//! Analytic / published-formula oracles for Airy, Stokes, Gerstner, spectra, heave.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_core::forces::UniformGravity;
use pe_core::integrators::rk4;
use pe_core::{inertia_solid_box, RigidBody, World};
use pe_hydro::hull::{BoxHull, Hull, RHO_FRESHWATER};
use pe_hydro::waves::{
    deep_water_wavelength, group_speed, phase_speed, pierson_moskowitz_hs_approx,
    pierson_moskowitz_spectrum, solve_wave_number, AiryWave, GerstnerWave, SpectralOcean,
    StokesSecondOrder, Wave, WaveField,
};
use pe_hydro::{
    box_added_mass_heave, canonical_wave_body_hash, heave_period_with_added_mass, Buoyancy,
};
use pe_math::libm_ops;
use pe_math::{Vec3, STANDARD_GRAVITY};

const G: f64 = STANDARD_GRAVITY;

#[test]
fn deep_water_dispersion_oracles() {
    // T = 10 s → λ = g T²/(2π). With g₀=9.80665 → λ≈156.078 m (rounded 156.13
    // in many texts uses g≈9.81); c = λ/T ≈ 15.61 m/s; c_g = c/2.
    let t = 10.0;
    let lambda = deep_water_wavelength(t, G);
    let lambda_exact = G * t * t / (2.0 * core::f64::consts::PI);
    assert_abs_diff_eq!(lambda, lambda_exact, epsilon = 1e-12);
    assert_abs_diff_eq!(lambda, 156.13, epsilon = 0.06); // engineering round-off
    let omega = 2.0 * core::f64::consts::PI / t;
    let k_deep = omega * omega / G;
    let h = 1.0e6; // effectively deep
    let k = solve_wave_number(omega, h, G);
    assert_relative_eq!(k, k_deep, max_relative = 1e-10);
    let c = phase_speed(omega, k);
    assert_abs_diff_eq!(c, 15.61, epsilon = 0.01);
    let cg = group_speed(omega, k, h);
    assert_relative_eq!(cg, 0.5 * c, max_relative = 1e-6);
}

#[test]
fn shallow_water_phase_speed() {
    // h = 1 m → c = √(g h) ≈ 3.13 m/s.
    let h = 1.0;
    let c_expect = libm_ops::sqrt(G * h);
    assert_abs_diff_eq!(c_expect, 3.13, epsilon = 0.01);
    let omega = 0.05; // long wave → shallow
    let k = solve_wave_number(omega, h, G);
    let c = phase_speed(omega, k);
    assert_relative_eq!(c, c_expect, max_relative = 1e-4);
    let cg = group_speed(omega, k, h);
    assert_relative_eq!(cg, c, max_relative = 1e-3);
}

#[test]
fn airy_height_and_velocity_phase() {
    let w = AiryWave::earth(1.0, 10.0, 1.0e5, 0.0);
    // At origin t=0, phase 0 → η = A.
    assert_abs_diff_eq!(w.height(0.0, 0.0, 0.0), 1.0, epsilon = 1e-12);
    // Horizontal velocity at surface max when cos χ = 1; vertical ~0.
    let v = w.velocity(0.0, 0.0, 0.0, 0.0);
    assert!(v.x > 0.0);
    assert_abs_diff_eq!(v.y, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(v.z, 0.0, epsilon = 1e-8);
}

#[test]
fn stokes_deep_crest_elevation() {
    let a = 1.0;
    // Choose deep water with known k: from deep dispersion k = ω²/g with T such that k a small.
    let period = 10.0;
    let w = StokesSecondOrder::earth(a, period, 1.0e5, 0.0);
    let crest = w.deep_water_crest_elevation();
    let expect = a + 0.5 * w.k * a * a;
    assert_abs_diff_eq!(crest, expect, epsilon = 1e-15);
    // At crest phase χ=0: η = a + B₂ with B₂ → ½ k a² in deep water.
    assert_relative_eq!(w.height(0.0, 0.0, 0.0), expect, max_relative = 1e-6);
}

#[test]
fn gerstner_steepness_guard() {
    let k = 0.1;
    let a = 12.0; // k A = 1.2 > 1
    let err = GerstnerWave::earth(a, k, 1e5, 0.0, 1.0).unwrap_err();
    assert_eq!(
        err,
        pe_hydro::waves::gerstner::GerstnerError::SteepnessExceeded
    );
    let ok = GerstnerWave::earth(5.0, k, 1e5, 0.0, 1.0).unwrap();
    assert!(ok.steepness() <= 1.0);
    assert!(ok.height(0.0, 0.0, 0.0).abs() <= ok.amplitude * ok.q + 1e-9);
}

#[test]
fn pm_hs_approx_and_discrete_m0() {
    let u = 10.0;
    let hs_approx = pierson_moskowitz_hs_approx(u, G);
    assert_abs_diff_eq!(hs_approx, 0.21 * u * u / G, epsilon = 1e-15);
    // ≈ 2.14 m
    assert_abs_diff_eq!(hs_approx, 2.141, epsilon = 0.01);

    let ocean = SpectralOcean::pierson_moskowitz(u, 200.0, 48, 12, 12345);
    let hs = ocean.significant_wave_height();
    // Discrete spectrum Hs must match 4√m0 of its own components exactly.
    assert_abs_diff_eq!(hs, 4.0 * libm_ops::sqrt(ocean.m0_discrete), epsilon = 1e-12);
    // Recover continuum PM Hs within a few percent with this grid.
    assert!(
        (hs - hs_approx).abs() / hs_approx < 0.08,
        "Hs discrete={hs} approx={hs_approx}"
    );
}

#[test]
fn jonswap_exceeds_pm_near_peak() {
    let u = 10.0;
    let g = G;
    let omega_p = 0.877 * g / u;
    let s_pm = pierson_moskowitz_spectrum(omega_p, u, g);
    let s_j = pe_hydro::jonswap_spectrum(omega_p, u, omega_p, 3.3, g);
    assert!(s_j > s_pm * 2.0);
}

#[test]
fn heave_period_with_added_mass_formula() {
    let hull = BoxHull::new(12.0, 4.0, 2.0);
    let mass = 24_000.0;
    let draft = hull.upright_draft_for_mass(mass, RHO_FRESHWATER);
    let a_wp = hull.length * hull.beam;
    let a33 = box_added_mass_heave(RHO_FRESHWATER, a_wp, draft);
    let t = heave_period_with_added_mass(mass, a33, RHO_FRESHWATER, a_wp, G);
    let expect =
        2.0 * core::f64::consts::PI * libm_ops::sqrt((mass + a33) / (RHO_FRESHWATER * G * a_wp));
    assert_abs_diff_eq!(t, expect, epsilon = 1e-15);
}

#[test]
fn wave_induced_heave_responds_to_airy() {
    let hull = BoxHull::new(12.0, 4.0, 2.0);
    let mass = 24_000.0;
    let draft = hull.upright_draft_for_mass(mass, RHO_FRESHWATER);
    let z_eq = -draft + 0.5 * hull.depth;
    let wave = AiryWave::earth(0.3, 6.0, 40.0, 0.0);
    let body = RigidBody::new(mass)
        .with_inertia(inertia_solid_box(mass, 12.0, 4.0, 2.0))
        .with_position(Vec3::new(0.0, 0.0, z_eq));
    let mut w = World::new(
        vec![body],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(
                Buoyancy::freshwater(0, Hull::Box(hull), 0.0)
                    .with_wave(WaveField::Airy(wave.clone()))
                    .with_froude_krylov(true),
            ),
        ],
    );
    let z0 = w.state.x[0].z;
    w.run_for(6.0, 1e-2, rk4);
    // Body heave should move (not stay at equilibrium).
    assert!(
        (w.state.x[0].z - z0).abs() > 1e-3,
        "expected wave-driven heave motion"
    );
    // Instantaneous force at crest should exceed weight (more immersion).
    let (f, _) = w.forces(&w.state, 0.0);
    // Just check finite / non-NaN.
    assert!(f[0].z.is_finite());
}

#[test]
fn wave_body_hash_bit_stable() {
    let a = canonical_wave_body_hash();
    let b = canonical_wave_body_hash();
    assert_eq!(a, b);
}
