//! Analytic oracles for M31 finite-strain multiplicative F_p / Hencky plasticity.
//!
//! Tolerances are locked from measured errors — never silently loosened.
//! See docs/MATERIALS.md / docs/SOFT.md.

use approx::assert_relative_eq;
use pe_math::libm_ops;
use pe_math::Vec3;
use pe_soft::constraint::{invert3, DistanceConstraint, Multiplier, TetraFemConstraint};
use pe_soft::finite_strain_plastic::{
    analytic_log_permanent_strain, analytic_log_uniaxial_stress, bar_log_strain,
    fs_plastic_canonical_hash, fs_plastic_hash_params, isochoric_project, make_fs_plastic_bar,
    quasistatic_log_stretch_unload, sym_exp3, update_fs_plastic_distances, update_fs_plastic_tets,
    FsPlasticSoftWorld, FsPlasticTet,
};
use pe_soft::particle::Particle;
use pe_soft::plastic_coupling::{analytic_permanent_strain, make_plastic_bar};
use pe_soft::plasticity::{return_map_uniaxial, J2Params, UniaxialPlasticState};
use pe_soft::solver::SoftParams;
use pe_soft::{kinetic_energy, stretch_potential};

fn soft_mat() -> J2Params {
    fs_plastic_hash_params()
}

fn steel_like() -> J2Params {
    J2Params {
        youngs_modulus: 2.0e11,
        poisson_ratio: 0.3,
        yield_strength: 2.5e8,
        hardening_modulus: 2.0e9,
    }
}

/// Fixed log-stretch rest update matches the closed-form return map.
#[test]
fn log_bar_rest_matches_analytic_return_map() {
    let p = soft_mat();
    let l0 = 1.0;
    let mut w = make_fs_plastic_bar(p, l0, 0.01, 1.0, 16);
    let lambda = 1.04;
    w.soft.particles[1].x = Vec3::new(l0 * lambda, 0.0, 0.0);
    update_fs_plastic_distances(
        &w.soft.particles,
        &mut w.soft.constraints.distances,
        &p,
        &mut w.bars,
    );
    let eps = libm_ops::log(lambda);
    let expect_ep = analytic_log_permanent_strain(&p, eps);
    let expect_sig = analytic_log_uniaxial_stress(&p, eps);
    let err_ep = (w.bars[0].state.plastic_strain - expect_ep).abs() / expect_ep.abs().max(1e-16);
    let err_sig = (w.bars[0].state.stress - expect_sig).abs() / expect_sig.abs().max(1e-16);
    // Measured: exact (same function). CI lock < 1e-12 rel.
    assert!(err_ep < 1e-12, "ε_p rel err {err_ep}");
    assert!(err_sig < 1e-12, "σ rel err {err_sig}");
    assert_relative_eq!(
        w.soft.constraints.distances[0].rest_length,
        l0 * libm_ops::exp(expect_ep),
        max_relative = 1e-12
    );
}

/// Below yield: unload returns to cold length (full springback).
#[test]
fn log_elastic_unload_full_springback() {
    let p = soft_mat();
    let l0 = 1.0;
    let area = 0.01;
    let eps_y = p.yield_strength / p.youngs_modulus;
    let lambda = libm_ops::exp(0.5 * eps_y);
    let (len, eps_p, analytic) =
        quasistatic_log_stretch_unload(p, l0, area, lambda, 40, 400, 1.0 / 240.0);
    assert!(eps_p.abs() < 1e-15, "no plastic set expected, got {eps_p}");
    assert!(analytic.abs() < 1e-15);
    let err = (len - l0).abs() / l0;
    // Measured ~1e-4…1e-3 with damping settle; CI lock < 0.5%.
    assert!(
        err < 0.005,
        "elastic springback rel err {err}: len={len} L0={l0}"
    );
}

/// Stretch past yield then unload: permanent set matches `L₀ exp(ε_p)`.
///
/// Measured (Linux x86_64): length rel err vs `L₀ exp(ε_p)` ≈ **1e-4…2e-3**.
/// CI lock: **&lt; 1%**.
#[test]
fn log_plastic_unload_permanent_set_within_1pct() {
    let p = soft_mat();
    let l0 = 1.0;
    let area = 0.01;
    let lambda = 1.06;
    let (len, eps_p, analytic) =
        quasistatic_log_stretch_unload(p, l0, area, lambda, 80, 800, 1.0 / 240.0);
    let map_err = (eps_p - analytic).abs() / analytic.abs().max(1e-16);
    assert!(
        map_err < 1e-12,
        "held ε_p must match analytic map, rel {map_err}"
    );
    let expect = l0 * libm_ops::exp(analytic);
    let len_err = (len - expect).abs() / expect;
    assert!(
        len_err < 0.01,
        "unload length rel err {len_err}: len={len} expect={expect}"
    );
}

/// At modest strain, Hencky permanent set ≈ engineering (M30) within 0.5%.
#[test]
fn small_strain_log_agrees_with_engineering() {
    let p = steel_like();
    let eps_y = p.yield_strength / p.youngs_modulus;
    let eps_eng = eps_y + 1.0e-3;
    let lambda = 1.0 + eps_eng;
    let eps_log = libm_ops::log(lambda);
    let ep_eng = analytic_permanent_strain(&p, eps_eng);
    let ep_log = analytic_log_permanent_strain(&p, eps_log);
    let err = (ep_log - ep_eng).abs() / ep_eng.abs().max(1e-16);
    // ln(1+e)−e ≈ −e²/2 ≈ 8e-7 absolute on ε≈0.002 → relative on ε_p ≪ 0.5%.
    assert!(
        err < 0.005,
        "small-strain log vs eng relative err {err}: log={ep_log} eng={ep_eng}"
    );
}

/// At large stretch, log and engineering permanent strains diverge measurably.
#[test]
fn large_strain_log_diverges_from_engineering() {
    let p = soft_mat();
    let lambda = 1.2;
    let eps_eng = lambda - 1.0;
    let eps_log = libm_ops::log(lambda);
    let ep_eng = analytic_permanent_strain(&p, eps_eng);
    let ep_log = analytic_log_permanent_strain(&p, eps_log);
    let rel = (ep_log - ep_eng).abs() / ep_eng.abs().max(1e-16);
    // Measured: tens of percent relative difference at λ=1.2 — CI lock > 5%.
    assert!(
        rel > 0.05,
        "expected finite-strain divergence >5%, got {rel}: log={ep_log} eng={ep_eng}"
    );
}

/// Continuum tangent under plastic log loading: E_t = E H/(E+H).
#[test]
fn log_hardening_slope_within_1pct() {
    let p = soft_mat();
    let eps_y = p.yield_strength / p.youngs_modulus;
    let e1 = eps_y + 0.01;
    let e2 = eps_y + 0.02;
    let s1 = analytic_log_uniaxial_stress(&p, e1);
    let s2 = analytic_log_uniaxial_stress(&p, e2);
    let slope = (s2 - s1) / (e2 - e1);
    let et = p.uniaxial_tangent();
    let err = (slope - et).abs() / et.abs().max(1e-16);
    assert!(
        err < 0.01,
        "hardening slope rel err {err}: slope={slope} Et={et}"
    );
}

/// After plastic unload, stretch energy ≪ elastic yield scale.
#[test]
fn unloaded_log_stretch_energy_small() {
    let p = soft_mat();
    let l0 = 1.0;
    let area = 0.01;
    let lambda = 1.06;
    let (len, eps_p, _) = quasistatic_log_stretch_unload(p, l0, area, lambda, 80, 800, 1.0 / 240.0);
    let mut w = make_fs_plastic_bar(p, l0, area, 1.0, 32);
    // Reconstruct settled state energy scale check on a fresh bar at plastic rest.
    w.bars[0].state.plastic_strain = eps_p;
    w.soft.constraints.distances[0].rest_length = l0 * libm_ops::exp(eps_p);
    w.soft.particles[1].x = Vec3::new(len, 0.0, 0.0);
    let u = stretch_potential(&w.soft);
    let k = kinetic_energy(&w.soft);
    let eps_y = p.yield_strength / p.youngs_modulus;
    let scale = 0.5 * p.youngs_modulus * area * l0 * eps_y * eps_y;
    let err = (u + k) / scale.max(1e-30);
    // Measured ≪ 2% of elastic yield energy scale. CI lock < 2%.
    assert!(
        err < 0.02,
        "unloaded energy / scale = {err} (U={u} K={k} scale={scale})"
    );
}

/// Multiplicative tet: pure dilatation does not yield; shear yields; det(F_p)=1.
#[test]
fn tet_multiplicative_isochoric_shear_yield() {
    let p = soft_mat();
    let mut w = FsPlasticSoftWorld::new(p);
    w.soft.params.gravity = Vec3::zero();
    let verts = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    for &v in &verts {
        w.soft.add_particle(Particle::pinned(v));
    }
    let dm = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let dm_inv = invert3(&dm).unwrap();
    let e = p.youngs_modulus;
    let nu = p.poisson_ratio;
    let mu = e / (2.0 * (1.0 + nu));
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    w.soft.constraints.tetra_fem.push(TetraFemConstraint {
        indices: [0, 1, 2, 3],
        dm_inv,
        rest_volume: 1.0 / 6.0,
        lame_lambda: lambda,
        lame_mu: mu,
        neo_hookean: false,
        lambda_dev: Multiplier::default(),
        lambda_vol: Multiplier::default(),
        broken: false,
    });
    w.track_all_tets();

    for (i, &v) in verts.iter().enumerate() {
        w.soft.particles[i].x = v * 1.05;
    }
    let s0 = update_fs_plastic_tets(
        &w.soft.particles,
        &mut w.soft.constraints.tetra_fem,
        &p,
        &mut w.tets,
    );
    assert_eq!(s0.n_plastic_tets, 0, "dilatation must not yield under J2");

    for (i, &v) in verts.iter().enumerate() {
        w.soft.particles[i].x = v;
    }
    w.soft.particles[1].x = Vec3::new(1.0, 0.15, 0.0);
    w.tets[0] = FsPlasticTet::new(0, dm_inv, 1.0 / 6.0);
    let s1 = update_fs_plastic_tets(
        &w.soft.particles,
        &mut w.soft.constraints.tetra_fem,
        &p,
        &mut w.tets,
    );
    assert!(
        s1.n_plastic_tets >= 1,
        "expected shear yield, α={}",
        w.tets[0].eq_plastic_strain
    );
    let det_fp = {
        let m = &w.tets[0].fp;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    // Measured: |det F_p − 1| ~ 1e-15 after projection. CI lock < 1e-10.
    assert!(
        (det_fp - 1.0).abs() < 1e-10,
        "F_p must be isochoric, det={det_fp}"
    );
    let seq = w.tets[0].stress.von_mises_eq();
    let sy = p.yield_strength + p.hardening_modulus * w.tets[0].eq_plastic_strain;
    let err = (seq - sy).abs() / sy;
    assert!(err < 0.01, "tet return-to-surface err {err}");
}

/// Isochoric helper + spectral exp smoke checks used by the tet path.
#[test]
fn spectral_exp_and_isochoric_helpers() {
    let m = [[0.0, 0.2, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let e = sym_exp3(&m);
    // exp of pure shear-like symmetric: det should be positive.
    let det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
        - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
        + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
    assert!(det > 0.0);
    let p = isochoric_project(&e);
    let det_p = p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1])
        - p[0][1] * (p[1][0] * p[2][2] - p[1][2] * p[2][0])
        + p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0]);
    assert!((det_p - 1.0).abs() < 1e-12);
}

/// Determinism: finite-strain plastic hash is bit-stable.
#[test]
fn fs_plastic_canonical_hash_stable_and_nonzero() {
    let a = fs_plastic_canonical_hash();
    let b = fs_plastic_canonical_hash();
    assert_eq!(a, b);
    assert_ne!(a, 0);
}

/// Log-strain helper matches ln(L/L0).
#[test]
fn log_strain_helper() {
    let l0 = 2.0;
    let particles = vec![
        Particle::pinned(Vec3::zero()),
        Particle::dynamic(Vec3::new(2.0 * libm_ops::exp(0.1), 0.0, 0.0), 1.0),
    ];
    let c = DistanceConstraint {
        i: 0,
        j: 1,
        rest_length: l0,
        compliance: 1e-6,
        lambda: Multiplier::default(),
        broken: false,
    };
    let eps = bar_log_strain(&particles, &c, l0);
    assert_relative_eq!(eps, 0.1, max_relative = 1e-12);
}

/// Analytic helpers match direct return map.
#[test]
fn analytic_log_helpers_match_direct_return_map() {
    let p = soft_mat();
    let eps_y = p.yield_strength / p.youngs_modulus;
    let eps = eps_y + 0.01;
    let r = return_map_uniaxial(&p, UniaxialPlasticState::default(), eps);
    assert_relative_eq!(
        analytic_log_permanent_strain(&p, eps),
        r.state.plastic_strain,
        max_relative = 1e-15
    );
    assert_relative_eq!(
        analytic_log_uniaxial_stress(&p, eps),
        r.state.stress,
        max_relative = 1e-15
    );
}

/// M30 engineering plastic hash path still compiles / runs (unchanged contract).
#[test]
fn m30_engineering_bar_still_available() {
    let p = steel_like();
    let mut w = make_plastic_bar(p, 1.0, 0.01, 1.0, 8);
    w.soft.params = SoftParams {
        substeps: 8,
        iterations: 2,
        gravity: Vec3::zero(),
        damping: 0.0,
    };
    assert_eq!(w.bars.len(), 1);
}
