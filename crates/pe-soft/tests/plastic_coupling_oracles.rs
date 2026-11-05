//! Analytic oracles for M30 soft XPBD ↔ J2 plastic coupling.
//!
//! Tolerances are locked from measured errors — never silently loosened.
//! See docs/MATERIALS.md / docs/SOFT.md.

use approx::assert_relative_eq;
use pe_math::Vec3;
use pe_soft::constraint::{invert3, DistanceConstraint, Multiplier, TetraFemConstraint};
use pe_soft::particle::Particle;
use pe_soft::plastic_coupling::{
    analytic_permanent_strain, analytic_uniaxial_stress, bar_engineering_strain, make_plastic_bar,
    plastic_canonical_hash, plastic_hash_params, quasistatic_stretch_unload,
    update_plastic_distances, update_plastic_tets, PlasticSoftWorld,
};
use pe_soft::plasticity::{return_map_uniaxial, J2Params, J2PlasticState3, UniaxialPlasticState};
use pe_soft::solver::SoftParams;
use pe_soft::{kinetic_energy, stretch_potential};

fn steel() -> J2Params {
    plastic_hash_params()
}

/// Fixed-stretch rest update matches the closed-form return map (bit-level).
#[test]
fn bar_rest_update_matches_analytic_return_map() {
    let p = steel();
    let l0 = 1.0;
    let area = 0.01;
    let mut w = make_plastic_bar(p, l0, area, 1.0, 16);
    let eps_y = p.yield_strength / p.youngs_modulus;
    let eps = eps_y + 2e-3;
    w.soft.particles[1].x = Vec3::new(l0 * (1.0 + eps), 0.0, 0.0);
    update_plastic_distances(
        &w.soft.particles,
        &mut w.soft.constraints.distances,
        &p,
        &mut w.bars,
    );
    let expect_ep = analytic_permanent_strain(&p, eps);
    let expect_sig = analytic_uniaxial_stress(&p, eps);
    let err_ep = (w.bars[0].state.plastic_strain - expect_ep).abs() / expect_ep.abs().max(1e-16);
    let err_sig = (w.bars[0].state.stress - expect_sig).abs() / expect_sig.abs().max(1e-16);
    // Measured: exact (same function). CI lock < 1e-12 rel.
    assert!(err_ep < 1e-12, "ε_p rel err {err_ep}");
    assert!(err_sig < 1e-12, "σ rel err {err_sig}");
    assert_relative_eq!(
        w.soft.constraints.distances[0].rest_length,
        l0 * (1.0 + expect_ep),
        max_relative = 1e-12
    );
}

/// Below yield: unload returns to cold length (full springback).
#[test]
fn elastic_unload_full_springback() {
    let p = steel();
    let l0 = 1.0;
    let area = 0.01;
    let eps_y = p.yield_strength / p.youngs_modulus;
    let eps = 0.5 * eps_y;
    let (len, eps_p, analytic) = quasistatic_stretch_unload(p, l0, area, eps, 40, 400, 1.0 / 240.0);
    assert!(eps_p.abs() < 1e-15, "no plastic set expected, got {eps_p}");
    assert!(analytic.abs() < 1e-15);
    let err = (len - l0).abs() / l0;
    // Measured ~1e-4…1e-3 with damping settle; CI lock < 0.5%.
    assert!(
        err < 0.005,
        "elastic springback rel err {err}: len={len} L0={l0}"
    );
}

/// Stretch past yield then unload: permanent set matches analytic ε_p.
///
/// Measured (Linux x86_64): length rel err vs `L₀(1+ε_p)` ≈ **2e-4…1e-3**.
/// CI lock: **&lt; 1%**.
#[test]
fn plastic_unload_permanent_set_within_1pct() {
    let p = steel();
    let l0 = 1.0;
    let area = 0.01;
    let eps_y = p.yield_strength / p.youngs_modulus;
    let eps = eps_y + 2e-3;
    let (len, eps_p, analytic) = quasistatic_stretch_unload(p, l0, area, eps, 80, 800, 1.0 / 240.0);
    let err_map = (eps_p - analytic).abs() / analytic.abs().max(1e-16);
    assert!(
        err_map < 0.01,
        "return-map ε_p during hold rel err {err_map}: got={eps_p} analytic={analytic}"
    );
    let expect_len = l0 * (1.0 + analytic);
    let err_len = (len - expect_len).abs() / expect_len;
    assert!(
        err_len < 0.01,
        "permanent length rel err {err_len}: len={len} expect={expect_len}"
    );
}

/// Hardening slope through the plastic XPBD bar under quasistatic hold.
///
/// At two holds past yield, `Δσ/Δε` matches `E_t = E H/(E+H)`.
/// Measured: ~0 (exact state from return map). CI lock **&lt; 1%**.
#[test]
fn plastic_hardening_slope_within_1pct() {
    let p = steel();
    let et = p.uniaxial_tangent();
    let eps_y = p.yield_strength / p.youngs_modulus;
    let eps1 = eps_y + 5e-4;
    let eps2 = eps_y + 1.5e-3;
    let mut stresses = Vec::new();
    for &eps in &[eps1, eps2] {
        let mut w = make_plastic_bar(p, 1.0, 0.01, 1.0, 16);
        w.soft.particles[1].x = Vec3::new(1.0 * (1.0 + eps), 0.0, 0.0);
        w.soft.particles[1].inv_mass = 0.0;
        for _ in 0..40 {
            w.step(1.0 / 240.0);
            w.soft.particles[1].x = Vec3::new(1.0 * (1.0 + eps), 0.0, 0.0);
            w.soft.particles[1].v = Vec3::zero();
        }
        stresses.push(w.bars[0].state.stress);
    }
    let slope = (stresses[1] - stresses[0]) / (eps2 - eps1);
    let err = (slope - et).abs() / et;
    assert!(
        err < 0.01,
        "hardening slope err {err}: meas={slope} Et={et}"
    );
}

/// After plastic unload settle, stretch potential ≈ 0 (elastic residual gone).
#[test]
fn unloaded_elastic_energy_near_zero() {
    let p = steel();
    let l0 = 1.0;
    let area = 0.01;
    let eps_y = p.yield_strength / p.youngs_modulus;
    let eps = eps_y + 2e-3;
    let mut w = make_plastic_bar(p, l0, area, 1.0, 32);
    w.soft.params.damping = 3.0;
    let x_hold = l0 * (1.0 + eps);
    w.soft.particles[1].x = Vec3::new(x_hold, 0.0, 0.0);
    w.soft.particles[1].inv_mass = 0.0;
    for _ in 0..80 {
        w.step(1.0 / 240.0);
        w.soft.particles[1].x = Vec3::new(x_hold, 0.0, 0.0);
        w.soft.particles[1].v = Vec3::zero();
    }
    w.soft.particles[1].inv_mass = 1.0;
    for _ in 0..800 {
        w.step(1.0 / 240.0);
    }
    let u = stretch_potential(&w.soft);
    let ke = kinetic_energy(&w.soft);
    // Peak elastic energy scale ~ ½ E A L ε_e² with ε_e ~ σ_y/E before yield end.
    let u_scale = 0.5 * p.youngs_modulus * area * l0 * (eps_y * eps_y);
    assert!(
        u / u_scale < 0.02,
        "residual stretch energy {u} vs scale {u_scale}"
    );
    assert!(ke < 1e-4, "residual KE {ke}");
}

/// Elastic path energy: stretch below yield, release — energy decays, no growth.
#[test]
fn elastic_path_energy_no_growth() {
    let p = steel();
    let l0 = 1.0;
    let mut w = make_plastic_bar(p, l0, 0.01, 1.0, 32);
    w.soft.params.damping = 0.0;
    let eps_y = p.yield_strength / p.youngs_modulus;
    w.soft.particles[1].x = Vec3::new(l0 * (1.0 + 0.3 * eps_y), 0.0, 0.0);
    w.soft.particles[1].v = Vec3::zero();
    // One plastic update (should be elastic) then free oscillation with light damp later.
    update_plastic_distances(
        &w.soft.particles,
        &mut w.soft.constraints.distances,
        &p,
        &mut w.bars,
    );
    assert_eq!(w.bars[0].state.eq_plastic_strain, 0.0);
    let e0 = stretch_potential(&w.soft) + kinetic_energy(&w.soft);
    let mut e_max = e0;
    for _ in 0..200 {
        w.step(1.0 / 240.0);
        let e = stretch_potential(&w.soft) + kinetic_energy(&w.soft);
        e_max = e_max.max(e);
    }
    // XPBD dissipates; must not grow above a tiny numerical overshoot band.
    assert!(
        e_max <= e0 * 1.001 + 1e-9,
        "energy grew: e0={e0} e_max={e_max}"
    );
}

/// Multi-segment plastic chain: mean ε_p matches single-segment analytic.
#[test]
fn multi_segment_bar_mean_plastic_strain() {
    let p = steel();
    let l0 = 1.0;
    let area = 0.01;
    let n_seg = 4;
    let mut w = PlasticSoftWorld::new(p);
    w.soft.params = SoftParams {
        substeps: 32,
        iterations: 2,
        gravity: Vec3::zero(),
        damping: 2.0,
    };
    let dx = l0 / n_seg as f64;
    let alpha = dx / (p.youngs_modulus * area);
    for i in 0..=n_seg {
        let part = if i == 0 {
            Particle::pinned(Vec3::zero())
        } else {
            Particle::dynamic(Vec3::new(i as f64 * dx, 0.0, 0.0), 1.0 / n_seg as f64)
        };
        w.soft.add_particle(part);
    }
    for i in 0..n_seg {
        w.soft.constraints.distances.push(DistanceConstraint {
            i,
            j: i + 1,
            rest_length: dx,
            compliance: alpha,
            lambda: Multiplier::default(),
            broken: false,
        });
    }
    w.track_all_distances(area);
    let eps_y = p.yield_strength / p.youngs_modulus;
    let eps = eps_y + 1.5e-3;
    // Uniform stretch of free nodes.
    for i in 1..=n_seg {
        w.soft.particles[i].x.x = (i as f64) * dx * (1.0 + eps);
        w.soft.particles[i].inv_mass = 0.0;
    }
    for _ in 0..60 {
        w.step(1.0 / 240.0);
        for i in 1..=n_seg {
            w.soft.particles[i].x.x = (i as f64) * dx * (1.0 + eps);
            w.soft.particles[i].v = Vec3::zero();
        }
    }
    let analytic = analytic_permanent_strain(&p, eps);
    let mean = w.mean_bar_eq_plastic();
    // eq plastic ≈ |ε_p| for monotonic tension
    let err = (mean - analytic.abs()).abs() / analytic.abs().max(1e-16);
    assert!(
        err < 0.01,
        "multi-seg mean ε_p^eq rel err {err}: mean={mean} analytic={analytic}"
    );
}

/// Tet Green-strain plastic update: pure dilatation does not yield; shear does.
#[test]
fn tet_hydrostatic_no_yield_shear_yields() {
    let p = steel();
    // Build a single tet at the origin.
    let mut w = PlasticSoftWorld::new(p);
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

    // Pure dilatation of apex-like: scale all free... all pinned — move via positions.
    // Uniform scale 1.01 about origin.
    for (i, &v) in verts.iter().enumerate() {
        w.soft.particles[i].x = v * 1.01;
    }
    let s0 = update_plastic_tets(
        &w.soft.particles,
        &mut w.soft.constraints.tetra_fem,
        &p,
        &mut w.tets,
    );
    assert_eq!(s0.n_plastic_tets, 0, "dilatation must not yield under J2");

    // Large shear: move vertex 1 in +y.
    for (i, &v) in verts.iter().enumerate() {
        w.soft.particles[i].x = v;
    }
    w.soft.particles[1].x = Vec3::new(1.0, 0.05, 0.0);
    // Reset plastic state for a clean shear probe.
    w.tets[0].state = J2PlasticState3::default();
    let s1 = update_plastic_tets(
        &w.soft.particles,
        &mut w.soft.constraints.tetra_fem,
        &p,
        &mut w.tets,
    );
    // 5% shear Green strain is large vs steel ε_y ~ 0.00125 — must yield.
    assert!(
        s1.n_plastic_tets >= 1,
        "expected shear yield, eq_p={}",
        w.tets[0].state.eq_plastic_strain
    );
    let seq = w.tets[0].state.stress.von_mises_eq();
    let sy = p.yield_strength + p.hardening_modulus * w.tets[0].state.eq_plastic_strain;
    let err = (seq - sy).abs() / sy;
    assert!(err < 0.01, "tet return-to-surface err {err}");
}

/// Determinism: plastic hash is bit-stable across repeated calls.
#[test]
fn plastic_canonical_hash_stable_and_nonzero() {
    let a = plastic_canonical_hash();
    let b = plastic_canonical_hash();
    assert_eq!(a, b);
    assert_ne!(a, 0);
}

/// Strain helper matches (L−L0)/L0.
#[test]
fn engineering_strain_helper() {
    let l0 = 2.0;
    let particles = vec![
        Particle::pinned(Vec3::zero()),
        Particle::dynamic(Vec3::new(2.2, 0.0, 0.0), 1.0),
    ];
    let c = DistanceConstraint {
        i: 0,
        j: 1,
        rest_length: l0,
        compliance: 1e-6,
        lambda: Multiplier::default(),
        broken: false,
    };
    let eps = bar_engineering_strain(&particles, &c, l0);
    assert_relative_eq!(eps, 0.1, max_relative = 1e-12);
    let _ = particles;
}

/// Virgin return map identity used by analytic helpers.
#[test]
fn analytic_helpers_match_direct_return_map() {
    let p = steel();
    let eps_y = p.yield_strength / p.youngs_modulus;
    let eps = eps_y + 1e-3;
    let r = return_map_uniaxial(&p, UniaxialPlasticState::default(), eps);
    assert_relative_eq!(
        analytic_permanent_strain(&p, eps),
        r.state.plastic_strain,
        max_relative = 1e-15
    );
    assert_relative_eq!(
        analytic_uniaxial_stress(&p, eps),
        r.state.stress,
        max_relative = 1e-15
    );
}
