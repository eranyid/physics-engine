//! Analytic oracles for M32 spectral Hencky / Kirchhoff return on `b_e`.
//!
//! Tolerances are locked from measured errors — never silently loosened.
//! See docs/MATERIALS.md / docs/SOFT.md.

use approx::assert_relative_eq;
use pe_math::libm_ops;
use pe_math::Vec3;
use pe_soft::constraint::{invert3, Multiplier, TetraFemConstraint};
use pe_soft::finite_strain_plastic::{
    fs_plastic_hash_params, update_fs_plastic_tets, FsPlasticSoftWorld,
};
use pe_soft::particle::Particle;
use pe_soft::plasticity::{J2Params, SymTensor3};
use pe_soft::spectral_hencky_plastic::{
    analytic_isochoric_uniaxial_kirchhoff, analytic_isochoric_uniaxial_seq,
    assemble_from_principal, be_plastic_canonical_hash, be_plastic_hash_params, left_cauchy_green,
    make_be_plastic_bar, quasistatic_be_log_stretch_unload, return_map_spectral_hencky_be,
    sym_spectral_decompose, update_be_plastic_tets, BePlasticSoftWorld, BePlasticTet,
};

fn soft_mat() -> J2Params {
    be_plastic_hash_params()
}

/// Isochoric uniaxial elastic Kirchhoff matches closed form (exact).
#[test]
fn isochoric_uniaxial_kirchhoff_analytic_oracle() {
    let p = soft_mat();
    let mu = p.shear_modulus();
    let lim = p.yield_strength / (3.0 * mu);
    let lam = libm_ops::exp(0.4 * lim);
    let be = [
        [lam * lam, 0.0, 0.0],
        [0.0, 1.0 / lam, 0.0],
        [0.0, 0.0, 1.0 / lam],
    ];
    let r = return_map_spectral_hencky_be(&p, &be, 0.0);
    assert!(!r.plastic, "must stay elastic for this λ");
    let (t1, t2, t3) = analytic_isochoric_uniaxial_kirchhoff(mu, lam);
    // Measured: bit-level agreement with principal closed form. CI lock < 1e-12 rel.
    let e1 = (r.kirchhoff.xx - t1).abs() / t1.abs().max(1e-16);
    let e2 = (r.kirchhoff.yy - t2).abs() / t2.abs().max(1e-16);
    let e3 = (r.kirchhoff.zz - t3).abs() / t3.abs().max(1e-16);
    assert!(e1 < 1e-12, "τ1 rel {e1}");
    assert!(e2 < 1e-12, "τ2 rel {e2}");
    assert!(e3 < 1e-12, "τ3 rel {e3}");
    let seq = r.kirchhoff.von_mises_eq();
    let seq_a = analytic_isochoric_uniaxial_seq(mu, lam);
    assert_relative_eq!(seq, seq_a, max_relative = 1e-12);
}

/// Past yield: Kirchhoff von Mises sits on `σ_y + H α`; det(b_e)=1.
#[test]
fn isochoric_uniaxial_plastic_surface_and_be_det() {
    let p = soft_mat();
    let mu = p.shear_modulus();
    let lam = 1.1;
    let be = [
        [lam * lam, 0.0, 0.0],
        [0.0, 1.0 / lam, 0.0],
        [0.0, 0.0, 1.0 / lam],
    ];
    assert!(analytic_isochoric_uniaxial_seq(mu, lam) > p.yield_strength);
    let r = return_map_spectral_hencky_be(&p, &be, 0.0);
    assert!(r.plastic);
    let seq = r.kirchhoff.von_mises_eq();
    let sy = p.yield_strength + p.hardening_modulus * r.eq_plastic_strain;
    let err = (seq - sy).abs() / sy;
    // Measured ~0 (spectral return algebra). CI lock < 1e-10.
    assert!(err < 1e-10, "surface err {err}");
    let det_be = {
        let m = &r.be;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    assert!((det_be - 1.0).abs() < 1e-10, "det b_e = {det_be}");
}

/// Pure dilatation of `b_e` never yields under J2.
#[test]
fn hydrostatic_be_no_yield() {
    let p = soft_mat();
    let s = 1.25_f64;
    let be = [[s * s, 0.0, 0.0], [0.0, s * s, 0.0], [0.0, 0.0, s * s]];
    let r = return_map_spectral_hencky_be(&p, &be, 0.0);
    assert!(!r.plastic);
    assert!(r.kirchhoff.von_mises_eq().abs() < 1e-9);
}

/// Soft XPBD tet: dilatation no yield; shear yields; det(F_p)=1; surface <1%.
#[test]
fn tet_spectral_isochoric_shear_yield() {
    let p = soft_mat();
    let mut w = BePlasticSoftWorld::new(p);
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
    let s0 = update_be_plastic_tets(
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
    w.tets[0] = BePlasticTet::new(0, dm_inv, 1.0 / 6.0);
    let s1 = update_be_plastic_tets(
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
    // Measured: |det F_p − 1| ~ 1e-15. CI lock < 1e-10.
    assert!(
        (det_fp - 1.0).abs() < 1e-10,
        "F_p must be isochoric, det={det_fp}"
    );
    let seq = w.tets[0].kirchhoff.von_mises_eq();
    let sy = p.yield_strength + p.hardening_modulus * w.tets[0].eq_plastic_strain;
    let err = (seq - sy).abs() / sy;
    // Measured ≪ 1%. CI lock < 1%.
    assert!(err < 0.01, "tet Kirchhoff return-to-surface err {err}");
}

/// At small shear, spectral Kirchhoff ≈ M31 Green+Hooke stress within 2%.
#[test]
fn small_strain_spectral_agrees_with_green_path() {
    let p = soft_mat();
    let gamma = 1.0e-3;
    // F = [[1, γ, 0], [0, 1, 0], [0, 0, 1]] → simple shear, F_p = I.
    let f = [[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let be = left_cauchy_green(&f);
    let r_be = return_map_spectral_hencky_be(&p, &be, 0.0);

    // M31 path via a one-shot tet update on pinned verts.
    let mut w = FsPlasticSoftWorld::new(p);
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
    // Apply F to tet: x' = F x with x columns from verts relative to v0.
    w.soft.particles[0].x = verts[0];
    w.soft.particles[1].x = Vec3::new(1.0, 0.0, 0.0);
    w.soft.particles[2].x = Vec3::new(gamma, 1.0, 0.0); // F * e2 = (γ, 1, 0)
    w.soft.particles[3].x = Vec3::new(0.0, 0.0, 1.0);
    update_fs_plastic_tets(
        &w.soft.particles,
        &mut w.soft.constraints.tetra_fem,
        &p,
        &mut w.tets,
    );
    let s_be = r_be.kirchhoff.frobenius();
    let s_gr = w.tets[0].stress.frobenius();
    let rel = (s_be - s_gr).abs() / s_gr.max(1e-30);
    // At γ=1e-3, Hencky ≈ Green; measured ≪ 2%. CI lock < 2%.
    assert!(
        rel < 0.02,
        "small-strain spectral vs Green rel {rel}: be={s_be} green={s_gr}"
    );
}

/// At large shear, spectral Kirchhoff diverges from Green+Hooke by >5%.
#[test]
fn large_strain_spectral_diverges_from_green_path() {
    let p = soft_mat();
    let gamma = 0.4;
    let f = [[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let be = left_cauchy_green(&f);
    let r_be = return_map_spectral_hencky_be(&p, &be, 0.0);

    let mut w = FsPlasticSoftWorld::new(p);
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
    w.soft.particles[0].x = verts[0];
    w.soft.particles[1].x = Vec3::new(1.0, 0.0, 0.0);
    w.soft.particles[2].x = Vec3::new(gamma, 1.0, 0.0);
    w.soft.particles[3].x = Vec3::new(0.0, 0.0, 1.0);
    update_fs_plastic_tets(
        &w.soft.particles,
        &mut w.soft.constraints.tetra_fem,
        &p,
        &mut w.tets,
    );
    let s_be = r_be.kirchhoff.frobenius();
    let s_gr = w.tets[0].stress.frobenius();
    let rel = (s_be - s_gr).abs() / s_gr.max(1e-30);
    // Measured: tens of percent at γ=0.4. CI lock > 5%.
    assert!(
        rel > 0.05,
        "expected large-strain divergence >5%, got {rel}: be={s_be} green={s_gr}"
    );
}

/// Hencky bar path (shared with M31) still unloads to `L₀ exp(ε_p)`.
#[test]
fn be_world_log_bar_unload_within_1pct() {
    let p = soft_mat();
    let l0 = 1.0;
    let area = 0.01;
    let lambda = 1.06;
    let (len, eps_p, analytic) =
        quasistatic_be_log_stretch_unload(p, l0, area, lambda, 80, 800, 1.0 / 240.0);
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

/// Spectral reconstruct identity: assemble(eig, Q) ≈ input.
#[test]
fn spectral_roundtrip_symmetric() {
    let m = [[2.0, 0.3, -0.1], [0.3, 1.5, 0.2], [-0.1, 0.2, 0.8]];
    let (eigs, q) = sym_spectral_decompose(&m);
    let recon = assemble_from_principal(&eigs, &q);
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (recon[i][j] - m[i][j]).abs() < 1e-10,
                "roundtrip fail at {i},{j}: {} vs {}",
                recon[i][j],
                m[i][j]
            );
        }
    }
}

/// Determinism: spectral Hencky plastic hash is bit-stable.
#[test]
fn be_plastic_canonical_hash_stable_and_nonzero() {
    let a = be_plastic_canonical_hash();
    let b = be_plastic_canonical_hash();
    assert_eq!(a, b);
    assert_ne!(a, 0);
}

/// M31 hash params still available (unchanged contract).
#[test]
fn m31_hash_params_unchanged_contract() {
    let a = fs_plastic_hash_params();
    let b = be_plastic_hash_params();
    assert_eq!(a, b);
}

/// Zero-plastic Δε_p when elastic.
#[test]
fn elastic_delta_eps_p_is_zero() {
    let p = soft_mat();
    let be = [[1.0001, 0.0, 0.0], [0.0, 1.0001, 0.0], [0.0, 0.0, 1.0001]];
    let r = return_map_spectral_hencky_be(&p, &be, 0.0);
    assert!(!r.plastic);
    assert_eq!(r.delta_eps_p, SymTensor3::zero());
}

/// make_be_plastic_bar constructs a tracked Hencky bar.
#[test]
fn make_be_bar_tracks_distance() {
    let p = soft_mat();
    let w = make_be_plastic_bar(p, 1.0, 0.01, 1.0, 8);
    assert_eq!(w.bars.len(), 1);
    assert_eq!(w.soft.constraints.distances.len(), 1);
}
