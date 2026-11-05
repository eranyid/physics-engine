//! M32 spectral Hencky / Kirchhoff (`b_e`) soft XPBD plastic demo.
//!
//! Prints the closed-form isochoric uniaxial Kirchhoff trial, a plastic
//! return-to-surface check, the locked `be_plastic_hash`, and confirms
//! M31 `fs_plastic_hash` / M30 `plastic_hash` are unchanged.
//!
//! ```text
//! cargo run -p pe-examples --bin soft_be_plastic_tet --release
//! ```

use pe_math::libm_ops;
use pe_soft::{
    analytic_isochoric_uniaxial_kirchhoff, analytic_isochoric_uniaxial_seq,
    be_plastic_canonical_hash, be_plastic_hash_params, fs_plastic_canonical_hash,
    left_cauchy_green, plastic_canonical_hash, return_map_spectral_hencky_be,
};

fn main() {
    let p = be_plastic_hash_params();
    let mu = p.shear_modulus();
    println!(
        "soft_be_plastic_tet: E={:.3e} ν={} σ_y={:.3e} H={:.3e} μ={:.3e}",
        p.youngs_modulus, p.poisson_ratio, p.yield_strength, p.hardening_modulus, mu
    );

    // Elastic isochoric uniaxial
    let lim = p.yield_strength / (3.0 * mu);
    let lam_el = libm_ops::exp(0.4 * lim);
    let be_el = [
        [lam_el * lam_el, 0.0, 0.0],
        [0.0, 1.0 / lam_el, 0.0],
        [0.0, 0.0, 1.0 / lam_el],
    ];
    let r_el = return_map_spectral_hencky_be(&p, &be_el, 0.0);
    let (t1, t2, _) = analytic_isochoric_uniaxial_kirchhoff(mu, lam_el);
    let seq_a = analytic_isochoric_uniaxial_seq(mu, lam_el);
    println!(
        "elastic λ={lam_el:.6}: plastic={} τ1={:.6e} (analytic {:.6e}) σ_eq={:.6e} (analytic {:.6e})",
        r_el.plastic,
        r_el.kirchhoff.xx,
        t1,
        r_el.kirchhoff.von_mises_eq(),
        seq_a
    );
    let _ = t2;

    // Plastic isochoric uniaxial
    let lam = 1.1;
    let be = [
        [lam * lam, 0.0, 0.0],
        [0.0, 1.0 / lam, 0.0],
        [0.0, 0.0, 1.0 / lam],
    ];
    let r = return_map_spectral_hencky_be(&p, &be, 0.0);
    let sy = p.yield_strength + p.hardening_modulus * r.eq_plastic_strain;
    let seq = r.kirchhoff.von_mises_eq();
    println!(
        "plastic λ={lam:.3}: Δγ={:.6e} α={:.6e} σ_eq={:.6e} sy_eff={:.6e} rel_err={:.4e}",
        r.delta_gamma,
        r.eq_plastic_strain,
        seq,
        sy,
        (seq - sy).abs() / sy
    );

    // Large shear: spectral vs identity left-CG sanity
    let gamma = 0.4;
    let f = [[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let be_sh = left_cauchy_green(&f);
    let r_sh = return_map_spectral_hencky_be(&p, &be_sh, 0.0);
    println!(
        "simple shear γ={gamma}: plastic={} ‖τ‖_F={:.6e} α={:.6e}",
        r_sh.plastic,
        r_sh.kirchhoff.frobenius(),
        r_sh.eq_plastic_strain
    );

    println!(
        "canonical be_plastic_hash = {:016x}",
        be_plastic_canonical_hash()
    );
    println!(
        "canonical fs_plastic_hash (M31, unchanged) = {:016x}",
        fs_plastic_canonical_hash()
    );
    println!(
        "canonical plastic_hash (M30, unchanged) = {:016x}",
        plastic_canonical_hash()
    );
}
