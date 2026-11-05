//! M31 finite-strain (Hencky) soft XPBD ↔ J2 plastic coupling demo.
//!
//! Quasistatic log-strain stretch past yield, unload, and compare permanent
//! set to `L₀ exp(ε_p)`. Also prints the locked `fs_plastic_hash` and confirms
//! the M30 engineering `plastic_hash` is unchanged.
//!
//! ```text
//! cargo run -p pe-examples --bin soft_fs_plastic_bar --release
//! ```

use pe_math::libm_ops;
use pe_soft::{
    analytic_log_permanent_strain, analytic_permanent_strain, fs_plastic_canonical_hash,
    fs_plastic_hash_params, plastic_canonical_hash, quasistatic_log_stretch_unload,
};

fn main() {
    let p = fs_plastic_hash_params();
    let l0 = 1.0;
    let area = 0.01;
    let eps_y = p.yield_strength / p.youngs_modulus;
    println!(
        "soft_fs_plastic_bar: E={:.3e} σ_y={:.3e} H={:.3e} ε_y={:.6e}",
        p.youngs_modulus, p.yield_strength, p.hardening_modulus, eps_y
    );

    // Elastic springback (log)
    let lambda_el = libm_ops::exp(0.5 * eps_y);
    let (len_el, eps_p_el, _) =
        quasistatic_log_stretch_unload(p, l0, area, lambda_el, 40, 400, 1.0 / 240.0);
    println!(
        "elastic unload: len={len_el:.9} (L0={l0}) rel_err={:.4e} ε_p={eps_p_el:.3e}",
        (len_el - l0).abs() / l0
    );

    // Plastic permanent set at finite stretch
    let lambda = 1.06;
    let (len, eps_p, analytic) =
        quasistatic_log_stretch_unload(p, l0, area, lambda, 80, 800, 1.0 / 240.0);
    let expect = l0 * libm_ops::exp(analytic);
    let map_err = (eps_p - analytic).abs() / analytic.abs().max(1e-16);
    let len_err = (len - expect).abs() / expect;
    println!(
        "plastic unload: λ={lambda:.6} ε_p={eps_p:.6e} analytic={analytic:.6e} map_rel={map_err:.4e}"
    );
    println!("  settled length={len:.9} expect={expect:.9} len_rel={len_err:.4e}");
    println!(
        "  analytic_log_permanent_strain(ln λ) = {:.6e}",
        analytic_log_permanent_strain(&p, libm_ops::log(lambda))
    );

    // Show finite-strain divergence vs engineering at λ=1.2
    let lam_big = 1.2;
    let ep_eng = analytic_permanent_strain(&p, lam_big - 1.0);
    let ep_log = analytic_log_permanent_strain(&p, libm_ops::log(lam_big));
    println!(
        "λ=1.2: ε_p^eng={ep_eng:.6e} ε_p^log={ep_log:.6e} rel_diff={:.4e}",
        (ep_log - ep_eng).abs() / ep_eng.abs().max(1e-16)
    );

    println!(
        "canonical fs_plastic_hash = {:016x}",
        fs_plastic_canonical_hash()
    );
    println!(
        "canonical plastic_hash (M30, unchanged) = {:016x}",
        plastic_canonical_hash()
    );
}
