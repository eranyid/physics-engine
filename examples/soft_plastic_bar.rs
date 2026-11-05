//! M30 soft XPBD ↔ J2 plastic coupling demo.
//!
//! Quasistatic stretch past yield, unload, and compare permanent set to the
//! analytic uniaxial return map. Also prints the locked plastic hash.
//!
//! ```text
//! cargo run -p pe-examples --bin soft_plastic_bar --release
//! ```

use pe_soft::{
    analytic_permanent_strain, plastic_canonical_hash, plastic_hash_params,
    quasistatic_stretch_unload,
};

fn main() {
    let p = plastic_hash_params();
    let l0 = 1.0;
    let area = 0.01;
    let eps_y = p.yield_strength / p.youngs_modulus;
    println!(
        "soft_plastic_bar: E={:.3e} σ_y={:.3e} H={:.3e} ε_y={:.6e}",
        p.youngs_modulus, p.yield_strength, p.hardening_modulus, eps_y
    );

    // Elastic springback
    let eps_el = 0.5 * eps_y;
    let (len_el, eps_p_el, _) =
        quasistatic_stretch_unload(p, l0, area, eps_el, 40, 400, 1.0 / 240.0);
    println!(
        "elastic unload: len={len_el:.9} (L0={l0}) rel_err={:.4e} ε_p={eps_p_el:.3e}",
        (len_el - l0).abs() / l0
    );

    // Plastic permanent set
    let eps = eps_y + 2e-3;
    let (len, eps_p, analytic) = quasistatic_stretch_unload(p, l0, area, eps, 80, 800, 1.0 / 240.0);
    let expect = l0 * (1.0 + analytic);
    let map_err = (eps_p - analytic).abs() / analytic.abs().max(1e-16);
    let len_err = (len - expect).abs() / expect;
    println!(
        "plastic unload: ε_target={eps:.6e} ε_p={eps_p:.6e} analytic={analytic:.6e} map_rel={map_err:.4e}"
    );
    println!("  settled length={len:.9} expect={expect:.9} len_rel={len_err:.4e}");
    println!(
        "  (analytic_permanent_strain helper = {:.6e})",
        analytic_permanent_strain(&p, eps)
    );

    println!("canonical plastic_hash = {:016x}", plastic_canonical_hash());
}
