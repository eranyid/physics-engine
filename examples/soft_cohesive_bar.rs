//! M33 bilinear cohesive-zone soft XPBD demo.
//!
//! Prints G_c, virgin peak, a softening damage check, XPBD failure / island
//! split, the locked `cohesive_hash`, and confirms prior plastic hashes are
//! unchanged.
//!
//! ```text
//! cargo run -p pe-examples --bin soft_cohesive_bar --release
//! ```

use pe_soft::{
    analytic_gc, be_plastic_canonical_hash, cohesive_canonical_hash, cohesive_hash_params,
    damage_from_delta_max, discrete_work_to_failure, eval_bilinear_czm, fs_plastic_canonical_hash,
    plastic_canonical_hash, quasistatic_cohesive_open, traction_virgin_envelope,
};

fn main() {
    let p = cohesive_hash_params();
    println!(
        "soft_cohesive_bar: σ_c={:.3e} δ_c={:.3e} δ_f={:.3e} G_c={:.6e}",
        p.sigma_c,
        p.delta_c,
        p.delta_f,
        analytic_gc(&p)
    );

    let t_peak = traction_virgin_envelope(&p, p.delta_c);
    println!("virgin peak t(δ_c)={t_peak:.6e} (σ_c={:.6e})", p.sigma_c);

    let w_int = discrete_work_to_failure(&p, 4096);
    let gc = analytic_gc(&p);
    println!(
        "∫ virgin envelope ≈ {w_int:.6e}; G_c={gc:.6e}; rel_err={:.3e}",
        (w_int - gc).abs() / gc
    );

    let d_star = 0.5 * (p.delta_c + p.delta_f);
    let ev = eval_bilinear_czm(&p, d_star, 0.0);
    let d_a = damage_from_delta_max(&p, d_star);
    println!(
        "softening δ*={d_star:.6e}: damage={:.6e} (analytic {d_a:.6e}) failed={}",
        ev.damage, ev.failed
    );

    let (opening, damage, traction, failed) =
        quasistatic_cohesive_open(p, 1.0, 0.01, 1.05 * p.delta_f, 80, 1.0 / 240.0);
    println!(
        "XPBD hold past δ_f: opening={opening:.6e} damage={damage:.6e} t={traction:.3e} failed={failed}"
    );

    println!(
        "canonical cohesive_hash = {:016x}",
        cohesive_canonical_hash()
    );
    println!(
        "unchanged be_plastic_hash = {:016x}",
        be_plastic_canonical_hash()
    );
    println!(
        "unchanged fs_plastic_hash = {:016x}",
        fs_plastic_canonical_hash()
    );
    println!(
        "unchanged plastic_hash    = {:016x}",
        plastic_canonical_hash()
    );
}
