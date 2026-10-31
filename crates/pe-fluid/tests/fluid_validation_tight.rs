//! Tightened validation: hydrostatic profile, gradient numbers, dam-break study.

use pe_fluid::eos::{TaitEos, TAIT_GAMMA_WATER};
use pe_fluid::kernels::{gradient_consistency_error, gradient_sum_norm, KernelKind};
use pe_fluid::oracles::ritter_front;
use pe_fluid::validation::{dam_break_convergence, tait_hydrostatic_profile_error};
use pe_fluid::Dim;
use pe_math::STANDARD_GRAVITY;

#[test]
fn tait_hydrostatic_column_profile_max_rel_under_3pct() {
    let eos = TaitEos::new(1000.0, 60.0, TAIT_GAMMA_WATER);
    let (max_rel, mean_rel, n) =
        tait_hydrostatic_profile_error(&eos, STANDARD_GRAVITY, 1.0, 0.0, 40);
    assert!(n >= 30, "n={n}");
    // Measured: max≈0.0081, mean≈0.0041 at c0=60, H=1
    assert!(
        max_rel < 0.03,
        "Tait hydrostatic profile max rel {max_rel} (mean {mean_rel})"
    );
    assert!(max_rel < 0.01, "expected ~0.81%; got max_rel={max_rel}");
}

#[test]
fn gradient_consistency_actual_numbers() {
    let h = 0.12;
    let e4 = gradient_consistency_error(KernelKind::CubicSpline, h, Dim::D2, h / 4.0);
    let e8 = gradient_consistency_error(KernelKind::CubicSpline, h, Dim::D2, h / 8.0);
    let g0 = gradient_sum_norm(KernelKind::CubicSpline, h, Dim::D2, h / 8.0);
    assert!(g0 < 1e-12, "|Σ∇W|={g0}");
    assert!(e4 < 5e-4, "e4={e4}");
    assert!(e8 < e4, "refinement: e8={e8} e4={e4}");
    assert!(e8 < 1e-5, "e8={e8}");
}

#[test]
fn dam_break_ritter_convergence_study() {
    // Geometry: h0=0.4, L_res=0.5, channel=6, t=0.22 (Ritter Δx≈0.87 m ≪ wall).
    // Default PBF path (reuse_neighbours=true). Toe front metric.
    //
    // Representative measured series (Linux x86_64):
    //   dx=0.050 → ~0.90
    //   dx=0.040 → ~0.52
    //   dx=0.030 → ~0.24
    //   dx=0.022 → ~0.17
    // Error falls with dx; finest CI-affordable still ≳15%. See docs/FLUID.md.
    let h0 = 0.4;
    let l_res = 0.5;
    let channel = 6.0;
    let g = 9.81;
    let t_end = 0.22;
    assert!(ritter_front(g, h0, t_end) < 0.5 * channel);

    let dxs = [0.05f64, 0.04, 0.03];
    let rows = dam_break_convergence(&dxs, h0, l_res, t_end, channel);
    println!("dam-break (dx, x_front, rel): {:?}", rows);
    for (dx, xf, rel) in &rows {
        assert!(xf.is_finite() && *rel < 2.5, "dx={dx} xf={xf} rel={rel}");
    }
    // Monotone improvement with refinement (5% absolute slack for tip noise).
    assert!(
        rows[1].2 <= rows[0].2 + 0.05,
        "no improvement 0.05→0.04: {:?}",
        rows
    );
    assert!(
        rows[2].2 <= rows[1].2 + 0.05,
        "no improvement 0.04→0.03: {:?}",
        rows
    );
    // Documented: still above 15% at dx=0.03 in this suite.
    assert!(
        rows[2].2 < 0.55,
        "unexpectedly large fine error: {:?}",
        rows
    );
}

#[test]
#[ignore = "slow: fine dam-break; run with --ignored"]
fn dam_break_ritter_fine_dx() {
    let rows = dam_break_convergence(&[0.022f64], 0.4, 0.5, 0.22, 6.0);
    println!("fine {:?}", rows);
    assert!(rows[0].2 < 0.35);
}
