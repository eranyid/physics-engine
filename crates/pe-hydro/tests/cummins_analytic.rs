//! Cummins retardation / impulse-response oracles.

use approx::assert_relative_eq;
use pe_hydro::cummins::{
    added_mass_delta_from_retardation, box_heave_b33_samples, box_heave_retardation,
    canonical_cummins_hash, damping_from_retardation, free_decay_cummins_vs_pole,
    free_decay_frozen_b_mismatch, heave_energy_proxy, relative_l2, retardation_from_damping,
    CumminsHeave,
};
use pe_hydro::hull::RHO_FRESHWATER;
use pe_hydro::seakeeping::box_barge_stations;
use pe_math::STANDARD_GRAVITY;

#[test]
fn k_t_consistent_with_b_omega() {
    // Build K from B(ω), reconstruct B̂(ω), compare on the interior band.
    let length = 40.0;
    let beam = 8.0;
    let draft = 2.0;
    let rho = RHO_FRESHWATER;
    let g = STANDARD_GRAVITY;
    let stations = box_barge_stations(length, beam, draft, 16);
    let (omega, b) = box_heave_b33_samples(&stations, rho, g, 0.05, 3.0, 96);
    let dt = 0.025;
    let n_t = 400;
    let t: Vec<f64> = (0..n_t).map(|i| i as f64 * dt).collect();
    let k_vals = retardation_from_damping(&omega, &b, &t);
    let kernel = pe_hydro::RetardationKernel {
        dt,
        k: k_vals,
        a_inf: 0.0,
        c: 0.0,
    };
    let b_hat = damping_from_retardation(&kernel, &omega);
    // Exclude the very lowest/highest ω where truncation is worst.
    let lo = 8usize;
    let hi = omega.len() - 8;
    let rel = relative_l2(&b[lo..hi], &b_hat[lo..hi]);
    eprintln!("K↔B consistency relative L2 (mid-band) = {rel:.4e}");
    // Measured mid-band relative L2 typically ≲ a few percent with this grid.
    assert!(rel < 0.08, "K(t)↔B(ω) mid-band rel L2 = {rel} (tol 8%)");
}

#[test]
fn k_t_consistent_with_a_omega_sine_transform() {
    // Ogilvie sine: A(ω)−A_∞ = −(1/ω)∫ K(t) sin(ωt) dt.
    // Build K numerically from B(ω)=e^{−αω}, then compare the discrete sine
    // transform to a fine trapezoid of the closed-form K(t)=(2/π)α/(α²+t²).
    let alpha = 0.5;
    let n_w = 2000;
    let w_max = 80.0;
    let omega: Vec<f64> = (0..n_w)
        .map(|i| w_max * (i as f64) / (n_w as f64 - 1.0))
        .collect();
    let b: Vec<f64> = omega
        .iter()
        .map(|&w| pe_math::libm_ops::exp(-alpha * w))
        .collect();
    let dt = 0.01;
    let n_t = 4000;
    let t: Vec<f64> = (0..n_t).map(|i| i as f64 * dt).collect();
    let k_vals = retardation_from_damping(&omega, &b, &t);
    let kernel = pe_hydro::RetardationKernel {
        dt,
        k: k_vals,
        a_inf: 0.0,
        c: 0.0,
    };
    // Probe frequencies away from 0 and ω_max.
    let probe = [0.5_f64, 1.0, 2.0, 4.0];
    let a_delta = added_mass_delta_from_retardation(&kernel, &probe);
    // Exact K sine transform via fine trapezoid of closed-form K.
    for (i, &w) in probe.iter().enumerate() {
        let mut acc = 0.0;
        let n_exact = 20000usize;
        let t_max = 200.0;
        let dte = t_max / (n_exact as f64);
        for j in 0..=n_exact {
            let tj = j as f64 * dte;
            let kj = (2.0 / std::f64::consts::PI) * alpha / (alpha * alpha + tj * tj);
            let weight = if j == 0 || j == n_exact { 0.5 } else { 1.0 };
            acc += weight * kj * pe_math::libm_ops::sin(w * tj) * dte;
        }
        let exact = -acc / w;
        let rel = (a_delta[i] - exact).abs() / exact.abs().max(1e-12);
        eprintln!(
            "sine A-delta(ω={w}): num={:.6e} exact={:.6e} rel={rel:.4e}",
            a_delta[i], exact
        );
        // Measured ≲ 5.3e-4 on this grid; CI tol ≈ 2× → 0.12%.
        assert!(
            rel < 0.0012,
            "A(ω)−A_∞ sine transform rel={rel} at ω={w} (tol 0.12%)"
        );
    }
}

#[test]
fn cummins_heave_decay_vs_laplace_pole() {
    // Dual path: time-stepped free decay vs Laplace root of the same K.
    let length = 40.0;
    let beam = 8.0;
    let draft = 2.0;
    let rho = RHO_FRESHWATER;
    let g = STANDARD_GRAVITY;
    let mass = rho * length * beam * draft;
    let out = free_decay_cummins_vs_pole(
        mass, length, beam, draft, rho, g, 0.15, 16, 0.02, 8000, 96, 5.0, 500,
    )
    .expect("pole / fit");
    let (beta_td, omega_td, beta_pole, omega_pole, rel_beta, rel_omega) = out;
    eprintln!(
        "Cummins TD vs Laplace pole: β_td={beta_td:.6} β_pole={beta_pole:.6} rel_β={rel_beta:.4e}; \
         ω_td={omega_td:.6} ω_pole={omega_pole:.6} rel_ω={rel_omega:.4e}"
    );

    // Evidence: frozen single-frequency B(ω_n) reference is ~20% off (not an oracle).
    let mismatch = free_decay_frozen_b_mismatch(
        mass, length, beam, draft, rho, g, 0.15, 16, 0.02, 8000, 96, 5.0, 500,
    )
    .expect("frozen mismatch");
    eprintln!(
        "evidence frozen-B mismatch: β_frozen={:.6} β_td={:.6} rel={:.4}",
        mismatch.0, mismatch.1, mismatch.2
    );
    assert!(
        mismatch.2 > 0.10,
        "expected frozen-B reference to disagree by >10% (got {}); if this fails, the \
         spectrum is accidentally single-frequency and the dual-path test is weak",
        mismatch.2
    );

    // Measured: rel_β ≈ 4.9e-3, rel_ω ≈ 1.9e-3. CI tol ≈ 2× → 1% / 0.4%.
    assert!(
        rel_beta < 0.01,
        "β TD vs Laplace pole rel={rel_beta} (tol 1%); td={beta_td} pole={beta_pole}"
    );
    assert!(
        rel_omega < 0.004,
        "ω_d TD vs Laplace pole rel={rel_omega} (tol 0.4%); td={omega_td} pole={omega_pole}"
    );
}

#[test]
fn cummins_free_decay_energy_monotonic() {
    let length = 30.0;
    let beam = 6.0;
    let draft = 1.5;
    let rho = RHO_FRESHWATER;
    let g = STANDARD_GRAVITY;
    let mass = rho * length * beam * draft;
    let kernel = box_heave_retardation(length, beam, draft, rho, g, 12, 1e-3, 4.0, 48, 0.05, 160);
    let z0 = 0.2;
    let mut td = CumminsHeave::new(mass, kernel.clone(), z0);
    // After a short transient (kernel filling), the instantaneous energy proxy
    // ½(m+A_∞)v² + ½Cz² must not grow across successive positive peaks.
    let n_steps = 800;
    let mut peak_e: Vec<f64> = Vec::new();
    let mut prev_z = td.z;
    let mut prev2_z = td.z;
    for step in 0..n_steps {
        td.step(0.0);
        if step < 40 {
            prev2_z = prev_z;
            prev_z = td.z;
            continue;
        }
        if prev_z > prev2_z && prev_z > td.z && prev_z > 1e-12 {
            // Energy at the peak sample (v≈0 near displacement peaks).
            peak_e.push(heave_energy_proxy(mass, &kernel, prev_z, 0.0));
        }
        prev2_z = prev_z;
        prev_z = td.z;
    }
    assert!(peak_e.len() >= 3, "need ≥3 peaks for envelope check");
    for i in 1..peak_e.len() {
        assert!(
            peak_e[i] <= peak_e[i - 1] * 1.02,
            "peak energy grew: E[{}]={} E[{}]={} (allow 2% numerical slack)",
            i - 1,
            peak_e[i - 1],
            i,
            peak_e[i]
        );
    }
    let z_end = td.z.abs();
    assert!(z_end < z0 * 0.5, "amplitude should decay substantially");
}

#[test]
fn cummins_hash_bit_stable() {
    let a = canonical_cummins_hash();
    let b = canonical_cummins_hash();
    assert_eq!(a, b);
    eprintln!("cummins hash = {a:016x}");
}

#[test]
fn known_transform_pair_cosine() {
    // Analytic: B(ω)=e^{-αω} → K(t)=(2/π) α/(α²+t²) for α>0 (Laplace identity).
    // Discrete check at a few t with fine ω grid.
    let alpha = 0.5;
    let n_w = 2000;
    let w_max = 80.0;
    let omega: Vec<f64> = (0..n_w)
        .map(|i| w_max * (i as f64) / (n_w as f64 - 1.0))
        .collect();
    let b: Vec<f64> = omega.iter().map(|&w| libm_ops_exp(-alpha * w)).collect();
    let ts = [0.0_f64, 0.5, 1.0, 2.0];
    let k = retardation_from_damping(&omega, &b, &ts);
    for (i, &t) in ts.iter().enumerate() {
        let exact = (2.0 / std::f64::consts::PI) * alpha / (alpha * alpha + t * t);
        let rel = (k[i] - exact).abs() / exact;
        eprintln!(
            "analytic K({t}) num={:.6} exact={:.6} rel={rel:.4e}",
            k[i], exact
        );
        assert!(rel < 0.02, "analytic cosine transform rel={rel} at t={t}");
    }
}

fn libm_ops_exp(x: f64) -> f64 {
    pe_math::libm_ops::exp(x)
}

#[test]
fn retardation_kernel_positive_near_zero() {
    let kernel = box_heave_retardation(
        40.0,
        8.0,
        2.0,
        RHO_FRESHWATER,
        STANDARD_GRAVITY,
        16,
        1e-3,
        4.0,
        64,
        0.05,
        100,
    );
    assert!(kernel.k[0] > 0.0, "K(0) should be positive for B>0");
    assert_relative_eq!(kernel.dt, 0.05, max_relative = 0.0);
}
