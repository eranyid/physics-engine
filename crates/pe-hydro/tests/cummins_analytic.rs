//! Cummins retardation / impulse-response oracles.

use approx::assert_relative_eq;
use pe_hydro::cummins::{
    box_heave_b33_samples, box_heave_retardation, canonical_cummins_hash, damping_from_retardation,
    relative_l2, retardation_from_damping, CumminsHeave,
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
fn cummins_heave_decay_vs_frequency_domain_damping() {
    let length = 40.0;
    let beam = 8.0;
    let draft = 2.0;
    let rho = RHO_FRESHWATER;
    let g = STANDARD_GRAVITY;
    let mass = rho * length * beam * draft;
    let stations = box_barge_stations(length, beam, draft, 16);
    let a_wp = length * beam;
    let mats_inf = pe_hydro::assemble_strip_matrices(&stations, rho, 1e3, g);
    let a_inf = mats_inf.a[0][0];
    let w_n = (rho * g * a_wp / (mass + a_inf)).sqrt();
    // FD reference uses A_∞ (same inertia as Cummins) + B(ω_n).
    let mats = pe_hydro::assemble_strip_matrices(&stations, rho, w_n, g);
    let beta_fd = mats.b[0][0] / (2.0 * (mass + a_inf));

    let dt = 0.02;
    let kernel = box_heave_retardation(length, beam, draft, rho, g, 16, 1e-3, 5.0, 96, dt, 500);
    let mut td = CumminsHeave::new(mass, kernel, 0.15);
    let samples = td.free_decay(8000);
    // Log-decrement on positive peaks (same method as free_decay_heave).
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for i in 1..samples.len() - 1 {
        if samples[i] > samples[i - 1] && samples[i] > samples[i + 1] && samples[i] > 1e-12 {
            peaks.push((i as f64 * dt, samples[i]));
        }
    }
    assert!(peaks.len() >= 3, "need ≥3 peaks, got {}", peaks.len());
    let mut sum = 0.0;
    let mut n = 0usize;
    for i in 0..peaks.len() - 1 {
        let (t0, a0) = peaks[i];
        let (t1, a1) = peaks[i + 1];
        if a0 > 0.0 && a1 > 0.0 && t1 > t0 {
            sum += pe_math::libm_ops::log(a0 / a1) / (t1 - t0);
            n += 1;
        }
    }
    let beta_fit = sum / n as f64;
    let rel = (beta_fit - beta_fd).abs() / beta_fd;
    eprintln!("Cummins vs FD(A∞,B(ωn)): beta_fd={beta_fd:.6} beta_fit={beta_fit:.6} rel={rel:.4}");
    // Measured ~0.15–0.20 with this grid (memory ≠ frozen B). Locked tol 30%.
    assert!(
        rel < 0.30,
        "Cummins β_fit vs FD β: rel={rel} (tol 30%); fd={beta_fd} fit={beta_fit}"
    );
}

#[test]
fn cummins_free_decay_energy_dissipates() {
    let length = 30.0;
    let beam = 6.0;
    let draft = 1.5;
    let rho = RHO_FRESHWATER;
    let g = STANDARD_GRAVITY;
    let mass = rho * length * beam * draft;
    let kernel = box_heave_retardation(length, beam, draft, rho, g, 12, 1e-3, 4.0, 48, 0.05, 160);
    let z0 = 0.2;
    let mut td = CumminsHeave::new(mass, kernel, z0);
    let samples = td.free_decay(800);
    let z_end = samples.last().copied().unwrap().abs();
    let z_mid = samples[samples.len() / 4].abs();
    assert!(z_mid < z0, "should decay from initial displacement");
    assert!(z_end < z_mid * 1.05, "late amplitude should not grow");
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
