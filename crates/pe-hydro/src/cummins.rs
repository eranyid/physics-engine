//! Cummins equation / impulse-response (retardation) memory for ship heave.
//!
//! Frequency-domain radiation damping `B(ω)` transforms to a retardation kernel
//! via the cosine transform (Cummins 1962; Ogilvie 1964):
//!
//! ```text
//! K(t) = (2/π) ∫_0^∞ [B(ω) − B_∞] cos(ω t) dω
//! ```
//!
//! with `B_∞ = 0` for floating bodies in potential theory. The infinite-frequency
//! added mass `A_∞` enters the instantaneous inertia. Time-domain heave:
//!
//! ```text
//! (m + A_∞) z̈(t) + ∫_0^t K(t−τ) ż(τ) dτ + C z(t) = F(t)
//! ```
//!
//! This module builds `K(t)` from strip-theory `B(ω)` samples, checks the inverse
//! transforms `B(ω) ↔ K(t)` (cosine) and `A(ω)−A_∞ ↔ K(t)` (sine / Ogilvie),
//! and advances a convolution memory ODE. Free-decay oracles compare the
//! time-stepped envelope to the Laplace characteristic pole of the same `K`
//! (not a single-frequency frozen `B(ω_n)` model — that mismatch is ~20%).
//!
//! Determinism: portable [`pe_math::libm_ops`] only; no FMA; fixed sample grids.
//!
//! # References
//! - Cummins, W. E. (1962). The impulse response function and ship motions.
//! - Ogilvie, T. F. (1964). Recent progress toward the understanding and
//!   prediction of ship motions.
//! - Sectional `B(ω)` from [`crate::seakeeping`] (nominal circle form).

use pe_math::libm_ops;

use crate::seakeeping::{
    assemble_strip_matrices, box_barge_stations, sectional_a33_inf, StripStation,
};

/// Retardation kernel samples `K(t_i)` on a uniform grid.
#[derive(Clone, Debug)]
pub struct RetardationKernel {
    /// Sample spacing `Δt` [s].
    pub dt: f64,
    /// `K(i·Δt)` [kg/s²] for heave (force / velocity·time).
    pub k: Vec<f64>,
    /// Infinite-frequency added mass `A_∞` [kg] used with this kernel.
    pub a_inf: f64,
    /// Restoring coefficient `C` [N/m] (heave).
    pub c: f64,
}

impl RetardationKernel {
    /// Number of samples.
    pub fn len(&self) -> usize {
        self.k.len()
    }

    /// True if the kernel has no samples.
    pub fn is_empty(&self) -> bool {
        self.k.is_empty()
    }

    /// Kernel value at sample index (0 if out of range).
    pub fn at(&self, i: usize) -> f64 {
        self.k.get(i).copied().unwrap_or(0.0)
    }

    /// Memory horizon `T = (n−1) Δt` [s].
    pub fn horizon(&self) -> f64 {
        if self.k.len() <= 1 {
            0.0
        } else {
            (self.k.len() - 1) as f64 * self.dt
        }
    }
}

/// Cosine-transform `K(t)` from frequency samples of radiation damping.
///
/// `B_omega[j]` is `B(ω_j)` with `omega[j]` strictly increasing, `omega[0]≥0`.
/// Uses the trapezoidal rule on a uniform-in-ω grid (caller should supply one).
///
/// `K(t) = (2/π) ∫_0^{ω_max} B(ω) cos(ω t) dω`.
pub fn retardation_from_damping(omega: &[f64], b_omega: &[f64], t_samples: &[f64]) -> Vec<f64> {
    assert_eq!(omega.len(), b_omega.len());
    assert!(omega.len() >= 2);
    let mut k = Vec::with_capacity(t_samples.len());
    let two_over_pi = 2.0 / core::f64::consts::PI;
    for &t in t_samples {
        let mut acc = 0.0;
        for j in 0..omega.len() - 1 {
            let w0 = omega[j];
            let w1 = omega[j + 1];
            let dw = w1 - w0;
            if dw <= 0.0 {
                continue;
            }
            let f0 = b_omega[j] * libm_ops::cos(w0 * t);
            let f1 = b_omega[j + 1] * libm_ops::cos(w1 * t);
            acc += 0.5 * (f0 + f1) * dw;
        }
        k.push(two_over_pi * acc);
    }
    k
}

/// Reconstruct `B(ω)` from `K(t)` via the inverse cosine transform:
/// `B(ω) = ∫_0^{T} K(t) cos(ω t) dt` (trapezoidal on the `K` grid).
pub fn damping_from_retardation(k: &RetardationKernel, omega: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(omega.len());
    let dt = k.dt;
    let n = k.k.len();
    for &w in omega {
        let mut acc = 0.0;
        for i in 0..n {
            let t = i as f64 * dt;
            let weight = if i == 0 || i + 1 == n { 0.5 } else { 1.0 };
            acc += weight * k.k[i] * libm_ops::cos(w * t) * dt;
        }
        out.push(acc);
    }
    out
}

/// Ogilvie sine transform: `A(ω) − A_∞ = −(1/ω) ∫_0^{T} K(t) sin(ω t) dt`.
///
/// Returns the added-mass increment `A(ω) − A_∞` at each sample frequency.
/// At `ω = 0` the limit is `−∫_0^{T} t K(t) dt`.
pub fn added_mass_delta_from_retardation(k: &RetardationKernel, omega: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(omega.len());
    let dt = k.dt;
    let n = k.k.len();
    for &w in omega {
        let mut acc = 0.0;
        for i in 0..n {
            let t = i as f64 * dt;
            let weight = if i == 0 || i + 1 == n { 0.5 } else { 1.0 };
            if w.abs() < 1e-14 {
                acc += weight * k.k[i] * t * dt;
            } else {
                acc += weight * k.k[i] * libm_ops::sin(w * t) * dt;
            }
        }
        if w.abs() < 1e-14 {
            out.push(-acc);
        } else {
            out.push(-acc / w);
        }
    }
    out
}

/// Laplace transform `K̂(s) = ∫_0^{T} K(t) e^{−s t} dt` with `s = σ + iω`
/// (returns `(Re, Im)`). Trapezoidal on the kernel grid.
pub fn laplace_retardation(k: &RetardationKernel, sigma: f64, omega: f64) -> (f64, f64) {
    let dt = k.dt;
    let n = k.k.len();
    let mut re = 0.0;
    let mut im = 0.0;
    for i in 0..n {
        let t = i as f64 * dt;
        let weight = if i == 0 || i + 1 == n { 0.5 } else { 1.0 };
        let decay = libm_ops::exp(-sigma * t);
        let c = libm_ops::cos(omega * t);
        let s = libm_ops::sin(omega * t);
        // e^{-(σ+iω)t} = e^{-σt}(cos(ωt) − i sin(ωt))
        re += weight * k.k[i] * decay * c * dt;
        im -= weight * k.k[i] * decay * s * dt;
    }
    (re, im)
}

/// Characteristic residual of the free Cummins ODE at pole `λ = −β + i ω_d`:
///
/// `(m + A_∞) λ² + λ K̂(λ) + C = 0`, `K̂(λ) = ∫_0^{T} K(t) e^{−λ t} dt`.
///
/// Returns `(Re f, Im f)`.
pub fn cummins_char_residual(
    mass: f64,
    kernel: &RetardationKernel,
    beta: f64,
    omega_d: f64,
) -> (f64, f64) {
    let m = mass + kernel.a_inf;
    // λ = −β + i ω_d ⇒ K̂(λ) = ∫ K e^{β t} e^{−i ω_d t} dt
    // = laplace_retardation(σ = −β, ω = ω_d). Finite horizon; K decays.
    let (k_re, k_im) = laplace_retardation(kernel, -beta, omega_d);
    // λ² = (−β + iω)² = β² − ω² − 2 i β ω
    let lam2_re = beta * beta - omega_d * omega_d;
    let lam2_im = -2.0 * beta * omega_d;
    // λ K̂ = (−β + iω)(k_re + i k_im)
    let lam_k_re = -beta * k_re - omega_d * k_im;
    let lam_k_im = -beta * k_im + omega_d * k_re;
    (m * lam2_re + lam_k_re + kernel.c, m * lam2_im + lam_k_im)
}

/// Find the underdamped free-decay pole `λ = −β + i ω_d` of the Cummins
/// characteristic equation by Newton iteration in `(β, ω_d)`.
///
/// `beta0`, `omega0` are initial guesses (e.g. from undamped `√(C/(m+A_∞))`
/// and a frozen-`B` damping estimate). Returns `(β, ω_d)` or `None` if the
/// iteration fails to converge.
pub fn free_decay_pole(
    mass: f64,
    kernel: &RetardationKernel,
    beta0: f64,
    omega0: f64,
) -> Option<(f64, f64)> {
    let mut beta = beta0.max(0.0);
    let mut omega = omega0.max(1e-6);
    let h = 1e-6;
    for _ in 0..40 {
        let (fr, fi) = cummins_char_residual(mass, kernel, beta, omega);
        let res = libm_ops::sqrt(fr * fr + fi * fi);
        if res < 1e-10 * (1.0 + kernel.c.abs()) {
            return Some((beta, omega));
        }
        // Jacobian via central differences
        let (fr_bp, fi_bp) = cummins_char_residual(mass, kernel, beta + h, omega);
        let (fr_bm, fi_bm) = cummins_char_residual(mass, kernel, (beta - h).max(0.0), omega);
        let (fr_wp, fi_wp) = cummins_char_residual(mass, kernel, beta, omega + h);
        let (fr_wm, fi_wm) = cummins_char_residual(mass, kernel, beta, (omega - h).max(1e-9));
        let j11 = (fr_bp - fr_bm) / (2.0 * h);
        let j21 = (fi_bp - fi_bm) / (2.0 * h);
        let j12 = (fr_wp - fr_wm) / (2.0 * h);
        let j22 = (fi_wp - fi_wm) / (2.0 * h);
        let det = j11 * j22 - j12 * j21;
        if libm_ops::fabs(det) < 1e-30 {
            return None;
        }
        // Solve J δ = −f
        let db = (-fr * j22 + fi * j12) / det;
        let dw = (-fi * j11 + fr * j21) / det;
        beta = (beta + db).max(0.0);
        omega = (omega + dw).max(1e-9);
        if libm_ops::sqrt(db * db + dw * dw) < 1e-12 {
            return Some((beta, omega));
        }
    }
    let (fr, fi) = cummins_char_residual(mass, kernel, beta, omega);
    let res = libm_ops::sqrt(fr * fr + fi * fi);
    if res < 1e-6 * (1.0 + kernel.c.abs()) {
        Some((beta, omega))
    } else {
        None
    }
}

/// Fit `(β, ω_d)` from a free-decay heave trace via parabolic peak interpolation
/// and log-decrement / mean period.
pub fn fit_free_decay_pole(samples: &[f64], dt: f64) -> Option<(f64, f64)> {
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for i in 1..samples.len() - 1 {
        if samples[i] > samples[i - 1] && samples[i] > samples[i + 1] && samples[i] > 1e-12 {
            let y0 = samples[i - 1];
            let y1 = samples[i];
            let y2 = samples[i + 1];
            let denom = y0 - 2.0 * y1 + y2;
            let (t_pk, a_pk) = if libm_ops::fabs(denom) > 1e-30 {
                let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
                let t = (i as f64 + delta) * dt;
                let a = y1 - 0.25 * (y0 - y2) * delta;
                (t, a)
            } else {
                (i as f64 * dt, y1)
            };
            peaks.push((t_pk, a_pk));
        }
    }
    if peaks.len() < 3 {
        return None;
    }
    let mut sum_beta = 0.0;
    let mut sum_omega = 0.0;
    let mut n = 0usize;
    for i in 0..peaks.len() - 1 {
        let (t0, a0) = peaks[i];
        let (t1, a1) = peaks[i + 1];
        if a0 > 0.0 && a1 > 0.0 && t1 > t0 {
            sum_beta += libm_ops::log(a0 / a1) / (t1 - t0);
            sum_omega += 2.0 * core::f64::consts::PI / (t1 - t0);
            n += 1;
        }
    }
    if n == 0 {
        None
    } else {
        Some((sum_beta / n as f64, sum_omega / n as f64))
    }
}

/// Instantaneous mechanical energy proxy `½(m+A_∞)v² + ½ C z²` (excludes the
/// hereditary memory reservoir; still monotone-decreasing for free decay of
/// lightly damped heave once the kernel has settled).
pub fn heave_energy_proxy(mass: f64, kernel: &RetardationKernel, z: f64, v: f64) -> f64 {
    let m = mass + kernel.a_inf;
    0.5 * m * v * v + 0.5 * kernel.c * z * z
}

/// Build a heave retardation kernel for a box barge from strip `B₃₃(ω)`.
///
/// - Frequency grid: `n_omega` points from `ω_min` to `ω_max` (rad/s).
/// - Time grid: `n_time` points with spacing `dt`.
/// - `A_∞` from strip `a₃₃∞` integrated over stations.
/// - `C = ρ g L B` (waterplane restoring).
#[allow(clippy::too_many_arguments)]
pub fn box_heave_retardation(
    length: f64,
    beam: f64,
    draft: f64,
    rho: f64,
    g: f64,
    n_strips: usize,
    omega_min: f64,
    omega_max: f64,
    n_omega: usize,
    dt: f64,
    n_time: usize,
) -> RetardationKernel {
    assert!(n_omega >= 8 && n_time >= 8 && dt > 0.0 && omega_max > omega_min);
    let stations = box_barge_stations(length, beam, draft, n_strips);
    let mut omega = Vec::with_capacity(n_omega);
    let mut b_w = Vec::with_capacity(n_omega);
    for i in 0..n_omega {
        let w = omega_min + (omega_max - omega_min) * (i as f64) / (n_omega as f64 - 1.0);
        let mats = assemble_strip_matrices(&stations, rho, w, g);
        omega.push(w);
        b_w.push(mats.b[0][0]);
    }
    let mut t_samples = Vec::with_capacity(n_time);
    for i in 0..n_time {
        t_samples.push(i as f64 * dt);
    }
    let k_vals = retardation_from_damping(&omega, &b_w, &t_samples);
    // A_∞ from high-frequency sectional a33∞
    let mut a_inf = 0.0;
    for s in &stations {
        a_inf += sectional_a33_inf(rho, s.beam, s.area_coeff) * s.dx;
    }
    let c = rho * g * length * beam;
    RetardationKernel {
        dt,
        k: k_vals,
        a_inf,
        c,
    }
}

/// Sample strip heave damping `B₃₃(ω)` on a uniform ω-grid (for oracles).
pub fn box_heave_b33_samples(
    stations: &[StripStation],
    rho: f64,
    g: f64,
    omega_min: f64,
    omega_max: f64,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut omega = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        let w = omega_min + (omega_max - omega_min) * (i as f64) / (n as f64 - 1.0);
        let mats = assemble_strip_matrices(stations, rho, w, g);
        omega.push(w);
        b.push(mats.b[0][0]);
    }
    (omega, b)
}

/// Relative L2 error between `B` and reconstructed `B̂` over the sample set.
pub fn relative_l2(reference: &[f64], approx: &[f64]) -> f64 {
    assert_eq!(reference.len(), approx.len());
    let mut num = 0.0;
    let mut den = 0.0;
    for (&r, &a) in reference.iter().zip(approx.iter()) {
        let d = r - a;
        num += d * d;
        den += r * r;
    }
    if den == 0.0 {
        return libm_ops::sqrt(num);
    }
    libm_ops::sqrt(num / den)
}

/// Heave Cummins ODE with trapezoidal convolution memory.
#[derive(Clone, Debug)]
pub struct CumminsHeave {
    /// Structural mass [kg].
    pub mass: f64,
    /// Retardation kernel (+ `A_∞`, `C`).
    pub kernel: RetardationKernel,
    /// Heave [m].
    pub z: f64,
    /// Heave rate [m/s].
    pub v: f64,
    /// Time [s].
    pub t: f64,
    /// Velocity history for convolution (newest at the end), length ≤ kernel.len().
    hist_v: Vec<f64>,
}

impl CumminsHeave {
    /// Construct at rest with initial displacement `z0`.
    pub fn new(mass: f64, kernel: RetardationKernel, z0: f64) -> Self {
        Self {
            mass,
            kernel,
            z: z0,
            v: 0.0,
            t: 0.0,
            hist_v: Vec::new(),
        }
    }

    /// Convolution `∫ K(t−τ) v(τ) dτ` using the stored velocity history + current `v`.
    fn memory_force(&self, v_now: f64) -> f64 {
        let dt = self.kernel.dt;
        let n_k = self.kernel.k.len();
        // History holds past velocities at the kernel grid; index 0 = oldest.
        // At the start of a step, hist has velocities up to the previous step.
        let mut acc = 0.0;
        // Include current velocity as the τ=t sample (weight ½ on ends).
        let n_hist = self.hist_v.len();
        // Samples of v at lag i·dt: i=0 → now, i=1 → one step ago, …
        for i in 0..n_k {
            let v_lag = if i == 0 {
                v_now
            } else if i <= n_hist {
                self.hist_v[n_hist - i]
            } else {
                0.0
            };
            let weight = if i == 0 || i + 1 == n_k { 0.5 } else { 1.0 };
            acc += weight * self.kernel.k[i] * v_lag * dt;
        }
        acc
    }

    /// One semi-implicit step of size `kernel.dt` with optional excitation `f` [N].
    pub fn step(&mut self, f: f64) {
        let dt = self.kernel.dt;
        let m = self.mass + self.kernel.a_inf;
        let mem = self.memory_force(self.v);
        // m a = F − mem − C z
        let a = (f - mem - self.kernel.c * self.z) / m;
        self.v += dt * a;
        self.z += dt * self.v;
        self.t += dt;
        self.hist_v.push(self.v);
        let max_hist = self.kernel.k.len();
        if self.hist_v.len() > max_hist {
            let drop = self.hist_v.len() - max_hist;
            self.hist_v.drain(0..drop);
        }
    }

    /// Free decay for `n_steps` from current state; returns `z` samples (incl. t=0).
    pub fn free_decay(&mut self, n_steps: usize) -> Vec<f64> {
        let mut samples = Vec::with_capacity(n_steps + 1);
        samples.push(self.z);
        for _ in 0..n_steps {
            self.step(0.0);
            samples.push(self.z);
        }
        samples
    }
}

/// Dual-path free-decay oracle: time-stepped Cummins vs Laplace characteristic
/// pole of the **same** kernel.
///
/// Returns `(β_td, ω_td, β_pole, ω_pole, rel_β, rel_ω)`.
///
/// # Why not `B(ω_n)/(2(m+A_∞))`?
///
/// A single-frequency frozen reference uses `A_∞` (or `A(ω_n)`) and `B` at one
/// frequency, while the convolution carries the full retardation spectrum and
/// the Kramers–Kronig `A(ω)−A_∞` content of `K(t)`. That mismatch is typically
/// ~20% on β for the box-barge grids used here — a loose bound, not an oracle.
/// The Laplace pole of the identical `K` is the matching independent solution.
#[allow(clippy::too_many_arguments)]
pub fn free_decay_cummins_vs_pole(
    mass: f64,
    length: f64,
    beam: f64,
    draft: f64,
    rho: f64,
    g: f64,
    z0: f64,
    n_strips: usize,
    dt: f64,
    n_steps: usize,
    n_omega: usize,
    omega_max: f64,
    n_kernel: usize,
) -> Option<(f64, f64, f64, f64, f64, f64)> {
    let kernel = box_heave_retardation(
        length, beam, draft, rho, g, n_strips, 1e-3, omega_max, n_omega, dt, n_kernel,
    );
    let m = mass + kernel.a_inf;
    let omega_undamped = libm_ops::sqrt(kernel.c / m);
    // Frozen-B guess for Newton only (not the oracle).
    let stations = box_barge_stations(length, beam, draft, n_strips);
    let mats = assemble_strip_matrices(&stations, rho, omega_undamped, g);
    let beta_guess = mats.b[0][0] / (2.0 * m);

    let mut td = CumminsHeave::new(mass, kernel.clone(), z0);
    let samples = td.free_decay(n_steps);
    let (beta_td, omega_td) = fit_free_decay_pole(&samples, dt)?;
    let (beta_pole, omega_pole) =
        free_decay_pole(mass, &kernel, beta_guess.max(1e-6), omega_undamped)?;
    let rel_beta = libm_ops::fabs(beta_td - beta_pole) / beta_pole.max(1e-30);
    let rel_omega = libm_ops::fabs(omega_td - omega_pole) / omega_pole.max(1e-30);
    Some((
        beta_td, omega_td, beta_pole, omega_pole, rel_beta, rel_omega,
    ))
}

/// Diagnostic: relative β error of the **incorrect** single-frequency reference
/// `B(ω_n)/(2(m+A_∞))` vs the time-domain fit (evidence of the old mismatch).
#[allow(clippy::too_many_arguments)]
pub fn free_decay_frozen_b_mismatch(
    mass: f64,
    length: f64,
    beam: f64,
    draft: f64,
    rho: f64,
    g: f64,
    z0: f64,
    n_strips: usize,
    dt: f64,
    n_steps: usize,
    n_omega: usize,
    omega_max: f64,
    n_kernel: usize,
) -> Option<(f64, f64, f64)> {
    let kernel = box_heave_retardation(
        length, beam, draft, rho, g, n_strips, 1e-3, omega_max, n_omega, dt, n_kernel,
    );
    let m = mass + kernel.a_inf;
    let omega_n = libm_ops::sqrt(kernel.c / m);
    let stations = box_barge_stations(length, beam, draft, n_strips);
    let mats = assemble_strip_matrices(&stations, rho, omega_n, g);
    let beta_frozen = mats.b[0][0] / (2.0 * m);
    let mut td = CumminsHeave::new(mass, kernel, z0);
    let samples = td.free_decay(n_steps);
    let (beta_td, _) = fit_free_decay_pole(&samples, dt)?;
    let rel = libm_ops::fabs(beta_td - beta_frozen) / beta_frozen.max(1e-30);
    Some((beta_frozen, beta_td, rel))
}

/// Canonical Cummins scenario hash (FNV-1a over packed heave samples).
///
/// Locked scenario (see `docs/DETERMINISM.md`):
/// - Box 40×8×2 m, freshwater, mass = ρ L B T
/// - Kernel: ω∈[1e-3, 4] rad/s, 64 freq samples; dt=0.05 s, 200 kernel samples
/// - Free decay z0=0.1 m, 400 steps
pub fn canonical_cummins_hash() -> u64 {
    use crate::hull::RHO_FRESHWATER;
    use pe_math::STANDARD_GRAVITY;

    let length = 40.0;
    let beam = 8.0;
    let draft = 2.0;
    let rho = RHO_FRESHWATER;
    let g = STANDARD_GRAVITY;
    let mass = rho * length * beam * draft;
    let dt = 0.05;
    let kernel = box_heave_retardation(length, beam, draft, rho, g, 16, 1e-3, 4.0, 64, dt, 200);
    let mut td = CumminsHeave::new(mass, kernel, 0.1);
    let samples = td.free_decay(400);
    fnv1a64_f64_slice(&samples)
}

fn fnv1a64_f64_slice(xs: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in xs {
        for b in x.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}
