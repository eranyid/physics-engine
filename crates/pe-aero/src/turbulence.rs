//! Steady wind and Dryden turbulence (seeded, deterministic).
//!
//! # References
//! - Dryden turbulence model: MIL-F-8785C / MIL-HDBK-1797; Hoblit, F. M.
//!   *Gust Loads on Aircraft*. Form filters on white noise for u_g, v_g, w_g.
//! - Longitudinal `u` uses the first-order Dryden filter; lateral `v` and
//!   vertical `w` use the **second-order** Dryden shaping filters
//!   (von Kármán/Dryden MIL-F-8785C form).
//! - Discrete-time updates use exact exponential / variance-matching one-step
//!   maps with portable `libm` (no FMA).
//!
//! RNG: [`pe_math::Xoshiro256StarStar`] (same PRNG as spectral waves).
//!
//! # Deterministic integrator coupling
//! [`HeldDrydenGust`] advances the filter once per discrete interval
//! `[n Δt, (n+1) Δt)` and **holds** the gust constant inside the interval.
//! RK4 (and any multi-stage method) therefore sees identical wind at every
//! stage that lands in the same interval, so replay of a fixed seed/`Δt`
//! schedule is bit-deterministic. See `docs/AERO.md`.

use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use pe_math::libm_ops;
use pe_math::{Vec3, Xoshiro256StarStar};

/// Constant / slowly varying wind in the world frame [m/s].
#[derive(Clone, Debug)]
pub struct SteadyWind {
    /// Wind velocity [m/s] (world frame; blowing *to* this direction).
    pub velocity: Vec3,
}

impl SteadyWind {
    /// Construct from components [m/s].
    pub fn new(vx: f64, vy: f64, vz: f64) -> Self {
        Self {
            velocity: Vec3::new(vx, vy, vz),
        }
    }

    /// Horizontal wind of speed `speed` [m/s] from heading `from_deg` [deg]
    /// meteorological (direction wind comes from, CW from north). World +x east,
    /// +y north, +z up.
    pub fn from_met(speed: f64, from_deg: f64) -> Self {
        let rad = from_deg * core::f64::consts::PI / 180.0;
        let vx = -speed * libm_ops::sin(rad);
        let vy = -speed * libm_ops::cos(rad);
        Self::new(vx, vy, 0.0)
    }
}

/// Dryden turbulence intensities / scale lengths (SI).
#[derive(Clone, Copy, Debug)]
pub struct DrydenParams {
    /// Longitudinal turbulence intensity σ_u [m/s].
    pub sigma_u: f64,
    /// Lateral intensity σ_v [m/s].
    pub sigma_v: f64,
    /// Vertical intensity σ_w [m/s].
    pub sigma_w: f64,
    /// Longitudinal scale L_u [m].
    pub l_u: f64,
    /// Lateral scale L_v [m].
    pub l_v: f64,
    /// Vertical scale L_w [m].
    pub l_w: f64,
}

impl DrydenParams {
    /// Light turbulence at low altitude (engineering defaults, not a flight-manual table).
    pub fn light_low_altitude() -> Self {
        Self {
            sigma_u: 1.5,
            sigma_v: 1.5,
            sigma_w: 1.0,
            l_u: 200.0,
            l_v: 200.0,
            l_w: 50.0,
        }
    }
}

/// First-order Dryden shaping state (longitudinal `u`).
#[derive(Clone, Debug)]
struct DrydenAxis1 {
    x: f64,
    sigma: f64,
    length: f64,
}

impl DrydenAxis1 {
    fn new(sigma: f64, length: f64) -> Self {
        Self {
            x: 0.0,
            sigma,
            length,
        }
    }

    /// `H_u(s) = σ √(2 L/(π V)) / (1 + s L/V)`.
    /// Discrete: `x ← a x + b w` with `a = exp(−V dt/L)`, `b = σ √(1−a²)`.
    fn step(&mut self, v: f64, dt: f64, w: f64) -> f64 {
        let v_abs = v.max(1.0);
        let a = libm_ops::exp(-v_abs * dt / self.length.max(1e-6));
        let b = self.sigma * libm_ops::sqrt((1.0 - a * a).max(0.0));
        self.x = a * self.x + b * w;
        self.x
    }
}

/// Second-order Dryden shaping state for `v` / `w` (MIL-F-8785C).
///
/// Continuous (Hoblit / MIL-F-8785C):
/// `H(s)/σ = √(L/(π V)) (1 + √3 τ s) / (1 + τ s)²`, `τ = L/V`.
/// Partial fractions: `√3/(1+τs) + (1−√3)/(1+τs)²`, realised as two
/// cascaded unit-variance OU stages with mix weights, then scaled by `σ`
/// using the stationary variance of the unit mix so `E[y²] → σ²`.
#[derive(Clone, Debug)]
struct DrydenAxis2 {
    /// First OU state (unit stationary variance).
    x1: f64,
    /// Second cascaded state.
    x2: f64,
    sigma: f64,
    length: f64,
}

impl DrydenAxis2 {
    fn new(sigma: f64, length: f64) -> Self {
        Self {
            x1: 0.0,
            x2: 0.0,
            sigma,
            length,
        }
    }

    fn step(&mut self, v: f64, dt: f64, w: f64) -> f64 {
        let v_abs = v.max(1.0);
        let l = self.length.max(1e-6);
        let a = libm_ops::exp(-v_abs * dt / l);
        let s = libm_ops::sqrt((1.0 - a * a).max(0.0));
        // Unit-variance OU approximating 1/(1+τ s).
        self.x1 = a * self.x1 + s * w;
        // Second pole: exponentially weighted copy of x1 (ZOH-style cascade).
        // x2 ← a x2 + (1−a) x1 keeps Cov/Var O(1) as a→1.
        self.x2 = a * self.x2 + (1.0 - a) * self.x1;
        // Partial fractions: (1+√3 τ s)/(1+τ s)² = √3/(1+τs) + (1−√3)/(1+τs)².
        let sqrt3 = libm_ops::sqrt(3.0);
        let c1 = sqrt3;
        let c2 = 1.0 - sqrt3;
        let y_unit = c1 * self.x1 + c2 * self.x2;
        // Stationary moments (R11=1) with x1⁺=a x1+s w, x2⁺=a x2+(1−a) x1⁺:
        //   R12 = 1/(1+a)
        //   R22 = (1−a)/(1+a) + 2 a² / (1+a)²
        let ap = (1.0 + a).max(1e-12);
        let r11 = 1.0;
        let r12 = 1.0 / ap;
        let r22 = (1.0 - a) / ap + 2.0 * a * a / (ap * ap);
        let var_unit = c1 * c1 * r11 + c2 * c2 * r22 + 2.0 * c1 * c2 * r12;
        let gain = self.sigma / libm_ops::sqrt(var_unit.max(1e-12));
        let _ = s;
        gain * y_unit
    }
}

/// Dryden turbulence generator (body/world gust velocities).
#[derive(Clone, Debug)]
pub struct DrydenTurbulence {
    /// Parameters.
    pub params: DrydenParams,
    rng: Xoshiro256StarStar,
    u: DrydenAxis1,
    v: DrydenAxis2,
    w: DrydenAxis2,
    /// Last gust [m/s] in wind axes as world-aligned `(u_g, v_g, w_g_up)`.
    last: Vec3,
}

impl DrydenTurbulence {
    /// Seeded Dryden turbulence (2nd-order `v`/`w`, 1st-order `u`).
    pub fn new(params: DrydenParams, seed: u64) -> Self {
        Self {
            u: DrydenAxis1::new(params.sigma_u, params.l_u),
            v: DrydenAxis2::new(params.sigma_v, params.l_v),
            w: DrydenAxis2::new(params.sigma_w, params.l_w),
            params,
            rng: Xoshiro256StarStar::seed(seed),
            last: Vec3::zero(),
        }
    }

    /// Advance filters; `airspeed` [m/s], `dt` [s]. Returns gust velocity in
    /// a north-east-up frame with components `(u_g, v_g, w_g_up)` [m/s].
    pub fn step(&mut self, airspeed: f64, dt: f64) -> Vec3 {
        let wu = self.rng.normal();
        let wv = self.rng.normal();
        let ww = self.rng.normal();
        let ug = self.u.step(airspeed, dt, wu);
        let vg = self.v.step(airspeed, dt, wv);
        let wg_down = self.w.step(airspeed, dt, ww);
        self.last = Vec3::new(ug, vg, -wg_down);
        self.last
    }

    /// Last gust sample [m/s].
    pub fn last_gust(&self) -> Vec3 {
        self.last
    }
}

/// Continuous-frequency Dryden PSD for the lateral/vertical component
/// (MIL-F-8785C), `Φ(ω)` [(m/s)²/(rad/s)]:
/// `Φ(ω) = σ² L/(π V) · (1 + 3 (L ω/V)²) / (1 + (L ω/V)²)²`.
pub fn dryden_vw_psd(omega: f64, sigma: f64, length: f64, airspeed: f64) -> f64 {
    let v = airspeed.max(1.0);
    let l = length.max(1e-6);
    let o = omega.abs();
    let lv = l / v;
    let x = lv * o;
    let x2 = x * x;
    let num = 1.0 + 3.0 * x2;
    let den = (1.0 + x2) * (1.0 + x2);
    sigma * sigma * (l / (core::f64::consts::PI * v)) * num / den
}

/// Longitudinal Dryden PSD `Φ_u(ω) = σ² (2 L)/(π V) / (1 + (L ω/V)²)`.
pub fn dryden_u_psd(omega: f64, sigma: f64, length: f64, airspeed: f64) -> f64 {
    let v = airspeed.max(1.0);
    let l = length.max(1e-6);
    let x = (l / v) * omega.abs();
    sigma * sigma * (2.0 * l / (core::f64::consts::PI * v)) / (1.0 + x * x)
}

/// Estimate one-sided PSD slope `d log Φ / d log ω` over a frequency band by
/// log–log linear regression on periodogram bins.
pub fn estimate_psd_log_slope(freqs: &[f64], psd: &[f64]) -> f64 {
    assert_eq!(freqs.len(), psd.len());
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut n = 0.0;
    for (&f, &p) in freqs.iter().zip(psd.iter()) {
        if f <= 0.0 || p <= 0.0 {
            continue;
        }
        let x = libm_ops::log(f);
        let y = libm_ops::log(p);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
        n += 1.0;
    }
    if n < 2.0 {
        return 0.0;
    }
    let den = n * sxx - sx * sx;
    if den.abs() < 1e-30 {
        return 0.0;
    }
    (n * sxy - sx * sy) / den
}

/// Periodogram PSD estimate (boxcar, one-sided) for a real signal sampled at
/// `dt` [s]. Returns `(freqs_hz_as_rad_per_s_angular, psd)`.
pub fn periodogram_psd(samples: &[f64], dt: f64) -> (Vec<f64>, Vec<f64>) {
    let n = samples.len();
    if n < 4 {
        return (Vec::new(), Vec::new());
    }
    // DFT via direct Goertzel-style for k=1..N/2 (portable, no FFT crate).
    let n_f = n as f64;
    let mut freqs = Vec::with_capacity(n / 2);
    let mut psd = Vec::with_capacity(n / 2);
    let scale = 2.0 * dt / n_f; // one-sided power scaling
    for k in 1..=(n / 2) {
        let omega = 2.0 * core::f64::consts::PI * (k as f64) / (n_f * dt);
        let mut re = 0.0;
        let mut im = 0.0;
        for (i, &x) in samples.iter().enumerate() {
            let ang = omega * dt * i as f64;
            re += x * libm_ops::cos(ang);
            im -= x * libm_ops::sin(ang);
        }
        let p = scale * (re * re + im * im);
        freqs.push(omega);
        psd.push(p);
    }
    (freqs, psd)
}

/// Sample RMS of a real sequence.
pub fn sample_rms(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let var = samples
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / samples.len() as f64;
    libm_ops::sqrt(var)
}

/// Combine steady wind + optional Dryden gust into a total wind vector [m/s].
#[derive(Clone, Debug)]
pub struct WindField {
    /// Steady component.
    pub steady: SteadyWind,
    /// Optional turbulence (None → steady only).
    pub dryden: Option<DrydenTurbulence>,
}

impl WindField {
    /// Steady only.
    pub fn steady(v: Vec3) -> Self {
        Self {
            steady: SteadyWind { velocity: v },
            dryden: None,
        }
    }

    /// Steady + Dryden.
    pub fn with_dryden(steady: SteadyWind, dryden: DrydenTurbulence) -> Self {
        Self {
            steady,
            dryden: Some(dryden),
        }
    }

    /// Total wind at this step (advances Dryden when present).
    pub fn step(&mut self, airspeed: f64, dt: f64) -> Vec3 {
        let gust = match &mut self.dryden {
            Some(d) => d.step(airspeed, dt),
            None => Vec3::zero(),
        };
        self.steady.velocity + gust
    }

    /// Current wind without advancing (steady + last gust).
    pub fn current(&self) -> Vec3 {
        let gust = self
            .dryden
            .as_ref()
            .map(|d| d.last_gust())
            .unwrap_or(Vec3::zero());
        self.steady.velocity + gust
    }
}

/// Piecewise-constant Dryden gust for immutable [`ForceGenerator::apply`].
///
/// Advances the underlying [`DrydenTurbulence`] when the discrete step index
/// `n = floor(t / dt)` increases, and holds the last sample otherwise. Thread-
/// safe via `Mutex` so `SixDofAero: ForceGenerator` stays `Sync`.
#[derive(Debug)]
pub struct HeldDrydenGust {
    inner: Mutex<HeldInner>,
    /// Fixed hold interval [s] (must match the integrator major step).
    pub dt: f64,
    /// Seed (retained for documentation / rebuild).
    pub seed: u64,
}

#[derive(Debug)]
struct HeldInner {
    dryden: DrydenTurbulence,
    /// Last completed step index (`−1` = none).
    step_idx: i64,
    last: Vec3,
}

impl Clone for HeldDrydenGust {
    fn clone(&self) -> Self {
        let g = self.inner.lock().expect("held dryden mutex");
        Self {
            inner: Mutex::new(HeldInner {
                dryden: g.dryden.clone(),
                step_idx: g.step_idx,
                last: g.last,
            }),
            dt: self.dt,
            seed: self.seed,
        }
    }
}

impl HeldDrydenGust {
    /// Build a held gust schedule with step `dt` [s] and RNG `seed`.
    pub fn new(params: DrydenParams, seed: u64, dt: f64) -> Self {
        Self {
            inner: Mutex::new(HeldInner {
                dryden: DrydenTurbulence::new(params, seed),
                step_idx: -1,
                last: Vec3::zero(),
            }),
            dt: dt.max(1e-9),
            seed,
        }
    }

    /// Gust at simulation time `t` [s] for the given airspeed [m/s].
    ///
    /// If `t` jumps backward relative to the internal index, the filter is
    /// rebuilt from the seed so replay from `t=0` stays deterministic.
    pub fn gust_at(&self, t: f64, airspeed: f64) -> Vec3 {
        let target = if t < 0.0 {
            0
        } else {
            libm_ops::floor(t / self.dt + 1e-12) as i64
        };
        let mut g = self.inner.lock().expect("held dryden mutex");
        if target < g.step_idx {
            // Replay from origin.
            let params = g.dryden.params;
            g.dryden = DrydenTurbulence::new(params, self.seed);
            g.step_idx = -1;
            g.last = Vec3::zero();
        }
        while g.step_idx < target {
            g.last = g.dryden.step(airspeed, self.dt);
            g.step_idx += 1;
        }
        g.last
    }
}

/// Monotonic scratch used only in tests / hash helpers (not required for Held).
#[allow(dead_code)]
static GUST_HASH_TICK: AtomicU64 = AtomicU64::new(0);

#[allow(dead_code)]
fn tick_gust_hash() -> u64 {
    GUST_HASH_TICK.fetch_add(1, Ordering::Relaxed)
}
