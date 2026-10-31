//! Spectral ocean: Pierson–Moskowitz / JONSWAP with `cos^{2s}` spreading.
//!
//! Surface is a **deterministic finite sum of Airy components** (no FFT), so the
//! elevation is bit-stable across OS when the seed and discretization match.
//!
//! # References
//! - Pierson, W. J. & Moskowitz, L. (1964). A proposed spectral form for fully
//!   developed wind seas. *J. Geophys. Res.* 69.
//! - Hasselmann et al. (1973). Measurements of wind-wave growth… (JONSWAP).
//! - Longuet-Higgins, Cartwright & Smith (1963); Mitsuyasu et al. — `cos^{2s}`
//!   directional spreading.
//! - Significant wave height `H_s = 4 √m₀` with `m₀ = ∫ S(ω) dω`.

use pe_math::libm_ops;
use pe_math::{Vec3, Xoshiro256StarStar, STANDARD_GRAVITY};

use super::airy::{dispersion_omega, solve_wave_number};
use super::Wave;

/// Spectral shape family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectrumKind {
    /// Pierson–Moskowitz fully developed sea.
    PiersonMoskowitz,
    /// JONSWAP fetch-limited sea.
    Jonswap,
}

/// Directional spreading parameters (`cos^{2s}` about `theta0`).
#[derive(Clone, Copy, Debug)]
pub struct SpreadingParams {
    /// Mean wave direction θ₀ [rad].
    pub theta0: f64,
    /// Spreading exponent s ≥ 0 (often ~1…16).
    pub s: f64,
}

impl Default for SpreadingParams {
    fn default() -> Self {
        Self {
            theta0: 0.0,
            s: 4.0,
        }
    }
}

/// One linear Airy component of a spectral sea.
#[derive(Clone, Debug)]
pub struct WaveComponent {
    /// Amplitude A [m].
    pub amplitude: f64,
    /// Wave number k [1/m].
    pub k: f64,
    /// Angular frequency ω [rad/s].
    pub omega: f64,
    /// Heading θ [rad].
    pub theta: f64,
    /// Phase φ [rad].
    pub phase: f64,
}

/// Pierson–Moskowitz spectrum `S(ω)` [m² s / rad] for wind speed `u19_5` [m/s].
///
/// `S(ω) = α g² ω⁻⁵ exp(−β (ω₀/ω)⁴)` with `α = 0.0081`, `β = 0.74`,
/// `ω₀ = g / U₁₉.₅` (Pierson & Moskowitz 1964).
pub fn pierson_moskowitz_spectrum(omega: f64, u19_5: f64, g: f64) -> f64 {
    if omega <= 0.0 || u19_5 <= 0.0 {
        return 0.0;
    }
    let alpha = 0.0081;
    let beta = 0.74;
    let omega0 = g / u19_5;
    let ratio = omega0 / omega;
    alpha * g * g / libm_ops::pow(omega, 5.0) * libm_ops::exp(-beta * libm_ops::pow(ratio, 4.0))
}

/// Approximate PM significant wave height `H_s ≈ 0.21 U² / g` [m]
/// (Pierson–Moskowitz; engineering fit for `U = U₁₉.₅`).
pub fn pierson_moskowitz_hs_approx(u19_5: f64, g: f64) -> f64 {
    0.21 * u19_5 * u19_5 / g
}

/// JONSWAP spectrum [m² s / rad].
///
/// `S_J = S_PM · γ^{exp(−(ω−ωp)²/(2 σ² ωp²))}` with peak enhancement `γ`
/// (Hasselmann et al. 1973). Peak frequency `ω_p` [rad/s]; `σ = 0.07` (ω≤ωp)
/// or `0.09` (ω>ωp).
pub fn jonswap_spectrum(omega: f64, u19_5: f64, omega_p: f64, gamma: f64, g: f64) -> f64 {
    let pm = pierson_moskowitz_spectrum(omega, u19_5, g);
    if pm == 0.0 || omega_p <= 0.0 {
        return pm;
    }
    let sigma = if omega <= omega_p { 0.07 } else { 0.09 };
    let dw = omega - omega_p;
    let arg = -(dw * dw) / (2.0 * sigma * sigma * omega_p * omega_p);
    pm * libm_ops::pow(gamma, libm_ops::exp(arg))
}

/// `cos^{2s}` directional spreading density (normalized over (−π, π]).
///
/// `D(θ) ∝ cos^{2s}((θ−θ₀)/2)` for `|θ−θ₀| ≤ π`, else 0. Normalization via
/// `∫ D dθ = 1` using `Γ(s+1) / (2 √π Γ(s+½))` for the half-angle form
/// (Longuet-Higgins / Goda).
pub fn spreading_density(theta: f64, params: SpreadingParams) -> f64 {
    let mut dtheta = theta - params.theta0;
    // wrap to (−π, π]
    let pi = core::f64::consts::PI;
    while dtheta > pi {
        dtheta -= 2.0 * pi;
    }
    while dtheta <= -pi {
        dtheta += 2.0 * pi;
    }
    let half = 0.5 * dtheta;
    let c = libm_ops::cos(half);
    if c <= 0.0 {
        return 0.0;
    }
    let s = params.s;
    let unnorm = libm_ops::pow(c, 2.0 * s);
    // Normalization constant: ∫_{-π}^{π} cos^{2s}(θ/2) dθ = 2π · Γ(s+1)/(√π Γ(s+½)) / 2π
    // For integer/half-integer s we use a recursive beta-function evaluation.
    let norm = spreading_norm(s);
    unnorm * norm
}

fn spreading_norm(s: f64) -> f64 {
    // ∫_0^{2π} cos^{2s}(φ/2) dφ with φ=θ … = 2 * √π * Γ(s+½) / Γ(s+1) * something
    // Standard result: D = [2^{2s−1}/π · Γ²(s+1)/Γ(2s+1)] cos^{2s}((θ−θ0)/2)
    // Mitsuyasu / ITTC form used here for s ≥ 0:
    // G(s) = 2^{2s−1} Γ²(s+1) / (π Γ(2s+1))
    let g1 = ln_gamma(s + 1.0);
    let g2 = ln_gamma(2.0 * s + 1.0);
    let log_g =
        (2.0 * s - 1.0) * libm_ops::log(2.0) + 2.0 * g1 - libm_ops::log(core::f64::consts::PI) - g2;
    libm_ops::exp(log_g)
}

/// Lanczos approximation for `ln Γ(x)` (x > 0). Clean-room coefficients from
/// Lanczos (1964) / Numerical Recipes style — portable via `libm`.
fn ln_gamma(x: f64) -> f64 {
    // For small positive x use reflection-free Lanczos with g=5, n=6.
    const COEFF: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.120_865_097_386_617_9e-2,
        -0.539_523_938_495_3e-5,
    ];
    let mut y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * libm_ops::log(tmp);
    let mut ser = 1.000_000_000_190_015;
    for c in COEFF {
        y += 1.0;
        ser += c / y;
    }
    -tmp + libm_ops::log(2.506_628_274_631_000_5 * ser / x)
}

/// Finite-component spectral ocean (deterministic component sum).
#[derive(Clone, Debug)]
pub struct SpectralOcean {
    /// Linear components (fixed order — no HashMap).
    pub components: Vec<WaveComponent>,
    /// Still-water depth h [m].
    pub depth: f64,
    /// Gravity g [m/s²].
    pub g: f64,
    /// Mean water level [m].
    pub mwl: f64,
    /// Zeroth spectral moment of the **discretized** spectrum `Σ ½ A_i²` [= m₀].
    pub m0_discrete: f64,
}

impl SpectralOcean {
    /// Significant wave height from discrete components: `H_s = 4 √m₀`.
    pub fn significant_wave_height(&self) -> f64 {
        4.0 * libm_ops::sqrt(self.m0_discrete.max(0.0))
    }

    /// Build a PM or JONSWAP sea as a regular ω×θ grid with seeded random phases.
    ///
    /// Amplitudes: `A = √(2 S(ω) D(θ) Δω Δθ)` so that `m₀ = Σ ½ A²` recovers the
    /// discrete spectral integral exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        kind: SpectrumKind,
        u19_5: f64,
        depth: f64,
        spreading: SpreadingParams,
        n_omega: usize,
        n_theta: usize,
        omega_min: f64,
        omega_max: f64,
        gamma: f64,
        omega_p: f64,
        seed: u64,
        g: f64,
        mwl: f64,
    ) -> Self {
        assert!(n_omega >= 1 && n_theta >= 1);
        assert!(omega_max > omega_min && omega_min > 0.0);
        let mut rng = Xoshiro256StarStar::seed(seed);
        let d_omega = (omega_max - omega_min) / n_omega as f64;
        let d_theta = 2.0 * core::f64::consts::PI / n_theta as f64;
        let mut components = Vec::with_capacity(n_omega * n_theta);
        let mut m0 = 0.0;
        for i in 0..n_omega {
            let omega = omega_min + (i as f64 + 0.5) * d_omega;
            let s_om = match kind {
                SpectrumKind::PiersonMoskowitz => pierson_moskowitz_spectrum(omega, u19_5, g),
                SpectrumKind::Jonswap => jonswap_spectrum(omega, u19_5, omega_p, gamma, g),
            };
            let k = solve_wave_number(omega, depth, g);
            // verify dispersion (omega may differ slightly — keep requested ω)
            let _ = dispersion_omega(k, depth, g);
            for j in 0..n_theta {
                let theta = -core::f64::consts::PI + (j as f64 + 0.5) * d_theta;
                let d = spreading_density(theta, spreading);
                let variance = s_om * d * d_omega * d_theta; // contribution to m₀
                if variance <= 0.0 {
                    continue;
                }
                let amp = libm_ops::sqrt(2.0 * variance);
                let phase = rng.uniform(0.0, 2.0 * core::f64::consts::PI);
                m0 += 0.5 * amp * amp;
                components.push(WaveComponent {
                    amplitude: amp,
                    k,
                    omega,
                    theta,
                    phase,
                });
            }
        }
        Self {
            components,
            depth,
            g,
            mwl,
            m0_discrete: m0,
        }
    }

    /// PM sea convenience (Earth g, deep-ish depth, default spreading).
    pub fn pierson_moskowitz(
        u19_5: f64,
        depth: f64,
        n_omega: usize,
        n_theta: usize,
        seed: u64,
    ) -> Self {
        let g = STANDARD_GRAVITY;
        // Frequency band covering the PM peak ω_p ≈ 0.877 g / U
        let omega_p = 0.877 * g / u19_5;
        let omega_min = 0.2 * omega_p;
        let omega_max = 5.0 * omega_p;
        Self::generate(
            SpectrumKind::PiersonMoskowitz,
            u19_5,
            depth,
            SpreadingParams::default(),
            n_omega,
            n_theta,
            omega_min,
            omega_max,
            1.0,
            omega_p,
            seed,
            g,
            0.0,
        )
    }

    /// JONSWAP sea convenience.
    pub fn jonswap(
        u19_5: f64,
        depth: f64,
        omega_p: f64,
        gamma: f64,
        n_omega: usize,
        n_theta: usize,
        seed: u64,
    ) -> Self {
        let g = STANDARD_GRAVITY;
        let omega_min = 0.2 * omega_p;
        let omega_max = 5.0 * omega_p;
        Self::generate(
            SpectrumKind::Jonswap,
            u19_5,
            depth,
            SpreadingParams::default(),
            n_omega,
            n_theta,
            omega_min,
            omega_max,
            gamma,
            omega_p,
            seed,
            g,
            0.0,
        )
    }
}

impl Wave for SpectralOcean {
    fn height(&self, x: f64, y: f64, t: f64) -> f64 {
        let mut eta = 0.0;
        for c in &self.components {
            let kx = c.k * libm_ops::cos(c.theta);
            let ky = c.k * libm_ops::sin(c.theta);
            let chi = kx * x + ky * y - c.omega * t + c.phase;
            eta += c.amplitude * libm_ops::cos(chi);
        }
        eta
    }

    fn velocity(&self, x: f64, y: f64, z: f64, t: f64) -> Vec3 {
        let z_rel = z - self.mwl;
        let mut u = 0.0;
        let mut v = 0.0;
        let mut w = 0.0;
        for c in &self.components {
            let kx = c.k * libm_ops::cos(c.theta);
            let ky = c.k * libm_ops::sin(c.theta);
            let chi = kx * x + ky * y - c.omega * t + c.phase;
            let kh = c.k * self.depth;
            let kz = c.k * (z_rel + self.depth);
            let (cf, sf) = if kh > 30.0 {
                let e = libm_ops::exp(c.k * z_rel.min(0.0));
                (e, e)
            } else {
                let sh = libm_ops::sinh(kh);
                if libm_ops::fabs(sh) < 1e-30 {
                    (1.0, 0.0)
                } else {
                    (libm_ops::cosh(kz) / sh, libm_ops::sinh(kz) / sh)
                }
            };
            let a_om = c.amplitude * c.omega;
            let cos_c = libm_ops::cos(chi);
            let sin_c = libm_ops::sin(chi);
            let uh = a_om * cf * cos_c;
            u += uh * libm_ops::cos(c.theta);
            v += uh * libm_ops::sin(c.theta);
            w += a_om * sf * sin_c;
        }
        Vec3::new(u, v, w)
    }

    fn mean_water_level(&self) -> f64 {
        self.mwl
    }
}
