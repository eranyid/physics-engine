//! Linear Airy (potential) waves with the full finite-depth dispersion relation.
//!
//! # References
//! - Airy, G. B. (1845). Tides and waves.
//! - Dean, R. G. & Dalrymple, R. A. *Water Wave Mechanics for Engineers and
//!   Scientists*. Dispersion: `ω² = g k tanh(k h)`.
//! - Deep-water limits: `λ = g T² / (2π)`, `c = g T / (2π)`, `c_g = c/2`.
//! - Shallow-water limit: `c = √(g h)`.

use pe_math::libm_ops;
use pe_math::{Vec3, STANDARD_GRAVITY};

use super::Wave;

/// Solve `ω² = g k tanh(k h)` for wave number `k` [1/m] by Newton iteration.
///
/// `omega` [rad/s], `h` [m] (>0), `g` [m/s²]. Deep-water start `k₀ = ω²/g`;
/// for shallow water the iterate converges to `ω/√(gh)`.
pub fn solve_wave_number(omega: f64, h: f64, g: f64) -> f64 {
    assert!(omega > 0.0 && h > 0.0 && g > 0.0);
    let omega2 = omega * omega;
    // Deep-water seed; clamp shallow seed if kh would be tiny.
    let mut k = omega2 / g;
    let k_shallow = omega / libm_ops::sqrt(g * h);
    if k < 0.5 * k_shallow {
        k = k_shallow;
    }
    for _ in 0..40 {
        let kh = k * h;
        let th = libm_ops::tanh(kh);
        // sech²(kh) = 1 − tanh²(kh)
        let sech2 = 1.0 - th * th;
        let f = g * k * th - omega2;
        let fp = g * th + g * k * h * sech2;
        if libm_ops::fabs(fp) < 1e-30 {
            break;
        }
        let dk = f / fp;
        k -= dk;
        if k <= 0.0 {
            k = 0.5 * k_shallow;
        }
        if libm_ops::fabs(dk) <= 1e-14 * k.max(1.0) {
            break;
        }
    }
    k
}

/// Angular frequency from dispersion: `ω = √(g k tanh(k h))` [rad/s].
pub fn dispersion_omega(k: f64, h: f64, g: f64) -> f64 {
    libm_ops::sqrt(g * k * libm_ops::tanh(k * h))
}

/// Phase speed `c = ω/k` [m/s].
pub fn phase_speed(omega: f64, k: f64) -> f64 {
    omega / k
}

/// Group speed `c_g = ½ c (1 + 2kh / sinh(2kh))` [m/s] (finite depth).
///
/// Deep water → `c/2`; shallow → `c`.
pub fn group_speed(omega: f64, k: f64, h: f64) -> f64 {
    let c = phase_speed(omega, k);
    let kh = k * h;
    let two_kh = 2.0 * kh;
    // Avoid sinh underflow: for large kh, 2kh/sinh(2kh) → 0.
    let factor = if two_kh > 40.0 {
        0.0
    } else {
        let s = libm_ops::sinh(two_kh);
        if libm_ops::fabs(s) < 1e-30 {
            1.0 // shallow: 2kh/sinh(2kh) → 1
        } else {
            two_kh / s
        }
    };
    0.5 * c * (1.0 + factor)
}

/// Deep-water wavelength `λ = g T² / (2π)` [m].
pub fn deep_water_wavelength(period: f64, g: f64) -> f64 {
    g * period * period / (2.0 * core::f64::consts::PI)
}

/// Progressive linear Airy wave.
#[derive(Clone, Debug)]
pub struct AiryWave {
    /// Amplitude A [m] (η = A cos χ).
    pub amplitude: f64,
    /// Wave number k [1/m].
    pub k: f64,
    /// Angular frequency ω [rad/s].
    pub omega: f64,
    /// Propagation heading θ [rad] from +x toward +y.
    pub theta: f64,
    /// Phase φ [rad] at the origin at t=0.
    pub phase: f64,
    /// Still-water depth h [m].
    pub depth: f64,
    /// Gravity g [m/s²].
    pub g: f64,
    /// Mean water level [m].
    pub mwl: f64,
}

impl AiryWave {
    /// Build from amplitude [m], period [s], depth [m], heading [rad].
    ///
    /// Solves the full dispersion relation for `k`.
    pub fn from_period(
        amplitude: f64,
        period: f64,
        depth: f64,
        theta: f64,
        phase: f64,
        g: f64,
        mwl: f64,
    ) -> Self {
        let omega = 2.0 * core::f64::consts::PI / period;
        let k = solve_wave_number(omega, depth, g);
        Self {
            amplitude,
            k,
            omega,
            theta,
            phase,
            depth,
            g,
            mwl,
        }
    }

    /// Convenience with Earth `g₀` and MWL = 0.
    pub fn earth(amplitude: f64, period: f64, depth: f64, theta: f64) -> Self {
        Self::from_period(amplitude, period, depth, theta, 0.0, STANDARD_GRAVITY, 0.0)
    }

    #[inline]
    fn phase_arg(&self, x: f64, y: f64, t: f64) -> f64 {
        let kx = self.k * libm_ops::cos(self.theta);
        let ky = self.k * libm_ops::sin(self.theta);
        kx * x + ky * y - self.omega * t + self.phase
    }

    /// Hyperbolic depth factor `cosh(k(z+h))/sinh(kh)` for horizontal velocity.
    #[inline]
    fn cosh_factor(&self, z_rel: f64) -> f64 {
        let kh = self.k * self.depth;
        let kz = self.k * (z_rel + self.depth);
        if kh > 30.0 {
            // Deep: ≈ e^{k z_rel}
            libm_ops::exp(self.k * z_rel)
        } else {
            libm_ops::cosh(kz) / libm_ops::sinh(kh)
        }
    }

    /// `sinh(k(z+h))/sinh(kh)` for vertical velocity.
    #[inline]
    fn sinh_factor(&self, z_rel: f64) -> f64 {
        let kh = self.k * self.depth;
        let kz = self.k * (z_rel + self.depth);
        if kh > 30.0 {
            libm_ops::exp(self.k * z_rel)
        } else {
            libm_ops::sinh(kz) / libm_ops::sinh(kh)
        }
    }

    /// Undisturbed Airy dynamic pressure contribution relative to hydrostatic
    /// under the mean surface: `p_dyn / (ρ g) = A cosh(k(z+h))/cosh(kh) · cos χ` [m]
    /// (Dean & Dalrymple). Used for long-wave Froude–Krylov.
    pub fn pressure_head(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        let z_rel = z - self.mwl;
        let kh = self.k * self.depth;
        let kz = self.k * (z_rel + self.depth);
        let chi = self.phase_arg(x, y, t);
        let factor = if kh > 30.0 {
            libm_ops::exp(self.k * z_rel)
        } else {
            libm_ops::cosh(kz) / libm_ops::cosh(kh)
        };
        self.amplitude * factor * libm_ops::cos(chi)
    }
}

impl Wave for AiryWave {
    fn height(&self, x: f64, y: f64, t: f64) -> f64 {
        self.amplitude * libm_ops::cos(self.phase_arg(x, y, t))
    }

    fn velocity(&self, x: f64, y: f64, z: f64, t: f64) -> Vec3 {
        let z_rel = z - self.mwl;
        let chi = self.phase_arg(x, y, t);
        let c = libm_ops::cos(chi);
        let s = libm_ops::sin(chi);
        let a_omega = self.amplitude * self.omega;
        let ct = libm_ops::cos(self.theta);
        let st = libm_ops::sin(self.theta);
        let u_h = a_omega * self.cosh_factor(z_rel) * c;
        let w = a_omega * self.sinh_factor(z_rel) * s;
        Vec3::new(u_h * ct, u_h * st, w)
    }

    fn mean_water_level(&self) -> f64 {
        self.mwl
    }
}
