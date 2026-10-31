//! Stokes second-order progressive waves (finite amplitude, irrotational).
//!
//! Deep-water crest elevation: `η_crest = a + ½ k a²`.
//!
//! # References
//! - Stokes, G. G. (1847). On the theory of oscillatory waves.
//! - Dean & Dalrymple, *Water Wave Mechanics*, §4 (Stokes expansion).
//! - Fenton, J. D. (1985). A fifth-order Stokes theory… (2nd-order subset).

use pe_math::libm_ops;
use pe_math::{Vec3, STANDARD_GRAVITY};

use super::airy::{dispersion_omega, solve_wave_number};
use super::Wave;

/// Stokes II progressive wave (elevation to second order; kinematics to first
/// order for immersion / FK use — Stokes drift omitted).
#[derive(Clone, Debug)]
pub struct StokesSecondOrder {
    /// First-order amplitude a [m].
    pub amplitude: f64,
    /// Wave number k [1/m].
    pub k: f64,
    /// Angular frequency ω [rad/s] (linear dispersion).
    pub omega: f64,
    /// Heading θ [rad].
    pub theta: f64,
    /// Phase φ [rad].
    pub phase: f64,
    /// Depth h [m].
    pub depth: f64,
    /// Gravity g [m/s²].
    pub g: f64,
    /// Mean water level [m].
    pub mwl: f64,
}

impl StokesSecondOrder {
    /// Build from amplitude [m], period [s], depth [m].
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

    /// Earth-g convenience, MWL = 0.
    pub fn earth(amplitude: f64, period: f64, depth: f64, theta: f64) -> Self {
        Self::from_period(amplitude, period, depth, theta, 0.0, STANDARD_GRAVITY, 0.0)
    }

    /// Build from prescribed `k` (ω from dispersion).
    pub fn from_k(
        amplitude: f64,
        k: f64,
        depth: f64,
        theta: f64,
        phase: f64,
        g: f64,
        mwl: f64,
    ) -> Self {
        let omega = dispersion_omega(k, depth, g);
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

    #[inline]
    fn phase_arg(&self, x: f64, y: f64, t: f64) -> f64 {
        let kx = self.k * libm_ops::cos(self.theta);
        let ky = self.k * libm_ops::sin(self.theta);
        kx * x + ky * y - self.omega * t + self.phase
    }

    /// Second-order elevation amplitude `B₂` such that `η = a cos χ + B₂ cos(2χ)`.
    ///
    /// Finite depth (Dean & Dalrymple):
    /// `B₂ = (k a² / 4) · (cosh(kh) / sinh³(kh)) · (2 + cosh(2 kh))`.
    /// Deep water → `½ k a²`.
    pub fn b2(&self) -> f64 {
        let a = self.amplitude;
        let kh = self.k * self.depth;
        if kh > 30.0 {
            return 0.5 * self.k * a * a;
        }
        let sh = libm_ops::sinh(kh);
        let ch = libm_ops::cosh(kh);
        let ch2 = libm_ops::cosh(2.0 * kh);
        (self.k * a * a / 4.0) * (ch / (sh * sh * sh)) * (2.0 + ch2)
    }

    /// Analytic deep-water crest elevation `a + ½ k a²` [m].
    pub fn deep_water_crest_elevation(&self) -> f64 {
        let a = self.amplitude;
        a + 0.5 * self.k * a * a
    }
}

impl Wave for StokesSecondOrder {
    fn height(&self, x: f64, y: f64, t: f64) -> f64 {
        let chi = self.phase_arg(x, y, t);
        let a = self.amplitude;
        let b2 = self.b2();
        a * libm_ops::cos(chi) + b2 * libm_ops::cos(2.0 * chi)
    }

    fn velocity(&self, x: f64, y: f64, z: f64, t: f64) -> Vec3 {
        // First-order Airy kinematics.
        let z_rel = z - self.mwl;
        let chi = self.phase_arg(x, y, t);
        let kh = self.k * self.depth;
        let kz = self.k * (z_rel + self.depth);
        let (cf, sf) = if kh > 30.0 {
            let e = libm_ops::exp(self.k * z_rel);
            (e, e)
        } else {
            let sh = libm_ops::sinh(kh);
            (libm_ops::cosh(kz) / sh, libm_ops::sinh(kz) / sh)
        };
        let a_omega = self.amplitude * self.omega;
        let c = libm_ops::cos(chi);
        let s = libm_ops::sin(chi);
        let u_h = a_omega * cf * c;
        let w = a_omega * sf * s;
        let ct = libm_ops::cos(self.theta);
        let st = libm_ops::sin(self.theta);
        Vec3::new(u_h * ct, u_h * st, w)
    }

    fn mean_water_level(&self) -> f64 {
        self.mwl
    }
}
