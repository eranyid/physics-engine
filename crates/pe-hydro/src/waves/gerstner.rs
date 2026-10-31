//! Gerstner / trochoidal waves (Lagrangian free-surface).
//!
//! Steepness guard: `Q · k · A ≤ 1` (no self-intersection). `Q ∈ (0, 1]` scales
//! the orbital radius; `Q = 1` is classical Gerstner.
//!
//! Deep-water form (surface):
//! `X = x − Q A sin χ`, `Z = Q A cos χ`, `χ = k x − ω t + φ`
//! with `ω² = g k`. Elevation reported in Eulerian `(x,y)` uses the implicit
//! surface map evaluated iteratively for `x₀`, then `η = Q A cos(k x₀ − ω t)`.
//!
//! # References
//! - Gerstner, F. J. (1802). Theorie der Wellen.
//! - Kinsman, B. *Wind Waves*; Lamb, *Hydrodynamics* §250 (trochoidal waves).
//! - Tessendorf, J. (2001). Simulating Ocean Water (steepness / Q factor).

use pe_math::libm_ops;
use pe_math::{Vec3, STANDARD_GRAVITY};

use super::airy::dispersion_omega;
use super::Wave;

/// Error when steepness violates the Gerstner guard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GerstnerError {
    /// `Q k A > 1` (would self-intersect).
    SteepnessExceeded,
    /// Non-positive amplitude, k, depth, or Q outside (0, 1].
    InvalidParameter,
}

/// Gerstner / trochoidal progressive wave.
#[derive(Clone, Debug)]
pub struct GerstnerWave {
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
    /// Steepness scale Q ∈ (0, 1].
    pub q: f64,
    /// Depth h [m] (dispersion); deep water if `k h ≫ 1`.
    pub depth: f64,
    /// Gravity g [m/s²].
    pub g: f64,
    /// Mean water level [m].
    pub mwl: f64,
}

impl GerstnerWave {
    /// Construct with steepness guard `Q k A ≤ 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        amplitude: f64,
        k: f64,
        depth: f64,
        theta: f64,
        phase: f64,
        q: f64,
        g: f64,
        mwl: f64,
    ) -> Result<Self, GerstnerError> {
        if !(amplitude > 0.0 && k > 0.0 && depth > 0.0 && q > 0.0 && q <= 1.0 && g > 0.0) {
            return Err(GerstnerError::InvalidParameter);
        }
        if q * k * amplitude > 1.0 + 1e-14 {
            return Err(GerstnerError::SteepnessExceeded);
        }
        let omega = dispersion_omega(k, depth, g);
        Ok(Self {
            amplitude,
            k,
            omega,
            theta,
            phase,
            q,
            depth,
            g,
            mwl,
        })
    }

    /// Earth-g convenience.
    pub fn earth(
        amplitude: f64,
        k: f64,
        depth: f64,
        theta: f64,
        q: f64,
    ) -> Result<Self, GerstnerError> {
        Self::try_new(amplitude, k, depth, theta, 0.0, q, STANDARD_GRAVITY, 0.0)
    }

    /// Product `Q k A` (must be ≤ 1).
    pub fn steepness(&self) -> f64 {
        self.q * self.k * self.amplitude
    }

    #[inline]
    fn dir(&self) -> (f64, f64) {
        (libm_ops::cos(self.theta), libm_ops::sin(self.theta))
    }

    /// Invert Eulerian horizontal position to Lagrangian `s = x₀·ê` along the
    /// propagation direction (fixed-point on `s − Q A sin(k s − ω t + φ) = r`).
    fn lagrangian_s(&self, x: f64, y: f64, t: f64) -> f64 {
        let (ct, st) = self.dir();
        let r = x * ct + y * st;
        let qa = self.q * self.amplitude;
        let mut s = r;
        for _ in 0..12 {
            let chi = self.k * s - self.omega * t + self.phase;
            // f(s) = s − Q A sin(χ) − r = 0; f' = 1 − Q A k cos(χ)
            let chi_sin = libm_ops::sin(chi);
            let chi_cos = libm_ops::cos(chi);
            let f = s - qa * chi_sin - r;
            let fp = 1.0 - qa * self.k * chi_cos;
            let ds = f / fp.max(0.05);
            s -= ds;
            if libm_ops::fabs(ds) < 1e-12 {
                break;
            }
        }
        s
    }
}

impl Wave for GerstnerWave {
    fn height(&self, x: f64, y: f64, t: f64) -> f64 {
        let s = self.lagrangian_s(x, y, t);
        let chi = self.k * s - self.omega * t + self.phase;
        self.q * self.amplitude * libm_ops::cos(chi)
    }

    fn velocity(&self, x: f64, y: f64, z: f64, t: f64) -> Vec3 {
        // Deep-water Gerstner particle velocity at the free surface orbit,
        // attenuated by e^{k z_rel} below MWL (Tessendorf / Kinsman).
        let z_rel = z - self.mwl;
        let s = self.lagrangian_s(x, y, t);
        let chi = self.k * s - self.omega * t + self.phase;
        let atten = if self.k * self.depth > 30.0 {
            libm_ops::exp(self.k * z_rel.min(0.0))
        } else {
            let kh = self.k * self.depth;
            let kz = self.k * (z_rel + self.depth);
            libm_ops::cosh(kz) / libm_ops::cosh(kh)
        };
        let qa = self.q * self.amplitude;
        let u_orb = qa * self.omega * libm_ops::cos(chi) * atten;
        let w = qa * self.omega * libm_ops::sin(chi) * atten;
        let (ct, st) = self.dir();
        Vec3::new(u_orb * ct, u_orb * st, w)
    }

    fn mean_water_level(&self) -> f64 {
        self.mwl
    }
}
