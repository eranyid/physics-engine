//! Equations of state for weakly compressible SPH.
//!
//! # Tait / Cole EOS (WCSPH)
//!
//! ```text
//! p = B [ (ρ/ρ₀)^γ − 1 ]
//! B = ρ₀ c₀² / γ
//! ```
//!
//! (Monaghan / Becker–Teschner WCSPH form. Linearised: `p ≈ ρ₀ c₀² (ρ/ρ₀ − 1)`.)
//!
//! Stiffness / speed-of-sound rule used here (Becker & Teschner / Monaghan
//! WCSPH practice): choose `c₀ ≥ 10 v_max` so density fluctuations stay
//! ≲ 1 %. Default γ = 7 (water).
//!
//! # References
//! - Cole, R. H. (1948). *Underwater Explosions*.
//! - Monaghan, J. J. (1994). Simulating free surface flows with SPH. *JCP*.
//! - Becker, M. & Teschner, M. (2007). Weakly compressible SPH for free surface
//!   flows. *SCA*.

use pe_math::libm_ops::pow;

/// Default Tait exponent for water.
pub const TAIT_GAMMA_WATER: f64 = 7.0;

/// Recommended minimum ratio `c₀ / v_max` for ~1 % compressibility.
pub const SOUND_SPEED_FACTOR: f64 = 10.0;

/// Tait EOS parameters.
#[derive(Clone, Copy, Debug)]
pub struct TaitEos {
    /// Rest density ρ₀ [kg/m³].
    pub rest_density: f64,
    /// Artificial speed of sound c₀ [m/s].
    pub sound_speed: f64,
    /// Exponent γ (dimensionless); water = 7.
    pub gamma: f64,
}

impl TaitEos {
    /// Construct with explicit sound speed.
    pub fn new(rest_density: f64, sound_speed: f64, gamma: f64) -> Self {
        Self {
            rest_density,
            sound_speed,
            gamma,
        }
    }

    /// Choose `c₀ = factor * v_max` (default factor [`SOUND_SPEED_FACTOR`]).
    pub fn from_vmax(rest_density: f64, v_max: f64, gamma: f64, factor: f64) -> Self {
        let c0 = factor * v_max.max(1e-6);
        Self::new(rest_density, c0, gamma)
    }

    /// Bulk modulus-like stiffness `B = ρ₀ c₀² / γ` [Pa].
    #[inline]
    pub fn stiffness_b(&self) -> f64 {
        self.rest_density * self.sound_speed * self.sound_speed / self.gamma
    }

    /// Pressure [Pa] from density [kg/m³].
    #[inline]
    pub fn pressure(&self, density: f64) -> f64 {
        let ratio = density / self.rest_density;
        let b = self.stiffness_b();
        b * (pow(ratio, self.gamma) - 1.0)
    }
}

/// Enforce the documented rule `c ≥ 10 v_max`; returns the sound speed used.
#[inline]
pub fn ensure_sound_speed(c: f64, v_max: f64) -> f64 {
    let min_c = SOUND_SPEED_FACTOR * v_max.max(0.0);
    if c < min_c {
        min_c
    } else {
        c
    }
}
