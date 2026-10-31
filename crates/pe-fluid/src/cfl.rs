//! CFL timestep control for SPH.
//!
//! ```text
//! dt ≤ λ_v  h / (c₀ + v_max)
//! dt ≤ λ_f  √(h / |a|_max)     (force / acceleration limit)
//! dt ≤ λ_visc  h² / ν_eff      when viscosity dominates (optional)
//! ```
//!
//! Default safety factors: `λ_v = 0.25`, `λ_f = 0.25`.
//!
//! # References
//! - Monaghan, J. J. & Kos, A. (1999). Solitary waves on a Cretan beach.

use pe_math::libm_ops::sqrt;
use pe_math::Vec3;

/// CFL safety factors.
#[derive(Clone, Copy, Debug)]
pub struct CflParams {
    /// Velocity / sound-speed factor λ_v.
    pub lambda_v: f64,
    /// Acceleration factor λ_f.
    pub lambda_f: f64,
}

impl Default for CflParams {
    fn default() -> Self {
        Self {
            lambda_v: 0.25,
            lambda_f: 0.25,
        }
    }
}

/// Suggested timestep [s] from smoothing length, sound speed, max speed, max |a|.
pub fn suggested_dt(h: f64, sound_speed: f64, v_max: f64, a_max: f64, params: CflParams) -> f64 {
    let dt_v = params.lambda_v * h / (sound_speed + v_max).max(1e-12);
    let dt_f = if a_max > 1e-12 {
        params.lambda_f * sqrt(h / a_max)
    } else {
        f64::INFINITY
    };
    dt_v.min(dt_f)
}

/// Max acceleration magnitude from an acceleration buffer.
pub fn max_accel(a: &[Vec3], fluid: &[bool]) -> f64 {
    let mut m = 0.0;
    for (i, ai) in a.iter().enumerate() {
        if fluid.get(i).copied().unwrap_or(true) {
            let s = ai.norm_squared();
            if s > m {
                m = s;
            }
        }
    }
    sqrt(m)
}
