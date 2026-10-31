//! Physical / geodetic constants used by the space module.
//!
//! Earth μ, R, J2 match the values requested for M7 validation
//! (WGS84-class; μ given as GM rather than G×M).

use pe_math::libm_ops;

/// 2π (τ).
pub const TWO_PI: f64 = core::f64::consts::TAU;

/// Earth gravitational parameter μ = GM [m³/s²].
///
/// Derived from the published value `398600.4418 km³/s²` (common WGS84/EGM96 GM).
pub const EARTH_MU: f64 = 3.986_004_418e14;

/// Earth μ in km³/s² (same constant; for documentation / km-side tests).
pub const EARTH_MU_KM3_S2: f64 = 398_600.441_8;

/// Earth equatorial radius [m] (WGS84).
pub const EARTH_RADIUS: f64 = 6_378_137.0;

/// Earth equatorial radius [km].
pub const EARTH_RADIUS_KM: f64 = 6_378.137;

/// Earth dynamical form factor J₂ [dimensionless] (EGM96-class value used in tests).
pub const EARTH_J2: f64 = 1.082_626_68e-3;

/// Approximate solar μ [m³/s²] (IAU 2015 GM_Sun ≈ 1.3271244e20).
///
/// Marked for third-body / SRP demos; not used as an unverified precision standard.
pub const MU_SUN: f64 = 1.327_124_4e20;

/// Degrees per day → rad/s helper (exact conversion factor).
#[inline]
pub fn deg_per_day_to_rad_per_s(deg_per_day: f64) -> f64 {
    deg_per_day * (core::f64::consts::PI / 180.0) / 86_400.0
}

/// Mean motion `n = √(μ / a³)` [rad/s] for a circular / osculating semi-major axis.
#[inline]
pub fn mean_motion_circular(mu: f64, a: f64) -> f64 {
    libm_ops::sqrt(mu / (a * a * a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn earth_mu_km_si_consistent() {
        assert_relative_eq!(EARTH_MU, EARTH_MU_KM3_S2 * 1e9, max_relative = 1e-15);
        assert_relative_eq!(EARTH_RADIUS, EARTH_RADIUS_KM * 1e3, max_relative = 1e-15);
    }
}
