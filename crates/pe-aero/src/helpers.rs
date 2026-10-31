//! Pure aero helpers: trim, terminal dive speed, Lanchester phugoid period.
//!
//! All inputs/outputs are SI. No compressibility / Mach corrections.

use pe_math::libm_ops;
use pe_math::STANDARD_GRAVITY;

use crate::polar::WingPolar;

/// Steady level-flight trim solution (SI).
#[derive(Clone, Copy, Debug)]
pub struct LevelFlightTrim {
    /// Trim airspeed [m/s].
    pub v: f64,
    /// Trim angle of attack [rad].
    pub alpha: f64,
    /// Required thrust (= drag) [N].
    pub thrust: f64,
    /// Lift coefficient [dimensionless].
    pub cl: f64,
    /// Drag coefficient [dimensionless].
    pub cd: f64,
    /// Lift-to-drag ratio [dimensionless].
    pub l_over_d: f64,
    /// Dynamic pressure [Pa].
    pub q_bar: f64,
    /// Minimum q to stall [Pa].
    pub q_min_stall: f64,
}

/// Solve steady level flight: `L = W`, `T = D` at best L/D `C_L` (parabolic polar).
///
/// `mass` [kg], `rho` [kg/m³], `g` [m/s²]. Assumes small α so lift ≈ vertical.
pub fn level_flight_trim(mass: f64, polar: &WingPolar, rho: f64, g: f64) -> LevelFlightTrim {
    let w = mass * g;
    let cl_stall = polar.cl(polar.alpha_stall);
    let q_min_stall = if cl_stall > 0.0 {
        w / (polar.s * cl_stall)
    } else {
        f64::INFINITY
    };
    let mut cl = libm_ops::sqrt(core::f64::consts::PI * polar.e * polar.aspect_ratio * polar.cd0);
    cl = cl.min(0.95 * cl_stall);
    let q = w / (polar.s * cl);
    let v = libm_ops::sqrt(2.0 * q / rho);
    let alpha = (cl - polar.cl0) / polar.cl_alpha;
    let cd = polar.cd(cl);
    let d = q * polar.s * cd;
    LevelFlightTrim {
        v,
        alpha,
        thrust: d,
        cl,
        cd,
        l_over_d: cl / cd,
        q_bar: q,
        q_min_stall,
    }
}

/// Vertical-dive terminal speed [m/s]: `V = √(2 m g / (ρ S C_D(α)))`.
///
/// `mass` [kg], `rho` [kg/m³], `alpha` [rad], `g` [m/s²].
pub fn terminal_velocity_vertical(
    mass: f64,
    polar: &WingPolar,
    rho: f64,
    alpha: f64,
    g: f64,
) -> f64 {
    let cd = polar.cd(polar.cl(alpha));
    libm_ops::sqrt(2.0 * mass * g / (rho * polar.s * cd))
}

/// Lanchester approximate phugoid period `T = π √2 · V / g` [s].
///
/// `v` [m/s], `g` [m/s²]. Long-period approximation for lightly damped phugoid.
pub fn lanchester_phugoid_period(v: f64, g: f64) -> f64 {
    core::f64::consts::PI * libm_ops::sqrt(2.0) * v / g
}

/// Convenience: Earth-g trim.
pub fn level_flight_trim_earth(mass: f64, polar: &WingPolar, rho: f64) -> LevelFlightTrim {
    level_flight_trim(mass, polar, rho, STANDARD_GRAVITY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::RHO0;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    #[test]
    fn trim_lift_equals_weight() {
        let polar = WingPolar::light_aircraft();
        let mass = 1000.0;
        let trim = level_flight_trim_earth(mass, &polar, RHO0);
        let w = mass * STANDARD_GRAVITY;
        let q = 0.5 * RHO0 * trim.v * trim.v;
        let l = q * polar.s * trim.cl;
        let d = q * polar.s * trim.cd;
        assert_relative_eq!(l, w, max_relative = 1e-12);
        assert_relative_eq!(trim.thrust, d, max_relative = 1e-12);
    }

    #[test]
    fn lanchester_phugoid_published() {
        let v = 50.0;
        let t = lanchester_phugoid_period(v, STANDARD_GRAVITY);
        assert_abs_diff_eq!(
            t,
            core::f64::consts::PI * libm_ops::sqrt(2.0) * v / STANDARD_GRAVITY,
            epsilon = 1e-15
        );
    }
}
