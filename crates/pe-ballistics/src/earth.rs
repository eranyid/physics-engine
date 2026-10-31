//! Rotating-Earth effects: Coriolis, Eötvös, gravity vs altitude.
//!
//! # References
//! - Coriolis / Eötvös: standard geophysical kinematics (e.g. McCoy, *Modern
//!   Exterior Ballistics*; Wikipedia / textbook forms).
//! - Gravity: inverse-square / linear height approximation `g(h) = g0 (R/(R+h))²`.

use pe_math::libm_ops;
use pe_math::{Vec3, STANDARD_GRAVITY};

/// Mean Earth radius [m] (WGS84 approx).
pub const EARTH_RADIUS_M: f64 = 6_378_137.0;
/// Earth sidereal angular rate [rad/s].
pub const EARTH_OMEGA: f64 = 7.292_115_e-5;

/// Local ENU frame at geodetic latitude `lat_rad` [rad]: +x east, +y north, +z up.
#[derive(Clone, Copy, Debug)]
pub struct LocalFrame {
    /// Geodetic latitude [rad].
    pub latitude_rad: f64,
    /// Earth angular velocity vector in local ENU [rad/s].
    pub omega_enu: Vec3,
}

impl LocalFrame {
    /// Build local frame; `omega` has components
    /// `(0, ω cos φ, ω sin φ)` in ENU.
    pub fn new(latitude_rad: f64) -> Self {
        let cos_l = libm_ops::cos(latitude_rad);
        let sin_l = libm_ops::sin(latitude_rad);
        Self {
            latitude_rad,
            omega_enu: Vec3::new(0.0, EARTH_OMEGA * cos_l, EARTH_OMEGA * sin_l),
        }
    }

    /// Coriolis acceleration [m/s²]: `a = −2 ω × v`.
    pub fn coriolis_accel(&self, velocity_enu: Vec3) -> Vec3 {
        (self.omega_enu.cross(velocity_enu)) * (-2.0)
    }

    /// Eötvös correction to effective gravity [m/s²] (vertical component).
    ///
    /// For eastward velocity `v_e` [m/s] at latitude φ:
    /// `Δg ≈ 2 ω v_e cos φ` (sign: eastward motion reduces apparent weight).
    pub fn eotvos_dg(&self, velocity_enu: Vec3) -> f64 {
        2.0 * EARTH_OMEGA * velocity_enu.x * libm_ops::cos(self.latitude_rad)
    }

    /// Gravity acceleration vector [m/s²] including altitude and Eötvös.
    pub fn gravity_accel(&self, altitude_m: f64, velocity_enu: Vec3) -> Vec3 {
        let g = gravity_magnitude(altitude_m) - self.eotvos_dg(velocity_enu);
        Vec3::new(0.0, 0.0, -g)
    }
}

/// `g(h) = g0 (R/(R+h))²` [m/s²] for altitude `h` [m] ≥ −R.
pub fn gravity_magnitude(altitude_m: f64) -> f64 {
    let r = EARTH_RADIUS_M;
    let h = altitude_m.max(-0.9 * r);
    STANDARD_GRAVITY * (r / (r + h)) * (r / (r + h))
}

/// Free-fall eastward Coriolis deflection [m] in the small-time limit:
/// `δ ≈ (1/3) ω g t³ cos(φ)` (McCoy / classical result).
pub fn freefall_eastward_deflection(t: f64, latitude_rad: f64) -> f64 {
    (1.0 / 3.0) * EARTH_OMEGA * STANDARD_GRAVITY * t * t * t * libm_ops::cos(latitude_rad)
}
