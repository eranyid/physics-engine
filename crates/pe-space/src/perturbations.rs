//! Orbital perturbations: J₂ zonal, cannonball SRP, third-body point mass,
//! and exponential-atmosphere drag (ISA density via `pe-aero` when applicable).
//!
//! Accelerations are returned in the same inertial frame as the state [m/s²].
//! J₂ uses the classic first-zonal formula in a planet-fixed equatorial frame
//! identified with inertial `(x,y,z)` (no Earth rotation / sidereal time yet).

use pe_aero::atmosphere::{density as isa_density, H_STRATO_TOP};
use pe_math::libm_ops;
use pe_math::Vec3;

use crate::constants::{EARTH_J2, EARTH_MU, EARTH_RADIUS};
use crate::state::NBodyState;

/// J₂ zonal harmonic model parameters.
#[derive(Clone, Copy, Debug)]
pub struct J2Model {
    /// Gravitational parameter μ [m³/s²].
    pub mu: f64,
    /// Reference equatorial radius R [m].
    pub radius: f64,
    /// J₂ coefficient [dimensionless].
    pub j2: f64,
}

impl Default for J2Model {
    fn default() -> Self {
        Self {
            mu: EARTH_MU,
            radius: EARTH_RADIUS,
            j2: EARTH_J2,
        }
    }
}

impl J2Model {
    /// Earth WGS84-class defaults used in M7 tests.
    pub fn earth() -> Self {
        Self::default()
    }
}

/// Acceleration due to J₂ on a spacecraft at `r` [m] (planet at origin).
///
/// ```text
/// a = −(μ/r³) [ x (1 + 1.5 J₂ (R/r)² (1 − 5 z²/r²)),
///               y (… same …),
///               z (1 + 1.5 J₂ (R/r)² (3 − 5 z²/r²)) ]
/// ```
/// (Vallado / standard first-zonal form.)
pub fn accel_j2(model: &J2Model, r: Vec3) -> Vec3 {
    let r2 = r.norm_squared();
    if r2 == 0.0 {
        return Vec3::zero();
    }
    let rn = libm_ops::sqrt(r2);
    let z2r2 = (r.z * r.z) / r2;
    let k = 1.5 * model.j2 * (model.radius / rn) * (model.radius / rn);
    let factor = -model.mu / (r2 * rn);
    Vec3::new(
        factor * r.x * (1.0 + k * (1.0 - 5.0 * z2r2)),
        factor * r.y * (1.0 + k * (1.0 - 5.0 * z2r2)),
        factor * r.z * (1.0 + k * (3.0 - 5.0 * z2r2)),
    )
}

/// Analytic mean nodal regression rate [rad/s] for near-circular orbits:
/// `Ω̇ = −1.5 n J₂ (R/p)² cos i` with `n = √(μ/a³)`, `p = a(1−e²)`.
pub fn j2_nodal_rate(model: &J2Model, a: f64, e: f64, i: f64) -> f64 {
    let n = libm_ops::sqrt(model.mu / (a * a * a));
    let p = a * (1.0 - e * e);
    -1.5 * n * model.j2 * (model.radius / p) * (model.radius / p) * libm_ops::cos(i)
}

/// Cannonball solar-radiation pressure model.
#[derive(Clone, Copy, Debug)]
pub struct SrpModel {
    /// Solar flux at 1 AU [W/m²] (≈ 1367; unverified precision beyond that).
    pub flux_w_m2: f64,
    /// Reflectivity coefficient CR [dimensionless] (1 = absorb, 2 = perfect reflect).
    pub cr: f64,
    /// Area / mass [m²/kg].
    pub area_over_mass: f64,
    /// 1 AU [m].
    pub au_m: f64,
}

impl Default for SrpModel {
    fn default() -> Self {
        Self {
            flux_w_m2: 1367.0,
            cr: 1.0,
            area_over_mass: 0.01,
            au_m: 149_597_870_700.0,
        }
    }
}

/// Cannonball SRP acceleration [m/s²].
///
/// `r_sat_wrt_sun` points from the Sun to the satellite [m]. Shadowing is **not**
/// modelled (always illuminated).
pub fn accel_cannonball_srp(model: &SrpModel, r_sat_wrt_sun: Vec3) -> Vec3 {
    let r2 = r_sat_wrt_sun.norm_squared();
    if r2 == 0.0 {
        return Vec3::zero();
    }
    let r = libm_ops::sqrt(r2);
    // P = flux/c ; c = 299792458 m/s
    let p0 = model.flux_w_m2 / 299_792_458.0;
    let scale = p0 * model.cr * model.area_over_mass * (model.au_m / r) * (model.au_m / r);
    // Acceleration away from the Sun: along +r_sat_wrt_sun
    r_sat_wrt_sun * (scale / r)
}

/// Point-mass third body (e.g. Moon / Sun) acting on a satellite about a primary.
#[derive(Clone, Copy, Debug)]
pub struct ThirdBody {
    /// Third-body μ [m³/s²].
    pub mu: f64,
}

/// Third-body acceleration on the satellite in the primary-centred frame:
/// `a = μ' ( (r'−r)/|r'−r|³ − r'/|r'|³ )` where `r` is sat wrt primary and
/// `r_third` is third body wrt primary.
pub fn accel_third_body(model: &ThirdBody, r_sat: Vec3, r_third: Vec3) -> Vec3 {
    let d = r_third - r_sat;
    let d2 = d.norm_squared();
    let t2 = r_third.norm_squared();
    if d2 == 0.0 || t2 == 0.0 {
        return Vec3::zero();
    }
    let inv_d3 = 1.0 / (d2 * libm_ops::sqrt(d2));
    let inv_t3 = 1.0 / (t2 * libm_ops::sqrt(t2));
    d * (model.mu * inv_d3) - r_third * (model.mu * inv_t3)
}

/// Exponential / ISA drag model for a cannonball spacecraft.
#[derive(Clone, Debug)]
pub struct DragModel {
    /// Drag coefficient CD [dimensionless].
    pub cd: f64,
    /// Area [m²].
    pub area: f64,
    /// Spacecraft mass [kg].
    pub mass: f64,
    /// Planet radius for altitude `h = |r| − R` [m].
    pub planet_radius: f64,
    /// If true, use `pe-aero` ISA density for `h ∈ [0, 20 km]`; above that, or on
    /// ISA error, fall back to the exponential model.
    pub use_isa_when_applicable: bool,
    /// Exponential reference density [kg/m³] at `h_ref`.
    pub rho0: f64,
    /// Reference altitude [m].
    pub h_ref: f64,
    /// Scale height [m].
    pub scale_height: f64,
    /// Atmosphere co-rotation angular velocity [rad/s] about +z (0 = inertially still).
    pub omega_atm: f64,
}

impl Default for DragModel {
    fn default() -> Self {
        Self {
            cd: 2.2,
            area: 1.0,
            mass: 100.0,
            planet_radius: EARTH_RADIUS,
            use_isa_when_applicable: true,
            rho0: 1.225,
            h_ref: 0.0,
            scale_height: 8500.0,
            omega_atm: 0.0,
        }
    }
}

fn density_at(model: &DragModel, h: f64) -> f64 {
    if h < 0.0 {
        return model.rho0;
    }
    if model.use_isa_when_applicable && h <= H_STRATO_TOP {
        if let Ok(rho) = isa_density(h) {
            return rho;
        }
    }
    model.rho0 * libm_ops::exp(-(h - model.h_ref) / model.scale_height)
}

/// Drag acceleration [m/s²] for body `i` (cannonball, relative to co-rotating air).
pub fn accel_drag_exponential(model: &DragModel, state: &NBodyState, i: usize) -> Vec3 {
    let r = state.x[i];
    let rn = r.norm();
    let h = rn - model.planet_radius;
    let rho = density_at(model, h);
    let v_atm = Vec3::new(-model.omega_atm * r.y, model.omega_atm * r.x, 0.0);
    let v_rel = state.v[i] - v_atm;
    let spd = v_rel.norm();
    if spd == 0.0 || model.mass == 0.0 {
        return Vec3::zero();
    }
    let coeff = -0.5 * rho * model.cd * model.area / model.mass * spd;
    v_rel * coeff
}

/// Convenience: J₂ acceleration for body 0 of an Earth-centred state.
pub fn accel_j2_body0(state: &NBodyState) -> Vec3 {
    if state.n() == 0 {
        Vec3::zero()
    } else {
        accel_j2(&J2Model::earth(), state.x[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn j2_equatorial_has_no_z_pref_at_xy() {
        // On the equator z=0, az formula still applies; ax,ay reduced vs point mass.
        let r = Vec3::new(6_778_000.0, 0.0, 0.0);
        let a = accel_j2(&J2Model::earth(), r);
        assert!(a.x < 0.0);
        assert_eq!(a.y, 0.0);
        assert_eq!(a.z, 0.0);
    }

    #[test]
    fn nodal_rate_formula_sso_ballpark() {
        let model = J2Model::earth();
        let a = EARTH_RADIUS + 800_000.0;
        // Solve for i such that Ω̇ = +0.9856 deg/day
        let want = 0.9856 * core::f64::consts::PI / 180.0 / 86_400.0;
        let n = libm_ops::sqrt(model.mu / (a * a * a));
        let factor = -1.5 * n * model.j2 * (model.radius / a) * (model.radius / a);
        let cosi = want / factor;
        let i = libm_ops::acos(cosi);
        assert_relative_eq!(i.to_degrees(), 98.6, max_relative = 5e-3);
        let rate = j2_nodal_rate(&model, a, 0.0, i);
        assert_relative_eq!(rate, want, max_relative = 1e-10);
    }
}
