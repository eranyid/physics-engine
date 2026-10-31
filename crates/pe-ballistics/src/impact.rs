//! Terminal ballistics / impact helpers (engineering approximations).
//!
//! # Models
//! - Kinetic energy / momentum at impact.
//! - Impulse into a [`pe_core::World`] rigid body at the contact point.
//! - **Poncelet** penetration depth (caller-supplied material constants).
//! - **Recht–Ipson** residual velocity after perforating a plate.
//!
//! Clearly labeled engineering approximations — not validated armor design tools.

use pe_core::World;
use pe_math::libm_ops;
use pe_math::Vec3;

/// Kinetic energy [J]: `½ m |v|²`.
pub fn kinetic_energy(mass_kg: f64, velocity: Vec3) -> f64 {
    0.5 * mass_kg * velocity.norm_squared()
}

/// Linear momentum [kg·m/s]: `m v`.
pub fn momentum(mass_kg: f64, velocity: Vec3) -> Vec3 {
    velocity * mass_kg
}

/// Apply impact impulse to a world body: transfers fraction `e_transfer` of
/// incident momentum along the impact direction (into the body).
///
/// - `point` [m] world contact point
/// - `incident_velocity` [m/s] of the projectile
/// - `mass_kg` [kg]
/// - `e_transfer` ∈ [0,1] momentum transfer fraction
pub fn apply_impact_impulse(
    world: &mut World,
    body: usize,
    point: Vec3,
    mass_kg: f64,
    incident_velocity: Vec3,
    e_transfer: f64,
) {
    let j = incident_velocity * (mass_kg * e_transfer.clamp(0.0, 1.0));
    world.apply_impulse(body, j, point);
}

/// Poncelet penetration depth [m].
///
/// Engineering form: `m dv/dx = −A (a + b v²)` integrated from `v0` to 0:
/// `x = (m / (2 A b)) ln(1 + (b/a) v0²)` for `a,b > 0`.
///
/// - `mass_kg` [kg], `v0` [m/s], `area` [m²]
/// - `a` [Pa] (strength-like), `b` [kg/m³] (inertia-like) — **caller-supplied**
pub fn poncelet_depth(mass_kg: f64, v0: f64, area: f64, a: f64, b: f64) -> Option<f64> {
    if !(mass_kg > 0.0 && area > 0.0 && a > 0.0 && b > 0.0 && v0 >= 0.0) {
        return None;
    }
    Some((mass_kg / (2.0 * area * b)) * libm_ops::log(1.0 + (b / a) * v0 * v0))
}

/// Recht–Ipson residual velocity [m/s] after plate perforation.
///
/// `v_r = a · (v0^p − v_bl^p)^{1/p}` for `v0 > v_bl`, else 0.
/// Classic: `p = 2`, `a = m_p / (m_p + m_plug)` style — here `a` and `p` are
/// **caller-supplied** (Recht & Ipson, 1963).
///
/// - `v0` [m/s] impact speed, `v_bl` [m/s] ballistic limit
pub fn recht_ipson_residual(v0: f64, v_bl: f64, a: f64, p: f64) -> f64 {
    if v0 <= v_bl || p <= 0.0 {
        return 0.0;
    }
    let inside = libm_ops::pow(v0, p) - libm_ops::pow(v_bl, p);
    if inside <= 0.0 {
        return 0.0;
    }
    a * libm_ops::pow(inside, 1.0 / p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn poncelet_zero_velocity() {
        assert_relative_eq!(
            poncelet_depth(0.01, 0.0, 1e-4, 1e8, 1e3).unwrap(),
            0.0,
            epsilon = 0.0
        );
    }

    #[test]
    fn recht_ipson_below_limit() {
        assert_eq!(recht_ipson_residual(100.0, 200.0, 1.0, 2.0), 0.0);
    }

    #[test]
    fn recht_ipson_p2_closed_form() {
        let v0 = 500.0;
        let vbl = 300.0;
        let a = 0.8;
        let vr = recht_ipson_residual(v0, vbl, a, 2.0);
        let expect = a * libm_ops::sqrt(v0 * v0 - vbl * vbl);
        assert_relative_eq!(vr, expect, epsilon = 1e-12);
    }
}
