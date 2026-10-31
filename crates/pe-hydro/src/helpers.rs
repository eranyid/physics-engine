//! Analytic hydrostatic helpers + IMO IS Code (MSC.267(85)) intact-stability checker.
//!
//! Wave kinematics live in [`crate::waves`]. Units are SI throughout.

use pe_math::libm_ops;
use pe_math::{Quat, Vec3, STANDARD_GRAVITY};

use crate::clip::convex_mesh_submerged;
use crate::hull::BoxHull;

/// Metacentric height [m]: `GM = KB + BM − KG` (all [m]).
pub fn metacentric_height(kb: f64, bm: f64, kg: f64) -> f64 {
    kb + bm - kg
}

/// Metacentric radius [m]: `BM = I_wp / ∇` (`I_wp` [m⁴], `∇` [m³] > 0).
pub fn bm_from_inertia(i_wp: f64, volume: f64) -> f64 {
    assert!(volume > 0.0, "volume must be positive");
    i_wp / volume
}

/// Small-angle righting arm [m]: `GZ = GM sin φ` (`φ` [rad]).
pub fn righting_arm(gm: f64, phi: f64) -> f64 {
    gm * libm_ops::sin(phi)
}

/// Righting moment [N m]: `ρ g ∇ GZ` (`ρ` [kg/m³], `∇` [m³], `GZ` [m], `g` [m/s²]).
pub fn righting_moment(rho: f64, volume: f64, gz: f64, g: f64) -> f64 {
    rho * g * volume * gz
}

/// Heave natural period [s]: `T = 2π √(m / (ρ g A_wp))` (no added mass).
///
/// `mass` [kg], `rho` [kg/m³], `A_wp` [m²], `g` [m/s²].
pub fn heave_period(mass: f64, rho: f64, a_wp: f64, g: f64) -> f64 {
    heave_period_with_added_mass(mass, 0.0, rho, a_wp, g)
}

/// Heave natural period [s]: `T = 2π √((m + a₃₃) / (ρ g A_wp))`.
///
/// `mass` [kg], `a33` heave added mass [kg], `rho` [kg/m³], `A_wp` [m²], `g` [m/s²].
pub fn heave_period_with_added_mass(mass: f64, a33: f64, rho: f64, a_wp: f64, g: f64) -> f64 {
    2.0 * core::f64::consts::PI * libm_ops::sqrt((mass + a33) / (rho * g * a_wp))
}

/// Roll natural period [s]: `T = 2π k / √(g GM)`.
///
/// `k` [m] radius of gyration, `GM` [m] > 0, `g` [m/s²].
pub fn roll_period(k: f64, gm: f64, g: f64) -> f64 {
    assert!(gm > 0.0, "GM must be positive for a stable roll period");
    2.0 * core::f64::consts::PI * k / libm_ops::sqrt(g * gm)
}

/// Large-angle GZ [m] for a rectangular barge at heel angles `phis` [rad].
///
/// Body origin = COM. Positive GZ is restoring. `draft` [m], `KG` [m] from keel.
pub fn box_righting_arm_curve(hull: &BoxHull, draft: f64, kg: f64, phis: &[f64]) -> Vec<f64> {
    let d = hull.depth;
    let z_com_geom = -0.5 * d + kg;
    let (corners, faces) = hull.mesh_topology(-z_com_geom);
    // mesh_topology applies z_shift to corners; we want corners[:,2] -= z_com_geom
    // which is z_shift = -z_com_geom. Good.
    let mut gz = Vec::with_capacity(phis.len());
    for &phi in phis {
        let q = Quat::from_axis_angle(Vec3::unit_x(), phi);
        let x = Vec3::new(0.0, 0.0, -draft + kg);
        let sub = convex_mesh_submerged(&corners, &faces, x, q, 0.0);
        if sub.volume <= 0.0 {
            gz.push(0.0);
            continue;
        }
        let g_body = q.inverse_rotate(Vec3::new(0.0, 0.0, -1.0));
        let cob_com = sub.cob_body;
        gz.push(cob_com.cross(g_body).x);
    }
    gz
}

/// Smallest `φ > 0` [rad] at which GZ crosses from positive to non-positive.
pub fn angle_of_vanishing_stability(
    hull: &BoxHull,
    draft: f64,
    kg: f64,
    phi_max: f64,
    n: usize,
) -> f64 {
    let mut phis = Vec::with_capacity(n);
    for i in 0..n {
        phis.push(phi_max * (i as f64) / ((n - 1) as f64).max(1.0));
    }
    let gz = box_righting_arm_curve(hull, draft, kg, &phis);
    for i in 1..phis.len() {
        if gz[i - 1] > 0.0 && gz[i] <= 0.0 {
            let t = gz[i - 1] / (gz[i - 1] - gz[i] + 1e-30);
            return phis[i - 1] + t * (phis[i] - phis[i - 1]);
        }
    }
    phi_max
}

/// Optional heave added mass [kg] ≈ `½ ρ A_wp min(draft, √A_wp)`.
pub fn box_added_mass_heave(rho: f64, a_wp: f64, draft: f64) -> f64 {
    0.5 * rho * a_wp * draft.min(libm_ops::sqrt(a_wp.max(0.0)))
}

/// Optional surge added mass [kg] ≈ `C_x ρ ∇`.
pub fn box_added_mass_surge(rho: f64, volume: f64, cx: f64) -> f64 {
    cx * rho * volume
}

// ----- IMO IS Code MSC.267(85) intact stability (pure function over a GZ curve) -----

/// Result of an IMO IS Code (MSC.267(85)) intact-stability check.
#[derive(Clone, Debug, PartialEq)]
pub struct IsCodeResult {
    /// Overall pass/fail.
    pub passed: bool,
    /// Area under GZ from 0 to 30° [m·rad]; criterion ≥ 0.055.
    pub area_0_30: f64,
    /// Area under GZ from 0 to 40° [m·rad]; criterion ≥ 0.09.
    pub area_0_40: f64,
    /// Area under GZ from 30° to 40° [m·rad]; criterion ≥ 0.03.
    pub area_30_40: f64,
    /// Max GZ in the range φ ≥ 30° [m]; criterion ≥ 0.20.
    pub gz_at_or_above_30: f64,
    /// Heel angle of maximum GZ [rad]; criterion ≥ 25°.
    pub angle_of_max_gz: f64,
    /// Initial metacentric height GM₀ [m]; criterion ≥ 0.15.
    pub gm0: f64,
    /// Human-readable failing criteria (empty if passed).
    pub failures: Vec<&'static str>,
}

/// Trapezoidal integral of `y(x)` over samples (`x` [rad], `y` [m]) → [m·rad].
fn trapz(x: &[f64], y: &[f64], x_lo: f64, x_hi: f64) -> f64 {
    if x.is_empty() || x_lo >= x_hi {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..x.len() - 1 {
        let (x0, x1) = (x[i], x[i + 1]);
        let (y0, y1) = (y[i], y[i + 1]);
        if x1 <= x_lo || x0 >= x_hi {
            continue;
        }
        let a = x0.max(x_lo);
        let b = x1.min(x_hi);
        if b <= a {
            continue;
        }
        // Linear interpolate y at a, b.
        let t0 = if (x1 - x0).abs() < 1e-30 {
            0.0
        } else {
            (a - x0) / (x1 - x0)
        };
        let t1 = if (x1 - x0).abs() < 1e-30 {
            0.0
        } else {
            (b - x0) / (x1 - x0)
        };
        let ya = y0 + t0 * (y1 - y0);
        let yb = y0 + t1 * (y1 - y0);
        area += 0.5 * (ya + yb) * (b - a);
    }
    area
}

/// Interpolate GZ [m] at angle `phi` [rad] from a sampled curve.
fn interp_gz(phis: &[f64], gz: &[f64], phi: f64) -> f64 {
    if phis.is_empty() {
        return 0.0;
    }
    if phi <= phis[0] {
        return gz[0];
    }
    if phi >= phis[phis.len() - 1] {
        return gz[gz.len() - 1];
    }
    for i in 0..phis.len() - 1 {
        if phis[i] <= phi && phi <= phis[i + 1] {
            let t = (phi - phis[i]) / (phis[i + 1] - phis[i] + 1e-30);
            return gz[i] + t * (gz[i + 1] - gz[i]);
        }
    }
    gz[gz.len() - 1]
}

/// IMO IS Code MSC.267(85) intact stability criteria over a GZ curve.
///
/// Inputs: heel samples `phis` [rad] (ascending, covering at least 0..40°),
/// corresponding `gz` [m], and initial `gm0` [m].
///
/// Criteria (all must pass):
/// - area 0→30° ≥ 0.055 m·rad
/// - area 0→40° ≥ 0.09 m·rad
/// - area 30→40° ≥ 0.03 m·rad
/// - GZ ≥ 0.20 m at some φ ≥ 30°
/// - angle of max GZ ≥ 25°
/// - GM₀ ≥ 0.15 m
pub fn check_imo_is_code(phis: &[f64], gz: &[f64], gm0: f64) -> IsCodeResult {
    assert_eq!(phis.len(), gz.len());
    let deg30 = 30.0_f64.to_radians();
    let deg40 = 40.0_f64.to_radians();
    let deg25 = 25.0_f64.to_radians();

    let area_0_30 = trapz(phis, gz, 0.0, deg30);
    let area_0_40 = trapz(phis, gz, 0.0, deg40);
    let area_30_40 = trapz(phis, gz, deg30, deg40);

    let mut gz_at_or_above_30: f64 = 0.0;
    let mut angle_of_max_gz = 0.0;
    let mut max_gz = f64::NEG_INFINITY;
    for i in 0..phis.len() {
        if phis[i] >= deg30 {
            gz_at_or_above_30 = gz_at_or_above_30.max(gz[i]);
        }
        if gz[i] > max_gz {
            max_gz = gz[i];
            angle_of_max_gz = phis[i];
        }
    }
    // Also sample exactly at 30° if within range.
    gz_at_or_above_30 = gz_at_or_above_30.max(interp_gz(phis, gz, deg30));

    let mut failures = Vec::new();
    if area_0_30 < 0.055 {
        failures.push("area_0_30 < 0.055 m·rad");
    }
    if area_0_40 < 0.09 {
        failures.push("area_0_40 < 0.09 m·rad");
    }
    if area_30_40 < 0.03 {
        failures.push("area_30_40 < 0.03 m·rad");
    }
    if gz_at_or_above_30 < 0.20 {
        failures.push("GZ < 0.20 m for all φ ≥ 30°");
    }
    if angle_of_max_gz < deg25 {
        failures.push("angle of max GZ < 25°");
    }
    if gm0 < 0.15 {
        failures.push("GM0 < 0.15 m");
    }

    IsCodeResult {
        passed: failures.is_empty(),
        area_0_30,
        area_0_40,
        area_30_40,
        gz_at_or_above_30,
        angle_of_max_gz,
        gm0,
        failures,
    }
}

/// Earth-g heave period convenience.
pub fn heave_period_earth(mass: f64, rho: f64, a_wp: f64) -> f64 {
    heave_period(mass, rho, a_wp, STANDARD_GRAVITY)
}

/// Earth-g roll period convenience.
pub fn roll_period_earth(k: f64, gm: f64) -> f64 {
    roll_period(k, gm, STANDARD_GRAVITY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn is_code_pass_synthetic() {
        // Construct a generous GZ curve that clearly passes all criteria.
        let n = 81;
        let mut phis = Vec::with_capacity(n);
        let mut gz = Vec::with_capacity(n);
        for i in 0..n {
            let phi = 50.0_f64.to_radians() * (i as f64) / ((n - 1) as f64);
            phis.push(phi);
            // GZ = 0.45 sin φ peaks past 25° and clears area / 0.20 m criteria.
            gz.push(0.45 * libm_ops::sin(phi));
        }
        let r = check_imo_is_code(&phis, &gz, 0.5);
        assert!(r.passed, "failures: {:?}", r.failures);
    }

    #[test]
    fn is_code_fail_low_gm_and_flat_gz() {
        let phis: Vec<f64> = (0..41).map(|i| (i as f64).to_radians()).collect();
        let gz: Vec<f64> = phis.iter().map(|&p| 0.05 * libm_ops::sin(p)).collect();
        let r = check_imo_is_code(&phis, &gz, 0.05);
        assert!(!r.passed);
        assert!(r.failures.len() >= 2);
    }

    #[test]
    fn roll_period_formula() {
        let hull = BoxHull::new(30.0, 8.0, 4.0);
        let draft = 2.0;
        let kg = 1.5;
        let gm = hull.upright_gm(draft, kg, true);
        let k = libm_ops::sqrt((hull.beam * hull.beam + hull.depth * hull.depth) / 12.0);
        let t = roll_period_earth(k, gm);
        assert_relative_eq!(
            t,
            2.0 * core::f64::consts::PI * k / libm_ops::sqrt(STANDARD_GRAVITY * gm),
            max_relative = 1e-15
        );
        assert!(t > 0.0);
    }
}
