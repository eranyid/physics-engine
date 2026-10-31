//! Spin drift (gyroscopic) and Magnus force — simple documented models.
//!
//! # References
//! - McCoy, R. L. *Modern Exterior Ballistics* — engineering spin-drift /
//!   Magnus approximations for point-mass + spin.
//! - These are **first-order game/sim models**, not 6-DOF aeroballistics.

use pe_math::libm_ops;
use pe_math::Vec3;

/// Spin-drift lateral acceleration scale.
///
/// Engineering model: horizontal drift rate accumulates roughly
/// `a_y ≈ k_sd · ω_spin · (v × g_hat)_horizontal_hat` style coupling.
/// Here we use a simple McCoy-like form:
/// `a_drift = k · (ω_z / |ω|_ref) · |v_h|` in the direction of `v × z` projected
/// horizontal — for small angles, right-hand twist drifts right.
///
/// - `k_sd` [1/s] caller-tuned (default ~1e-4…1e-3 for rifle-like).
/// - `spin_rad_s` [rad/s] (positive = right-hand when looking downrange).
/// - `velocity` [m/s] world/ENU.
///
/// Returns acceleration [m/s²].
pub fn spin_drift_accel(velocity: Vec3, spin_rad_s: f64, k_sd: f64) -> Vec3 {
    let vh = Vec3::new(velocity.x, velocity.y, 0.0);
    let speed_h = vh.norm();
    if speed_h < 1e-9 {
        return Vec3::zero();
    }
    // Horizontal perpendicular to velocity (right of track for +z-up, +spin).
    let right = Vec3::new(-vh.y, vh.x, 0.0).normalize();
    right * (k_sd * spin_rad_s.signum() * speed_h * libm_ops::fabs(spin_rad_s).min(1e5) * 1e-5)
}

/// Magnus acceleration [m/s²]: `a = (k_m / m) · (ω × v) · ρ` engineering form
/// reduced to `a = k_magnus · (ω_hat × v) · ρ · |ω|`.
///
/// - `k_magnus` [m³/kg] · style combined coefficient (caller-tuned).
/// - `spin_axis` unit axis of rotation (often ≈ velocity direction).
/// - `spin_rad_s` [rad/s], `velocity` [m/s], `rho` [kg/m³].
pub fn magnus_accel(
    velocity: Vec3,
    spin_axis: Vec3,
    spin_rad_s: f64,
    rho: f64,
    k_magnus: f64,
) -> Vec3 {
    let axis = spin_axis.normalize();
    if axis.norm_squared() < 1e-20 {
        return Vec3::zero();
    }
    axis.cross(velocity) * (k_magnus * rho * spin_rad_s)
}

/// Default spin-drift coefficient [dimensionless scale] for rifle-like sims.
pub const DEFAULT_SPIN_DRIFT_K: f64 = 1.0;
/// Default Magnus coefficient [m³/kg] engineering scale (small).
pub const DEFAULT_MAGNUS_K: f64 = 1e-5;
