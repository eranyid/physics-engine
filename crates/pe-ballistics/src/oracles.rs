//! Closed-form external-ballistics oracles for validation.
//!
//! All SI. Used by tests and docs — not approximate numerical trajectories.

use pe_math::libm_ops;
use pe_math::STANDARD_GRAVITY;

use crate::earth::freefall_eastward_deflection;

/// Vacuum projectile range [m]: `R = v² sin(2θ) / g` on flat ground.
pub fn vacuum_range(v0: f64, theta_rad: f64, g: f64) -> f64 {
    v0 * v0 * libm_ops::sin(2.0 * theta_rad) / g.max(1e-30)
}

/// Vacuum max height [m]: `H = v² sin²θ / (2g)`.
pub fn vacuum_max_height(v0: f64, theta_rad: f64, g: f64) -> f64 {
    let s = libm_ops::sin(theta_rad);
    v0 * v0 * s * s / (2.0 * g.max(1e-30))
}

/// Linear-drag horizontal velocity [m/s]: `v_x = v0x e^{−(k/m) t}`.
pub fn linear_drag_vx(v0x: f64, k_over_m: f64, t: f64) -> f64 {
    v0x * libm_ops::exp(-k_over_m * t)
}

/// Linear-drag vertical velocity under gravity [m/s]:
/// `v_z = −(mg/k) + (v0z + mg/k) e^{−(k/m)t}` with downward −z.
pub fn linear_drag_vz(v0z: f64, k: f64, mass: f64, g: f64, t: f64) -> f64 {
    let terminal = -mass * g / k.max(1e-30);
    terminal + (v0z - terminal) * libm_ops::exp(-(k / mass) * t)
}

/// Quadratic-drag terminal speed [m/s]: `√(m g / k)` with `F = −k |v| v`.
pub fn quadratic_terminal_speed(mass: f64, k: f64, g: f64) -> f64 {
    libm_ops::sqrt(mass * g / k.max(1e-30))
}

/// Coriolis free-fall eastward deflection oracle [m].
pub fn coriolis_freefall_east(t: f64, latitude_rad: f64) -> f64 {
    freefall_eastward_deflection(t, latitude_rad)
}

/// Energy balance: `ΔKE + ΔPE + W_drag = 0` residual for a path sample.
///
/// `W_drag` is positive work done **against** the body (energy dissipated).
pub fn energy_residual(ke0: f64, pe0: f64, ke1: f64, pe1: f64, drag_work_j: f64) -> f64 {
    (ke1 + pe1) - (ke0 + pe0) + drag_work_j
}

/// Flat-ground vacuum flight time [s]: `T = 2 v sinθ / g`.
pub fn vacuum_flight_time(v0: f64, theta_rad: f64, g: f64) -> f64 {
    2.0 * v0 * libm_ops::sin(theta_rad) / g.max(1e-30)
}

/// Default `g` [m/s²].
pub fn g0() -> f64 {
    STANDARD_GRAVITY
}
