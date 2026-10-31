//! Analytic oracles for fluid validation (clean-room textbook formulas).
//!
//! # Ritter (1892) dry-bed dam-break
//!
//! Ideal frictionless shallow-water dam-break with still water depth `h0` for
//! `x < 0` and dry bed for `x > 0`:
//!
//! - `c0 = √(g h0)`
//! - Front position: `x_f(t) = 2 c0 t`
//! - For `−c0 t ≤ x ≤ 2 c0 t`: `u = (2/3)(x/t + c0)`, `c = (1/3)(2 c0 − x/t)`,
//!   `h = c²/g`
//!
//! # References
//! - Ritter, A. (1892). Die Fortpflanzung der Wasserwellen. *Z. Verein. Deutsch.
//!   Ing.* **36**(33), 947–954.
//! - Stoker, J. J. (1957). *Water Waves*. (textbook form of Ritter solution).
//!
//! Martin & Moyce (1952) experimental dam-break tables are **not** transcribed
//! here (no verified primary table in-tree) — use Ritter for the analytic check.
//!
//! # Hydrostatics
//!
//! `p = ρ g h` with `h` the depth below the free surface.
//!
//! # Poiseuille (2-D channel)
//!
//! Steady laminar flow between plates at `y = 0` and `y = H` driven by constant
//! body force `f` [m/s²] in +x with kinematic viscosity `ν`:
//! `u(y) = (f/(2ν)) y (H − y)`.
//!
//! Couette: top wall at `U`, no pressure gradient: `u(y) = U y / H`.

use pe_math::libm_ops::sqrt;

/// Ritter wave celerity `c0 = √(g h0)` [m/s].
#[inline]
pub fn ritter_c0(g: f64, h0: f64) -> f64 {
    sqrt(g * h0)
}

/// Ritter wet–dry front position `x_f = 2 c0 t` [m].
#[inline]
pub fn ritter_front(g: f64, h0: f64, t: f64) -> f64 {
    2.0 * ritter_c0(g, h0) * t
}

/// Ritter height [m] at position `x` and time `t` (0 outside the rarefaction).
pub fn ritter_height(g: f64, h0: f64, x: f64, t: f64) -> f64 {
    if t <= 0.0 {
        return if x < 0.0 { h0 } else { 0.0 };
    }
    let c0 = ritter_c0(g, h0);
    let x_l = -c0 * t;
    let x_f = 2.0 * c0 * t;
    if x <= x_l {
        h0
    } else if x >= x_f {
        0.0
    } else {
        let c = (2.0 * c0 - x / t) / 3.0;
        (c * c) / g
    }
}

/// Ritter velocity [m/s] at `(x, t)`.
pub fn ritter_velocity(g: f64, h0: f64, x: f64, t: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let c0 = ritter_c0(g, h0);
    let x_l = -c0 * t;
    let x_f = 2.0 * c0 * t;
    if x <= x_l || x >= x_f {
        0.0
    } else {
        (2.0 / 3.0) * (x / t + c0)
    }
}

/// Hydrostatic pressure [Pa]: `p = ρ g h`.
#[inline]
pub fn hydrostatic_pressure(rho: f64, g: f64, depth: f64) -> f64 {
    rho * g * depth
}

/// Poiseuille plane-channel profile `u(y)` [m/s].
#[inline]
pub fn poiseuille_u(f: f64, nu: f64, y: f64, height: f64) -> f64 {
    (f / (2.0 * nu)) * y * (height - y)
}

/// Couette profile `u(y) = U y / H`.
#[inline]
pub fn couette_u(u_top: f64, y: f64, height: f64) -> f64 {
    u_top * y / height
}
