//! Steady resistance: ITTC-1957 friction line and simple residual models.
//!
//! # ITTC-1957 (verified formula)
//!
//! The International Towing Tank Conference 1957 ship-model correlation line:
//!
//! ```text
//! C_f = 0.075 / (log₁₀(R_n) − 2)²
//! ```
//!
//! with Reynolds number `R_n = V L / ν` (length-based). This closed form is
//! the published ITTC-57 definition (ITTC Recommended Procedures; also
//! Newman / Faltinsen textbook statements of the line). Implemented exactly.
//!
//! # Residual / wave resistance
//!
//! A caller-supplied residual coefficient `C_r` (or a Froude-linear surrogate)
//! may be added: `C_t = C_f + C_r`. **Holtrop–Mennen** (1982) full resistance
//! decomposition is **intentionally unimplemented** — the empirical coefficient
//! tables were not verified against a primary/reproduction source accessible
//! for transcription in this tree (M23). Any residual defaults below are marked
//! **unverified**.
//!
//! # References
//! - ITTC Recommended Procedures — 1957 ship-model correlation line.
//! - Freshwater / seawater kinematic viscosity: standard engineering defaults
//!   (20 °C freshwater `ν ≈ 1.004e-6` m²/s) — **nominal** unless overridden.

use pe_math::libm_ops;

/// Nominal freshwater kinematic viscosity at ~20 °C [m²/s] (**unverified**
/// handbook order; override for exact Oracle work).
pub const NU_FRESHWATER_20C_NOMINAL: f64 = 1.004e-6;

/// Nominal seawater kinematic viscosity at ~15 °C [m²/s] (**unverified**).
pub const NU_SEAWATER_15C_NOMINAL: f64 = 1.19e-6;

/// ITTC-1957 frictional resistance coefficient [dimensionless].
///
/// `C_f = 0.075 / (log₁₀(R_n) − 2)²` for `R_n > 10²` (the formula domain;
/// returns `0` for non-positive speed / length / viscosity).
///
/// # Panics
/// None — out-of-domain inputs return `0`.
pub fn ittc57_cf(reynolds: f64) -> f64 {
    if reynolds <= 100.0 {
        return 0.0;
    }
    let log10_rn = libm_ops::log(reynolds) / libm_ops::log(10.0);
    let den = log10_rn - 2.0;
    if libm_ops::fabs(den) < 1e-15 {
        return 0.0;
    }
    0.075 / (den * den)
}

/// Length Reynolds number `R_n = V L / ν`.
pub fn reynolds_number(speed: f64, length: f64, nu: f64) -> f64 {
    if speed <= 0.0 || length <= 0.0 || nu <= 0.0 {
        return 0.0;
    }
    speed * length / nu
}

/// ITTC-57 `C_f` from speed [m/s], length [m], kinematic viscosity [m²/s].
pub fn ittc57_cf_from_speed(speed: f64, length: f64, nu: f64) -> f64 {
    ittc57_cf(reynolds_number(speed, length, nu))
}

/// Frictional resistance [N]: `R_f = ½ ρ V² S C_f` with ITTC-57 `C_f`.
pub fn ittc57_friction_force(rho: f64, speed: f64, s_wetted: f64, length: f64, nu: f64) -> f64 {
    let cf = ittc57_cf_from_speed(speed, length, nu);
    0.5 * rho * speed * speed * s_wetted * cf
}

/// Total calm-water resistance [N] with ITTC-57 friction plus residual `C_r`.
///
/// `C_r` is caller-supplied (may be 0). Form factor `(1+k)` multiplies `C_f`
/// only when `form_factor_1_plus_k` is set; the ITTC correlation usually absorbs
/// form into model–ship extrapolation — treat `1+k ≠ 1` as an engineering
/// option (**document the source** if non-unity).
pub fn calm_water_resistance(
    rho: f64,
    speed: f64,
    s_wetted: f64,
    length: f64,
    nu: f64,
    c_r: f64,
    form_factor_1_plus_k: f64,
) -> f64 {
    let cf = ittc57_cf_from_speed(speed, length, nu);
    let ct = form_factor_1_plus_k * cf + c_r;
    0.5 * rho * speed * speed * s_wetted * ct
}

/// Froude number `Fr = V / √(g L)`.
pub fn froude_number(speed: f64, length: f64, g: f64) -> f64 {
    if speed <= 0.0 || length <= 0.0 || g <= 0.0 {
        return 0.0;
    }
    speed / libm_ops::sqrt(g * length)
}

/// **Unverified** residual coefficient surrogate `C_r = k_r · Fr` (not Holtrop).
///
/// Marked unverified: no primary Holtrop–Mennen / ITTC residual table is
/// transcribed in-tree. Use only as a tunable placeholder.
pub fn residual_cf_linear_froude_unverified(fr: f64, k_r: f64) -> f64 {
    k_r * fr.max(0.0)
}
