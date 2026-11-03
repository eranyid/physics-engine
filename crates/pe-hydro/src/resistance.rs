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
//! may be added: `C_t = C_f + C_r`. Full Holtrop–Mennen (1982) decomposition
//! lives in [`crate::holtrop`] and is verified against the published ISP §5
//! numerical example (M27). The linear Froude residual below remains marked
//! **unverified**.
//!
//! # References
//! - ITTC Recommended Procedures — 1957 ship-model correlation line.
//! - Freshwater / seawater kinematic viscosity: standard engineering defaults
//!   (20 °C freshwater `ν ≈ 1.004e-6` m²/s) — **nominal** unless overridden.

use pe_math::libm_ops;

use crate::error::{require_finite, require_non_negative, require_positive, HydroError};

/// Nominal freshwater kinematic viscosity at ~20 °C [m²/s] (**unverified**
/// handbook order; override for exact Oracle work).
pub const NU_FRESHWATER_20C_NOMINAL: f64 = 1.004e-6;

/// Nominal seawater kinematic viscosity at ~15 °C [m²/s] (**unverified**).
///
/// Holtrop & Mennen (1982) numerical example uses seawater at 15 °C with
/// density 1025 kg/m³; this nominal `ν` reproduces the published `C_F` to
/// ~0.001390 for that case.
pub const NU_SEAWATER_15C_NOMINAL: f64 = 1.19e-6;

/// ITTC-1957 frictional resistance coefficient [dimensionless].
///
/// `C_f = 0.075 / (log₁₀(R_n) − 2)²` for `R_n > 10²` (the formula domain;
/// returns `0` for non-positive speed / length / viscosity).
///
/// # Panics
/// None — out-of-domain inputs return `0`. Prefer [`try_ittc57_cf`] at boundaries.
pub fn ittc57_cf(reynolds: f64) -> f64 {
    try_ittc57_cf(reynolds).unwrap_or(0.0)
}

/// Fallible ITTC-57 `C_f` (`reynolds` finite; returns `0` for `R_n ≤ 100`).
pub fn try_ittc57_cf(reynolds: f64) -> Result<f64, HydroError> {
    require_finite(reynolds, "reynolds")?;
    if reynolds <= 100.0 {
        return Ok(0.0);
    }
    let log10_rn = libm_ops::log(reynolds) / libm_ops::log(10.0);
    let den = log10_rn - 2.0;
    if libm_ops::fabs(den) < 1e-15 {
        return Ok(0.0);
    }
    let cf = 0.075 / (den * den);
    require_finite(cf, "ittc57_cf")?;
    Ok(cf)
}

/// Length Reynolds number `R_n = V L / ν`.
pub fn reynolds_number(speed: f64, length: f64, nu: f64) -> f64 {
    try_reynolds_number(speed, length, nu).unwrap_or(0.0)
}

/// Fallible Reynolds number.
pub fn try_reynolds_number(speed: f64, length: f64, nu: f64) -> Result<f64, HydroError> {
    require_finite(speed, "speed")?;
    require_finite(length, "length")?;
    require_positive(nu, "nu")?;
    if speed <= 0.0 || length <= 0.0 {
        return Ok(0.0);
    }
    Ok(speed * length / nu)
}

/// ITTC-57 `C_f` from speed [m/s], length [m], kinematic viscosity [m²/s].
pub fn ittc57_cf_from_speed(speed: f64, length: f64, nu: f64) -> f64 {
    ittc57_cf(reynolds_number(speed, length, nu))
}

/// Fallible ITTC-57 `C_f` from speed / length / ν.
pub fn try_ittc57_cf_from_speed(speed: f64, length: f64, nu: f64) -> Result<f64, HydroError> {
    try_ittc57_cf(try_reynolds_number(speed, length, nu)?)
}

/// Frictional resistance [N]: `R_f = ½ ρ V² S C_f` with ITTC-57 `C_f`.
pub fn ittc57_friction_force(rho: f64, speed: f64, s_wetted: f64, length: f64, nu: f64) -> f64 {
    try_ittc57_friction_force(rho, speed, s_wetted, length, nu).unwrap_or(0.0)
}

/// Fallible ITTC-57 frictional force [N].
pub fn try_ittc57_friction_force(
    rho: f64,
    speed: f64,
    s_wetted: f64,
    length: f64,
    nu: f64,
) -> Result<f64, HydroError> {
    require_positive(rho, "rho")?;
    require_non_negative(speed, "speed")?;
    require_non_negative(s_wetted, "s_wetted")?;
    require_positive(length, "length")?;
    let cf = try_ittc57_cf_from_speed(speed, length, nu)?;
    let rf = 0.5 * rho * speed * speed * s_wetted * cf;
    require_finite(rf, "ittc57_friction_force")?;
    Ok(rf)
}

/// Total calm-water resistance [N] with ITTC-57 friction plus residual `C_r`.
///
/// `C_r` is caller-supplied (may be 0). Form factor `(1+k)` multiplies `C_f`
/// only when `form_factor_1_plus_k` is set; the ITTC correlation usually absorbs
/// form into model–ship extrapolation — treat `1+k ≠ 1` as an engineering
/// option (**document the source** if non-unity).
///
/// For Holtrop–Mennen full decomposition see [`crate::holtrop`].
pub fn calm_water_resistance(
    rho: f64,
    speed: f64,
    s_wetted: f64,
    length: f64,
    nu: f64,
    c_r: f64,
    form_factor_1_plus_k: f64,
) -> f64 {
    try_calm_water_resistance(rho, speed, s_wetted, length, nu, c_r, form_factor_1_plus_k)
        .unwrap_or(0.0)
}

/// Fallible ITTC + residual calm-water resistance [N].
pub fn try_calm_water_resistance(
    rho: f64,
    speed: f64,
    s_wetted: f64,
    length: f64,
    nu: f64,
    c_r: f64,
    form_factor_1_plus_k: f64,
) -> Result<f64, HydroError> {
    require_positive(rho, "rho")?;
    require_non_negative(speed, "speed")?;
    require_non_negative(s_wetted, "s_wetted")?;
    require_positive(length, "length")?;
    require_finite(c_r, "c_r")?;
    require_finite(form_factor_1_plus_k, "form_factor_1_plus_k")?;
    let cf = try_ittc57_cf_from_speed(speed, length, nu)?;
    let ct = form_factor_1_plus_k * cf + c_r;
    let r = 0.5 * rho * speed * speed * s_wetted * ct;
    require_finite(r, "calm_water_resistance")?;
    Ok(r)
}

/// Froude number `Fr = V / √(g L)`.
pub fn froude_number(speed: f64, length: f64, g: f64) -> f64 {
    try_froude_number(speed, length, g).unwrap_or(0.0)
}

/// Fallible Froude number.
pub fn try_froude_number(speed: f64, length: f64, g: f64) -> Result<f64, HydroError> {
    require_finite(speed, "speed")?;
    require_positive(length, "length")?;
    require_positive(g, "g")?;
    if speed <= 0.0 {
        return Ok(0.0);
    }
    Ok(speed / libm_ops::sqrt(g * length))
}

/// **Unverified** residual coefficient surrogate `C_r = k_r · Fr` (not Holtrop).
///
/// Tunable placeholder only — prefer [`crate::holtrop`] for verified Holtrop–
/// Mennen decomposition.
pub fn residual_cf_linear_froude_unverified(fr: f64, k_r: f64) -> f64 {
    k_r * fr.max(0.0)
}
