//! Portable elementary functions via the pure-Rust [`libm`] crate.
//!
//! The deterministic step path must not call platform `libm` / MSVCRT or rely on
//! FMA / `target-cpu=native`. Use these wrappers everywhere sin/cos/sqrt/exp
//! (and friends) appear in dynamics.

/// √x via pure-Rust libm [same units as x].
#[inline]
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

/// sin(x) with x in radians.
#[inline]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

/// cos(x) with x in radians.
#[inline]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

/// exp(x).
#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// tanh(x) (analytic closed forms / tests; also available for force models).
#[inline]
pub fn tanh(x: f64) -> f64 {
    libm::tanh(x)
}

/// ln(x) via pure-Rust libm.
#[inline]
pub fn log(x: f64) -> f64 {
    libm::log(x)
}

/// |x|.
#[inline]
pub fn fabs(x: f64) -> f64 {
    libm::fabs(x)
}

/// Hypotenuse √(x²+y²) without intermediate overflow where possible.
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    libm::hypot(x, y)
}

/// atan2(y, x) [rad].
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

/// copysign(x, y).
#[inline]
pub fn copysign(x: f64, y: f64) -> f64 {
    libm::copysign(x, y)
}

/// `x^y` via pure-Rust libm (used by jet thrust density scaling).
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

/// asin(x) [rad].
#[inline]
pub fn asin(x: f64) -> f64 {
    libm::asin(x)
}

/// acos(x) [rad].
#[inline]
pub fn acos(x: f64) -> f64 {
    libm::acos(x)
}

/// atan(x) [rad].
#[inline]
pub fn atan(x: f64) -> f64 {
    libm::atan(x)
}

/// tan(x) with x in radians.
#[inline]
pub fn tan(x: f64) -> f64 {
    libm::tan(x)
}

/// Cube root via pure-Rust libm.
#[inline]
pub fn cbrt(x: f64) -> f64 {
    libm::cbrt(x)
}

/// Floor to integer as f64.
#[inline]
pub fn floor(x: f64) -> f64 {
    libm::floor(x)
}

/// Truncate toward zero.
#[inline]
pub fn trunc(x: f64) -> f64 {
    libm::trunc(x)
}
