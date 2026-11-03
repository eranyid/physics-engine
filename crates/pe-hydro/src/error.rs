//! Typed errors for hydro construction and finite-input validation (M27).

use core::fmt;

/// Error returned by fallible hydro constructors and validated entry points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HydroError {
    /// A scalar or vector component was NaN or ±∞.
    NonFinite(&'static str),
    /// Length / area / volume / draft was non-finite or not strictly positive.
    InvalidSize(&'static str),
    /// Density, viscosity, mass, or similar physical coefficient failed its domain.
    InvalidParameter(&'static str),
    /// Holtrop–Mennen input fell outside the published applicability guard.
    OutOfRange(&'static str),
}

impl fmt::Display for HydroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HydroError::NonFinite(what) => write!(f, "non-finite value in {what}"),
            HydroError::InvalidSize(what) => {
                write!(f, "{what} must be finite and strictly positive")
            }
            HydroError::InvalidParameter(what) => write!(f, "invalid parameter: {what}"),
            HydroError::OutOfRange(what) => write!(f, "out of published range: {what}"),
        }
    }
}

impl std::error::Error for HydroError {}

/// True when `x` is finite (not NaN / ±∞).
#[inline]
pub fn require_finite(x: f64, what: &'static str) -> Result<(), HydroError> {
    if x.is_finite() {
        Ok(())
    } else {
        Err(HydroError::NonFinite(what))
    }
}

/// Finite and strictly positive.
#[inline]
pub fn require_positive(x: f64, what: &'static str) -> Result<(), HydroError> {
    if x.is_finite() && x > 0.0 {
        Ok(())
    } else {
        Err(HydroError::InvalidSize(what))
    }
}

/// Finite and `≥ 0`.
#[inline]
pub fn require_non_negative(x: f64, what: &'static str) -> Result<(), HydroError> {
    if x.is_finite() && x >= 0.0 {
        Ok(())
    } else {
        Err(HydroError::InvalidParameter(what))
    }
}
