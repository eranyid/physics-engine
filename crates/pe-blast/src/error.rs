//! Typed errors for blast construction and finite-input validation (M27).

use core::fmt;

/// Error returned by fallible blast constructors and validated entry points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlastError {
    /// A scalar or vector component was NaN or ±∞.
    NonFinite(&'static str),
    /// Mass / charge / TNT-equivalence was non-finite or not strictly positive.
    InvalidMass(&'static str),
    /// Length, area, duration, pressure, or similar size was non-finite or not > 0.
    InvalidSize(&'static str),
    /// Coefficient or dimensionless factor failed its domain check.
    InvalidParameter(&'static str),
}

impl fmt::Display for BlastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlastError::NonFinite(what) => write!(f, "non-finite value in {what}"),
            BlastError::InvalidMass(what) => {
                write!(f, "{what} must be finite and strictly positive")
            }
            BlastError::InvalidSize(what) => {
                write!(f, "{what} must be finite and strictly positive")
            }
            BlastError::InvalidParameter(what) => write!(f, "invalid parameter: {what}"),
        }
    }
}

impl std::error::Error for BlastError {}

/// True when `x` is finite (not NaN / ±∞).
#[inline]
pub fn require_finite(x: f64, what: &'static str) -> Result<(), BlastError> {
    if x.is_finite() {
        Ok(())
    } else {
        Err(BlastError::NonFinite(what))
    }
}

/// Finite and strictly positive.
#[inline]
pub fn require_positive(x: f64, what: &'static str) -> Result<(), BlastError> {
    if x.is_finite() && x > 0.0 {
        Ok(())
    } else {
        Err(BlastError::InvalidSize(what))
    }
}

/// Finite and `≥ 0`.
#[inline]
pub fn require_non_negative(x: f64, what: &'static str) -> Result<(), BlastError> {
    if x.is_finite() && x >= 0.0 {
        Ok(())
    } else {
        Err(BlastError::InvalidParameter(what))
    }
}
