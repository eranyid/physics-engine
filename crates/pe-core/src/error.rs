//! Typed errors for rigid-body construction and validation.

use core::fmt;

/// Error returned by fallible rigid-body / inertia constructors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BodyError {
    /// Mass was non-finite or not strictly positive.
    InvalidMass,
    /// Inertia tensor failed symmetry / positive-definiteness / finiteness checks.
    InvalidInertia(&'static str),
    /// A state vector contained NaN or ±∞.
    NonFinite(&'static str),
    /// Orientation quaternion was zero or non-finite.
    InvalidOrientation,
    /// 3×3 matrix was singular (non-invertible).
    SingularMatrix,
}

impl fmt::Display for BodyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BodyError::InvalidMass => write!(f, "mass must be finite and strictly positive"),
            BodyError::InvalidInertia(detail) => write!(f, "invalid inertia: {detail}"),
            BodyError::NonFinite(what) => write!(f, "non-finite value in {what}"),
            BodyError::InvalidOrientation => {
                write!(f, "orientation quaternion must be finite and non-zero")
            }
            BodyError::SingularMatrix => write!(f, "singular 3x3 matrix"),
        }
    }
}

impl std::error::Error for BodyError {}
