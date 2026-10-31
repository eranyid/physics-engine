//! Scene I/O errors. Malformed input never panics — callers get [`SceneError`].

use core::fmt;

/// Error returned by scene load / validate / migrate / instantiate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SceneError {
    /// Unknown or unsupported schema version.
    UnsupportedVersion {
        /// Version found in the document.
        found: u32,
        /// Highest version this build can read.
        max_supported: u32,
    },
    /// Magic / framing mismatch for the binary format.
    BadMagic,
    /// Truncated or inconsistent binary framing.
    Truncated {
        /// Human-readable detail.
        detail: String,
    },
    /// JSON / RON parse failure (never panics).
    Parse {
        /// Format name (`json`, `ron`, `binary`).
        format: &'static str,
        /// Underlying message.
        message: String,
    },
    /// Schema validation failure (lengths, NaN, invariants).
    Invalid {
        /// Field or rule that failed.
        detail: String,
    },
    /// Migration path not implemented for a version hop.
    MigrationStub {
        /// From version.
        from: u32,
        /// To version.
        to: u32,
    },
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion {
                found,
                max_supported,
            } => write!(
                f,
                "unsupported scene schema version {found} (max {max_supported})"
            ),
            Self::BadMagic => write!(f, "bad scene binary magic"),
            Self::Truncated { detail } => write!(f, "truncated scene binary: {detail}"),
            Self::Parse { format, message } => write!(f, "scene {format} parse error: {message}"),
            Self::Invalid { detail } => write!(f, "invalid scene: {detail}"),
            Self::MigrationStub { from, to } => {
                write!(f, "migration stub: no path from v{from} to v{to}")
            }
        }
    }
}

impl std::error::Error for SceneError {}
