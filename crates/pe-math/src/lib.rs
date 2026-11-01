//! Scalar f64 math primitives for the physics engine.
//!
//! Units are SI throughout. Quaternion convention matches the Python oracle:
//! scalar-first `[w, x, y, z]`, Hamilton product, active body→world rotation.
//!
//! Hot-path ops are `#[inline]`. Explicit SIMD / FMA is intentionally omitted on
//! the deterministic path (would change rounding vs locked cross-OS hashes).

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod libm_ops;
pub mod quat;
pub mod rng;
pub mod units;
pub mod vec3;

pub use quat::Quat;
pub use rng::Xoshiro256StarStar;
pub use units::{G, STANDARD_GRAVITY};
pub use vec3::Vec3;
