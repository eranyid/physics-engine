//! Versioned scene serialization for the physics engine (M8 / scene I/O).
//!
//! Formats: pretty/compact JSON, RON, and a compact binary wrapper around JSON
//! (`PESC` magic + schema version). State + solver runtime snapshots round-trip
//! so that save → load → continue matches an uninterrupted run bit-exactly
//! (same [`pe_core::diagnostics::state_hash64`]).
//!
//! Malformed input returns [`SceneError`] — loaders never panic.

#![deny(missing_docs)]

pub mod binary;
/// Conversions between scene descriptors and engine types.
pub mod convert;
pub mod document;
pub mod error;
pub mod migrate;
pub mod types;

pub use convert::force_to_box;

pub use document::{
    canonical_scene_convex_roundtrip_hash, canonical_scene_roundtrip_hash,
    convex_round_trip_continue_hashes, godot_drop_box_result, round_trip_continue_hashes,
    SceneDocument, CANONICAL_SCENE_CONVEX_ROUNDTRIP_HASH, CANONICAL_SCENE_ROUNDTRIP_HASH,
};
pub use error::SceneError;
pub use types::{
    BodyDesc, ForceDesc, JointDesc, JointKindDesc, MaterialDesc, RuntimeSnapshot, ShapeDesc,
    SolverSettings, StateSnapshot, StaticDesc, SCENE_BINARY_MAGIC, SCENE_SCHEMA_V1,
    SCENE_SCHEMA_VERSION,
};
