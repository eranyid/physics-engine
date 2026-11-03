//! PyO3 bindings for M24 simulation baking.

use pe_scene::{
    bake_scene, canonical_bake_json_hash as bake_hash, canonical_bake_scene, BakeOptions,
    BakeReduceParams, SceneDocument, BAKE_SCHEMA_VERSION, CANONICAL_BAKE_JSON_HASH,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Bake schema version.
#[pyfunction]
fn bake_schema_version() -> u32 {
    BAKE_SCHEMA_VERSION
}

/// Locked FNV-1a of the canonical bake JSON.
#[pyfunction]
fn canonical_bake_json_hash() -> u64 {
    bake_hash()
}

/// Locked constant (must match [`canonical_bake_json_hash`]).
#[pyfunction]
fn canonical_bake_json_hash_locked() -> u64 {
    CANONICAL_BAKE_JSON_HASH
}

/// Canonical projectile bake scene as pretty JSON.
#[pyfunction]
fn bake_canonical_scene_json() -> PyResult<String> {
    canonical_bake_scene()
        .to_json_pretty()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Bake a scene JSON string into bake-curve JSON.
///
/// Parameters match [`pe_scene::BakeOptions`]. Pass `reduce_max_pos_error < 0`
/// to disable keyframe reduction.
#[pyfunction]
#[pyo3(signature = (
    scene_json,
    duration,
    physics_dt,
    sample_fps,
    include_velocities=true,
    reduce_max_pos_error=-1.0,
    reduce_max_rot_error=0.0,
))]
#[allow(clippy::too_many_arguments)]
fn scene_bake_json(
    py: Python<'_>,
    scene_json: &str,
    duration: f64,
    physics_dt: f64,
    sample_fps: f64,
    include_velocities: bool,
    reduce_max_pos_error: f64,
    reduce_max_rot_error: f64,
) -> PyResult<String> {
    let reduce = if reduce_max_pos_error >= 0.0 {
        Some(BakeReduceParams {
            max_position_error: reduce_max_pos_error,
            max_rotation_error: reduce_max_rot_error,
        })
    } else {
        None
    };
    let opt = BakeOptions {
        duration,
        physics_dt,
        sample_fps,
        include_velocities,
        reduce,
    };
    py.allow_threads(|| {
        let doc = SceneDocument::from_json(scene_json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let bake = bake_scene(&doc, &opt).map_err(|e| PyValueError::new_err(e.to_string()))?;
        bake.to_json()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

/// Register bake helpers on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bake_schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_bake_json_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_bake_json_hash_locked, m)?)?;
    m.add_function(wrap_pyfunction!(bake_canonical_scene_json, m)?)?;
    m.add_function(wrap_pyfunction!(scene_bake_json, m)?)?;
    m.add("BAKE_SCHEMA_VERSION", BAKE_SCHEMA_VERSION)?;
    m.add("CANONICAL_BAKE_JSON_HASH", CANONICAL_BAKE_JSON_HASH)?;
    Ok(())
}
