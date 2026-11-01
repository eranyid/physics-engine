//! PyO3 bindings for `pe-scene` (schema v2 load/save).

use pe_scene::{
    canonical_scene_convex_roundtrip_hash as scene_convex_hash,
    canonical_scene_roundtrip_hash as scene_pyramid_hash, godot_drop_box_result as drop_box_result,
    SceneDocument, SCENE_SCHEMA_VERSION,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

/// Current scene schema version written by save helpers.
#[pyfunction]
fn scene_schema_version() -> u32 {
    SCENE_SCHEMA_VERSION
}

/// Locked pyramid save/load/continue hash.
#[pyfunction]
fn canonical_scene_roundtrip_hash() -> u64 {
    scene_pyramid_hash()
}

/// Locked convex scene save/load/continue hash.
#[pyfunction]
fn canonical_scene_convex_roundtrip_hash() -> u64 {
    scene_convex_hash()
}

/// Godot CI drop scenario `(final_y, state_hash)`.
#[pyfunction]
fn godot_drop_box_result() -> (f64, u64) {
    drop_box_result()
}

/// Load a scene document from a JSON string; returns pretty JSON at current schema.
#[pyfunction]
fn scene_load_json(s: &str) -> PyResult<String> {
    let doc = SceneDocument::from_json(s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    doc.to_json_pretty()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Validate + re-serialize JSON (round-trip). Returns compact JSON.
#[pyfunction]
fn scene_roundtrip_json(s: &str) -> PyResult<String> {
    let doc = SceneDocument::from_json(s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    doc.validate()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    doc.to_json()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Load JSON and return the post-migration `schema_version`.
#[pyfunction]
fn scene_json_schema_version(s: &str) -> PyResult<u32> {
    let doc = SceneDocument::from_json(s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(doc.schema_version)
}

/// Save a scene from JSON text to binary `PESC` bytes (after migrate/validate).
#[pyfunction]
fn scene_json_to_binary<'py>(py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyBytes>> {
    let doc = SceneDocument::from_json(s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    doc.validate()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let bytes = doc
        .to_binary()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &bytes))
}

/// Load binary `PESC` bytes and return pretty JSON at current schema.
#[pyfunction]
fn scene_load_binary(bytes: &[u8]) -> PyResult<String> {
    let doc =
        SceneDocument::from_binary(bytes).map_err(|e| PyValueError::new_err(e.to_string()))?;
    doc.to_json_pretty()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Canonical pyramid scene as pretty JSON (schema v2).
#[pyfunction]
fn scene_canonical_pyramid_json() -> PyResult<String> {
    SceneDocument::canonical_pyramid()
        .to_json_pretty()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Canonical convex (hull + capsule + rounded) scene as pretty JSON (schema v2).
#[pyfunction]
fn scene_canonical_convex_json() -> PyResult<String> {
    SceneDocument::canonical_convex_document()
        .to_json_pretty()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Register scene helpers on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scene_schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(scene_load_json, m)?)?;
    m.add_function(wrap_pyfunction!(scene_roundtrip_json, m)?)?;
    m.add_function(wrap_pyfunction!(scene_json_schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(scene_json_to_binary, m)?)?;
    m.add_function(wrap_pyfunction!(scene_load_binary, m)?)?;
    m.add_function(wrap_pyfunction!(scene_canonical_pyramid_json, m)?)?;
    m.add_function(wrap_pyfunction!(scene_canonical_convex_json, m)?)?;
    m.add_function(wrap_pyfunction!(godot_drop_box_result, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_scene_roundtrip_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_scene_convex_roundtrip_hash, m)?)?;
    m.add("SCENE_SCHEMA_VERSION", SCENE_SCHEMA_VERSION)?;
    Ok(())
}
