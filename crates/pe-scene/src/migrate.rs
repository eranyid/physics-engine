//! Schema migration. v1 documents load and bump to the current schema (v2).

use crate::document::SceneDocument;
use crate::error::SceneError;
use crate::types::SCENE_SCHEMA_VERSION;

/// Bump toward [`SCENE_SCHEMA_VERSION`].
///
/// - Current version → no-op.
/// - Future versions → [`SceneError::UnsupportedVersion`].
/// - v1 → v2: additive (convex/rounded shapes, fuller CCD fields, body bullet);
///   documents without those fields remain valid.
/// - Other older hops → [`SceneError::MigrationStub`].
pub fn migrate_to_current(schema_version: u32) -> Result<u32, SceneError> {
    if schema_version == SCENE_SCHEMA_VERSION {
        return Ok(schema_version);
    }
    if schema_version > SCENE_SCHEMA_VERSION {
        return Err(SceneError::UnsupportedVersion {
            found: schema_version,
            max_supported: SCENE_SCHEMA_VERSION,
        });
    }
    // v1 → v2: field defaults cover missing convex/rounded/CCD extensions.
    if schema_version == 1 && SCENE_SCHEMA_VERSION >= 2 {
        return Ok(SCENE_SCHEMA_VERSION);
    }
    Err(SceneError::MigrationStub {
        from: schema_version,
        to: SCENE_SCHEMA_VERSION,
    })
}

/// Run migrations on a loaded document.
pub fn migrate_document(mut doc: SceneDocument) -> Result<SceneDocument, SceneError> {
    let from = doc.schema_version;
    let to = migrate_to_current(from)?;
    doc.schema_version = to;
    Ok(doc)
}
