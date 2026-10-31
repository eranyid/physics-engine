//! Schema migration stubs. v1 is the only live version; older hops return
//! [`crate::SceneError::MigrationStub`] until a real path is implemented.

use crate::document::SceneDocument;
use crate::error::SceneError;
use crate::types::SCENE_SCHEMA_VERSION;

/// Bump toward [`SCENE_SCHEMA_VERSION`].
///
/// Currently a no-op for v1→v1. Any other hop is an explicit stub error so
/// callers never silently accept unknown layouts.
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
    // Future: v0 → v1 transforms go here.
    Err(SceneError::MigrationStub {
        from: schema_version,
        to: SCENE_SCHEMA_VERSION,
    })
}

/// Run migrations on a loaded document (stub until schema > 1 exists).
pub fn migrate_document(mut doc: SceneDocument) -> Result<SceneDocument, SceneError> {
    migrate_to_current(doc.schema_version)?;
    doc.schema_version = SCENE_SCHEMA_VERSION;
    Ok(doc)
}
