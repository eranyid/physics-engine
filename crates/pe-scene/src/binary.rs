//! Compact binary framing: `PESC` + u32 version + u64 JSON-len + UTF-8 JSON.
//!
//! State / warm-cache floats are stored as `u64` IEEE bit patterns inside the
//! JSON payload, so the binary path is bit-exact even though the envelope is JSON.

use crate::document::SceneDocument;
use crate::error::{map_format_error, SceneError};
use crate::migrate::migrate_document;
use crate::types::{SCENE_BINARY_MAGIC, SCENE_SCHEMA_VERSION};

/// Encode document to compact binary.
pub fn encode(doc: &SceneDocument) -> Result<Vec<u8>, SceneError> {
    let json = serde_json::to_string(doc).map_err(|e| map_format_error("binary", e.to_string()))?;
    let payload = json.as_bytes();
    let mut out = Vec::with_capacity(4 + 4 + 8 + payload.len());
    out.extend_from_slice(&SCENE_BINARY_MAGIC);
    out.extend_from_slice(&SCENE_SCHEMA_VERSION.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

/// Decode binary; returns [`SceneError`] on any malformation (no panic).
pub fn decode(bytes: &[u8]) -> Result<SceneDocument, SceneError> {
    if bytes.len() < 16 {
        return Err(SceneError::Truncated {
            detail: format!("need ≥16 header bytes, got {}", bytes.len()),
        });
    }
    if bytes[0..4] != SCENE_BINARY_MAGIC {
        return Err(SceneError::BadMagic);
    }
    let ver = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let len = u64::from_le_bytes([
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    ]) as usize;
    if bytes.len() < 16 + len {
        return Err(SceneError::Truncated {
            detail: format!(
                "payload len {len} but buffer {}",
                bytes.len().saturating_sub(16)
            ),
        });
    }
    if bytes.len() != 16 + len {
        return Err(SceneError::Truncated {
            detail: format!("trailing {} bytes after payload", bytes.len() - 16 - len),
        });
    }
    let s = core::str::from_utf8(&bytes[16..])
        .map_err(|e| map_format_error("binary", format!("utf-8: {e}")))?;
    let mut doc: SceneDocument =
        serde_json::from_str(s).map_err(|e| map_format_error("binary", e.to_string()))?;
    // Binary header version is authoritative when it disagrees with the payload.
    if doc.schema_version != ver {
        doc.schema_version = ver;
    }
    migrate_document(doc)
}
