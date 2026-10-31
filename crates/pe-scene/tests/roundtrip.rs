//! Scene round-trip and malformed-input tests (never panic on bad bytes).

use pe_scene::{
    canonical_scene_roundtrip_hash, round_trip_continue_hashes, SceneDocument, SceneError,
    CANONICAL_SCENE_ROUNDTRIP_HASH, SCENE_SCHEMA_VERSION,
};
use proptest::prelude::*;

#[test]
fn json_ron_binary_round_trip_pyramid() {
    let doc = SceneDocument::canonical_pyramid();
    let json = doc.to_json().unwrap();
    let doc2 = SceneDocument::from_json(&json).unwrap();
    assert_eq!(doc.bodies.len(), doc2.bodies.len());
    assert_eq!(doc.schema_version, doc2.schema_version);

    let ron = doc.to_ron_pretty().unwrap();
    let doc3 = SceneDocument::from_ron(&ron).unwrap();
    assert_eq!(doc.shapes, doc3.shapes);

    let bin = doc.to_binary().unwrap();
    let doc4 = SceneDocument::from_binary(&bin).unwrap();
    assert_eq!(doc.materials, doc4.materials);
}

#[test]
fn save_load_continue_matches_uninterrupted() {
    let (h_rt, h_full) = round_trip_continue_hashes().unwrap();
    assert_eq!(h_rt, h_full);
    assert_eq!(h_rt, CANONICAL_SCENE_ROUNDTRIP_HASH);
    assert_eq!(
        canonical_scene_roundtrip_hash(),
        CANONICAL_SCENE_ROUNDTRIP_HASH
    );
}

#[test]
fn malformed_json_is_error() {
    let err = SceneDocument::from_json("{not json").unwrap_err();
    assert!(matches!(err, SceneError::Parse { format: "json", .. }));
}

#[test]
fn malformed_ron_is_error() {
    let err = SceneDocument::from_ron("SceneDocument(").unwrap_err();
    assert!(matches!(err, SceneError::Parse { format: "ron", .. }));
}

#[test]
fn malformed_binary_never_panics() {
    assert!(matches!(
        SceneDocument::from_binary(&[]).unwrap_err(),
        SceneError::Truncated { .. }
    ));
    assert!(matches!(
        SceneDocument::from_binary(b"XXXX").unwrap_err(),
        SceneError::Truncated { .. } | SceneError::BadMagic
    ));
    assert!(matches!(
        SceneDocument::from_binary(b"PESC").unwrap_err(),
        SceneError::Truncated { .. }
    ));
    let mut bad = Vec::from(*b"PESC");
    bad.extend_from_slice(&SCENE_SCHEMA_VERSION.to_le_bytes());
    bad.extend_from_slice(&8u64.to_le_bytes());
    bad.extend_from_slice(b"{badjson");
    let err = SceneDocument::from_binary(&bad).unwrap_err();
    assert!(matches!(
        err,
        SceneError::Parse { .. } | SceneError::Truncated { .. }
    ));
}

#[test]
fn unsupported_version_errors() {
    let mut doc = SceneDocument::canonical_pyramid();
    doc.schema_version = 999;
    let json = serde_json::to_string(&doc).unwrap();
    // from_json runs migrate which rejects future versions
    let err = SceneDocument::from_json(&json).unwrap_err();
    assert!(matches!(
        err,
        SceneError::UnsupportedVersion { found: 999, .. } | SceneError::MigrationStub { .. }
    ));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn fuzz_binary_no_panic(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        let _ = SceneDocument::from_binary(&bytes);
    }

    #[test]
    fn fuzz_json_no_panic(s in ".*") {
        let _ = SceneDocument::from_json(&s);
    }

    #[test]
    fn fuzz_ron_no_panic(s in ".*") {
        let _ = SceneDocument::from_ron(&s);
    }
}
