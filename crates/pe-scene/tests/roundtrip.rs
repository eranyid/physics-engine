//! Scene round-trip, v1 migration, unknown-field, and malformed-input tests.

use pe_scene::{
    canonical_scene_convex_roundtrip_hash, canonical_scene_roundtrip_hash,
    convex_round_trip_continue_hashes, godot_drop_box_result, round_trip_continue_hashes,
    SceneDocument, SceneError, CANONICAL_SCENE_CONVEX_ROUNDTRIP_HASH,
    CANONICAL_SCENE_ROUNDTRIP_HASH, SCENE_SCHEMA_V1, SCENE_SCHEMA_VERSION,
};
use proptest::prelude::*;

#[test]
fn json_ron_binary_round_trip_pyramid() {
    let doc = SceneDocument::canonical_pyramid();
    assert_eq!(doc.schema_version, SCENE_SCHEMA_VERSION);
    let json = doc.to_json().unwrap();
    let doc2 = SceneDocument::from_json(&json).unwrap();
    assert_eq!(doc.bodies.len(), doc2.bodies.len());
    assert_eq!(doc2.schema_version, SCENE_SCHEMA_VERSION);

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
fn convex_v2_round_trip_hash_locked() {
    let (h_rt, h_full) = convex_round_trip_continue_hashes().unwrap();
    assert_eq!(h_rt, h_full);
    assert_eq!(h_rt, CANONICAL_SCENE_CONVEX_ROUNDTRIP_HASH);
    assert_eq!(
        canonical_scene_convex_roundtrip_hash(),
        CANONICAL_SCENE_CONVEX_ROUNDTRIP_HASH
    );
}

#[test]
fn convex_rounded_capsule_serialize_in_v2() {
    let doc = SceneDocument::canonical_convex_document();
    assert_eq!(doc.schema_version, SCENE_SCHEMA_VERSION);
    let json = doc.to_json().unwrap();
    assert!(json.contains("\"convex\"") || json.contains("\"type\":\"convex\""));
    assert!(json.contains("rounded") || json.contains("capsule"));
    let doc2 = SceneDocument::from_json(&json).unwrap();
    assert_eq!(doc.shapes.len(), doc2.shapes.len());
    let cw = doc2.instantiate().unwrap();
    assert_eq!(cw.world.n, 3);
}

#[test]
fn frozen_v1_fixture_migrates_and_instantiates() {
    let raw = include_str!("fixtures/v1_box_ground.json");
    assert!(raw.contains("\"schema_version\": 1") || raw.contains("\"schema_version\":1"));
    // Raw parse without migrate must see v1.
    let parsed: serde_json::Value = serde_json::from_str(raw).unwrap();
    assert_eq!(parsed["schema_version"], SCENE_SCHEMA_V1);

    let doc = SceneDocument::from_json(raw).unwrap();
    assert_eq!(doc.schema_version, SCENE_SCHEMA_VERSION);
    assert!(!doc.bodies[0].bullet);
    assert!(!doc.solver.ccd_toi);
    let mut cw = doc.instantiate().unwrap();
    for _ in 0..60 {
        cw.step(1.0 / 60.0);
    }
    assert!(cw.world.state.x[0].z.is_finite());
    assert!(cw.world.state.x[0].z > 0.0);
}

#[test]
fn unknown_field_is_clear_error() {
    let doc = SceneDocument::canonical_pyramid();
    let mut v: serde_json::Value = serde_json::from_str(&doc.to_json().unwrap()).unwrap();
    v["not_a_real_field"] = serde_json::json!(123);
    let err = SceneDocument::from_json(&v.to_string()).unwrap_err();
    assert!(
        matches!(err, SceneError::UnknownField { format: "json", .. }),
        "expected UnknownField, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("unknown field") || msg.contains("not_a_real_field"),
        "unclear message: {msg}"
    );
}

#[test]
fn unsupported_version_errors_clearly() {
    let mut doc = SceneDocument::canonical_pyramid();
    doc.schema_version = 999;
    let json = serde_json::to_string(&doc).unwrap();
    let err = SceneDocument::from_json(&json).unwrap_err();
    assert!(matches!(
        err,
        SceneError::UnsupportedVersion {
            found: 999,
            max_supported: SCENE_SCHEMA_VERSION
        }
    ));
    let msg = err.to_string();
    assert!(msg.contains("999"));
    assert!(msg.contains(&SCENE_SCHEMA_VERSION.to_string()));
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
        SceneError::Parse { .. } | SceneError::Truncated { .. } | SceneError::UnknownField { .. }
    ));
}

#[test]
fn godot_drop_box_settles_near_half_extent() {
    let (y, hash) = godot_drop_box_result();
    assert!(y.is_finite(), "y={y}");
    // Resting on y=0 plane with half-extent 0.25 → COM ≈ 0.25 (small penetration OK).
    assert!(
        (y - 0.25).abs() < 0.08,
        "final y={y} outside tolerance of 0.25±0.08"
    );
    assert_ne!(hash, 0);
    // Bit-stable on repeat.
    let (y2, hash2) = godot_drop_box_result();
    assert_eq!(hash, hash2);
    assert_eq!(y.to_bits(), y2.to_bits());
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
