"""Scene schema v2 load/save bindings (physics_engine_rs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pe = pytest.importorskip("physics_engine_rs")

V1_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "crates"
    / "pe-scene"
    / "tests"
    / "fixtures"
    / "v1_box_ground.json"
)


def test_scene_schema_version_is_v2():
    assert pe.SCENE_SCHEMA_VERSION == 2
    assert pe.scene_schema_version() == 2


def test_canonical_scene_hashes_locked():
    assert pe.canonical_scene_roundtrip_hash() == 0x50D52B2193AA528C
    assert pe.canonical_scene_convex_roundtrip_hash() == 0x4CA980096C036ADE
    assert pe.canonical_scene_roundtrip_hash() == pe.canonical_collision_hash()


def test_v1_fixture_migrates_to_v2():
    raw = V1_FIXTURE.read_text()
    assert '"schema_version": 1' in raw or '"schema_version":1' in raw
    assert pe.scene_json_schema_version(raw) == 2
    pretty = pe.scene_load_json(raw)
    doc = json.loads(pretty)
    assert doc["schema_version"] == 2
    assert doc["bodies"][0]["name"] == "box0"


def test_json_binary_roundtrip_pyramid():
    src = pe.scene_canonical_pyramid_json()
    doc = json.loads(src)
    assert doc["schema_version"] == 2
    assert len(doc["bodies"]) == 10
    compact = pe.scene_roundtrip_json(src)
    blob = pe.scene_json_to_binary(compact)
    assert blob[:4] == b"PESC"
    back = pe.scene_load_binary(blob)
    doc2 = json.loads(back)
    assert doc2["schema_version"] == 2
    assert len(doc2["shapes"]) == 10


def test_convex_scene_has_v2_shapes():
    src = pe.scene_canonical_convex_json()
    doc = json.loads(src)
    types = {s["type"] for s in doc["shapes"]}
    assert "convex" in types
    assert "capsule" in types
    assert "rounded" in types
    # Rounded box inner.
    rounded = next(s for s in doc["shapes"] if s["type"] == "rounded")
    assert rounded["inner"]["type"] == "box"
    assert rounded["radius"] > 0.0


def test_unknown_field_errors():
    src = pe.scene_canonical_pyramid_json()
    doc = json.loads(src)
    doc["not_a_real_field"] = 123
    with pytest.raises(ValueError, match="unknown field"):
        pe.scene_load_json(json.dumps(doc))


def test_unsupported_version_errors():
    src = pe.scene_canonical_pyramid_json()
    doc = json.loads(src)
    doc["schema_version"] = 999
    with pytest.raises(ValueError, match="999"):
        pe.scene_load_json(json.dumps(doc))


def test_godot_drop_box_result_matches_locked():
    y, h = pe.godot_drop_box_result()
    assert abs(y - 0.25) < 0.08
    assert h == 0xA0D625F02641A420
