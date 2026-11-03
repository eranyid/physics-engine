"""Python oracle harness for M24 simulation baking."""

from __future__ import annotations

import json

import pytest

pe = pytest.importorskip("physics_engine_rs")


def test_canonical_bake_hash_matches_locked():
    assert pe.canonical_bake_json_hash() == pe.CANONICAL_BAKE_JSON_HASH
    assert pe.CANONICAL_BAKE_JSON_HASH == 0x3E9A97A85DA0CBC6


def test_scene_bake_json_projectile_schema():
    scene = pe.bake_canonical_scene_json()
    out = pe.scene_bake_json(scene, 1.0, 1.0 / 240.0, 24.0, True, -1.0, 0.0)
    doc = json.loads(out)
    assert doc["schema_version"] == pe.BAKE_SCHEMA_VERSION
    assert doc["sample_fps"] == 24.0
    assert doc["interpolation"]["position"] == "hermite"
    assert doc["interpolation"]["rotation"] == "slerp"
    body = doc["bodies"][0]
    assert len(body["times"]) == 25
    assert body["times"][0] == 0.0
    assert body["times"][-1] == 1.0
    assert len(body["position"]) == 25
    assert "linear_velocity" in body


def test_coincident_fps_python():
    scene = pe.bake_canonical_scene_json()
    a = json.loads(pe.scene_bake_json(scene, 1.0, 1.0 / 240.0, 24.0))
    b = json.loads(pe.scene_bake_json(scene, 1.0, 1.0 / 240.0, 48.0))
    # t=0 and t=1 endpoints must match (JSON may lose ULPs mid-curve).
    assert a["bodies"][0]["position"][0] == b["bodies"][0]["position"][0]
    assert a["bodies"][0]["position"][-1] == b["bodies"][0]["position"][-1]
