# Scene I/O (`pe-scene`)

Versioned serialization for bodies, shapes, materials, statics, joints, force
generators, solver settings (including CCD), packed state, and solver runtime
(warm starts).

## Formats

| Format | API | Bit-exact state? |
| --- | --- | --- |
| JSON | `SceneDocument::to_json` / `from_json` | Yes for state/runtime (`u64` IEEE bits) |
| RON | `to_ron_pretty` / `from_ron` | Same |
| Binary | `to_binary` / `from_binary` (`PESC` + version + JSON payload) | Yes (preferred for save/load/continue) |

Schema field: `schema_version` (current = **2**). Migration:
`pe_scene::migrate::migrate_to_current` — v1 documents load and bump to v2;
unknown/future versions error with a clear `UnsupportedVersion` message.
Documents reject unknown fields (`deny_unknown_fields`) via
`SceneError::UnknownField`.

### Schema v2 additions

| Feature | Descriptor |
| --- | --- |
| Convex hulls | `ShapeDesc::Convex { vertices, radius }` (faces rebuilt on load) |
| Rounded shapes | `ShapeDesc::Rounded { inner, radius }` (e.g. rounded box); capsule remains first-class |
| Body bullet / TOI | `BodyDesc.bullet` |
| CCD params | `ccd_speculative`, `ccd_speculative_distance`, `ccd_toi`, `ccd_toi_max_iters`, `ccd_toi_tolerance` |
| Joints | Unchanged kinds (distance, revolute, prismatic, weld, wheel, mouse) — already in v1 |

v1 files that omit the new fields still load (defaults applied). Frozen fixture:
`crates/pe-scene/tests/fixtures/v1_box_ground.json`.

Malformed input returns [`SceneError`] — loaders **never panic** (fuzzed in
`crates/pe-scene/tests/roundtrip.rs`).

## Bit-exact continue

`StateSnapshot` / `RuntimeSnapshot` store floats as `f64::to_bits()` so that

```text
run 300 steps → capture → binary save/load → instantiate → 300 steps
```

matches an uninterrupted 600-step run (same `state_hash64`).

Locked hashes (`cargo run -p pe-scene --bin scene_hash --release`):

| Line | Scenario | Hash |
| --- | --- | --- |
| 1 | 10-box pyramid save/load/continue | `50d52b2193aa528c` |
| 2 | M6 convex (hull+capsule+rounded) save/load/continue | `4ca980096c036ade` |
| 3 | Godot Y-up box drop (180×1/60) | `a0d625f02641a420` |

See [`DETERMINISM.md`](DETERMINISM.md).

## Python

`physics_engine_rs` exposes `scene_load_json` / `scene_json_to_binary` /
`scene_load_binary` / `scene_canonical_*_json` / locked hash helpers. Tests:
`tests/test_scene_rust.py`.

## Limits / deferred

- Force generators are a fixed stage-1 enum (`UniformGravity`, drag, spring, …).
  Aero/hydro/space generators are **not** in the scene schema yet.
- Trait-object forces on a live `World` cannot be reverse-engineered —
  `SceneDocument::capture` requires the caller to pass the `forces` list.
- JSON decimal floats for body mass/inertia/solver tuning can lose ULPs if edited
  by hand; bit-exact continue relies on the `*_bits` fields and binary framing.
- Nested `Rounded` shapes are rejected (flatten radii instead). Soft-body assets
  remain out of scope (other tracks).
