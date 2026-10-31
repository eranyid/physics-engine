# Scene I/O (`pe-scene`)

Versioned serialization for bodies, shapes, materials, statics, joints, force
generators, solver settings, packed state, and solver runtime (warm starts).

## Formats

| Format | API | Bit-exact state? |
| --- | --- | --- |
| JSON | `SceneDocument::to_json` / `from_json` | Yes for state/runtime (`u64` IEEE bits) |
| RON | `to_ron_pretty` / `from_ron` | Same |
| Binary | `to_binary` / `from_binary` (`PESC` + version + JSON payload) | Yes (preferred for save/load/continue) |

Schema field: `schema_version` (current = **1**). Migration stub:
`pe_scene::migrate::migrate_to_current` — unknown/future versions error;
older hops return `SceneError::MigrationStub` until a real path exists.

Malformed input returns [`SceneError`] — loaders **never panic** (fuzzed in
`crates/pe-scene/tests/roundtrip.rs`).

## Bit-exact continue

`StateSnapshot` / `RuntimeSnapshot` store floats as `f64::to_bits()` so that

```text
run 300 steps → capture → binary save/load → instantiate → 300 steps
```

matches an uninterrupted 600-step run (same `state_hash64`).

Locked hash (equals stage-2 pyramid hash):

```bash
cargo run -p pe-scene --bin scene_hash --release
# expected: 50d52b2193aa528c
```

See [`DETERMINISM.md`](DETERMINISM.md).

## Limits / deferred

- Force generators are a fixed stage-1 enum (`UniformGravity`, drag, spring, …).
  Aero/hydro/space generators are **not** in the scene schema yet.
- Trait-object forces on a live `World` cannot be reverse-engineered —
  `SceneDocument::capture` requires the caller to pass the `forces` list.
- JSON decimal floats for body mass/inertia/solver tuning can lose ULPs if edited
  by hand; bit-exact continue relies on the `*_bits` fields and binary framing.
- Soft-body / GJK assets: out of scope (other tracks). Capturing a live world
  that already uses `Shape::Convex` / `Shape::Rounded` returns
  `SceneError::Invalid` until schema v2 adds those descriptors.
