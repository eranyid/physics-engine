# Godot 4 GDExtension (`godot/`)

Thin GDExtension built with **godot-rust / gdext** (`godot` crate, **MPL-2.0** —
see [`THIRD_PARTY.md`](../THIRD_PARTY.md)). Exposes:

| Class | Role |
| --- | --- |
| `PePhysicsWorld` (`Node3D`) | Owns a `ContactWorld`; steps in `_physics_process` or via `step_once` |
| `PeRigidBody3D` (`Node3D`) | Syncs a body index transform from a parent world |

Gravity defaults to **Godot Y-up** `(0, −g, 0)`. Ground plane API uses normal +Y.

## Build

```bash
cargo build -p pe-godot          # debug → target/debug/libpe_godot.so
cargo build -p pe-godot --release
```

Point `godot/example/pe_godot.gdextension` at those artifacts (paths are relative
to the example project). Open `godot/example/` in **Godot 4.3+**.

## Headless CI drop test (required on Linux)

CI downloads **Godot 4.3-stable** Linux x86_64 from the official GitHub release,
verifies the **SHA-512** from `SHA512-SUMS.txt`, caches `.godot-ci/`, builds the
extension, and runs:

```bash
./godot/ci/run_headless_drop.sh
# stages target/debug/libpe_godot.so → godot/example/bin/,
# runs godot --import (registers GDExtension on a fresh checkout),
# verifies "Initialize godot-rust", then runs res://ci_drop_test.tscn
```

`.gdextension` keys include `linux.x86_64` (required for headless) as well as
`linux.debug/release.x86_64`. Script creates world, ground plane, drops a box
(`half=0.25` from `y=2`), steps 180×`dt=1/60`, asserts:

| Check | Measured (Linux x86_64, Godot 4.3.stable) |
| --- | --- |
| Final `y` | `0.249900` (tolerance ±0.08 around 0.25) |
| `state_hash_hex` | `a0d625f02641a420` (matches `pe_scene::godot_drop_box_result`) |

Pinned binary checksum (official `SHA512-SUMS.txt` entry for
`Godot_v4.3-stable_linux.x86_64.zip`):

```text
fd52bb4ba8acc30ca5accd1c566d470ad7282f891ccc0995dfafabcf92bcf76280ce182bf9d80ebd885f3ed2165d01e1fc3f2928436b15498dfbd98656c2a45a
```

## CI status (honest)

| Check | Status |
| --- | --- |
| `cargo build -p pe-godot` on Linux | **Done in CI** |
| Godot 4.3-stable headless drop + hash | **Done in CI** (required job `pe-godot-headless`) |
| Godot editor interactive play | **Unverified** in CI (use local editor) |

## Demo APIs

- `add_sphere_body` / `add_box_body` / `add_ground_plane` (+Y)
- `step_once(dt)` / `state_hash_hex()` for headless scripts
- `get_body_position` / `get_body_quaternion`

## Limits

- f64 engine state → Godot `f32` transforms (precision loss at the boundary;
  hash stays on the f64 engine state).
- No full collider mesh sync.
- Soft bodies / GJK editor tooling: deferred (other tracks).
- Reloadable GDExtension flagged in `.gdextension`; hot-reload not CI-tested.
