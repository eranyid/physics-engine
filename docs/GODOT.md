# Godot 4 GDExtension (`godot/`)

Thin GDExtension built with **godot-rust / gdext** (`godot` crate, **MPL-2.0** —
see [`THIRD_PARTY.md`](../THIRD_PARTY.md)). Exposes:

| Class | Role |
| --- | --- |
| `PePhysicsWorld` (`Node3D`) | Owns a `ContactWorld`; steps in `_physics_process` |
| `PeRigidBody3D` (`Node3D`) | Syncs a body index transform from a parent world |

## Build (Linux compile check)

```bash
cargo build -p pe-godot          # debug → target/debug/libpe_godot.so
cargo build -p pe-godot --release
```

Point `godot/example/pe_godot.gdextension` at those artifacts (paths are relative
to the example project). Open `godot/example/` in **Godot 4.3+**.

Example scene `main.tscn` is a stub root; attach scripts or call
`add_sphere_body` / `add_ground_plane` from GDScript after load.

## CI status (honest)

| Check | Status |
| --- | --- |
| `cargo build -p pe-godot` on Linux | **Done in CI** (compile check) |
| Godot headless import / run | **Not run** — Godot editor/binary is not installed in this CI image |
| Manual play in Godot editor | **Unverified** in this milestone |

Do **not** claim editor-tested behaviour until someone runs the example project
locally in Godot 4.3+.

## Limits

- f64 engine state → Godot `f32` transforms (precision loss at the boundary).
- No full collider mesh sync; demo APIs are sphere + ground plane.
- Soft bodies / GJK convex: deferred (other tracks).
- Reloadable GDExtension flagged in `.gdextension`; hot-reload not CI-tested.
