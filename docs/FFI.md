# C ABI (`pe-ffi`)

Stable C surface for engines and the Godot GDExtension. Header:
[`crates/pe-ffi/include/pe_ffi.h`](../crates/pe-ffi/include/pe_ffi.h)
(cbindgen-generated; CI diffs against a fresh generation).

## ABI stability policy

- **v0.x:** additive only within a minor release when possible. Removing or
  changing a symbol / struct layout is a **minor** bump at minimum; document in
  this file.
- Error codes (`PE_OK`, `PE_ERR_*`) are stable integers.
- Opaque `PeWorld*`: never dereference from C; only pass through API functions.
- SoA transform buffers: caller allocates; library never allocates per-body
  host buffers on get/set transforms.

## Thread safety

- One `PeWorld` → one owner thread. No internal mutex.
- Distinct worlds may run on distinct threads.
- Do not call into a world from two threads concurrently (including
  destroy vs step).

## Unwinding

Every `extern "C"` entry uses `catch_unwind`. Panics become `PE_ERR_PANIC` and
**never** unwind across the FFI boundary.

## Input validation (M22)

`pe_world_add_body` / `pe_world_add_static` / `pe_world_set_transforms` /
`pe_world_step` reject non-finite inputs and non-positive shape sizes with
`PE_ERR_ARG` (including NaN/Inf pose, zero quaternions, non-PD inertia, and
`dt ≤ 0`). Oversized SoA counts return `PE_ERR_BOUNDS`. Null required pointers
return `PE_ERR_NULL`. Rust regression coverage: `crates/pe-ffi/tests/boundary.rs`.

## Quick C usage

```c
#include "pe_ffi.h"

PeWorld *w = pe_world_create();
uint32_t id;
pe_world_add_body(w, 1.0, 0,0,2, 0,0,0, 1,0,0,0, 0,0,0, 0.4,0.4,0.4,
                  0 /*sphere*/, 0.25, 0,0, 0.5, 0.3, &id);
pe_world_add_static(w, 3 /*plane*/, 0,0,1, 0, 0,0,0, 0,1, NULL);
pe_world_step(w, 1.0/60.0);
double pos[3], quat[4];
pe_world_get_transforms(w, pos, quat, 1);
pe_world_destroy(w);
```

Build/run the CI smoke test:

```bash
./crates/pe-ffi/c_tests/run_c_test.sh
```

## Queries & events

Implementations live in [`pe_collision::queries`](../crates/pe-collision/src/queries.rs)
(shared with the Godot GDExtension). The C ABI wraps the same functions:

- `pe_world_raycast` — spheres/planes exact; box/capsule/convex use AABB hits
  (**approximate normals**; documented as unverified for exact OBB).
- `pe_world_overlap_aabb` — broad AABB overlap indices.
- `pe_world_get_contacts` — contacts from the **last** `pe_world_step`.

Rust also exposes `sphere_cast` (analytic vs sphere/plane; expanded-AABB vs
other shapes) for Godot / native callers; not yet in the C header (additive
follow-up).

## Limits / deferred

- No multi-world islands API beyond what `ContactWorld` exposes.
- Hit-scan / blast event channels not wired (use domain crates directly).
- Header regen requires `cbindgen` 0.28+ on PATH for the drift check.
