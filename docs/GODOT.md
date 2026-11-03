# Godot 4 GDExtension (`godot/`)

Thin GDExtension built with **godot-rust / gdext** (`godot` crate, **MPL-2.0** —
see [`THIRD_PARTY.md`](../THIRD_PARTY.md)). f64 engine state with f32 at the
Godot transform boundary; `state_hash_hex` hashes the f64 packed state.

Gravity defaults to **Godot Y-up** `(0, −g, 0)`. Ground plane API uses normal +Y.

## Classes

| Class | Role |
| --- | --- |
| `PePhysicsWorld` (`Node3D`) | Owns a `ContactWorld`; fixed-step accumulator + optional visual interpolation; queries; contact signals |
| `PeRigidBody3D` (`Node3D`) | Authored dynamic/static/kinematic body (shape exports or child `PeCollisionShape3D`); syncs transforms; `apply_force` / impulse / torque; sleep helpers; `body_entered` / `body_exited` |
| `PeStaticBody3D` (`Node3D`) | Authored static collider |
| `PeCollisionShape3D` (`Node3D`) | Optional child shape override (sphere / box / capsule / convex) |
| `PeDistanceJoint3D` / `PeRevoluteJoint3D` / `PePrismaticJoint3D` / `PeWeldJoint3D` / `PeWheelJoint3D` / `PeMouseJoint3D` | Joint nodes for kinds in [`JOINTS.md`](JOINTS.md) |

## Build

```bash
cargo build -p pe-godot          # debug → target/debug/libpe_godot.so
cargo build -p pe-godot --release
```

Point `godot/example/pe_godot.gdextension` at those artifacts (paths are relative
to the example project). Open `godot/example/` in **Godot 4.3+**.

Demo scene: `main.tscn` (box stack + pendulum + raycast label).

## Stepping / determinism

- `fixed_dt > 0` (default `1/60`): accumulator in `_physics_process`, capped by
  `max_substeps` (default 8). Physics result does not depend on render rate.
- `interpolate` (default on): body visuals nlerp between previous/current poses;
  **does not** affect f64 state or `state_hash_hex`.
- Headless scripts should set `fixed_dt = 0` and call `step_once(dt)` for exact
  step counts (same as the locked drop hash path).
- `allow_sleep` defaults **off** on Godot worlds: core wake-on-gravity still
  assumes Z-up weight subtraction (see `ContactWorld::step`).

## Scene queries (GDScript)

- `raycast(origin, direction, max_distance) → Dictionary`
  keys: `hit`, `position`, `normal`, `body`, `distance`
  (sphere/plane exact; box/capsule/convex = AABB hit, approximate normal)
- `sphere_cast(origin, direction, radius, max_distance) → Dictionary`
  keys: `hit`, `position`, `normal`, `body`, `toi`
- `overlap_aabb(lo, hi) → PackedInt32Array` (statics encoded as `n + si`)

Rust implementations live in `pe_collision::queries` with analytic unit tests.

## Contact signals

- `PePhysicsWorld.contact_started(body_a, body_b)` /
  `contact_ended(body_a, body_b)` — unique body pairs from the last step’s
  contact list vs the previous step (`BTreeSet`, Godot-side only).
- `PeRigidBody3D.body_entered(other)` / `body_exited(other)` when the body was
  registered through the node path.

## Headless CI (Linux required)

CI downloads **Godot 4.3-stable** Linux x86_64 from the official GitHub release,
verifies the **SHA-512** from `SHA512-SUMS.txt`, caches `.godot-ci/`, builds the
extension, and runs:

```bash
./godot/ci/run_headless_suite.sh
# drop + shapes + joint + raycast + contact_started
# (legacy single-scene: ./godot/ci/run_headless_drop.sh)
```

### Locked drop test (unchanged)

`.gdextension` keys include `linux.x86_64` as well as
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

### M25 headless suite (measured on Linux x86_64, Godot 4.3.stable)

| Scene | Oracle | Measured | Bound | Status |
| --- | --- | --- | --- | --- |
| `ci_drop_test` | final `y`, hash | `y=0.249900`, `a0d625f02641a420` | ±0.08 / exact hash | pass |
| `ci_shapes_test` | rest `y` sphere/box/capsule/convex | `[0.2499, 0.2499, 0.1999, 0.2251]` | ±0.12 vs `[0.25,0.25,0.20,0.25]` | pass |
| `ci_joint_test` | pendulum \|r\| after 180 steps | `dist=1.0083` | `[0.5 L, 1.15 L]` | pass |
| `ci_raycast_test` | ray→sphere `t` | `4.5` | ±1e-4 vs analytic `4.5` | pass |
| `ci_contact_signal_test` | `contact_started` count | `1` | ≥ 1 | pass |

## CI status (honest)

| Check | Status |
| --- | --- |
| `cargo build -p pe-godot` on Linux | **Done in CI** (`rust` job builds workspace members; headless job rebuilds) |
| `cargo build -p pe-godot` on Windows / macOS | **Done in CI** (job `pe-godot-build`, **build-only**) |
| Godot 4.3-stable headless suite + drop hash | **Done in CI** (required job `pe-godot-headless`, Linux) |
| Headless Godot on Windows / macOS | **Not run** (no pinned Godot runner in CI for those OS) |
| Godot editor interactive play | **Unverified** in CI (use local editor + `main.tscn`) |

## Demo / script APIs

- `add_sphere_body` / `add_box_body` / `add_capsule_body` / `add_convex_body` / `add_ground_plane`
- `add_distance_joint(body_a, body_b, …)` (`body_b < 0` → world)
- `step_once(dt)` / `state_hash_hex()` / `get_body_position` / `get_body_quaternion`
- `raycast` / `sphere_cast` / `overlap_aabb`
- Body nodes: mass / friction / restitution / shape exports / velocities /
  `apply_force` / `apply_impulse` / `apply_torque` / sleep helpers

## Limits / out of scope (M25)

- f64 engine → Godot `f32` transforms (precision loss at the boundary).
- Exact OBB / convex ray normals (AABB approximation).
- True kinematic mass coupling (kinematic = scripted static pose sync).
- Soft bodies / GJK editor tooling.
- Sleep wake-on-gravity assumes Z-up in the core (Godot defaults sleep off).
- Hot-reload not CI-tested (`reloadable = true` in `.gdextension`).
