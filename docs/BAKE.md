# Simulation baking (M24)

Offline export of per-body transform curves for animation tools (Toon Studio /
Swift) and Godot. Implemented in `pe-scene::bake`, exposed through `pe-ffi`
(C ABI) and `physics_engine_rs` (Python).

## Pipeline

1. Load a versioned [`SceneDocument`](SCENE.md) (JSON / RON / binary).
2. Instantiate a `ContactWorld` (sleep forced off for the bake).
3. Step at fixed `physics_dt` for `duration` seconds.
4. Sample every body at times `tᵢ = i / sample_fps` (plus `duration` if needed)
   using exact-time interpolation between physics knots.
5. Optionally run keyframe reduction with a stated max error bound.
6. Write a versioned bake JSON document.

## Interpolation

Between physics knots `(t₀, x₀, q₀, v₀, ω₀)` and `(t₁, x₁, q₁, v₁, ω₁)` with
`Δt = t₁ − t₀` and `s = (t − t₀) / Δt ∈ [0, 1]`:

| Channel | Method |
| --- | --- |
| Position | Cubic **Hermite**: `p = h₀₀ x₀ + h₁₀ Δt v₀ + h₀₁ x₁ + h₁₁ Δt v₁` with `h₀₀=2s³−3s²+1`, `h₁₀=s³−2s²+s`, `h₀₁=−2s³+3s²`, `h₁₁=s³−s²` |
| Rotation | Unit-quaternion **slerp** (short arc; portable `libm` `acos`/`sin`) |
| Linear / angular velocity (optional) | Linear lerp |

When `t` lands on a physics knot, the stored sample is copied (no interpolant).
Resampling the **same** scene at different `sample_fps` therefore agrees
**bit-exactly** at coincident times (same knots + same Hermite/slerp).

Hermite recovers free-fall / projectile quadratics **exactly** when the knots
themselves are exact. Contact-world stepping uses semi-implicit Euler, so knots
carry the integrator’s truncation error (see Oracles).

## Keyframe reduction

Optional Douglas–Peucker over the dense output samples. Deviation of each
interior sample is measured against Hermite (position) and slerp (rotation)
between candidate endpoints. A sample is kept when either error exceeds the
caller’s bounds:

* `max_position_error` [m] — Euclidean
* `max_rotation_error` [rad] — short-arc quaternion angle

**Guarantee:** every discarded dense sample reconstructs from the reduced keys
within those bounds (plus a few ULPs of floating-point noise). Prefer
`include_velocities = true` when reducing so Hermite uses real endpoint
velocities.

## Bake JSON schema (v1)

```json
{
  "schema_version": 1,
  "physics_dt": 0.004166666666666667,
  "duration": 1.0,
  "sample_fps": 24.0,
  "include_velocities": true,
  "interpolation": { "position": "hermite", "rotation": "slerp" },
  "reduction": null,
  "bodies": [
    {
      "index": 0,
      "name": "projectile",
      "times": [0.0, 0.041666666666666664, "..."],
      "position": [[0.0, 0.0, 10.0], "..."],
      "rotation": [[1.0, 0.0, 0.0, 0.0], "..."],
      "linear_velocity": [[3.0, 0.0, 4.0], "..."],
      "angular_velocity": [[0.0, 0.0, 0.0], "..."]
    }
  ]
}
```

Unknown fields are rejected. `reduction` is omitted or `null` when unused.
Velocity arrays are omitted when `include_velocities` is false.

**Note:** decimal JSON may lose ULPs on re-parse (same caveat as scene mass /
inertia fields). The locked cross-OS hash is FNV-1a over the **producer’s**
compact JSON bytes (`cargo run -p pe-scene --bin bake_hash --release`).

| Platform | Canonical bake JSON hash | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `3e9a97a85da0cbc6` | locked |
| macOS (CI) | same | CI asserts equality |
| Windows (CI) | same | CI asserts equality |

Canonical scenario: unit-mass sphere at `(0,0,10)` with `v=(3,0,4)`, Earth
gravity, no contacts; `duration=1`, `physics_dt=1/240`, `sample_fps=24`,
velocities on, no reduction.

## Oracles (tolerances not loosened)

| Oracle | Bound | Why |
| --- | --- | --- |
| Free fall / projectile @ 24 fps vs continuous analytic | `½ g T Δt` on position; `½ g Δt` on velocity | Semi-implicit Euler global truncation for constant accel |
| Uniform principal-axis spin slerp vs `exp(ω t)` | `1e-10` rad | Implicit-midpoint exp map + portable slerp |
| Small-angle pendulum period from bake samples | `< 1%` vs `2π√(L/g)` | Same stiff distance-joint oracle as M5b |
| Keyframe reduction | ≤ stated `(max_pos, max_rot)` on every dense sample | Douglas–Peucker + Hermite/slerp measure |
| Bit-identical re-bake + locked JSON hash | exact | Deterministic step + serde field order |
| Coincident times across 24/30/48 fps | bit-identical poses | Shared physics knots + identical interpolant |

## Rust API

```rust
use pe_scene::{bake_scene, BakeOptions, SceneDocument};

let scene = SceneDocument::from_json(&json)?;
let bake = bake_scene(&scene, &BakeOptions {
    duration: 2.0,
    physics_dt: 1.0 / 240.0,
    sample_fps: 24.0,
    include_velocities: true,
    reduce: None,
})?;
std::fs::write("out.bake.json", bake.to_json_pretty()?)?;
```

CLI demo: `cargo run -p pe-examples --bin bake_scene --release`.

## C ABI (`pe-ffi`)

```c
int pe_scene_bake(
    const char *scene_json,
    double duration,
    double physics_dt,
    double sample_fps,
    int include_velocities,
    double reduce_max_pos_error,   /* < 0 disables reduction */
    double reduce_max_rot_error,
    uint8_t *out_json,
    size_t out_cap,
    size_t *out_len);

int pe_canonical_bake_json_hash(uint64_t *out_hash);
```

Finite-input validation matches M22: non-finite / non-positive timing args →
`PE_ERR_ARG`; null required pointers → `PE_ERR_NULL`; size probe →
`PE_ERR_BOUNDS`; parse/bake failure → `PE_ERR_SCENE`. Never unwinds.

## Swift usage note (Toon Studio)

No Swift package is built in CI. Link `libpe_ffi` (static or dylib) and the
generated header `pe_ffi.h`, then:

```swift
import Foundation

func bakeScene(sceneJSON: String, duration: Double, dt: Double, fps: Double) throws -> Data {
    let cScene = sceneJSON.cString(using: .utf8)!
    var need: Int = 0
    var rc = pe_scene_bake(
        cScene, duration, dt, fps,
        1,          // include velocities
        -1.0, 0.0,  // no reduction
        nil, 0, &need)
    guard rc == PE_ERR_BOUNDS, need > 0 else {
        throw BakeError.probeFailed(rc)
    }
    var buf = [UInt8](repeating: 0, count: need)
    rc = buf.withUnsafeMutableBufferPointer { bp in
        pe_scene_bake(
            cScene, duration, dt, fps, 1, -1.0, 0.0,
            bp.baseAddress, bp.count, &need)
    }
    guard rc == PE_OK else { throw BakeError.bakeFailed(rc) }
    return Data(buf.prefix(need))
}
```

Map `PE_ERR_*` to your app’s error type. Drive Godot the same way via the
existing GDExtension (`PePhysicsWorld`) or by shelling out to the bake CLI and
importing the JSON curves as `Animation` tracks.

## Python

```python
import physics_engine_rs as pe
scene = pe.bake_canonical_scene_json()
curves = pe.scene_bake_json(scene, 1.0, 1.0 / 240.0, 24.0)
assert pe.canonical_bake_json_hash() == pe.CANONICAL_BAKE_JSON_HASH
```

## Limits / deferred

* Bake always uses `ContactWorld::step` (semi-implicit Euler / soft-step as
  configured by the scene). Contact-free RK4 bake is **not** wired.
* Soft bodies / fluids / blast fragments are outside the rigid bake schema.
* No glTF / USD export yet — JSON curves only.
