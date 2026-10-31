# Determinism policy (stages 1–2 / M2–M3)

## Guarantees

On the **Rust step path** (`pe-core` integrators + force apply + quaternion `exp_rotvec`,
and `pe-collision` broad/narrow/contact):

1. **f64 only**; no `target-cpu=native`, no explicit FMA (`mul_add` unused).
2. Elementary functions go through the pure-Rust [`libm`](https://crates.io/crates/libm) crate
   (`pe_math::libm_ops`: `sin`/`cos`/`sqrt`/`exp`/`log`/`tanh`/`fabs`/`hypot`/`atan2`/`copysign`).
3. Fixed body and generator order (`Vec`, never `HashMap` iteration).
4. Contact pairs sorted by `(min_id, max_id)` before narrow phase; warm-start cache is a
   `BTreeMap` keyed by `(a, b, feature)` (lookups only — never unordered iteration).
5. Identical inputs → **bit-identical** packed state on the same ISA across OS when (1)–(4) hold.

Cross-check helpers:

- Stage 1: `cargo run -p pe-core --bin canonical_hash --release` /
  `physics_engine_rs.canonical_determinism_hash()`
- Stage 2: `cargo run -p pe-collision --bin collision_hash --release` /
  `physics_engine_rs.canonical_collision_hash()`

### Canonical stage-1 scenario hash (locked)

Scenario (must stay in sync in the binary, Rust test, and Python helper):

- Bodies: mass 1 at `(0,0,10)` with `v=(1,0,0)`; mass 2 at `(1,0,0)` with
  `v=(0,0.5,0)`, `I=diag(0.4,0.5,0.6)`, `ω=(0.1,0,0.2)`.
- Generators: `UniformGravity`, `LinearDrag(0.01)`, `Spring(0→(0,0,10), k=5)`,
  `ConstantTorque(1, (0,0,0.05))`.
- Integrate with **RK4**, `t_end=2`, `dt=1e-3`.
- Hash: FNV-1a 64 over little-endian IEEE bits of `State::pack()`.

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `ca172fe3e3e57c79` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

### Canonical stage-2 collision hash (locked)

Scenario:

- 10-box pyramid (rows of 1+2+3+4 half-extent-0.5 boxes, μ=0.5, e=0) on a static
  `z=0` plane (μ=1), uniform gravity, 20 PGS iterations.
- 600 contact steps at `dt = 1/240`.
- Hash: FNV-1a 64 over packed dynamic state after the run.

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `50d52b2193aa528c` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

If a platform ever diverges, **do not silently loosen**: record the OS, toolchain,
exact first differing packed index / float bits, and root cause (usually accidental
platform `libm` or FMA) in this table.

## Oracle vs Python reference

The Python package uses NumPy/SciPy (platform libm, vectorized reductions). Therefore:

| Scenario | Measured max \|Rust−Python\| (Linux x86_64) | Notes |
| --- | --- | --- |
| Projectile / free fall (RK4 / Verlet) | **0** (bit-identical in practice) | Polynomial / near-exact |
| Linear drag (RK4, 12 s) | **0** | Same force algebra; libm≈NumPy here |
| Harmonic oscillator (RK4) | **0** | Scalar spring |
| Torque-free `L` / KE (5 s RK4) | ~4e-16 abs / ~7e-16 rel KE | Quaternion `exp_rotvec` libm vs NumPy |
| Kepler half-period (RK4) | ≤ 1e-9 rel on separation (slow) | NumPy `einsum` pair loop ≠ scalar `i<j` order |
| Quadratic drag | ≤ 1e-10 rel (oracle test bound) | `sqrt` in \|u\| |
| Head-on equal-mass e=1 spheres (contact) | **0** | Same SI algebra + pair order |
| Block on incline (slide / hold) | ~1e-15 | Contact PGS; libm `sqrt` in μ combine |
| Rolling sphere → 5/7 v₀ | ~1e-15 | Same friction disk projection |
| 3-box stack 0.5 s | ~1e-14 | Manifold clipping + warm start |

Aim for near bit-exact on contacts; residual ~1 ULP differences come from portable
`libm` vs NumPy/`math` on a handful of `sqrt`/`sin`/`cos` calls in the shared
quaternion / SAT path — not from algorithm or ordering drift.

Within Rust, `test_rust_determinism_repeat`, `determinism_bitwise_repeat`, and
`collision_hash_is_bit_stable_on_repeat` require **bit-identical** packed state on repeat.
