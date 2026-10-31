# Determinism policy (stages 1–3 / M2–M5a + M7 space + M9 waves/aero)

## Guarantees

On the **Rust step path** (`pe-core` integrators + force apply + quaternion `exp_rotvec`,
`pe-collision` broad/narrow/contact, `pe-aero` / `pe-hydro` force apply, and `pe-space`
N-body / Kepler / perturbation paths):

1. **f64 only**; no `target-cpu=native`, no explicit FMA (`mul_add` unused).
2. Elementary functions go through the pure-Rust [`libm`](https://crates.io/crates/libm) crate
   (`pe_math::libm_ops`: `sin`/`cos`/`tan`/`sqrt`/`cbrt`/`exp`/`log`/`tanh`/`fabs`/`hypot`/`atan2`/`atan`/`copysign`/`pow`/`asin`/`acos`).
3. Fixed body and generator order (`Vec`, never `HashMap` iteration).
4. Contact pairs sorted by `(min_id, max_id)` before narrow phase; warm-start cache is a
   `BTreeMap` keyed by `(a, b, feature)` (lookups only — never unordered iteration).
5. Hydro mesh waterplane points sorted by `atan2` about the centroid (`total_cmp`) — no hash maps.
6. Spectral-wave / Dryden randomness uses the portable seeded **xoshiro256\*\*** in
   `pe-math` (no external RNG crate; fixed component order — never `HashMap`).
7. Identical inputs → **bit-identical** packed state on the same ISA across OS when (1)–(6) hold.

Cross-check helpers:

- Stage 1: `cargo run -p pe-core --bin canonical_hash --release` /
  `physics_engine_rs.canonical_determinism_hash()`
- Stage 2: `cargo run -p pe-collision --bin collision_hash --release` /
  `physics_engine_rs.canonical_collision_hash()`
- Stage 3: `cargo run -p pe-hydro --bin aero_hydro_hash --release` /
  `physics_engine_rs.canonical_aero_hydro_hash()`
- M5a soft-step: `cargo run -p pe-collision --bin soft_step_hash --release` /
  `physics_engine_rs.canonical_soft_step_collision_hash()`
- M5b joints: `cargo run -p pe-collision --bin joints_hash --release` /
  `physics_engine_rs.canonical_joints_hash()`
- M7 space: `cargo run -p pe-space --bin space_hash --release` /
  `physics_engine_rs.canonical_space_hash()`
- M9 wave-body: `cargo run -p pe-hydro --bin wave_body_hash --release` /
  `physics_engine_rs.canonical_wave_body_hash()`

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

### Canonical M5a soft-step collision hash (locked)

Scenario: same 10-box partial pyramid as stage-2, but `SolverMode::SoftStep`
(default SoftStepParams: 4 substeps, Hertz 30, ζ 10, restitution threshold 1 m/s),
sleep disabled, 600 steps @ `dt = 1/240`.

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `676d67a223120e34` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

### Canonical M5b joints hash (locked)

Scenario (must stay in sync in `pe_collision::canonical_joints_hash`, the
`joints_hash` binary, and `physics_engine_rs.canonical_joints_hash()`):

- Double pendulum: two unit spheres, lengths 0.5 m, soft distance joints
  (Hertz 90, ζ 0.8) body0↔world and body1↔body0; soft-step (8 substeps);
  sleep off; uniform gravity.
- 600 steps @ `dt = 1/240`.
- Hash: FNV-1a 64 over packed dynamic state.

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `62156ada3fa770ad` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

### Canonical stage-3 aero+hydro hash (locked)

Scenario:

- Body 0: GA-like aircraft (`WingPolar` S=16, CD0=0.025, e=0.8, …), mass 1000 kg at
  `z=500` m, `v=(40,0,0)`, nose-up α≈0.05 rad; `AircraftAero` (constant ρ=ρ₀) +
  `FixedThrust(800 N)` + uniform gravity.
- Body 1: box barge 12×4×2 m, mass 24 000 kg at Archimedes draft, small heel 0.02 rad;
  `Buoyancy` (freshwater) + shared gravity.
- Integrate with **RK4**, `t_end=1`, `dt=1e-2`.
- Hash: FNV-1a 64 over little-endian IEEE bits of `State::pack()`.

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `25ecc6e6a8911d8c` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

### Canonical wave-body hash (locked)

Scenario (must stay in sync in `pe_hydro::canonical_wave_body_hash`, the
`wave_body_hash` binary, and `physics_engine_rs.canonical_wave_body_hash()`):

- Box barge 12×4×2 m, mass 24 000 kg at Archimedes draft (freshwater).
- Airy wave: A=0.25 m, T=8 s, h=50 m, θ=0; buoyancy reads local η + long-wave FK.
- Integrate with **RK4**, `t_end=2`, `dt=1e-2`.
- Hash: FNV-1a 64 over little-endian IEEE bits of `State::pack()`.

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `3f1f6102a4606fbe` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

Helper: `cargo run -p pe-hydro --bin wave_body_hash --release`.

### Canonical M7 space hash (locked)

Scenario (must stay in sync in the binary, Rust scenario helper, and Python helper):

- Chenciner–Montgomery figure-eight, `G = m = 1`
- `x₁ = (0.97000436, −0.24308753, 0)`, `x₂ = −x₁`, `x₃ = 0`
- `v₃ = (−0.93240737, −0.86473146, 0)`, `v₁ = v₂ = −v₃/2`
- Integrate **2000** leapfrog steps over one period `T = 6.32591398`
- Hash: FNV-1a 64 over little-endian IEEE bits of packed `(x‖v)`

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `4633eb9e6412be18` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

### Canonical M7 IAS15 hash (locked)

Scenario (must stay in sync in the binary, Rust scenario helper, and Python helper):

- Earth-μ Kepler, `a = 1e7 m`, `e = 0.5`, `i = 0.2`, Ω = 0.1, ω = 0.3, ν = 0
- Compensated positions on; IAS15 with `ε_b = 1e-9`, initial `dt = T/100`
- **64 accepted** adaptive steps; hash FNV-1a 64 over packed `(x‖v)`

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `b918884ca264b70d` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

`cargo run -p pe-space --bin space_hash --release` prints figure-eight then IAS15
(two lines).

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
| ISA T/p/ρ/a (0–20 km) | **0** / ≤1e-14 | Same constants + `pow`/`exp` via libm |
| Wing polar CL/CD (incl. stall) | **0** | Pure algebra |
| Level-flight trim dict | ≤1e-14 | `sqrt` in V |
| Aero forces at α=0 | ≤1e-12 | Same force algebra |
| Trim net force (world) | ≤1e-5 N abs | Matches Python atol |
| Buoyancy equilibrium force | ≤1e-8 N abs | Archimedes upright closed form |
| Box GZ curve (small φ) | ≤1e-9 rel | Mesh clip path vs NumPy |
| Roll period formula | **0** / ≤1e-15 | Pure algebra |

Aim for near bit-exact on contacts and aero/hydro algebra; residual ~1 ULP differences
come from portable `libm` vs NumPy/`math` on a handful of `sqrt`/`sin`/`cos`/`pow`/`exp`
calls — not from algorithm or ordering drift.

Within Rust, `test_rust_determinism_repeat`, `determinism_bitwise_repeat`, and
`collision_hash_is_bit_stable_on_repeat` require **bit-identical** packed state on repeat.
The stage-3 hash is asserted equal on repeat via `canonical_aero_hydro_hash`.
The M7 figure-eight hash is asserted equal on repeat via `canonical_space_hash` /
`figure_eight_hash_bit_stable_on_repeat`.
