# Determinism policy (stages 1–3 / M2–M9 + M8 soft + fluids + M11 Morison/seakeeping/BEM)

## Guarantees

On the **Rust step path** (`pe-core` integrators + force apply + quaternion `exp_rotvec`,
`pe-collision` broad/narrow/contact, `pe-aero` / `pe-hydro` force apply, `pe-space`
N-body / Kepler / perturbation paths, and `pe-soft` XPBD):

1. **f64 only**; no `target-cpu=native`, no explicit FMA (`mul_add` unused).
2. Elementary functions go through the pure-Rust [`libm`](https://crates.io/crates/libm) crate
   (`pe_math::libm_ops`: `sin`/`cos`/`tan`/`sqrt`/`cbrt`/`exp`/`log`/`tanh`/`fabs`/`hypot`/`atan2`/`atan`/`copysign`/`pow`/`asin`/`acos`/`sinh`/`cosh`/`floor`).
3. Fixed body and generator order (`Vec`, never `HashMap` iteration).
4. Contact pairs sorted by `(min_id, max_id)` before narrow phase; warm-start cache is a
   `BTreeMap` keyed by `(a, b, feature)` (lookups only — never unordered iteration).
5. Hydro mesh waterplane points sorted by `atan2` about the centroid (`total_cmp`) — no hash maps.
6. Spectral-wave / Dryden randomness uses the portable seeded **xoshiro256\*\*** in
   `pe-math` (no external RNG crate; fixed component order — never `HashMap`).
7. Soft-body self-collision uses a dense `Vec` spatial hash; pairs sorted by
   `(min_id, max_id)` before resolve (never `HashMap` iteration).
8. Identical inputs → **bit-identical** packed state on the same ISA across OS when (1)–(7) hold.

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
- M6 convex: `cargo run -p pe-collision --bin convex_hash --release` /
  `physics_engine_rs.canonical_convex_hash()`
- M7 space: `cargo run -p pe-space --bin space_hash --release` /
  `physics_engine_rs.canonical_space_hash()`
- Blast: `cargo run -p pe-blast --bin blast_hash --release` /
  `physics_engine_rs.canonical_blast_hash()`
- Ballistics: `cargo run -p pe-ballistics --bin ballistics_hash --release` /
  `physics_engine_rs.canonical_ballistics_hash()`
- Fluids (PBF): `cargo run -p pe-fluid --bin fluid_hash --release` /
  `physics_engine_rs.canonical_fluid_hash()`
- Fluids (DFSPH): `cargo run -p pe-fluid --bin dfsph_hash --release` /
  `physics_engine_rs.canonical_dfsph_hash()`
- M9 wave-body: `cargo run -p pe-hydro --bin wave_body_hash --release` /
  `physics_engine_rs.canonical_wave_body_hash()`
- M8 soft bodies: `cargo run -p pe-soft --bin soft_hash --release` /
  `physics_engine_rs.canonical_soft_hash()`
- M11 Morison+BEM: `cargo run -p pe-hydro --bin m11_hash --release` /
  `physics_engine_rs.canonical_m11_hash()`

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

### Canonical scene save/load/continue hash (locked)

Scenario (must stay in sync with `pe_scene::canonical_scene_roundtrip_hash` /
`scene_hash` binary):

- Same 10-box pyramid as the stage-2 collision hash.
- Run **300** contact steps @ `dt = 1/240`, capture scene (state + warm cache) to
  binary, load, run **300** more.
- Must match an uninterrupted **600**-step run bit-exactly.
- Hash: FNV-1a 64 over packed dynamic state (equals stage-2 collision hash when
  restore is correct).

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `50d52b2193aa528c` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

Helper: `cargo run -p pe-scene --bin scene_hash --release`.

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

### Canonical M6 convex hash (locked)

Scenario (must stay in sync in `pe_collision::canonical_convex_hash`, the
`convex_hash` binary, and `physics_engine_rs.canonical_convex_hash()`):

- Dynamic: convex box-hull (half 0.25), capsule (r=0.1, hh=0.2), rounded box
  (half 0.2 ⊕ r=0.05); materials μ=0.4, e=0; sleep off; 20 PGS iterations.
- Static: box half-extents (2,2,0.1) at z=−0.1, μ=0.8.
- Uniform gravity; 600 steps @ `dt = 1/240`.
- Hash: FNV-1a 64 over packed dynamic state.

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `4ca980096c036ade` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

Legacy stage-2 / M5a / M5b hashes are **unchanged** when M6 shapes are unused.

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

### Canonical M11 Morison + BEM hash (locked)

Scenario (must stay in sync in `pe_hydro::canonical_m11_hash`, the `m11_hash`
binary, and `physics_engine_rs.canonical_m11_hash()`):

- Body 0: vertical slender cylinder D=0.5 m, L=10 m, mass 2000 kg; Morison
  (`C_m=2`, `C_d=1` nominal) in Airy A=0.5 m, T=8 s, h=30 m (32 segments).
- Body 1: 500 kg at z=50 m; BEM rotor R=4 m, 2 blades, Ω=40 rad/s, thrust +z,
  constant ρ=ρ₀.
- Integrate with **RK4**, `t_end=1`, `dt=1e-2`.
- Hash: FNV-1a 64 over little-endian IEEE bits of `State::pack()`.

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `83aebe96f5cb8e4c` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

Helper: `cargo run -p pe-hydro --bin m11_hash --release`.

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

### Canonical blast-pyramid hash (locked)

Scenario (`pe_blast::blast_pyramid_hash`, `blast_hash` binary,
`physics_engine_rs.canonical_blast_hash()`):

- 6-box pyramid (rows 1+2+3), mass 10 kg, half-extent 0.25 m, uniform gravity.
- Friedlander blast load from `W_eq=1 kg` at `(-2,0,0.5)` m on exposed −x faces.
- 200× RK4 steps at `dt=1/240` with per-step blast impulses.
- Hash: FNV-1a 64 over packed `State`.

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `b51e11e89d4dd055` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

### Canonical ballistics volley hash (locked)

Scenario (`pe_ballistics::ballistics_volley_hash`, `ballistics_hash` binary,
`physics_engine_rs.canonical_ballistics_hash()`):

- 100 bullets mixed across 5 bundled ammo classes; Dryden + steady wind; lat≈0.7 rad.
- 120 steps at `dt=1/120`.
- Hash: FNV-1a 64 over `(id, x, v, alive)` in ascending shot-id order.

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `c839907696748e9a` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

### Canonical M8 soft-body hash (locked)

Scenario (`pe_soft::soft_canonical_hash`, `soft_hash` binary,
`physics_engine_rs.canonical_soft_hash()`):

- Merged rope (8 seg) + 4×4 cloth + 2-div jelly cube above a `z=0` plane.
- 120 outer steps at `dt=1/60`, 10 XPBD substeps, 1 iteration/substep.
- Hash: FNV-1a 64 over packed `(x‖v)` particle state (IEEE bits, LE).

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `adfc0a4c1209e5dc` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

Updated from `7e7d0d3af68735c6` when tet FEM switched to StVK/SNH energy XPBD
(`C=√(2Ψ)`, `α=1/V₀`; jelly cube in the canonical scene uses neo-Hookean tets).

### Canonical fluid-tank hash (locked)

Scenario (`pe_fluid::fluid_tank_hash`, `fluid_hash` binary,
`physics_engine_rs.canonical_fluid_hash()`):

- PBF 3-D tank: 4×4×4 fluid lattice (`dx=0.05`, `h=2dx`, `ρ₀=1000`) in an Akinci
  boundary shell; gravity −z; 4 constraint iterations; artificial pressure off.
- 40 steps @ `dt = 1/120`.
- Hash: FNV-1a 64 over fluid particle `(x, v)` IEEE bits in index order.
- Neighbour search: sorted uniform-grid cell lists (no `HashMap` iteration).

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `9d9e0da856fd43d4` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

Helper: `cargo run -p pe-fluid --bin fluid_hash --release`.

### Canonical DFSPH-tank hash (locked)

Scenario (`pe_fluid::dfsph_tank_hash`, `dfsph_hash` binary,
`physics_engine_rs.canonical_dfsph_hash()`):

- DFSPH 3-D tank: 4×4×4 fluid lattice (`dx=0.05`, `h=2dx`, `ρ₀=1000`, mass
  calibrated to cubic-spline lattice rest density) in an Akinci boundary shell
  with recomputed `ψ`; gravity −z; compression-only density/divergence solvers;
  warm start on; XSPH `ε=0.05`; max 8 density + 8 divergence iterations;
  tolerances 1%.
- 40 steps @ `dt = 1/120` (adaptive CFL off for the hash).
- Hash: FNV-1a 64 over fluid particle `(x, v)` IEEE bits in index order.
- Neighbour search: sorted uniform-grid cell lists (no `HashMap` iteration).
- Elementary functions: portable `pe_math::libm_ops` only; no FMA on the step path.

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `f1f2825b5eeaab0d` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

Helper: `cargo run -p pe-fluid --bin dfsph_hash --release`.

### Canonical fluid↔rigid coupling hash (locked)

Scenario (`pe_fluid::coupling_tank_hash`, `coupling_hash` binary,
`physics_engine_rs.canonical_coupling_hash()`):

- DFSPH 2-D floating box via Akinci samples (`make_floating_box_2d`, `dx=0.05`,
  `ρ_b=500`, tank/water as in `coupling_hash_world`).
- 80 steps @ `dt = 1/240`.
- Hash: FNV-1a 64 over fluid particle `(x, v)` IEEE bits, then rigid packed state.
- Reactions via `pe_core::ExternalLoads`; no `HashMap` iteration; `libm_ops` only.

| Platform | Hash (hex) | Status |
| --- | --- | --- |
| Linux x86_64 (CI + local) | `e9a6a84fef4ce9d5` | locked |
| macOS (CI) | same expected | CI asserts equality |
| Windows (CI) | same expected | CI asserts equality |

Helper: `cargo run -p pe-fluid --bin coupling_hash --release`.

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
