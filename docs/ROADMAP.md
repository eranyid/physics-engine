# physics-engine — Roadmap

**Goal.** A general-purpose, full-simulation physics engine for studio, game, and
research use: space, air, sea, solids, soft bodies, fluids, explosions, materials.
**Decision.** Rust core + Python bindings (PyO3 / maturin); Godot GDExtension later.
The existing Python package remains the **reference oracle**: Rust results are
cross-checked against it, and both are validated against analytic / published
benchmarks. SI units; correctness before throughput.

This document plans architecture, domains, milestones, reference-project survey,
and risks. Implementation detail of stages 1–3 stays in [DESIGN.md](DESIGN.md).

---

## 1. Target architecture

### Workspace layout (planned)

```
physics-engine/
  src/physics_engine/          # Python reference oracle (stages 1–3, kept)
  tests/                       # analytic + oracle tests (Python)
  crates/
    pe-math/                   # f64 vectors, quaternions, frames, units helpers
    pe-core/                   # World, bodies, forces, integrators, diagnostics
    pe-collision/              # shapes, broad/narrow phase, contacts, CCD
    pe-constraints/            # joints, motors, soft constraints, islands
    pe-aero/                   # ISA atmosphere, aero force models
    pe-hydro/                  # buoyancy, waves, ship/fluid interaction hooks
    pe-space/                  # n-body, orbital helpers, gravity fields
    pe-blast/                  # Hopkinson–Cranz / Friedlander / Sedov / Euler blast
    pe-ballistics/             # point-mass ballistics, ammo profiles, projectile SoA
    pe-materials/              # constitutive models (elastic/plastic/thermal)
    pe-soft/                   # soft bodies / FEM / XPBD cloth-style (later)
    pe-fluids/                 # particles / SPH / blast coupling (later)
    pe-scene/                  # serialization, scene graph, sleeping metadata
    pe-ffi/                    # stable C ABI for GDExtension and others
  python/                      # PyO3 crate → maturin wheel (`physics_engine_rs`)
  godot/                       # GDExtension (post-Python bindings)
  docs/
  THIRD_PARTY.md               # attribution when any third-party code is reused
```

Feature flags on `pe-core` gate optional domains so game builds stay lean.

### Core data model

- **SoA primary storage** (positions, quaternions `[w,x,y,z]`, linear velocity,
  world-frame angular momentum **L**, mass / inertia / flags) matching the Python
  `World` layout for oracle diffs.
- Body handles are stable indices with generational IDs; force generators and
  constraints reference handles, not raw pointers.
- Shape / material tables are separate; colliders attach to bodies (or statics).
- Step pipeline (contact worlds): integrate forces → broad → narrow → solve →
  integrate positions → positional correction / CCD TOI substeps as enabled.

### Determinism and units

- **SI only** (m, kg, s, N, rad). No silent unit conversion in the core.
- **f64 first** everywhere in the deterministic path. Optional `f32` feature later
  for games; never the validation default.
- Deterministic mode: fixed body/constraint iteration order, no unordered
  hash maps in the step, no nondeterministic parallelism, bit-stable across
  identical inputs on the same ISA (document platform caveats: FMA, libm).
- Randomness (fracture seeds, turbulence) is opt-in and seeded.

### SIMD / parallelism

1. Scalar f64 correctness.
2. Explicit SIMD (e.g. `wide` / hand-written) on narrow loops and batch force eval.
3. Parallelism only on **independent islands** or embarrassingly parallel phases
   (broad phase pairs, particle neighborhoods), behind a feature flag; default
   CI remains single-threaded deterministic.

### Collision: broad / narrow / CCD

| Stage | Default choice | Notes |
| --- | --- | --- |
| Broad | Sweep-and-prune (as in Python) → dynamic BVH / tree | Brute force kept as test oracle |
| Narrow | Dedicated primitives first; GJK/EPA for convex | Port existing sphere/box/capsule/plane |
| Contacts | Clipped manifolds, warm start | Match Python contact layout for diffs |
| CCD | Speculative contacts + TOI sweep for fast bodies | Required before thin-feature trust |

### Solver strategy

Phased, not exclusive:

1. **Sequential impulses / PGS** — port of current `ContactWorld` (ship first;
   oracle parity).
2. **Constraint graph + islands + sleeping** — Box2D/Rapier-style; needed for
   large scenes and joints.
3. **XPBD** — soft bodies, cloth, some joints; complementary module, not a
   wholesale replacement of rigid SI.
4. Direct / MLCP alternatives only if PGS proves insufficient for stiff stacks.

Joints: ball, hinge, slider, fixed, distance, motors/limits; breakable limits
later. Same impulse/XPBD machinery as contacts where possible.

### Scene / serialization

Versioned binary + JSON/RON scene format: bodies, shapes, materials, constraints,
force generators, domain parameters (atmosphere, waterplane). Round-trip tests
required before editor/Godot workflows.

### Bindings

| Binding | When | Role |
| --- | --- | --- |
| Rust API | M0+ | Native consumers, benchmarks |
| Python (PyO3/maturin) | M1–M2 | Oracle harness, studio tooling, notebooks |
| C ABI (`pe-ffi`) | M3+ | Stable surface for engines |
| Godot GDExtension | After C ABI | Game integration |

Python keeps importing `physics_engine` (oracle). Rust wheel is a separate
package (e.g. `physics_engine_rs`) until APIs converge; a thin facade may later
prefer Rust when installed.

---

## 2. Physics domains (priority order)

Effort: **S** ≈ small module / few person-weeks of focused work; **M** ≈ solid
subsystem; **L** ≈ large multi-milestone; **XL** ≈ research-grade ongoing program.
Sizes are relative engineering scope, not calendar schedules.

### D1 — Rigid body dynamics & integrators (exists in Python)

- **Models:** Newtonian translation; rotational L storage; gravity, drag, springs;
  Euler / semi-implicit / Verlet / RK4; diagnostics (E, P, L).
- **Validation:** free fall, drag closed forms, Kepler, HO convergence orders,
  torque-free tops, N-body momentum, determinism (existing suite).
- **Effort:** S to port; ongoing oracle maintenance.

### D2 — Collision & contact (exists in Python)

- **Models:** primitives, SAP, SAT manifolds, sequential impulses, friction,
  restitution, positional correction.
- **Validation:** mass properties, SAP≡brute, bounce \(e^2 h\), cradle, incline
  stick/slide, rolling \(5/7\), stack drift bounds (existing + slow markers).
- **Effort:** M to port + CCD + missing pairs (capsule–box, convex/GJK).

### D3 — Constraints / joints / islands / sleep

- **Models:** bilateral joints, motors, limits; island graph; sleep/wake.
- **Validation:** pendulum period \(2\pi\sqrt{L/g}\); motor tracking error;
  sleeping conserves resting stacks; no energy pump on locked pairs.
- **Effort:** M–L.

### D4 — Atmosphere & aerodynamics (exists in Python)

- **Models:** ISA; wing polar + stall; thrust; pitch derivatives.
- **Validation:** ISA tables; trim L=W, T=D; \((L/D)_{\max}\); dive terminal V;
  published polars as optional fixtures.
- **Effort:** S port; M for 6-DOF aero / compressibility later.

### D5 — Hydrostatics & ships (exists in Python)

- **Models:** submerged volume, buoyancy, GM/GZ, heave/roll, hull drag, added mass.
- **Validation:** Archimedes draft; GM analytic; GZ vs small-angle; periods;
  ITTC/published GZ curves when available.
- **Effort:** S port; L for seakeeping/waves coupling.

### D6 — Space / gravity / orbits

- **Models:** point-mass n-body; optional central + J2; Kepler/Lambert helpers;
  SRP / drag / third-body; non-inertial frames later. (**M7 shipped** in `pe-space` —
  IAS15 + WHFast landed; DOP853 still RKF78 stand-in.)
- **Validation:** Kepler two-body; figure-eight / Pythagorean conservation; J₂ nodal
  rate; sun-synchronous inclination; energy/L property tests. SPICE ephemeris later.
- **Effort:** M.

### D7 — Hydrodynamics & waves

- **Models:** linear Airy / Stokes II / Gerstner + spectral (PM/JONSWAP) in
  `pe-hydro` (**shipped**); Morison forces; strip-theory or panel hooks later.
- **Validation:** Airy elevation/kinematics + dispersion oracles; spectral `H_s`;
  wave-coupled buoyancy heave; Morison / RAO later.
- **Effort:** L (v1 waves done; Morison/seakeeping remain).

### D8 — Materials (elasticity, plasticity, friction, thermal)

- **Models:** Hookean / neo-Hookean continuum hooks; Mohr–Coulomb / von Mises
  yield; frictional contact already in D2; heat capacity + conduction later.
- **Validation:** uniaxial stress–strain; elastic wave speed \(\sqrt{E/\rho}\);
  plastic limit loads on simple geometries; Stefan / 1-D heat solutions.
- **Effort:** L (start M for rigid material params only).

### D9 — Soft bodies

- **Models:** XPBD or FEM tet/shell; cloth; fracture coupling to D10.
- **Validation:** cantilever deflection; hanging cable catenary; stretching modes
  vs analytic strings; energy dissipation bounds.
- **Effort:** L–XL.

### D10 — Fluids, particles, explosions / blast

- **Models:** SPH or hybrid particles; blast: Friedlander / Hopkinson–Cranz
  scaling as engineering loads before full CFD; debris as rigid/soft bodies.
- **Validation:** hydrostatic SPH rest; dam-break vs literature; Sod shock (if
  compressible); blast peak overpressure vs scaled distance charts.
- **Effort:** XL.

Domain priority for shipping: **D1→D2→D3→D4→D5→D6→D7→D8→D9→D10**, with D4/D5
ported early because the Python oracle already exists.

---

## 3. Phased milestones (each shippable & testable)

| ID | Milestone | Exit criteria | Status |
| --- | --- | --- | --- |
| **M0** | Docs & policy | This roadmap + DESIGN dual-track; `THIRD_PARTY.md` policy; CI still green on Python | **done** |
| **M1** | Rust skeleton | Workspace builds; `pe-math` + `pe-core`; maturin abi3 wheel; CI: `cargo test`/`clippy`/`fmt` + pytest | **done** |
| **M2** | Port stage 1 | Rust matches Python oracle on stage-1 analytic tests (tolerances documented); bit-stable deterministic mode | **done** |
| **M3** | Port stage 2 | Contacts/friction/stacks cross-validated; missing narrow pairs tracked; SAP≡brute in Rust | **done** |
| **M4** | Port stage 3 | ISA/aero/hydro parity with Python suite | **done** (this PR) |
| **M5** | Joints + islands + sleep | Pendulum/motor tests; large resting scene sleeps; Python may gain thin joint oracle or shared fixtures | **M5a done**; **M5b done** (joints + speculative/TOI CCD; convex GJK remains M6) |
| **M6** | CCD + convex | TOI tests (no tunnel at configured speeds); GJK/EPA convex; capsule–box | **done** (GJK/EPA, hulls, capsule–box, convex CA CCD; see `docs/COLLISION.md`) |
| **M7** | Space module | Kepler + n-body conservation suite in Rust (+ Python helpers as needed) | **done** (this PR) |
| **M8** | Scene I/O + C ABI | Round-trip serialization; `pe-ffi` headers; smoke GDExtension optional | **done** (pe-scene + pe-ffi + godot/; soft-body track also shipped as **M8 soft** / `pe-soft`) |
| **M9** | Waves / hydrodynamics v1 | Airy + Stokes/Gerstner/spectra + buoyancy coupling (Morison later) | **done** (Morison deferred) |
| **M10** | Materials + soft v1 | Elastic validation; one soft demo with analytic check | **done** (shipped as M8 soft / `pe-soft`) |
| **M11** | Fluids / blast v1 | Particle hydrostatic + one blast engineering model vs scaling law | planned (blast engineering track shipped early — see below) |

### M1–M2 as built (summary)

- Workspace: `crates/pe-math`, `crates/pe-core`, `python/` → `physics_engine_rs` (PyO3 **abi3-py310**).
- Determinism: pure-Rust `libm`, no `target-cpu=native` / FMA, fixed generator order; cross-OS
  hash CI on Linux/macOS/Windows (`docs/DETERMINISM.md`).
- Bindings: GIL released in `step`/`run`; state moved into NumPy via contiguous pack.
- Tests: Rust analytic suite + Python oracle (`tests/test_oracle_rust.py`); Criterion baselines in
  `docs/BENCHMARKS.md`.

### M3 as built (summary)

- New crate `crates/pe-collision`: shapes, `BroadPhase` (SAP + brute), `NarrowPhase` (dedicated
  primitives + box–box SAT), sequential-impulse `ContactWorld` (Coulomb friction, restitution,
  warm start via `FeatureId`, positional correction). Soft-step / CCD / joints / islands deferred.
- Extension hooks only (not implemented): dynamic AABB tree behind `BroadPhase`; GJK/EPA or
  SAT-with-feature-caching behind `NarrowPhase`; geometric feature ids on manifold points.
- Capsule–box still `NotImplemented` (tracked for M6 with convex/GJK).
- Determinism: sorted `(min id, max id)` pairs; `BTreeMap` warm-start cache; cross-OS collision
  hash CI (10-box pyramid, 600 steps) — `docs/DETERMINISM.md`.
- Tests: Rust unit + proptest narrow-phase; analytic contact suite; Python oracle
  `tests/test_oracle_contacts_rust.py` (near bit-exact). Criterion pyramid benches in
  `docs/BENCHMARKS.md`. Default CI stays fast (no Criterion; slow pytest marker unchanged).

### M5a as built (summary)

- Soft-step solver (Box2D v3 / Solver2D clean-room): sub-steps (default 4), soft constraints
  (contact Hertz 30, ζ 10, capped to 0.25×substep rate), per-substep warm start, relax without
  bias, restitution from pre-solve relative velocity (~1 m/s threshold). **PGS remains the
  default** (oracle-compatible).
- Dynamic AABB tree broad phase (SAH insert, fat AABBs, rotations) ≡ SAP ≡ brute; sleeping
  islands (0.05 m/s, 2°/s, 0.5 s) with wake on new contact / non-gravity force.
- Determinism: soft-step pyramid hash CI; Criterion benches for soft-step vs PGS and tree vs SAP.

### M5b as built (summary)

- Maximal-coordinate joints in `pe-collision` (distance, revolute+limits/motor/spring,
  prismatic+limits/motor, weld, wheel/suspension, mouse/target): soft Hertz/ζ, warm start,
  breakable force/torque thresholds, reaction queries; solved in soft-step sub-steps and on
  the PGS path. Joints merge into islands (sleeping respects them).
- CCD: speculative contacts (Box2D v3 style, off by default) + bullet TOI vs static planes
  (analytic sphere–plane; support binary search for box/capsule). No tunnel at 1000 m/s vs
  plane at 60 Hz (tested).
- Docs: [`docs/JOINTS.md`](JOINTS.md). Determinism: joints double-pendulum hash CI.
- **Deferred (M5b):** gear/pulley, hard XPBD joints.
- **M6 done:** GJK/EPA, convex hulls, capsule–box, convex–convex CA CCD
  (`docs/COLLISION.md`). Remaining CCD gap: angular CA bounds.

### M4 as built (summary)

- New crates `crates/pe-aero` (ISA through 20 km, `WingPolar` + stall, Fixed/Propeller/Jet
  thrust, `AircraftAero` force generator, trim / terminal-V / Lanchester phugoid helpers) and
  `crates/pe-hydro` (Box/Cylinder/Sphere/ConvexMesh hulls, buoyancy, GM/GZ, heave/roll periods,
  hull drag, added-mass helpers, **IMO IS Code MSC.267(85)** intact-stability checker).
- Exposed through `physics_engine_rs` (PyO3); generators register into `World` like stage 1–2.
- Determinism: portable `libm` only; no `HashMap` in the step; cross-OS aero+hydro hash CI
  (aircraft + floating barge, 1 s RK4) — `docs/DETERMINISM.md`.
- Tests: Rust published-value + analytic suites; Python oracle
  `tests/test_oracle_aero_hydro_rust.py`. Criterion benches in `docs/BENCHMARKS.md`.
  Default CI stays fast (no Criterion; slow pytest marker unchanged).
- **Out of scope (M4):** waves, 6-DOF aero check-cases, blade element, compressibility
  — **waves + check-cases + compressibility + Dryden shipped in M9 follow-on** (this PR).

### M8 as built (summary)

- New crates `pe-scene` (JSON/RON/`PESC` binary; bit-exact state+warm-cache restore)
  and `pe-ffi` (opaque `PeWorld*`, SoA transforms, raycast/overlap, contacts,
  `catch_unwind`, cbindgen header + C smoke in CI).
- `godot/` GDExtension (`godot` / gdext, MPL-2.0 dependency): `PePhysicsWorld` /
  `PeRigidBody3D`; Linux **compile check** in CI. Godot editor/headless **not** run
  (`docs/GODOT.md`).
- Docs: [`SCENE.md`](SCENE.md), [`FFI.md`](FFI.md), [`GODOT.md`](GODOT.md).
- Determinism: scene save/load/continue hash locked (= collision pyramid hash).

### M7 as built (summary)

- New crate `crates/pe-space`: `GravityField` + N-body `Integrator` traits (SI; μ used
  directly); Leapfrog, Yoshida4, RK4, adaptive **RKF78** (DOP853 stand-in), **IAS15**
  (Rein & Spiegel 2015), **WHFast**-style Wisdom–Holman (Jacobi / democratic-heliocentric);
  compensated Neumaier positions + origin rebase; Kepler elements ↔ state, Stumpff /
  universal propagation, Lambert (Izzo 2015 geometry); J₂, cannonball SRP, third-body,
  ISA/exponential drag.
- Clean-room from published papers only — no REBOUND / nyx-space / JSBSim code
  (`docs/SPACE.md`, `THIRD_PARTY.md`).
- Determinism: portable `libm`, no FMA / `HashMap` iteration; cross-OS figure-eight +
  IAS15 hash CI.
- Tests: Rust analytic + proptest; Python `tests/test_space_rust.py`. Criterion benches in
  `docs/BENCHMARKS.md`.
- Follow-up (deferred): Hairer DOP853 tableau + order tests; WHFast symplectic correctors.

### Blast / ballistics track

- New crates `crates/pe-blast` and `crates/pe-ballistics` (clean-room literature;
  SI; `libm`; no FMA / `HashMap` iteration). Docs: [`BLAST.md`](BLAST.md),
  [`BALLISTICS.md`](BALLISTICS.md), [`AMMO.md`](AMMO.md).
- Blast: Hopkinson–Cranz, Kinney–Graham, Friedlander, Swisdak/Kingery–Bulmash
  (verified coeffs), Rankine–Hugoniot, Sedov–Taylor (`ξ₀` from energy integral),
  1-D Euler HLLC/MUSCL/SSP-RK2 (Sod), rigid-body Friedlander loads, Gurney/Mott.
- Ballistics: `AmmoProfile` (serde), SoA `ProjectileSystem` (per-shot RNG), ISA +
  Dryden wind, Coriolis/Eötvös, spin/Magnus models, hit-scan, Poncelet /
  Recht–Ipson, ballistic tables; cross-OS 100-bullet volley hash.
- PyO3 + Python tests; Criterion benches recorded in [`BENCHMARKS.md`](BENCHMARKS.md).

### M8 soft / materials (`pe-soft`) as built

- New crate `crates/pe-soft`: XPBD particles + compliance (`α̃ = α/dt²`), substeps
  (Macklin 2019), distance / bend / volume / tetra StVK & stable neo-Hookean
  (Macklin & Müller 2021) / shape matching / pressure; static plane/sphere/box +
  deterministic spatial-hash self-collision. Docs: [`SOFT.md`](SOFT.md).
- `Material` table (density, E, ν, friction, restitution, yield, thermal) — all
  built-ins `verified: false` / nominal; E,ν → Lamé and XPBD compliance helpers.
- Analytic oracles: catenary rope, spring period, axial `FL/(EA)`, tet volume,
  cantilever vs Euler–Bernoulli (honest tolerances), cloth energy, zero-compliance
  rigid limit, energy bounds, substep study. Cross-OS soft hash CI.
- PyO3 `SoftWorld` returning NumPy `(n,3)` arrays; pytest in
  `tests/test_soft_rust.py`. Criterion benches in [`BENCHMARKS.md`](BENCHMARKS.md).
- **Deferred:** corotational FEM, plasticity, fracture, thermal coupling, rigid
  two-way coupling.

**Rule.** No milestone merges without: (1) analytic or published benchmark tests,
(2) oracle or cross-impl diff where a Python reference exists, (3) CI green.

---

## 4. Reference project survey

**Reuse rule.** Prefer learning algorithms and structure over copying code. Any
copied or derived source must keep copyright/license notices and be listed in
[`THIRD_PARTY.md`](../THIRD_PARTY.md) (created when the first reuse lands; until
then the file may state “none yet”). This project is MIT; Apache-2.0
dependencies are compatible with attribution; do not take code under licenses
incompatible with MIT distribution without an explicit owner decision.

| Repository | License (verified in-repo) | Learn / reuse | Relevance |
| --- | --- | --- | --- |
| [erincatto/box2d](https://github.com/erincatto/box2d) | **MIT** (`LICENSE`) | Islands, sleeping, contact caching, sequential impulses, determinism discipline, CCD ideas | **Primary** 2D rigid reference (lift ideas to 3D) |
| [dimforge/rapier](https://github.com/dimforge/rapier) | **Apache-2.0** (`LICENSE`, `Cargo.toml`) | Rust crate layout, f64 feature split, broadphase, joints, SIMD/parallelism patterns, Python binding precedent | **Primary** Rust architecture reference |
| [google/liquidfun](https://github.com/google/liquidfun) | **zlib** (Box2D `License.txt` / `NOTICE`; `MODULE_LICENSE_BSD_LIKE`) | Particle / liquid particle coupling to rigid body engine | **Relevant** for D10 particle fluids |
| [idmillington/cyclone-physics](https://github.com/idmillington/cyclone-physics) | **MIT** | Educational rigid-body & contact structure; force generators; clear math presentation | **Relevant** pedagogy / API clarity |
| [liabru/matter-js](https://github.com/liabru/matter-js) | **MIT** | 2D engine ergonomics, constraint demos, broadphase practicalities | **Secondary** (JS, 2D only) |
| [galacean/engine](https://github.com/galacean/engine) | **MIT** | How a full engine embeds physics; component/scene patterns — not the solver itself | **Secondary** integration patterns only |

### Not relevant (do not treat as physics-engine references)

| Repository | License | Why excluded |
| --- | --- | --- |
| [pmndrs/react-spring](https://github.com/pmndrs/react-spring) | **MIT** | UI / React animation springs — not mechanical simulation |
| [melonjs/melonJS](https://github.com/melonjs/melonJS) | **MIT** (`LICENSE.md`) | HTML5 game framework; physics is not the product |
| [TheAlgorithms/C-Plus-Plus](https://github.com/TheAlgorithms/C-Plus-Plus) | **MIT** | Teaching algorithm collection; not a simulation architecture |

---

## 5. Risks and open questions

### Risks

- **Scope creep.** “Full simulation of everything” is larger than a game physics
  engine. Mitigation: milestones M1–M6 before soft/fluids; cut lines explicit.
- **Oracle drift.** Python and Rust diverging silently. Mitigation: shared
  fixture files (JSON states) and CI job diffing trajectories.
- **Determinism vs speed.** Parallelism and FMA break bit-stability. Mitigation:
  deterministic feature as default in tests; fast mode documented separately.
- **License hygiene.** Accidental copy without attribution. Mitigation:
  `THIRD_PARTY.md` gate in review checklist.
- **Aero/hydro fidelity expectations.** Studio use may need CFD-class results;
  engineering models will disagree with RANS/LES. Mitigation: document model
  class and validation envelope per module.
- **Single maintainer bandwidth.** Honest total scope for D1–D10 at production
  quality is on the order of **many person-months to multiple person-years** of
  focused work (M1–M6 alone is a substantial program; D9–D10 are open-ended).
  Ship vertical slices, not a big-bang engine.

### Open questions

1. **Solver of record for rigid 3D:** stay on sequential impulses long-term, or
   migrate islands to a Rapier-like pipeline after oracle parity?
2. **Soft body method:** XPBD vs FEM first?
3. **Fluid method:** SPH in-process vs coupling to an external solver?
4. **Godot timing:** after M8 only, or prototype FFI earlier?
5. **Public API name / versioning:** keep `physics_engine` for Python oracle
   forever, or eventually unify under one package with a `reference=` backend?
6. **f32 games mode:** required for Godot targets, or f64 acceptable?

### Effort snapshot (honest)

| Band | Contents | Relative effort |
| --- | --- | --- |
| Near-term | M0–M4 (docs, skeleton, port stages 1–3 + oracle CI) | M–L |
| Mid-term | M5–M8 (joints, CCD, space, scene, FFI) | L |
| Long-term | M9–M11 + production hardening of soft/fluids/blast | XL, ongoing |

Success for the redefined goal is not “feature-complete universe physics,” but
a **credible, validated, extensible core** that grows domain modules behind
tests — with Python as the correctness conscience for as long as it remains
valuable.
