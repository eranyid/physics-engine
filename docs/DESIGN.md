# physics-engine — Design

Rigorous, reusable real-world physics simulation. Correctness first: every model
ships with tests against closed-form results (and, where noted, published tables).

**Strategic direction (owner decision).** The long-term product is a
**Rust core** with **Python bindings** (PyO3 / maturin), later usable from Godot
via GDExtension. The Python package in `src/physics_engine/` remains the
**reference oracle**: stages 1–3 stay maintained for analytic validation and for
cross-checking Rust ports. Roadmap, domain priorities, milestones, reference
survey, and risks: [ROADMAP.md](ROADMAP.md).

This document describes the **as-built Python reference** (stages 1–3) and the
invariants that the Rust port must preserve (units, frames, numerics, test bar).

## Architecture / module layout

### Dual-track (as of M1–M4)

| Track | Role |
| --- | --- |
| `src/physics_engine/` (Python) | Reference oracle; analytic tests; slow/energy runs |
| `crates/pe-math`, `pe-core`, `pe-collision`, `pe-aero`, `pe-hydro` (Rust) | Stages 1–3 production core; analytic + Criterion |
| `python/` → `physics_engine_rs` | PyO3 abi3 wheel; GIL released on step; oracle harness |
| Godot GDExtension | `godot/` + `pe-ffi` C ABI (M8; compile-checked — see docs/GODOT.md) |

Determinism / oracle tolerances: [DETERMINISM.md](DETERMINISM.md). Baselines:
[BENCHMARKS.md](BENCHMARKS.md).

### Rust stages 1–3 (as built)

```
crates/pe-math/       Vec3, Quat [w,x,y,z], SI constants, portable libm wrappers
crates/pe-core/       World SoA, forces, integrators, diagnostics, canonical_hash bin
crates/pe-collision/  shapes, BroadPhase/NarrowPhase traits, ContactWorld, collision_hash bin
crates/pe-aero/       ISA atmosphere, WingPolar/stall, thrust, AircraftAero
crates/pe-hydro/      Box/Cylinder/Sphere/Mesh hulls, Buoyancy, GM/GZ, IMO IS Code, aero_hydro_hash
crates/pe-fluid/      SPH (WCSPH) + PBF particle fluids (3-D / 2-D tests), fluid_hash
python/               PyO3 module physics_engine_rs (abi3 ≥ 3.10) — World + ContactWorld + aero/hydro/fluid
```

Invariants preserved from the Python oracle: SI units; world-frame `L`; quaternion
body→world; fixed-step Euler / symplectic Euler / velocity Verlet / RK4 with the same
implicit-midpoint rotational drift; force set including springs and constant torque;
stage-2 sequential impulses with Coulomb friction, restitution threshold, warm start,
and translational positional correction (β, slop). Pair order sorted by `(min id, max id)`.
Stage-3 aero/hydro match Python equations and constants (ISA, polar, trim, Archimedes,
GM/GZ, periods). Extra vs Python: IMO IS Code MSC.267(85) intact-stability checker;
`SphereHull` closed-form upright buoyancy. At `V → 0`, α/β clamp to 0 (no divide-by-zero);
Mach ≈ 1 singularities are not modelled (no compressibility).

**M9 waves / aero extensions:** `pe-hydro::waves` (Airy, Stokes II, Gerstner,
PM/JONSWAP spectral sum) coupled into `Buoyancy`; `pe-aero` compressibility
helpers, Dryden/steady wind, NASA TM flat-Earth check-cases 1–3.

**M11 Morison / seakeeping / BEM:** `pe-hydro::morison`, `pe-hydro::seakeeping`
(STF heave/pitch strip theory); `pe-aero::bem` (Glauert BEMT + Prandtl tip loss).
Docs: [`SEAKEEPING.md`](SEAKEEPING.md), [`ROTOR.md`](ROTOR.md). Soft bodies /
particle fluids remain separate tracks. Joints/islands/sleep (M5) and CCD +
convex/GJK (M6) are shipped; see [`JOINTS.md`](JOINTS.md) and
[`COLLISION.md`](COLLISION.md).

### Python reference (as built)

```
src/physics_engine/
  units.py        SI constants (from scipy.constants), unit type aliases, dimension algebra for tests
  quaternion.py   Hamilton quaternions: multiply, normalize, to/from rotation matrix, exp map
  body.py         RigidBody (user-facing description) + inertia helpers (sphere, box, cylinder)
  world.py        World: owns bodies + force generators, packs state into flat arrays (SoA)
  forces.py       Force generators: UniformGravity, NewtonianGravity, LinearDrag, QuadraticDrag, Spring
  integrators.py  explicit Euler, semi-implicit (symplectic) Euler, velocity Verlet, RK4
  diagnostics.py  kinetic/potential energy, linear and angular momentum
  shapes.py       Sphere, Box (OBB), Capsule, Plane (static half-space), Material; volume, inertia, AABB
  broadphase.py   sweep-and-prune on x (+ brute-force O(N^2) reference used in tests)
  narrowphase.py  sphere/sphere, sphere/plane, sphere/box, box/plane, box/box (SAT + clipping),
                  capsule/plane, capsule/sphere, capsule/capsule; dispatcher `collide`
  contact.py      ContactWorld: wraps a World, adds shapes/statics, sequential-impulse solver
  atmosphere.py   ISA / ICAO standard atmosphere (T, p, ρ, a vs geopotential altitude)
  aero.py         Wing polar (stall + parabolic drag), AircraftAero, Fixed/Propeller/Jet thrust
  hydro.py        Box/cylinder/convex-mesh hulls, buoyancy, GM/GZ, hull drag, added-mass helpers
```

Data flow: `World.step(dt, integrator)` → integrator queries `World.accelerations(state)` which
zeroes force/torque accumulators and lets each generator add to them. Bodies are stored as
structure-of-arrays (N×3 positions, N×4 quaternions, …) so later stages vectorize.

## Units

SI only (m, kg, s, N, rad). No unit conversion inside the engine. Enforcement:
* `units.py` exports `typing` aliases (`Meters`, `Kilograms`, `Newtons`, …) used in signatures;
  parameter names carry units in docstrings (e.g. drag `k` in kg/m).
* A small `Dim(M, L, T)` algebra is used in tests to check that formulas
  (terminal velocity, orbital period, energies) have the right dimensions.
* Constants come from `scipy.constants` (CODATA), never hand-typed.

## Frames and rotation convention

* World frame: right-handed, inertial. Gravity direction is a parameter (default −z).
* Body frame: origin at centre of mass, axes = principal axes when inertia is diagonal.
* Orientation quaternion `q = [w, x, y, z]` (scalar first, Hamilton product, active rotation
  body→world: `v_world = R(q) v_body`). Unit norm is re-imposed after every step.
* Rotational state is stored as **world-frame angular momentum L**; ω = R I_b⁻¹ Rᵀ L.
  This makes torque-free L conservation structural and handles the gyroscopic term exactly.

## Numerical choices

* Integrators (all fixed step):
  * explicit Euler — baseline, order 1, energy grows.
  * semi-implicit Euler — symplectic, order 1, bounded energy error.
  * velocity Verlet (kick-drift-kick leapfrog) — symplectic, order 2 for position-dependent forces;
    with velocity-dependent forces (drag) it uses the half-step velocity (still stable, lower order).
  * RK4 — order 4, not symplectic: secular energy drift on long runs.
  * Rotation in Euler/Verlet: L kicked by torque, q drifted with the exponential map evaluated at
    the midpoint attitude (implicit midpoint, fixed-point iterated):
    `q ← exp(dt ω(q_mid)) q`; using ω(q_n) instead gave ~2 % kinetic-energy drift in 20 s. RK4 integrates q̇ = ½ (0,ω) ⊗ q then renormalises.
* Timestep policy: fixed dt chosen by the caller (default target 1/240 s for games-scale,
  physically chosen for orbital tests). Adaptive stepping (scipy `solve_ivp`) may be added
  as a reference integrator, never in the deterministic core loop.
* Determinism: float64, fixed body/generator order, no threading, no randomness in the step;
  identical inputs give bit-identical outputs (tested). The Rust core inherits this policy
  for its deterministic feature (see ROADMAP § Determinism and units); optional parallelism
  must not be the validation default.
* Precision: **f64 first** in both tracks. An f32 games mode, if added, is a separate
  feature and is never the oracle.

## Test plan

Stage 1 (this release): free fall and projectile vs closed form; linear drag vs exponential
approach to mg/b; quadratic drag vs tanh solution and √(mg/k); Kepler circular and e=0.5
orbits (period, energy, angular momentum); symplectic vs RK4 long-run energy behaviour;
observed convergence order on the harmonic oscillator (Euler 1, symplectic Euler 1, Verlet 2,
RK4 4); torque-free rigid body (L and T conserved, |q|=1, symmetric-top precession rate
(I₃−I₁)ω₃/I₁, intermediate-axis instability); momentum conservation in isolated N-body;
determinism; unit dimension checks.

Stage 2 (collisions): shapes (sphere, box, capsule, convex hull) with mass-property tests vs
analytic inertia; broad phase (sweep-and-prune / BVH) tested against brute-force O(N²) pairs;
narrow phase (GJK/EPA, SAT) tested on known penetrations and separating distances;
contact resolution (sequential impulses) tested with restitution e (bounce height e²h),
Newton's cradle momentum transfer, resting stacks (no drift/jitter bound); Coulomb friction:
block on incline sticks iff tanθ ≤ μ_s, sliding acceleration g(sinθ − μ_k cosθ), rolling sphere
a = 5/7 g sinθ.

Stage 3 (vehicles): aircraft — lift/drag from C_L(α), C_D = C_D0 + C_L²/(πeAR), thrust;
tests: steady level flight trim (L = W, T = D), glide ratio = L/D, ISA atmosphere density vs
tables. Ship — buoyancy from submerged volume (Archimedes), tests: box/cylinder floating draft
= m/(ρA), metacentric height GM = KB + BM − KG with BM = I_wp/V, small-angle roll period
T = 2π k/√(g GM), righting-arm sign change beyond vanishing stability.

## Stage 2 as built (collisions)

**Pipeline** (`ContactWorld.step(dt)`, semi-implicit Euler with contacts):
1. kick `v += dt F/m`, `L += dt T` from the stage-1 force generators;
2. broad phase (sweep-and-prune over all AABBs, static–static pairs skipped) + narrow phase at the
   current poses; contacts carry world point, unit normal A→B and depth;
3. sequential impulses (projected Gauss–Seidel, default 20 iterations) on (v, L): an impulse P at
   lever r changes v by P/m and the stored world angular momentum by r×P (exact, no I⁻¹ round-trip);
   * normal: accumulated λ ≥ 0, target `vn ≥ −e·vn₀` when approach speed > `restitution_threshold`
     (0.2 m/s), else 0 (resting contacts don't bounce);
   * friction: two tangents, accumulated tangential impulse projected onto the Coulomb disk
     |jt| ≤ μλ (isotropic); μ combined as √(μa μb), e as max(ea, eb); one μ (static = kinetic);
   * warm starting: impulses cached per (pair, contact index) and reused if the point moved < 1 cm;
4. drift x and q (stage-1 implicit-midpoint exponential map);
5. positional correction: re-detect, per pair translate bodies apart by β(depth − slop)
   (β = 0.2, slop = 0.1 mm) split by inverse mass. Velocities are untouched, so no kinetic energy is
   injected (Baumgarte velocity bias is *not* used); linear momentum and the COM are preserved.

Statics (`StaticCollider`: plane, box, …) have infinite mass. Box–box SAT tests 15 axes, prefers
face axes (edge axis chosen only if 5 % shallower), clips the incident face against the reference
face side planes (Sutherland–Hodgman) and reduces to ≤ 4 points; edge–edge yields one point.

**Tests vs analytic results** (`tests/test_shapes_broadphase.py`, `test_narrowphase.py`,
`test_contacts.py`): box/sphere mass properties, capsule inertia vs 10⁶-sample Monte Carlo (0.5 %);
SAP = brute force on 10 random 200-box scenes incl. touching faces; known depths/normals for every
narrow-phase pair incl. edge–edge and an edge-axis separation; 1-D head-on collisions for
e = 1, 0.5, 0 (velocities, momentum, ΔKE = ½μ(1−e²)v_rel², and e itself all to 1e-12); off-centre
sphere→box hit with friction conserves P and L about the contact point to 1e-12; bounce apex
ratio e² and bounce times (dt = 1e-4: ≤ 0.23 % / 1.2 ms after 4 bounces, error is contact detection
lag ≤ dt per impact); block on incline slides at g(sinθ − μcosθ) (rel. error ~1e-14) and holds for
tanθ < μ (≤ 0.1 mm settling, then no creep); sphere launched sliding reaches v = 5/7 v₀ at
t* = 2v₀/(7μg) and then ω r = v; sphere rolls down incline at 5/7 g sinθ; box at rest for 5 s
(no xy drift, |R − I| < 1e-7); bit-identical reruns; dimension checks.
Slow (`pytest -m slow`): 3-box stack 5 s (xy drift < 0.1 mm, measured ~34 µm), tilted box drop
settles; plus the two long stage-1 energy runs. Default `pytest` skips them (`addopts = -m "not slow"`).

**Known limitations / risks**
* Pure Python per-contact loops: fine for tests and tens of bodies, too slow for large scenes
  (vectorisation or a compiled solver is future work).
* Only the semi-implicit Euler step supports contacts; RK4/Verlet remain contact-free.
* Discrete detection, no CCD: fast thin objects can tunnel; impact detected up to one step late.
* Single friction coefficient, decoupled tangent effective masses, no rolling/spin friction.
* Capsule–box not implemented (raises NotImplementedError); no convex hulls or GJK/EPA yet.
* Positional correction is translational only (rotational penetration is resolved by the
  velocity solver), and lifting bodies out of penetration adds ≤ m g β(depth − slop) potential
  energy per step. Tall stacks drift slowly (tens of µm over 5 s for 3 boxes); no sleeping.
* Warm-start matching by contact index assumes a stable contact order per pair.

## Stage 3 as built (vehicles — library layer, no game/combat/visuals)

Reusable force generators and pure helpers on top of stage-1 `World`. No coupling to contacts
beyond sharing rigid bodies; SI only.

### Atmosphere (`atmosphere.py`)
ISA / ICAO Doc 7488 through 20 km geopotential altitude: troposphere lapse 6.5 K/km then
isothermal lower stratosphere. `temperature`, `pressure`, `density`, `speed_of_sound`.
Sea-level constants T₀=288.15 K, p₀=101325 Pa, ρ₀=1.225 kg/m³; R = p₀/(ρ₀ T₀); a=√(γ R T).

### Aerodynamics (`aero.py`)
Body axes: +x nose, +y right, +z up (z-up world when level). α = atan2(−v_z, v_x) in body
frame. `WingPolar`: linear C_L(α) to α_stall then a linear stall break; parabolic
C_D = C_D0 + C_L²/(π e AR); pitching C_m = C_m0 + C_mα α + C_mq (q c)/(2V).
`AircraftAero` applies ½ρV²S (C_L ê_L + C_D ê_D) and M = ½ρV² S c C_m about +y.
Thrust: `FixedThrust`, `PropellerThrust` (min of static and ηP/V, ·ρ/ρ_sl), `JetThrust`
(T_sl · throttle · (ρ/ρ_sl)^n). Helpers: `level_flight_trim` (L=W, T=D at best L/D CL),
`terminal_velocity_vertical`.

### Hydrostatics (`hydro.py`)
`BoxHull`, `CylinderHull` (vertical), `ConvexMeshHull`: `submerged(x,q,water_level)` →
volume, COB, A_wp, I_xx/I_yy. Upright box/cylinder use closed form; general pose clips a
convex mesh against z=water_level (Sutherland–Hodgman) and closes with the waterplane face.
`Buoyancy` force generator: F = ρ g ∇ (+z), torque (r_COB−r_COM)×F. Analytic helpers:
KB, BM=I/∇, GM=KB+BM−KG, GZ≈GM sin φ, heave period 2π√(m/(ρ g A_wp)), roll period
2π k/√(g GM), large-angle `box_righting_arm_curve` / `angle_of_vanishing_stability`.
Optional: `HullDrag` (viscous C_f + residual/wave C_w(Fr)), `box_added_mass_heave/surge`
(bake into `RigidBody.mass`; not a force generator — acceleration is unavailable in `apply`).

**Tests vs analytic results** (`tests/test_atmosphere.py`, `test_aero.py`, `test_hydro.py`):
ISA table at 0/1/5/11/20 km vs published T, p, ρ, a (rel ~2e-4); drag polar and
(L/D)_max = 1/(2√(C_D0 K)); stall break below linear extrapolation; level-flight trim
L=W and T=D to 1e-12, net force ≈0 with freestream-aligned thrust (atol 1e-5 N); pitch
C_mα restoring moment to 1e-9; vertical-dive terminal velocity within 0.5 % after 90 s;
Archimedes draft T=m/(ρA) for box and cylinder (volume to 1e-12); barge GM=KB+BM−KG with
BM=B²/(12T); numerical GZ vs GM sin φ within ~2 % at φ≤5°; GZ sign change past AVS on a
low-freeboard barge; heave period from zero-crossings within 2 %; roll-period formula exact,
simulated small-heel period within ~8 % (slow). Default `pytest` keeps these fast; slow
markers cover trimmed level flight (5 s) and roll oscillation.

**Known limitations / risks**
* Longitudinal / planar aero only (no sideslip, no lateral-directional derivatives, no
  ground effect, no compressibility; stall model is a simple linear break).
* Propeller/jet models are first-order (no advance ratio map, no intake momentum drag).
* ISA stops at 20 km; geometric vs geopotential altitude not distinguished.
* Cylinder hull is vertical; heeled cylinder uses an N-gon prism (facet error).
* Convex-mesh buoyancy is for closed convex polyhedra with consistent outward winding;
  non-convex / open meshes are unsupported. Waterplane moments from clipped edges are
  approximate under coarse faceting.
* Added mass is a mass-property helper only (no radiation damping, no frequency dependence).
* No coupling of aero/hydro to the contact solver; vehicles do not yet “land” or “dock”
  through `ContactWorld`. No visuals, control systems, or combat logic.

## Beyond stage 3

Stages 1–3 close the original Python validation arc. Further work (Rust port,
joints/islands/CCD, space, waves, materials, soft bodies, fluids/blast, bindings)
is planned in [ROADMAP.md](ROADMAP.md). Design invariants above (SI, frames,
L-based rotation, f64 determinism, analytic test bar) remain binding unless this
document is explicitly revised.

**Third-party code.** Algorithms may be learned from surveyed engines; any reused
source must retain license notices and be recorded in `THIRD_PARTY.md` (see ROADMAP
§ Reference project survey).
