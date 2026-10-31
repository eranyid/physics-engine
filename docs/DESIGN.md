# physics-engine — Design

Rigorous, reusable real-world physics simulation in Python (NumPy/SciPy, CPU only).
Correctness first: every model ships with tests against closed-form results.

## Architecture / module layout

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
  # stage 3: aero.py, hydro.py
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
  identical inputs give bit-identical outputs (tested).

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
