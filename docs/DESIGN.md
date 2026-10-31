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
  # stage 2: shapes.py, broadphase.py, narrowphase.py, contact.py
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
