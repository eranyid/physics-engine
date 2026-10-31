# Joints and CCD (M5b)

Maximal-coordinate joints solved inside the existing soft-step / PGS contact
solvers, plus speculative contacts and bullet TOI CCD. Clean-room from published
Catto / Box2D v3 descriptions — **no Box2D or Rapier source copied**. See
[`THIRD_PARTY.md`](../THIRD_PARTY.md).

## Softness

Joint bilateral rows use the same Hertz / ζ map as contacts
([`Softness::from_hertz_zeta`](../crates/pe-collision/src/softness.rs)):

```text
ω = 2π f
a1 = 2ζ + ω h
a2 = ω h · a1
a3 = 1 / (1 + a2)
bias_rate     = ω / a1
mass_scale    = a2 · a3
impulse_scale = a3
```

`h` is the **sub-step** size for soft-step (`dt / substeps`) or the full `dt` on
the PGS path. Hertz is capped at `0.25 / h` (same Nyquist-style cap as contacts).

Defaults: `f = 60 Hz`, `ζ = 0.7` ([`DEFAULT_JOINT_HERTZ`] /
[`DEFAULT_JOINT_ZETA`]).

## Joint kinds (maximal coordinates)

| Kind | Constraints (approx.) | Notes |
| --- | --- | --- |
| Distance | 1 linear | Rest length; soft spring about length |
| Revolute | 3 point + 2 axis-align (+ limit / motor / spring) | Hinge about `local_axis_a` |
| Prismatic | 2 lateral + 2 angular (+ limit / motor) | Slide along axis |
| Weld | 3 linear + 3 angular | Soft relative pose lock |
| Wheel | 2 lateral + 1 soft axial spring | Suspension; free spin about axis |
| Mouse | 3 soft to world target | Optional `max_force` clamp |

Warm starting stores per-row impulses on the `Joint` and reapplies each step /
sub-step. Breakable joints zero out when `|F|` or `|τ|` exceeds thresholds
(reaction force/torque are averages over the last sub-step: `λ / h`).

Islands: active joints contribute sorted `(min,max)` pairs so connected bodies
share an island; sleeping respects the merged graph.

## Limits of the approach

- Soft PGS / soft-step is **not** a hard XPBD or reduced-coordinate solver.
  Stiff limits and welds can overshoot; use higher Hertz / more substeps.
- Prismatic spring–mass period vs Hertz is only approximately `T ≈ 1/f` under
  light damping in soft-step (see tests; **partially unverified** as an exact
  oscillator identity).
- Double-pendulum energy is **bounded** but not conserved (soft dissipation) —
  exact conservation is **unverified** / not claimed.
- Revolute angular limit uses a soft unilateral row; large impacts can overshoot
  the band by a few tenths of a radian at default stiffness.
- Motor target speed is matched about the hinge axis with a force-limited hard
  velocity row; sign convention: positive `target_speed` drives `ω_a · axis`
  (body A) toward that value when B is world/fixed.

## CCD

1. **Speculative contacts** (off by default): for dynamic–static planes within
   `speculative_distance`, emit a unilateral contact with separation `sep ≥ 0`
   and bias `sep/h` so approach cannot close faster than available gap
   (Box2D v3 style).
2. **Bullet TOI** (off by default): for `bullet`-flagged bodies, analytic
   sphere–plane TOI (and support-based binary search for box/capsule vs plane)
   advances the body to impact before the discrete step. Verified: **no tunnel**
   for a sphere at 1000 m/s against a plane (“1 cm wall” class) at 60 Hz.

Full convex–convex GJK CCD is **deferred** (tracked with M6 GJK/EPA).

## Analytic oracles (tests)

- Small-angle pendulum: `T = 2π √(L/g)` (within ~5 %).
- Large amplitude: `T = 4 √(L/g) K(sin(θ₀/2))` via AGM elliptic `K` (~8 %).
- Chain of 50 distance links hangs stably.
- Breakable distance joint breaks under weight when `F_break < mg`.
- Locked cross-OS joints hash: see [`DETERMINISM.md`](DETERMINISM.md).

## References (algorithms only)

- Erin Catto, [Solver2D](https://box2d.org/posts/2024/02/solver2d/) (2024-02-05);
  GDC talks on soft constraints / sequential impulses.
- Box2D v3 release notes (speculative contacts, soft joints) — ideas only.
- Abramowitz & Stegun — complete elliptic integral via AGM (pendulum period).
