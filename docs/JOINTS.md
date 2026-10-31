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

## Pendulum period validation (error budget)

A planar distance-joint pendulum is the joint oracle. Period is measured from
successive positive-\(x\) zero-crossings (first interval skipped).

**Dominant error sources (diagnosed, release host):**

1. **Incorrect initial placement** (off the circle of radius \(L\)) injects a
   soft-joint stretch transient and was the main reason early tests needed 5–8%
   bands. Oracles now place the bob at
   \((L\sin\theta_0,\,0,\,-L\cos\theta_0)\).
2. **Comparing small-angle formula at finite \(\theta_0\)**: the exact period is
   \(T = 4\sqrt{L/g}\,K(\sin(\theta_0/2))\). Relative to \(2\pi\sqrt{L/g}\),
   \(T/T_0 - 1 \approx \theta_0^2/16\). At \(\theta_0 = 0.05\) that alone is
   ~0.016%; tests that claim “small-angle” use \(\theta_0 = 0.01\) (~0.0006%
   bias) or compare to the elliptic formula.
3. **Joint softness / Hertz**: soft distance bias + CFM shifts the effective
   length and damps. Raising Hertz (and substeps so the Nyquist cap allows it)
   shrinks period error monotonically (see
   `pendulum_period_converges_with_stiffness`).
4. **Sub-step size / semi-implicit soft-step**: once Hertz is high enough that
   length error is \(\lesssim 10^{-6}\,L\), further substeps change period at
   the \(10^{-5}\) relative level — phase error is **not** the leading term for
   these oracles.
5. **Relax pass**: hard relax after soft solve does not dominate period error
   once (1)–(3) are fixed.

**Tight oracles (stiff joints):**

| Case | Settings | Theory | Measured rel. err (release) | Bound |
| --- | --- | --- | --- | --- |
| Small angle \(\theta_0=0.01\) | 16 substeps, 500 Hz, \(\zeta=0.1\), \(dt=1/240\) | \(2\pi\sqrt{L/g}\) | \(\ll 1\%\) (~0.02% class) | **&lt; 1%** |
| Large amplitude \(\theta_0=1\) | 32 substeps, 1000 Hz, \(\zeta=0.05\), \(dt=1/480\) | \(4\sqrt{L/g}\,K(\sin(\theta_0/2))\) (AGM \(K\)) | \(\ll 1\%\) (~0.001% class) | **&lt; 2%** |

Default softness (8 substeps, 120 Hz, \(\zeta=0.7\)) still matches the elliptic
period to about 1% when ICs are on-circle.

## Double-pendulum energy drift

Soft distance joints + soft-step **dissipate**; exact conservation is not
claimed. With circular ICs (\(\theta_1=0.2\), \(\theta_2=0.3\)), two unit masses,
\(L=0.5\), stiff settings (32 substeps, 500 Hz, \(\zeta=0.1\), \(dt=1/480\)):

| Window | \(\lvert\Delta E\rvert\) | Energy span | Notes |
| --- | --- | --- | --- |
| 20 s after 0.5 s settle | ~\(2\times 10^{-5}\) J | ~\(2\times 10^{-4}\) J | ~0.0001% of \(\lvert E\rvert\) |
| 60 s (same settings) | ~\(9\times 10^{-5}\) J | ~\(2\times 10^{-4}\) J | slow secular drain |

Looser joints (8 substeps, 90 Hz, \(\zeta=0.8\), \(dt=1/240\)) over 20 s:
\(\lvert\Delta E\rvert\sim 3.5\times 10^{-4}\) J, span \(\sim 1.7\times 10^{-3}\) J.
Earlier “span ~14 J” figures came from off-circle ICs (stretch → PE dump), not
from the integrator alone.

## Limits of the approach

- Soft PGS / soft-step is **not** a hard XPBD or reduced-coordinate solver.
  Stiff limits and welds can overshoot; use higher Hertz / more substeps.
- Prismatic spring–mass period vs Hertz is only approximately `T ≈ 1/f` under
  light damping in soft-step (see tests; **partially unverified** as an exact
  oscillator identity).
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

- Small-angle pendulum (stiff): \(T = 2\pi\sqrt{L/g}\) within **1%**
  (`pendulum_small_angle_period_stiff_under_1pct`).
- Large amplitude (stiff): \(T = 4\sqrt{L/g}\,K(\sin(\theta_0/2))\) with AGM
  \(K\) within **2%** (`pendulum_large_amplitude_elliptic_stiff_under_2pct`).
- Stiffness convergence of period error
  (`pendulum_period_converges_with_stiffness`).
- Double-pendulum energy span/drift bounds over 20 s
  (`double_pendulum_energy_drift_documented`).
- Chain of 50 distance links hangs stably.
- Breakable distance joint breaks under weight when `F_break < mg`.
- Locked cross-OS joints hash: see [`DETERMINISM.md`](DETERMINISM.md).

## References (algorithms only)

- Erin Catto, [Solver2D](https://box2d.org/posts/2024/02/solver2d/) (2024-02-05);
  GDC talks on soft constraints / sequential impulses.
- Box2D v3 release notes (speculative contacts, soft joints) — ideas only.
- Abramowitz & Stegun — complete elliptic integral via AGM (pendulum period).
