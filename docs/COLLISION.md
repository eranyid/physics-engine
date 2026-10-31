# Collision — general convex (M6)

GJK distance / intersection, EPA penetration, convex hulls, capsule–box
manifolds, and convex–convex CCD (conservative advancement). Clean-room from
published algorithms — **no** Box2D / Bullet / Rapier source copied. See
[`THIRD_PARTY.md`](../THIRD_PARTY.md).

## Shapes

| Shape | Notes |
| --- | --- |
| Sphere / Box / Capsule / Plane | Unchanged dedicated pairs (locked hashes) |
| `Convex(ConvexHull)` | Quickhull-built polyhedron; optional Minkowski `radius` |
| `Rounded(inner, radius)` | Minkowski sum of an inner convex with a sphere |

`order_index` appends Convex=4, Rounded=5 so sphere/capsule/box/plane ordering
(and the stage-2 pyramid hash) stay bit-identical when new shapes are unused.

## Tolerances (honest)

Do **not** silent-loosen. Document changes here and in
[`DETERMINISM.md`](DETERMINISM.md).

| Symbol | Default | Role |
| --- | --- | --- |
| `GjkTol::abs` | `1e-10` m | Absolute distance / origin-in-simplex |
| `GjkTol::rel` | `1e-12` | Relative vs support scale |
| `GjkTol::area` / `volume` | `1e-20` / `1e-30` | Degenerate simplex collapse |
| `GjkTol::max_iters` | 64 | Hard iteration cap |
| `EpaTol::abs` | `1e-8` m | Face-distance convergence |
| `EpaTol::rel` | `1e-10` | Relative EPA convergence |
| `EpaTol::max_iters` / `max_faces` | 64 / 64 | Caps |
| `HULL_EPS` | `1e-12` m | Coplanarity / hull validity |
| `CcdParams::toi_tolerance` | `1e-4` m | CA / TOI contact threshold |

Sphere–sphere EPA uses the closed form `ra+rb−‖c‖` (same numeric target as EPA).
Box–box face manifolds still use the dedicated SAT clipper for oracle parity;
GJK/EPA is used for general convex and mixed pairs.

## GJK / EPA

- **GJK**: van den Bergen distance sub-algorithm; terminates when support no
  longer improves `‖C‖² − C·v`, or the origin lies in the simplex.
- **EPA**: origin-containing tetrahedron from diagonal support seeds (avoids
  origin-on-edge tets for AABBs), then expanding polytope to the closest face.
- **Support maps**: sphere, OBB, capsule (segment⊕sphere), convex hull,
  rounded (`support_core + r n̂`).

## Manifolds

- **Capsule–box**: segment samples → sphere–box contacts (≤4), GJK/EPA fallback
  when deeply embedded.
- **Capsule–capsule**: closest-segment spheres + optional parallel endpoint
  contacts; feature ids for warm start.
- **Convex–convex**: EPA normal/depth; reference/incident face clipping
  (Sutherland–Hodgman) when a hull/box face is available; else single EPA point.
  Cap: **4** contacts in 3D. (No separate 2D solver path in this engine.)

Feature ids: `feature_pair(ref, inc) = (ref << 16) | inc`.

## CCD

1. Speculative contacts + bullet sphere–plane TOI (M5b) — unchanged.
2. **M6**: GJK conservative advancement for bullet bodies vs static convex
   (boxes/hulls). Verified: fast box vs thin static box at 60 Hz does **not**
   tunnel (`m6_convex::ccd_fast_box_vs_thin_static_no_tunnel`).
3. Angular velocity in CA is **deferred** (translation CA + documented gap);
   rotating bullets may still tunnel at extreme rates.

## Hull validity

`ConvexHull::validate`: closed manifold (each edge shared by 2 faces), Euler
`V − E + F = 2`, and convexity (no vertex strictly outside a face plane).

## Determinism hash

Scene: convex box-hull + capsule + rounded box on a static thick box, gravity,
600 steps @ 1/240, sleep off. Locked value:
[`CANONICAL_CONVEX_HASH`](../crates/pe-collision/src/lib.rs) /
[`DETERMINISM.md`](DETERMINISM.md).

Helper: `cargo run -p pe-collision --bin convex_hash --release`.

## Deferred

- Full angular CA (`ω` bounds in TOI).
- Gear / pulley joints (M5b deferral).
- Exact sphere-swept polyhedron inertia / volume for rounded shapes.
- 2D polygon clip path (engine is 3D-only).

## References (algorithms only)

- Gino van den Bergen, *Collision Detection in Interactive 3D Environments*
  (Morgan Kaufmann, 2003) — GJK + EPA.
- Erin Catto, GDC talks on GJK / contact manifolds / CCD.
- Barber, Dobkin, Huhdanpaa — Quickhull (ACM TOMS 1996).
- Ericson, *Real-Time Collision Detection* — segment/triangle closest points.
- Mirtich — conservative advancement (CCD).
