# Integrator and contact hot-loop baselines (Criterion)

Measured on the Cloud Agent host (Linux x86_64, release, Rust 1.99.1).
Numbers are wall time for the named workload; re-run locally after numerical changes.

Default CI does **not** run Criterion (`cargo test` omits harness benches). Use
`cargo bench -p pe-core --bench integrators`,
`cargo bench -p pe-collision --bench contact_pyramids`,
`cargo bench -p pe-aero --bench aero_forces`,
`cargo bench -p pe-hydro --bench hydrostatics`, and
`cargo bench -p pe-space --bench space`,
`cargo bench -p pe-blast --bench blast`,
`cargo bench -p pe-ballistics --bench ballistics`, and
`cargo bench -p pe-fluid --bench fluid`.

## Stage 1 — integrators (`pe-core`)

| Benchmark | Workload | Median time |
| --- | --- | --- |
| `ho_rk4_1k_steps` | 1 body spring HO, 1000× RK4 @ dt=1e-3 | ~533 µs |
| `ho_verlet_1k_steps` | same, velocity Verlet | ~472 µs |
| `ho_symplectic_1k_steps` | same, symplectic Euler | ~298 µs |
| `projectile_verlet_1k` | 1 body + uniform g, 1000× Verlet | ~472 µs |
| `nbody8_rk4_200_steps` | 8-body Newtonian, 200× RK4 @ dt=0.1 | ~454 µs |

## Stage 2 / M5a — contacts (`pe-collision`)

| Benchmark | Workload | Median time |
| --- | --- | --- |
| `pyramid_10_rows_60_steps_pgs` | 55 boxes, 60× PGS @ 1/240 | ~39.0 ms |
| `pyramid_10_rows_60_steps_soft` | 55 boxes, 60× soft-step @ 1/240 | ~126 ms |
| `pyramid_20_rows_60_steps_pgs` | 210 boxes, 60× PGS | ~171 ms |
| `pyramid_20_rows_60_steps_soft` | 210 boxes, 60× soft-step | ~596 ms |
| `pyramid_10_sap` | 10-row pyramid, SAP broad phase | ~36.4 ms |
| `pyramid_10_tree` | 10-row pyramid, dynamic AABB tree | ~39.2 ms |

## M5a — broad phase only

| Benchmark | Workload | Median time |
| --- | --- | --- |
| `broadphase/sap/1000` | 1k random AABBs | ~140 µs |
| `broadphase/tree/1000` | 1k random AABBs | ~511 µs |
| `broadphase/brute/1000` | 1k random AABBs | ~2.34 ms |
| `broadphase/sap/10000` | 10k random AABBs | ~11.4 ms |
| `broadphase/tree/10000` | 10k random AABBs | ~9.27 ms |

Tree rebuild-from-scratch is slower than SAP at 1k (query overhead) but wins at 10k.

## M5a — soft-step vs PGS quality comparison

5-row pyramid, 1 s @ dt=1/240, sleep off, SAP (from
`m5a_oracles::soft_step_vs_pgs_comparison_metrics`):

| Solver | Max contact depth [m] | Kinetic energy [J] |
| --- | --- | --- |
| PGS (20 iter) | ~0.30 mm (3.0e-4 m) | ~4.37 J |
| Soft-step (4 sub, Hertz 30, ζ 10) | ~2.29 mm (2.3e-3 m) | ~4.00 J |

Soft-step 1000:1 sphere stack penetration ~1.7 mm (under 5 mm + 5 mm slop;
PGS meets the 5 mm oracle). Soft-step KE is at or below PGS on this pyramid.

**Known limitation:** soft-step max contact depth on the 5-row pyramid remains
~2.29 mm vs PGS ~0.30 mm (~7.6× deeper). Soft-step recovers mass-ratio stacks and
matches/beats PGS on kinetic energy, but equal-mass pyramid residual penetration
is still larger than sequential-impulse PGS with 20 iterations. Values are
comparative / engineering bounds — not closed-form oracles.

Soft-step integrates forces per sub-step with `h`, warm-starts each sub-step, and
hard-holds high mass-ratio / heavy-static / single-body static contacts within
linear slop so soft spring equilibrium cannot park under a 1000:1 stack.

## M5b — joints / CCD (`pe-collision`)

| Benchmark | Workload | Median time |
| --- | --- | --- |
| `joints_double_pendulum_600` | soft-step double pendulum, 600× @ 1/240 | ~5.36 ms |
| `ccd_bullet_toi_sphere_plane` | 1000 m/s sphere vs plane TOI query | ~0.61 ns |

Re-record after changing joint softness or CCD paths. Analytic oracles live in
`tests/m5b_joints_ccd.rs` / `docs/JOINTS.md` (not throughput targets).

## M6 — GJK / EPA / convex CCD (`pe-collision`)

| Benchmark | Workload | Median time |
| --- | --- | --- |
| `gjk_sphere_sphere_1k` | 1000× GJK distance (separated spheres) | ~152 µs |
| `epa_box_box_1k` | 1000× EPA penetration (overlapping boxes) | ~5.24 ms |
| `ccd_ca_box_thin` | CA TOI fast box vs thin static box | ~1.43 µs |
| `convex_hash_scene` | canonical M6 convex hash scenario | ~19.0 ms |

Oracles: `tests/m6_convex.rs` / `docs/COLLISION.md`. Re-record medians on the
Cloud Agent host after GJK/EPA numerical changes.

## M9 — aero / hydro / waves (`pe-aero`, `pe-hydro`)

| Benchmark | Workload | Median time |
| --- | --- | --- |
| `forces_eval_1k` | aero+thrust+gravity force eval (iter) | ~85 ns |
| `rk4_1k_steps` | trimmed aircraft, 1000× RK4 @ dt=1e-3 | ~1.17 ms |
| `normal_shock_1k` | 1000× normal-shock evaluations | ~83 µs |
| `prandtl_glauert_1k` | 1000× PG corrections | ~4.0 µs |
| `dryden_1k_steps` | 1000× Dryden filter steps | ~70 µs |
| `buoyancy_forces_1k` | floating barge force eval (iter) | ~857 ns |
| `heave_rk4_500_steps` | floating barge, 500× RK4 @ dt=1e-3 | ~2.27 ms |
| `gz_curve_60_angles` | box GZ curve, 60 heel samples | ~51 µs |
| `airy_height_1k` | 1000× Airy η evaluations | ~12.7 µs |
| `spectral_height_eval` | one PM spectral η (24×8 components) | ~3.0 µs |
| `wave_buoyancy_rk4_200` | wave-coupled barge, 200× RK4 @ 1e-2 | ~211 µs |
| `morison_vertical_64` | Morison integrate, 64 segments | ~8.3 µs |
| `seakeeping_rao_one` | one heave/pitch RAO eval (16 strips) | ~940 ns |
| `bem_hover_eval` | one BEM hover evaluation (16 stations) | ~55 µs |
| `m11_hash_scenario` | canonical M11 Morison+BEM hash | ~17.8 ms |

## M7 — space (`pe-space`)

| Benchmark | Workload | Median time |
| --- | --- | --- |
| `leo_leapfrog_2k_steps` | circular LEO, 2000× leapfrog | ~105 µs |
| `leo_yoshida4_2k_steps` | same, Yoshida4 | ~337 µs |
| `leo_rk4_2k_steps` | same, RK4 | ~1.48 ms |
| `figure8_leapfrog_2k_steps` | figure-eight, 2000× leapfrog | ~141 µs |
| `ias15_e09_one_orbit` | `e=0.9` Kepler, IAS15 one period | ~1.12 ms |
| `whfast_5body_10k_steps` | synthetic 5-body WHFast Jacobi, 10k steps | ~14.2 ms |

## Blast / ballistics (`pe-blast`, `pe-ballistics`)

| Benchmark | Workload | Median time |
| --- | --- | --- |
| `kinney_1k` | 1000 Kinney–Graham evaluations | ~6.3 µs |
| `friedlander_10k` | 10k Friedlander samples | ~28 µs |
| `sod_n200_to_0.2` | Sod tube n=200 to t=0.2 (CFL 0.4) | ~3.9 ms |
| `sedov_radius_1k` | 1000 Sedov radius evaluations | ~45 µs |
| `blast_pyramid_hash` | canonical blast hash scenario | ~469 µs |
| `analytic_cd_100k` | 100k analytic C_D(M) samples | ~602 µs |
| `step_10k_bullets` | one ProjectileSystem step, 10k bullets | ~15.1 ms |
| `volley_hash` | canonical 100-bullet hash scenario | ~14.5 ms |

## Fluids (`pe-fluid`)

Measured on the Cloud Agent host (Linux x86_64, release).

Neighbour-list reuse across PBF iterations (default on):

| Benchmark | Workload | Before reuse | After reuse | Approx. rate (after) |
| --- | --- | --- | --- | --- |
| `fluid_tank_hash` | canonical PBF 4³ tank, 40 steps | ~86.4 ms | **~67.5 ms** (−22%) | — |
| `pbf_1k_particles_10_steps` | ~1000 fluid + boundary, 10 PBF steps | ~250 ms | **~183 ms** (−27%) | ~5.5×10⁴ particle·steps/s |
| `wcsph_256_particles_20_steps` | 8×8×4 WCSPH, 20 steps | ~41.9 ms | ~41.4 ms (noise) | ~1.2×10⁵ particle·steps/s |

Re-record with `cargo bench -p pe-fluid --bench fluid` after solver changes.

These are foundation baselines for M2–M9 + blast/ballistics/fluids — not
throughput targets. Re-record when changing `libm` usage, SoA layout, integrator
math, the contact solver, aero/hydro force models, waves, space integrators,
blast/ballistics, or SPH/PBF loops.
