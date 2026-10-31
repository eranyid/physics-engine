# Integrator and contact hot-loop baselines (Criterion)

Measured on the Cloud Agent host (Linux x86_64, release, Rust 1.99.1).
Numbers are wall time for the named workload; re-run locally after numerical changes.

Default CI does **not** run Criterion (`cargo test` omits harness benches). Use
`cargo bench -p pe-core --bench integrators` and
`cargo bench -p pe-collision --bench contact_pyramids`.

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
| PGS (20 iter) | ~3.0e-4 | ~4.37 |
| Soft-step (4 sub, Hertz 30, ζ 10) | ~9.2e-3 | ~10.0 |

Soft-step 1000:1 sphere stack penetration remains ~0.52 m (PGS meets the 5 mm
oracle). Values are comparative / engineering bounds — not closed-form oracles.

These are foundation baselines — not throughput targets. Re-record when changing
`libm` usage, SoA layout, integrator math, or the contact solver.
