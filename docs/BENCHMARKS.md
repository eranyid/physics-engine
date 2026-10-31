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

## Stage 2 — contacts (`pe-collision`)

| Benchmark | Workload | Median time |
| --- | --- | --- |
| `pyramid_10_rows_60_steps` | 55 boxes (10 rows) + ground, 60× contact @ 1/240 | ~39.3 ms |
| `pyramid_20_rows_60_steps` | 210 boxes (20 rows) + ground, 60× contact @ 1/240 | ~180 ms |

These are foundation baselines for M2/M3 — not throughput targets. Re-record when
changing `libm` usage, SoA layout, integrator math, or the contact solver.
