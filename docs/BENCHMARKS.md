# Integrator, contact, and aero/hydro hot-loop baselines (Criterion)

Measured on the Cloud Agent host (Linux x86_64, release, Rust 1.99.1).
Numbers are wall time for the named workload; re-run locally after numerical changes.

Default CI does **not** run Criterion (`cargo test` omits harness benches). Use
`cargo bench -p pe-core --bench integrators`,
`cargo bench -p pe-collision --bench contact_pyramids`,
`cargo bench -p pe-aero --bench aero_forces`,
`cargo bench -p pe-hydro --bench hydrostatics`, and
`cargo bench -p pe-space --bench space`.

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

## Stage 3 — aero / hydro (`pe-aero`, `pe-hydro`)

| Benchmark | Workload | Median time |
| --- | --- | --- |
| `forces_eval_1k` | aero+thrust+gravity force eval (iter) | ~77 ns |
| `rk4_1k_steps` | trimmed aircraft, 1000× RK4 @ dt=1e-3 | ~1.09 ms |
| `buoyancy_forces_1k` | floating barge force eval (iter) | ~821 ns |
| `heave_rk4_500_steps` | floating barge, 500× RK4 @ dt=1e-3 | ~2.23 ms |
| `gz_curve_60_angles` | box GZ curve, 60 heel samples | ~54 µs |

## M7 — space (`pe-space`)

| Benchmark | Workload | Median time |
| --- | --- | --- |
| `leo_leapfrog_2k_steps` | circular LEO, 2000× leapfrog | ~105 µs |
| `leo_yoshida4_2k_steps` | same, Yoshida4 | ~337 µs |
| `leo_rk4_2k_steps` | same, RK4 | ~1.48 ms |
| `figure8_leapfrog_2k_steps` | figure-eight, 2000× leapfrog | ~141 µs |
| `ias15_e09_one_orbit` | `e=0.9` Kepler, IAS15 one period | ~1.12 ms |
| `whfast_5body_10k_steps` | synthetic 5-body WHFast Jacobi, 10k steps | ~14.2 ms |

These are foundation baselines for M2–M7 — not throughput targets. Re-record when
changing `libm` usage, SoA layout, integrator math, the contact solver, aero/hydro
force models, or space integrators.
