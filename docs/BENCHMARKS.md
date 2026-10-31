# Integrator, contact, and aero/hydro hot-loop baselines (Criterion)

Measured on the Cloud Agent host (Linux x86_64, release, Rust 1.99.1).
Numbers are wall time for the named workload; re-run locally after numerical changes.

Default CI does **not** run Criterion (`cargo test` omits harness benches). Use
`cargo bench -p pe-core --bench integrators`,
`cargo bench -p pe-collision --bench contact_pyramids`,
`cargo bench -p pe-aero --bench aero_forces`,
`cargo bench -p pe-hydro --bench hydrostatics`, and
`cargo bench -p pe-space --bench space`,
`cargo bench -p pe-blast --bench blast`, and
`cargo bench -p pe-ballistics --bench ballistics`.

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

## M7 — space (`pe-space`)

| Benchmark | Workload | Median time |
| --- | --- | --- |
| `leo_leapfrog_2k_steps` | circular LEO, 2000× leapfrog | ~101 µs |
| `leo_yoshida4_2k_steps` | same, Yoshida4 | ~312 µs |
| `leo_rk4_2k_steps` | same, RK4 | ~1.68 ms |
| `figure8_leapfrog_2k_steps` | figure-eight, 2000× leapfrog | ~162 µs |

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

These are foundation baselines for M2–M7 + blast/ballistics — not throughput
targets. Re-record when changing `libm` usage, SoA layout, integrator math, the
contact solver, aero/hydro force models, space integrators, or blast/ballistics.
