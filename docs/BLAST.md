# Blast / shock module (`pe-blast`)

Engineering blast and shock models for games and studio simulation. Clean-room
from published formulas (see [`THIRD_PARTY.md`](../THIRD_PARTY.md)). **No**
explosive chemistry, recipes, weapon design, or manufacture — TNT-equivalence
and Gurney energy are caller-supplied parameters.

## Units and frames

| Quantity | Unit | Notes |
| --- | --- | --- |
| Distance / radius | m | |
| Mass (TNT-eq) | kg | `W_eq = η · W` |
| Scaled distance `Z` | m·kg^{-1/3} | Hopkinson–Cranz |
| Pressure | Pa | overpressure = gauge |
| Impulse | Pa·s | |
| Time | s | |
| Energy | J | Sedov blast energy |
| Density | kg/m³ | ambient |

World frame: right-handed, same as `pe-core`. Blast impulses update `v` and `L`
via [`World::apply_impulse`](../crates/pe-core/src/world.rs).

## Models

1. **Hopkinson–Cranz** — `Z = R / W_eq^{1/3}`.
2. **Kinney & Graham (1985)** — closed-form free-air peak overpressure.
3. **Friedlander** — `p(t)=p_s(1−t/t_d)exp(−b t/t_d)`; analytic positive impulse.
4. **Swisdak (1994)** simplified Kingery–Bulmash — coefficients transcribed from
   the NSWC IHTR seminar paper (NTIS ADA526744). Valid `Z` ranges enforced by
   clamp + `in_range` flag.
5. **Rankine–Hugoniot** — shock strength, dynamic pressure, normal reflection
   ratio (limits 2 weak / 8 strong for `γ=1.4`).
6. **Sedov–Taylor** — `R=ξ₀(E t²/ρ)^{1/5}`; `ξ₀` from numerical energy integral
   over closed-form profiles (`ξ₀≈1.033` for `γ=1.4` spherical).
7. **1-D Euler** — FV HLLC + MUSCL(minmod) + SSP-RK2; Sod vs exact Riemann;
   spherical geometric sources.
8. **Rigid-body loading** — Friedlander on exposed faces → impulses; generic P–I
   damage curve (caller parameters).
9. **Gurney / Mott** — fragment speed from caller `√(2E)` and `C/M`; Mott mass
   sampling via seeded `xoshiro256**`.

## Limits

- Analytic models are engineering / similarity fits; the 3-D solver (below)
  is an inviscid ideal-gas Euler code at game resolution, not a certified
  blast-design tool.
- Swisdak curves are hemispherical **surface** burst fits.
- Sedov assumes strong point blast in uniform ambient gas.
- The 1-D Euler solver is a validation aid and the near-field source of the
  3-D remap.

## Validation highlights

- Kinney `Z=1` → `p_s≈1008.79 kPa`; `Z=10` → `≈9.985 kPa` (`p0=101.325 kPa`).
- Sedov `ξ₀≈1.032` (target ~1.033); `R ∝ t^{0.4}`.
- Sod L1 density error decreases under grid refinement (CFL 0.4, `t=0.2`).
- RH reflection limits 2 / 8 for `γ=1.4`.
- Cross-OS hash: blast-on-pyramid scenario (`docs/DETERMINISM.md`).

## 3-D blast simulation (`euler3d`, `blast3d`)

A compressible Euler solver on a uniform voxel grid: shock reflection from
walls, ground and ceiling, channeling through doors and corridors, shadowing
behind cover, venting, and the confined gas-pressure phase in rooms. These
are things the analytic curves cannot do.

```rust
use pe_blast::*;
let mut s = BlastScene::new(min, max, 0.1);           // 10 cm voxels
s.fill_ground(0.1);
s.fill_aabb(wall_min, wall_max);                      // walls
s.clear_aabb(door_min, door_max);                     // openings
let g = s.add_probe(point);                           // pressure gauge
s.detonate(origin, 0.25);                             // kg TNT-equivalent
s.run_until_with(0.02, 1_000_000, |s, dt| { s.apply_to_world(&mut world, &faces, dt); });
let rec = &s.probes[g];  // peak_overpressure, positive/negative_impulse, arrival_time
std::fs::write("slice.ppm", s.slice_ppm(2, k, 3e5)); // pressure image
```

| Piece | Method |
| --- | --- |
| Fluxes | 3-D HLLC (Toro §10.4), Davis wave speeds, tangential momenta through the star states |
| Reconstruction | MUSCL on primitives, `Limiter::{MonotonizedCentral (default), VanLeer, Minmod}`; optional first-order fallback above a pressure jump (`first_order_jump`, off by default) |
| Time | SSP-RK2, CFL on `Σ_d (|u_d| + a)/dx` (default 0.4) |
| Walls | solid voxels and `Boundary::Wall` faces are mirror ghosts (exact symmetry planes); `Boundary::Outflow` is zero-gradient |
| Source | 1-D → 3-D **remap**: a TNT-density sphere holding `W_eq · 4.184 MJ/kg` is expanded on a fine 1-D spherical grid (8 cells per charge radius) until the shock reaches the clearance to the nearest wall minus 2 cells, then written into the 3-D grid by 4×4×4 sub-sampling and rescaled to the exact energy. If the clearance is under 3 cells, the energy is deposited directly as a ball. `Detonation::surface()` uses the `2W` mirror image for ground bursts |
| Coupling | `ProbeRecord` gauges (peak, ± impulse, arrival, history); `face_pressure` / `apply_to_world` apply `p·A·dt` impulses on `ExposedFace`s of `pe-core` bodies; `PressureField` trait is shared by the grid and the analytic `BlastSource` |
| Determinism | per-face fluxes, then per-cell gather in fixed order. The `parallel` feature (rayon over z-slabs) is bit-identical to serial. Canonical `blast3d_hash` is locked in CI |

**Resolution.** Accuracy is set by the *scaled* cell size `dx / W^{1/3}`
[m·kg^{-1/3}]. Free-field peaks are smeared by roughly 3 cells, so they are
low when the positive-phase length is only a few cells. Measured peak ratio
(simulated / Kinney–Graham) along the worst (axis) and best (diagonal)
directions, with the remap:

CALIBRATION_TABLE

Guideline: `dx / W^{1/3} ≤ 0.1` for peak pressures (≈ 10 cm voxels for a
1 kg charge). Impulses converge faster than peaks. For far-field peaks in
open air the analytic Kingery / Kinney curves are cheaper and more accurate;
use the 3-D grid where geometry matters.

**Cost.** About 0.6–0.8 µs per cell-step on one core (`cargo run -p pe-blast
--release --example blast3d_bench`). A 64³ grid at 10 cm runs a 20 ms event
(≈ 800 steps) in a few minutes on one core, less with `--features parallel`.
This suits cut-scenes, slow motion, pre-baked events and server-side
resolution, not per-frame real time.

### Validation (`crates/pe-blast/tests/euler3d.rs`)

| Test | Check |
| --- | --- |
| Sod on x, y, z | L1 density error < 0.012 vs exact Riemann; all three axes agree to 1e-12 |
| Closed box with baffle | mass and energy conserved to 1e-12; energy deposit exact |
| Centered blast | symmetric under x↔y, x↔z, y↔z and mirror (1e-11 of peak) |
| Sedov–Taylor | captured shock radius vs `ξ₀ (E t²/ρ)^{1/5}` within 12 % at 16³ (`#[ignore]` fine grid: 5 %) |
| Normal reflection | planar shock on a wall: peak matches Rankine–Hugoniot reflected overpressure within 5 % |
| Shielding | gauge behind a 1.2 m wall: lower peak, later arrival than the open gauge at the same range |
| Venting | sealed room settles at the quasi-static `(γ−1)E/V` (±25 %); a door lowers it by more than 40 % |
| Free air | probe peaks vs Kinney within the calibration band (`#[ignore]` fine-grid twin) |
| Rigid coupling | a voxelized box is pushed away from the charge |
| Determinism | hash scene reproducible; serial = parallel (CI hash) |

Examples: `blast3d_room` (room + doorway + corridor, gauges, PPM slices),
`blast3d_calibrate`, `blast3d_bench`.
