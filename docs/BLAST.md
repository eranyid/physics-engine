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

- Engineering / similarity models, not full 3-D CFD.
- Swisdak curves are hemispherical **surface** burst fits.
- Sedov assumes strong point blast in uniform ambient gas.
- Euler solver is 1-D validation / coupling aid, not production blast CFD.

## Validation highlights

- Kinney `Z=1` → `p_s≈1008.79 kPa`; `Z=10` → `≈9.985 kPa` (`p0=101.325 kPa`).
- Sedov `ξ₀≈1.032` (target ~1.033); `R ∝ t^{0.4}`.
- Sod L1 density error decreases under grid refinement (CFL 0.4, `t=0.2`).
- RH reflection limits 2 / 8 for `γ=1.4`.
- Cross-OS hash: blast-on-pyramid scenario (`docs/DETERMINISM.md`).
