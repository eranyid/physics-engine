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
via [`World::apply_impulse`](../crates/pe-core/src/world.rs). Continuous loads
write into [`ExternalLoads`](../crates/pe-core/src/external_loads.rs) and are
integrated by the world’s force generators (RK4 / symplectic / …).

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
8. **Rigid-body loading (M20)** — Friedlander on exposed faces with:
   - Kinney **or** Kingery–Bulmash peaks / TOA / `t_d`
   - incident + reflected blend (`p_i + (p_r−p_i) cos²θ`)
   - Rankine–Hugoniot dynamic-pressure drag (`C_d · q`, caller `C_d`)
   - AABB ray shadowing / occlusion (conservative; no diffraction)
   - per-step impulse **or** continuous [`ExternalLoads`] integration
9. **Gurney / Mott (M20)** — fragment speed from caller / table `√(2E)` and
   `C/M` (flat/sheet, cylindrical, spherical); Mott mass sampling via seeded
   `xoshiro256**`; fragments spawn into `pe-ballistics::ProjectileSystem`.
10. **Cohesive / brittle fracture (M20)** — deterministic Voronoi pre-fracture
    of a box into rigid chunks joined by soft distance joints with
    `break_force = σ_f · A` from the materials table’s fracture stress; bonds
    break on impulse / reaction and leave islands/sleeping correct.

## Model limits (honest)

- Engineering / similarity models, **not** full 3-D CFD.
- Swisdak curves are hemispherical **surface** burst fits (valid-`Z` clamp).
- Incidence blend is a face-on engineering wall model — no clearing, Mach stem,
  or multiple reflections.
- Dynamic-pressure `C_d` is caller-tuned (**unverified** aero database).
- AABB shadowing over-reports occlusion and ignores diffraction / clearing.
- Sedov assumes strong point blast in uniform ambient gas.
- Euler solver is 1-D validation / coupling aid, not production blast CFD.
- Voronoi voxel chunks use solid-box inertia proxies; bond area is a shared
  voxel-face estimate — not continuum LEFM.
- Built-in Gurney constants (TNT / Comp B / HMX) are labelled
  `verified: false` until locked to a fetched primary table cell.

## Validation / oracles (M20)

| Oracle | Measured vs | Tolerance | Why |
| --- | --- | --- | --- |
| Friedlander `I⁺` | closed form `(b−1+e^{−b})/b²` | **1e-12** | algebra + `libm::exp` |
| Hopkinson invariance | same `Z` → same `p_s` | **0** | identical formula path |
| Gurney cylindrical `C/M=0.5`, `√(2E)=2440` | `2440√0.4` | **1e-12 rel** | closed Gurney; `√(2E)` itself unverified |
| Mott mean | `E[m]=2μ` | **0** closed form; sample **&lt;2%** @ N=5e4 | Exp(1) second moment |
| Fragment KE budget | `½ M V²` after mass normalize | **1e-12 rel** | identical `V` |
| Antipodal momentum | `|P|/(M V)` | **&lt;1e-14** | equal-mass ± pairs |
| Bond break threshold | strict `\|J\| > J*` | **exact** | boolean predicate |
| RH wall reflection | weak 2 / strong 8 | **1e-5 / 1e-6** | γ=1.4 limits |
| KB vs RH mid-Z ratio | chart vs ideal gas | **&lt;15% rel** | documented chart residual |

Legacy: Kinney `Z=1/10`, Sedov `ξ₀`/`R∝t^{0.4}`, Sod L1 refinement, Swisdak
`in_range`, cross-OS hashes below.

## Determinism hashes

`cargo run -p pe-blast --bin blast_hash --release` prints four lines:

| Line | Scenario | Hash |
| --- | --- | --- |
| 1 | Pyramid Kinney impulse (legacy) | `b51e11e89d4dd055` **unchanged** |
| 2 | ExternalLoads KB+RH+drag | `693b60be33f9b8f2` |
| 3 | Gurney/Mott ballistic fragments | `36d3f61d77c14859` |
| 4 | Voronoi cohesive destruction | `38f55bac9ab3d16f` |

See [`DETERMINISM.md`](DETERMINISM.md).

## Scene (schema v3)

`SceneDocument.blast: Vec<BlastLoadingDesc>` and
`fracture_compounds: Vec<FractureCompoundDesc>` — applied by the caller via
`pe_scene::apply_blast_loading_desc` + `ExternalLoads`. Bonds remain ordinary
joints with `break_force`. See [`SCENE.md`](SCENE.md).
