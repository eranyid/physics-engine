# Materials track (`pe-soft`) — plasticity, fracture, thermal

SI units; `f64`; portable `libm`; no FMA; no `HashMap` iteration. Constitutive
helpers and mesh coupling live in `pe-soft` (plasticity / fracture / thermal
modules + extended `Material` table). Soft XPBD stepper: [`SOFT.md`](SOFT.md).

## Plasticity — J2 von Mises + linear isotropic hardening

**1-D:** trial `σ_tr = E(ε − ε_p)`, yield `f = |σ_tr| − (σ_y + H α)`.
Return: `Δγ = f/(E+H)`, `ε_p ← ε_p + Δγ sgn(σ_tr)`, `α ← α + Δγ`.
Plastic continuum tangent `E_t = E H/(E+H)`.

**3-D:** radial return on `s = dev(σ)`, `σ_eq = √(3/2)‖s‖`,
`Δγ = f/(3μ+H)`, `s ← (1 − 3μ Δγ/σ_eq) s_tr`. Hydrostatic stress never yields.

### Oracle (uniaxial tension) — measured

| Check | Measured | CI lock |
| --- | --- | --- |
| Yield onset `σ(ε_y)=σ_y` | **~0** rel | **&lt; 1%** |
| Elastic slope `E` | **~0** | **&lt; 1%** |
| Plastic slope vs `E_t` | **~3e-15** rel | **&lt; 1%** |
| Unload slope `E` | **~0** | **&lt; 1%** |
| Permanent strain `ε − σ/E` | **0** | **&lt; 1%** |
| 3-D hydrostatic no yield | exact | — |
| 3-D return-to-surface | **&lt; 1e-12** | **&lt; 1%** |

API: `pe_soft::return_map_uniaxial`, `return_map_j2_3d`, `J2Params`.

## Fracture — max principal stress / strain

Distance constraints use axial engineering stress `σ = E (L−L₀)/L₀` (or strain
`(L−L₀)/L₀`). Tet FEM uses max principal PK2 (StVK) / Green strain. Candidates
are sorted by `(kind, index)` before breaking — deterministic. Broken
constraints are skipped by XPBD. Islands = union-find over remaining active
distance + tet edges.

### Oracle — measured

| Check | Measured | CI lock |
| --- | --- | --- |
| Bar break stress vs threshold | **0.50%** (first sample past thresh) | **&lt; 2%** |
| Energy released finite, ≥ 0 | yes | — |
| No NaN after break | yes | — |
| Split-body islands (2 nodes, 1 link) | 2 islands | — |
| Event order stable | by index | — |

API: `apply_fracture`, `constraint_islands`, `FractureCriterion`.

## Thermal — lumped conduction + expansion

Graph Laplacian / lumped FEM on undirected edges:
`C_i Ṫ_i = Σ_j κ_ij (T_j − T_i)`, `κ_ij = k A / L` on a 1-D rod.
Explicit Euler with CFL `dt ≤ 0.4 C_i/Σκ` (oracle uses `0.5×` that bound).

Analytic Fourier mode (Dirichlet ends):
`T(x,t) = T₀ sin(πx/L) exp(−κ (π/L)² t)`, `κ = k/(ρ c)`.

Free expansion: `ΔL = α L₀ ΔT`.

### Oracle — measured

| Check | Setup | Measured | CI lock |
| --- | --- | --- | --- |
| Transient Fourier mode-1 | n=65, Cu-like, t≈500 s, dt=0.5·CFL | **~0.89%** max rel | **&lt; 2%** |
| Steady linear profile | Dirichlet ends, 17 nodes | **~2e-13** abs | **&lt; 0.05 K** |
| Insulated energy `Σ C T` | 400 steps | **&lt; 1e-12** rel | **&lt; 1e-11** |
| Free expansion `ΔL=α L ΔT` | exact | **0** | **&lt; 1%** coupling |

API: `ThermalMesh`, `rod_fourier_mode1`, `apply_thermal_expansion_from_cold`.

## Material table

Built-ins in `builtin_materials()` — **fixed order**. Fields:

`density, E, ν, friction, restitution, σ_y, H, k, c_p, α, fracture_stress,
fracture_strain, verified, source_note`.

**Rule:** `verified: true` only with a non-empty citation of the exact numbers.

| Name | `verified` | Notes |
| --- | --- | --- |
| `rubber_soft` … `concrete_nominal` | **false** | Prior soft demos; nominal |
| `copper_nominal` … `polycarbonate_nominal` | **false** | Prior nominal additions |
| `steel_aisi1020` | **true** | MatWeb AISI 1020 CR: ρ=7870, E=186 GPa, ν=0.29, σ_y=350 MPa, k=51.9, c_p=486, α=11.7e-6 (retrieved 2026-11-01) |
| `aluminum_6061_t6` | **true** | ASM MatWeb 6061-T6 AA typicals: ρ=2700, E=68.9 GPa, ν=0.33, σ_y=276 MPa, k=167, c_p=896, α=23.6e-6 |
| `copper_c11000` | **true** | CRC/MatWeb pure Cu: ρ=8960, E=110 GPa, ν=0.34, k=401, c_p=385, α=16.5e-6 |
| `titanium_ti6al4v` | **true** | MatWeb Grade 5 annealed: ρ=4430, E=113.8 GPa, ν=0.34, σ_y=880 MPa, k=6.7, c_p=526.3, α=8.6e-6 |
| `glass_soda_lime` | **true** | Handbook: ρ=2500, E=70 GPa, ν=0.23, k=1.05, c_p=840, α=9e-6 |
| `concrete_normal_weight` | **true** | ACI/CRC ~28 MPa: ρ=2400, E=30 GPa, ν=0.20, k=1.4, c_p=880, α=10e-6 |
| `water_20c` | **true** | IAPWS/CRC 20 °C: ρ=998.21, k=0.5984, c_p=4181.8; E≈K=2.18 GPa, ν=0.499 |
| `wood_pine_along_grain` | **false** | Species/MC dependent — not locked |
| `rubber_natural` | **false** | Compound-dependent — not locked |

## Determinism hash

`cargo run -p pe-soft --bin materials_hash --release` → locked in
[`DETERMINISM.md`](DETERMINISM.md) (`materials_canonical_hash`). Packs J2 stress
history + rod temperatures + fractured bar state.

## Deferred

- Coupled XPBD + J2 on tet deformation gradients each step (constitutive return
  map is shipped; full FEM plastic soft-step integration deferred)
- Cohesive fracture / crack-path tracking
- Implicit / Crank–Nicolson thermal; radiation / convection BCs
- Sourced / verified handbook rows (`verified: true`)
