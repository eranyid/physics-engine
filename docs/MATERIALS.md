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

| Name | `verified` | Notes |
| --- | --- | --- |
| `rubber_soft` … `concrete_nominal` | **false** | Prior soft demos; nominal |
| `copper_nominal` | **false** | k≈400 W/(m·K), α≈16.5e-6 textbook order-of-magnitude |
| `glass_soda_lime_nominal` | **false** | nominal |
| `titanium_ti6al4v_nominal` | **false** | nominal |
| `lead_nominal` | **false** | nominal |
| `ice_nominal` | **false** | nominal near 0 °C |
| `polycarbonate_nominal` | **false** | nominal |

No handbook row is marked `verified: true` in this milestone (no primary table
fetched). Cite a source and flip the flag when a value is transcribed from a
verified reference.

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
