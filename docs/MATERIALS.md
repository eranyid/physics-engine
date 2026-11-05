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

## Thermal — lumped / implicit conduction + expansion

Graph Laplacian / lumped FEM on undirected edges (1-D rods) and linear-tet
conductivity assembly (`ThermalMesh::from_tets`):
`C_i Ṫ_i = Σ_j κ_ij (T_j − T_i) + Q_i + h A_i (T_∞ − T_i)`,
`κ_ij = k A / L` on a rod (or `−K_ij` from tet `K_ij = k |V| ∇N_i·∇N_j`).

**Time schemes:** explicit Euler (CFL `dt ≤ 0.4 C_i/(Σκ+hA)`); **backward Euler**
(θ=1, unconditionally stable); **Crank–Nicolson** (θ=½, 2nd-order). Implicit
solves use Thomas on 1-D paths or deterministic CG (fixed iteration order; no
`HashMap`; no `mul_add`).

**Boundaries:** Dirichlet (`fixed`), Neumann nodal power `Q`, insulated
(default), Robin / convection `h A`, `T_∞`.

Analytic Fourier mode (Dirichlet ends):
`T(x,t) = T₀ sin(πx/L) exp(−κ (π/L)² t)`, `κ = k/(ρ c)`.

Semi-infinite surface step (Incropera / Carslaw & Jaeger):
`T = T_i + (T_s − T_i) erfc(x/(2√(κ t)))`.

Free expansion: `ΔL = α L₀ ΔT`. Clamped bar: `σ = −E α ΔT`.

### Oracle — measured (explicit, materials track)

| Check | Setup | Measured | CI lock |
| --- | --- | --- | --- |
| Transient Fourier mode-1 | n=65, Cu-like, t≈500 s, dt=0.5·CFL | **~0.89%** max rel | **&lt; 2%** |
| Steady linear profile | Dirichlet ends, 17 nodes | **~2e-13** abs | **&lt; 0.05 K** |
| Insulated energy `Σ C T` | 400 steps | **&lt; 1e-12** rel | **&lt; 1e-11** |
| Free expansion `ΔL=α L ΔT` | exact | **0** | **&lt; 1%** coupling |

### Oracle — measured (M29 implicit)

| Check | Setup | Measured | CI lock |
| --- | --- | --- | --- |
| BE Fourier mode-1 | n=65, dt=25 s ≫ CFL, t=500 s | **~5.4e-4** max rel | **&lt; 0.5%** |
| Semi-infinite erfc | n=257, L=4 m, dt=0.25 s, t=20 s | **~0.21 K** max abs | **&lt; 2 K** |
| Steady linear (BE) | 17 nodes, long settle | **~5e-12** abs | **&lt; 1e-9 K** |
| Insulated energy (BE) | dt ≫ CFL | **&lt; 1e-12** rel | **&lt; 1e-11** |
| BE stability | dt = 100× CFL | bounded, monotone | — |
| CN order in dt | self-conv. vs dt=1, ratio e(8)/e(4) | **~4.20** | **∈ [3, 5.5]** |
| Robin lumped Bi≪1 | single node, BE | **~0.67%** rel | **&lt; 2%** |
| Free expansion coupling | XPBD rest from cold | **0** | **&lt; 1%** |
| Clamped stress `E α ΔT` | pinned–pinned distance bar | **~7e-14** rel | **&lt; 2%** |

API: `ThermalMesh`, `ThermalScheme`, `step_backward_euler` /
`step_crank_nicolson`, `rod_fourier_mode1`, `semi_infinite_erfc`,
`robin_lumped_exp`, `apply_thermal_expansion_from_cold`,
`apply_thermal_expansion_tets_from_cold`, `clamped_thermal_stress`.
Python: `ThermalRod`, `canonical_thermal_hash()`.

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

M29 implicit thermal: `cargo run -p pe-soft --bin thermal_hash --release` →
`thermal_canonical_hash` (`becda3e8805b5b90`). Existing `materials_hash` /
`soft_hash` / `soft_coupling_hash` unchanged.

## Soft XPBD ↔ J2 plastic coupling (M30)

After each XPBD outer step, `PlasticSoftWorld` measures total strain from the
**cold** rest configuration, applies the constitutive return map, and updates
XPBD rests so only the elastic residual remains:

| Kind | Strain | Rest update |
| --- | --- | --- |
| Distance bar | `ε = (L − L₀)/L₀` | `rest ← L₀ (1 + ε_p)` |
| Tet FEM | Green–Lagrange from `F = Ds Dm_cold⁻¹` | `F_p ≈ I+ε_p`; `Dm_inv ← Dm_cold⁻¹ F_p⁻¹`; `V₀ ← V₀ det(F_p)` |

Compliance stays elastic (`α = L₀/(E A)` / Lamé). Plastic flow lives in the rest.

### Oracle — measured (M30)

| Check | Setup | Measured | CI lock |
| --- | --- | --- | --- |
| Fixed-stretch rest ≡ return map | hold `ε = ε_y+2e-3` | **~0** (same fn) | **&lt; 1e-12** rel |
| Elastic springback | hold `0.5 ε_y`, release | **0** | **&lt; 0.5%** |
| Plastic permanent set | hold `ε_y+2e-3`, release | length rel **0**; map **~1e-14** | **&lt; 1%** |
| Hardening slope `Δσ/Δε` vs `E_t` | two holds past yield | **~0** | **&lt; 1%** |
| Unloaded stretch energy | after plastic settle | ≪ 2% of `½ E A L ε_y²` | **&lt; 2%** of scale |
| Multi-seg mean `ε_p` | 4-seg uniform stretch | **&lt; 1%** | **&lt; 1%** |
| Tet dilatation / shear | unit tet | no yield / surface **&lt; 1%** | **&lt; 1%** |

API: `PlasticSoftWorld`, `make_plastic_bar`, `update_plastic_distances`,
`update_plastic_tets`, `quasistatic_stretch_unload`, `plastic_canonical_hash`.
Python: `PlasticBar`, `soft_quasistatic_stretch_unload`,
`canonical_plastic_hash()`.

M30 plastic hash: `cargo run -p pe-soft --bin plastic_hash --release` →
`plastic_canonical_hash` (`73d9ebdd6677ec8c`). Existing soft / materials /
soft_coupling / thermal hashes **unchanged**.

## Finite-strain multiplicative F_p (M31)

`FsPlasticSoftWorld` upgrades the M30 rest-update path for large stretch:

| Kind | Strain measure | Rest / intermediate update |
| --- | --- | --- |
| Distance bar | Hencky `ε = ln(L/L₀)` | `rest ← L₀ exp(ε_p)` |
| Tet FEM | Green of `F_e = F F_p⁻¹` | `F_p ← exp(Δε_p) F_p`, then `F_p ← F_p/∛det(F_p)`; `Dm_inv ← Dm_cold⁻¹ F_p⁻¹` |

At small strain (`ε ≲ 10⁻³`) Hencky recovers the M30 engineering rest update
within the locked 0.5% oracle. At `λ = 1.2` the two permanent-strain measures
diverge by tens of percent — the finite-strain path is required.

### Oracle — measured (M31)

| Check | Setup | Measured | CI lock |
| --- | --- | --- | --- |
| Log rest ≡ return map | hold `λ=1.04` | **~0** (same fn) | **&lt; 1e-12** rel |
| Elastic log springback | hold `exp(0.5 ε_y)`, release | **~0** | **&lt; 0.5%** |
| Plastic unload | hold `λ=1.06`, release | length vs `L₀ exp(ε_p)` | **&lt; 1%** |
| Small-strain log≡eng | steel `ε_y+1e-3` | **≪ 0.5%** | **&lt; 0.5%** |
| Large-strain divergence | soft mat `λ=1.2` | **≫ 5%** | **&gt; 5%** |
| Hardening slope vs `E_t` | two log holds | **~0** | **&lt; 1%** |
| Unloaded stretch energy | after plastic settle | ≪ 2% of scale | **&lt; 2%** |
| Tet dilatation / shear | unit tet, `γ=0.15` | no yield / `|det F_p−1|~0` / surface | **&lt; 1e-10** det; **&lt; 1%** surface |

API: `FsPlasticSoftWorld`, `make_fs_plastic_bar`, `update_fs_plastic_distances`,
`update_fs_plastic_tets`, `quasistatic_log_stretch_unload`,
`fs_plastic_canonical_hash`, `sym_exp3`, `isochoric_project`.
Python: `FsPlasticBar`, `soft_quasistatic_log_stretch_unload`,
`soft_analytic_log_permanent_strain`, `canonical_fs_plastic_hash()`.

M31 hash: `cargo run -p pe-soft --bin fs_plastic_hash --release` →
`fs_plastic_canonical_hash` (`7c902cd304aea5ae`). M30 `plastic_hash`
(`73d9ebdd6677ec8c`) and soft / materials / soft_coupling / thermal hashes
**unchanged**.

## Spectral Hencky / Kirchhoff on `b_e` (M32)

`BePlasticSoftWorld` upgrades the M31 tet trial from Green-of-`F_e` + Hooke to
the full isotropic spectral algorithm on the elastic left Cauchy–Green tensor:

1. `F_e^{tr} = F F_p⁻¹`, `b_e^{tr} = F_e^{tr}(F_e^{tr})ᵀ`
2. Spectral: `b_e^{tr} = Σ λ_a² n_a ⊗ n_a`, Hencky `ε_a = ln(λ_a)`
3. Kirchhoff `τ_a = λ tr(ε) + 2μ ε_a`; J2 return in principal Kirchhoff space
4. `b_e ← Σ exp(2 ε_a) n_a ⊗ n_a`; `F_p ← exp(Δε_p) F_p` + isochoric projection

Distance bars reuse the M31 Hencky rest update. At small shear (`γ = 10⁻³`) the
Kirchhoff Frobenius agrees with the M31 Green+Hooke stress within 2%; at
`γ = 0.4` they diverge by &gt;5%.

### Oracle — measured (M32)

| Check | Setup | Measured | CI lock |
| --- | --- | --- | --- |
| Isochoric uniaxial Kirchhoff | `λ` below yield | **~0** vs `τ₁=2μ ln λ` | **&lt; 1e-12** |
| Isochoric uniaxial `σ_eq` | same | **~0** vs `3μ\|ln λ\|` | **&lt; 1e-12** |
| Plastic surface | `λ=1.1` isochoric | **~0** | **&lt; 1e-10** |
| `det b_e` after return | isochoric plastic | **~0** vs 1 | **&lt; 1e-10** |
| Hydrostatic `b_e` | uniform stretch | no yield | — |
| Tet dilatation / shear | unit tet, `γ=0.15` | no yield / surface | **&lt; 1e-10** det; **&lt; 1%** |
| Small-shear ≡ Green | `γ=1e-3` | **≪ 2%** | **&lt; 2%** |
| Large-shear divergence | `γ=0.4` | **≫ 5%** | **&gt; 5%** |
| Hencky bar unload | `λ=1.06` | length vs `L₀ exp(ε_p)` | **&lt; 1%** |

API: `BePlasticSoftWorld`, `return_map_spectral_hencky_be`,
`update_be_plastic_tets`, `left_cauchy_green`, `sym_spectral_decompose`,
`analytic_isochoric_uniaxial_kirchhoff`, `analytic_isochoric_uniaxial_seq`,
`be_plastic_canonical_hash`. Python: `BePlasticBar`,
`soft_analytic_isochoric_uniaxial_seq`, `canonical_be_plastic_hash()`.

M32 hash: `cargo run -p pe-soft --bin be_plastic_hash --release` →
`be_plastic_canonical_hash` (`8562bc39f064698e`). M31 `fs_plastic_hash`
(`7c902cd304aea5ae`), M30 `plastic_hash` (`73d9ebdd6677ec8c`), and soft /
materials / soft_coupling / thermal hashes **unchanged**.

## Deferred

- Cohesive fracture / crack-path tracking
- Radiation BCs; temperature-dependent `k(T)` / `c_p(T)`
- Additional sourced / verified handbook rows
