# Particle fluids (`pe-fluid`)

Weakly compressible SPH (WCSPH) and Position Based Fluids (PBF) for 3-D liquids,
with a 2-D mode for analytic tests. Clean-room from published formulas — see
[`THIRD_PARTY.md`](../THIRD_PARTY.md). **No** surface tension, **no** free-surface
mesh reconstruction, **no** DFSPH, **no** fluid↔rigid coupling (`pe-core` untouched).

## Units and frames

| Quantity | Unit |
| --- | --- |
| Position / smoothing length `h` | m |
| Density ρ | kg/m³ |
| Pressure | Pa |
| Mass | kg |
| Velocity | m/s |
| Time | s |

World frame: right-handed, same as `pe-core`. Gravity default −z (3-D) or −y (2-D tests).

## SPH kernels

| Kernel | Support | Role |
| --- | --- | --- |
| Monaghan cubic spline | `2h` | WCSPH density / forces |
| Wendland C2 | `2h` | Optional smooth density kernel |
| Müller poly6 | `h` | PBF density |
| Müller spiky | `h` | PBF / pressure gradients |

**Partition of unity** (rectangle rule): cubic 3-D `∫W` error &lt; 1% at `n_half=24`.

**Gradient consistency** (2-D cubic spline, `h=0.12`, lattice; measured Linux x86_64):

| Spacing | max \|A − I\| | \|Σ V ∇W\| |
| --- | --- | --- |
| `dx = h/4` | **1.51×10⁻⁴** | ~1e-15 |
| `dx = h/8` | **2.5×10⁻⁷** | ~1e-15 |
| `dx = h/12` | ~3×10⁻⁶ | ~1e-15 |

Zeroth-order sum is numerically zero; first-order error vs **+I** is ≪ 1% on a
full lattice. **Kernel correction / renormalisation** (Bonet–Lok, CSPM, etc.) is
**not** implemented — optional for free-surface / truncated neighbourhoods where
the discrete identity degrades. On the interior lattice above it is unnecessary.

## Neighbour search

Uniform grid with **sorted cell lists** (no `HashMap` iteration). PBF default
`reuse_neighbours = true`: one grid build + cached neighbour lists per step,
reused across all constraint iterations (avoids per-iteration rebuild). Set
`false` for validation runs that rebuild after every λ/Δp iteration.

## WCSPH

- Density summation with Akinci (2012) boundary volume `ψ_b`.
- Tait EOS: `p = B[(ρ/ρ₀)^γ − 1]`, `B = ρ₀ c₀²/γ`, `γ = 7`.
- Sound-speed rule: **`c₀ ≥ 10 v_max`**.
- Monaghan artificial viscosity + optional XSPH; CFL helper.

## PBF (Macklin & Müller 2013)

- `Cᵢ = ρᵢ/ρ₀ − 1`, λ with CFM `ε ≈ 0.01`.
- Artificial pressure (ratio clamped ≤ 1); tank settles often use `k = 0`.
- Optional vorticity confinement; XSPH-like viscosity after the solve.
- SoA particle buffers; scratch `dv`/`ω`/`neigh_lists` reused across steps.

## Deferred

| Item | Notes |
| --- | --- |
| DFSPH (Bender & Koschier 2015) | Divergence-free / density-invariant SPH |
| Fluid → rigid buoyancy/drag | Would sample onto `pe-core` bodies |
| Surface tension / free-surface meshing | Out of scope |
| Kernel correction (Bonet–Lok / CSPM) | Optional; see gradient table above |
| GPU / multi-thread | Deterministic single-thread default |

## Validation (measured numbers)

### Tait hydrostatic profile `p ≈ ρ₀ g (H − z)`

Linearised density `ρ = ρ₀ (1 + g (H−z)/c₀²)` → Tait pressure, 40 samples on
`z ∈ [0, H]`, `H = 1` m, `ρ₀ = 1000`, `γ = 7`:

| `c₀` [m/s] | max rel error | mean rel error |
| --- | --- | --- |
| 40 | 1.83% | 0.93% |
| **60** | **0.81%** | **0.41%** |
| 100 | 0.29% | 0.15% |

**Target &lt; 3% max relative error: met** at `c₀ = 60` (CI asserts `max_rel < 0.03`
and `< 0.01` for this case). Nonlinearity of Tait (`γ=7`) sets the residual vs
the linear `ρ₀ g h` formula; it shrinks as `gH/c₀² → 0`.

### PBF settled tank

Mean `|ρ − ρ₀|/ρ₀` **&lt; 1%** (2-D, `k=0`, 10 iterations) after settle.

### Dam-break front vs Ritter (1892)

**Setup:** 2-D PBF, `h₀ = 0.4` m, reservoir length `L_res = 0.5` m, channel
`6` m, sample at `t = 0.22` s (after the initial transient; ideal Ritter travel
`2√(g h₀) t ≈ 0.87` m ≪ channel). Front = near-bed **toe** (`y < 0.3 h₀`,
`ρ > 0.5 ρ₀`). `reuse_neighbours = false`, light viscosity `ε = 0.015`.

| `dx` [m] | rel. error vs Ritter (measured) |
| --- | --- |
| 0.050 | **1.20** |
| 0.040 | **0.71** |
| 0.030 | **0.44** |
| 0.022 | ~0.17–0.25 (finer / ignored runs) |

Error **decreases under refinement** (CI asserts monotone improvement on
`0.05 → 0.04 → 0.03`). At `dx = 0.03` the relative error is still **~44%**,
**not under 15%**.

**Why &lt;15% vs ideal Ritter is not claimed**

1. **Model mismatch:** Ritter is inviscid 1-D shallow-water on an *infinite*
   dry bed with an instantaneous dam removal. We solve 2-D PBF (position-based
   constraints), not the Saint-Venant equations.
2. **Kernel support:** the discontinuous wet/dry front is smoothed over
   `O(h) = O(2 dx)`, biasing the toe location by a fraction of a smoothing length.
3. **Boundaries:** Akinci wall samples on the bed add effective friction /
   impermeability forces absent from the frictionless Ritter ideal.
4. **Viscosity / XSPH:** even mild `viscosity_eps` dissipates momentum at the tip;
   PBF still systematically **overshoots** Ritter here, while WCSPH **lags**
   (~49–58% undershoot in a parallel study) — neither asymptotes to the SWE front
   at practical `dx`.
5. **Finite reservoir:** still water to the left is a finite column (`L_res=0.5` m),
   not the infinite left state of the classical Riemann problem.

Martin & Moyce (1952) tables are **not** transcribed. Ritter remains the analytic
reference with the limits above stated explicitly — tolerances are **not** loosened
silently.

### Other checks

| Check | Result |
| --- | --- |
| Mass conservation | bit-exact (constant particle mass) |
| Internal pair forces | Σ F = 0 (equal-mass pairs) |
| Viscosity-only pair kick | KE non-increasing |
| Determinism hash | [`DETERMINISM.md`](DETERMINISM.md) |

## Performance

Hot path: neighbour gather × PBF iterations. Optimisations in this crate:

- Cached neighbour lists across PBF iterations (`reuse_neighbours`, default on)
- SoA particle state; reused scratch buffers (`dv`, `ω`, `neigh_lists`)
- No `HashMap` on the step path

**Criterion** (`cargo bench -p pe-fluid --bench fluid`), Linux x86_64 release:

| Workload | Before reuse | After reuse |
| --- | --- | --- |
| `fluid_tank_hash` (4³, 40 steps) | ~86.4 ms | **~67.5 ms (−22%)** |
| `pbf_1k_particles_10_steps` | ~250 ms | **~183 ms (−27%)** ≈ 5.5×10⁴ part·steps/s |
| `wcsph_256 × 20` | ~41.9 ms | ~41.4 ms (unchanged path) |

Neighbour-list reuse avoids one uniform-grid rebuild per PBF iteration (typically
`iterations − 1` rebuilds saved per step).
