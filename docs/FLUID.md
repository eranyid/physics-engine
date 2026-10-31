# Particle fluids (`pe-fluid`)

Weakly compressible SPH (WCSPH), Position Based Fluids (PBF), and divergence-free
SPH (DFSPH) for 3-D liquids, with a 2-D mode for analytic tests. Clean-room from
published formulas — see [`THIRD_PARTY.md`](../THIRD_PARTY.md). Optional kernel
correction (Shepard / gradient) and Akinci (2013) surface tension. **No**
free-surface mesh reconstruction; **no** fluid↔rigid coupling (`pe-core` untouched).

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
| Monaghan cubic spline | `2h` | WCSPH / DFSPH density / forces |
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
full lattice.

### Kernel correction (optional)

Shepard filter and Bonet–Lok / CSPM gradient correction are available for
**truncated** neighbourhoods (free surface / walls), where the discrete identity
degrades.

Measured 2-D cubic, `h=0.12`, `dx=h/6`, **half-space** truncation (Linux x86_64):

| Quantity | Before correction | After gradient correction |
| --- | --- | --- |
| max \|A − I\| | **0.500** | **~5.6×10⁻¹⁶** |
| Shepard `Σ V W` | **0.557** (full lattice: **1.000**) | — (normalise by sum) |

## Neighbour search

Uniform grid with **sorted cell lists** (no `HashMap` iteration). PBF default
`reuse_neighbours = true`: one grid build + cached neighbour lists per step,
reused across all constraint iterations. DFSPH rebuilds once after advection
(neighbours + `ρ` + `α` shared by both pressure solvers).

## WCSPH

- Density summation with Akinci (2012) boundary volume `ψ_b` (provisional `dx^d`,
  or `recomputed_akinci_psi` for closed shells).
- Tait EOS: `p = B[(ρ/ρ₀)^γ − 1]`, `B = ρ₀ c₀²/γ`, `γ = 7`.
- Sound-speed rule: **`c₀ ≥ 10 v_max`**.
- Monaghan artificial viscosity + optional XSPH; CFL helper.
- Optional Akinci (2013) surface tension.

## PBF (Macklin & Müller 2013)

- `Cᵢ = ρᵢ/ρ₀ − 1`, λ with CFM `ε ≈ 0.01`.
- Artificial pressure (ratio clamped ≤ 1); tank settles often use `k = 0`.
- Optional vorticity confinement; XSPH-like viscosity after the solve.
- SoA particle buffers; scratch `dv`/`ω`/`neigh_lists` reused across steps.

## DFSPH (Bender & Koschier 2015 / TVCG 2017)

Clean-room from the published algorithm:

1. Non-pressure forces (gravity, XSPH, optional surface tension).
2. Adaptive CFL: `Δt ≤ 0.4 d / ‖v‖_max` (and force limit).
3. Predict `v*`; **constant-density** Jacobi solver (warm-started `κ`).
4. Advect `x`; rebuild neighbours; recompute `ρ`, `α`.
5. **Divergence-free** Jacobi solver (warm-started `κᵛ`).
6. `v ← v*`.

Factors `α_i = ρ_i / (|Σ m∇W|² + Σ |m∇W|²)` with an SI-scaled denominator floor
(the paper’s absolute `1e−6` clamp is for their mass scale). **Compression-only**
`(ρ*−ρ₀)⁺` / `(Dρ/Dt)⁺` — required for free-surface stability (two-sided form
causes tensile blow-up on under-dense SPH surface particles).

Cubic spline for both `W` and `∇W` (paper requirement).

## Deferred

| Item | Notes |
| --- | --- |
| Fluid → rigid buoyancy/drag | Would sample onto `pe-core` bodies |
| Free-surface meshing | Out of scope |
| GPU / multi-thread | Deterministic single-thread default |
| Tight Laplace `Δp` lock | Shape locked; Tait+AV smear — see droplet section |

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
and `< 0.01` for this case).

### PBF settled tank

Mean `|ρ − ρ₀|/ρ₀` **&lt; 1%** (2-D, `k=0`, 10 iterations) after settle.

### DFSPH settled tank

Closed 2-D tank, calibrated mass ×1.03, Akinci `ψ` recomputed, compression-only
DFSPH, `dt=2.5×10⁻⁴`, 1200 steps (Linux x86_64):

| Metric | Measured | CI lock |
| --- | --- | --- |
| Mean compression `(ρ−ρ₀)⁺/ρ₀` | **0.000** | **&lt; 0.5%** |
| Interior `|ρ−ρ₀|/ρ₀` (lower 50% by y) | ~0.28 | informational (SPH surface truncation) |

Target 0.1–0.5% **compression** (paper volume-compression metric): **met**.
Absolute `|ρ−ρ₀|` including free-surface under-density is **not** claimed under
0.5% — that would silently loosen the wrong metric.

Wall penetration: 0/64 fluid particles crossed floor/left by more than `dx` in the
CI tank drop (measured).

Mass conserved (constant particle mass). Isolated equal-mass pair: momentum drift
`‖ΔP‖ < 1e−6` over 20 steps (no gravity / boundaries).

### Dam-break front vs Ritter (1892)

**Setup:** 2-D, `h₀ = 0.4` m, reservoir length `L_res = 0.5` m, channel `6` m,
sample at `t = 0.22` s (Ritter travel `2√(g h₀) t ≈ 0.87` m ≪ channel).
Front = near-bed **toe** (`y < 0.35 h₀`, `ρ > 0.4 ρ₀`).

#### PBF (default path)

| `dx` [m] | rel. error vs Ritter (measured) |
| --- | --- |
| 0.050 | **1.20** |
| 0.040 | **0.71** |
| 0.030 | **0.44** |
| 0.022 | **0.13** (`--ignored` fine run) |

Error **decreases under refinement**. At `dx = 0.03` still **~44%** (not under 15%).
At `dx = 0.022` ~13% — under 15% only at finer-than-CI resolution.

#### DFSPH (compression-only, `h=2.2 dx`, calibrated mass)

| `dx` [m] | `x_front` [m] | rel. error vs Ritter |
| --- | --- | --- |
| 0.050 | 0.738 | **0.462** |
| 0.040 | 0.805 | **0.421** |
| 0.030 | 0.840 | **0.392** |

Monotone improvement; finest CI `dx=0.03` still **~39%** — **not under 15%**.

#### WCSPH self-consistency reference (labelled)

Parallel WCSPH study (same geometry, `c₀=50`, mild AV/XSPH) **undershoots** Ritter
by ~50–60% at these `dx` (toe lags). High-resolution WCSPH is a **self-consistency**
reference for SPH discretisation, not a substitute for experiment.

Martin & Moyce (1952) / Koshizuka–Oka (1996) surge-front tables are **not**
transcribed (no verified primary table obtainable in this environment).

#### Root cause (evidence)

Probe at `t=0.22`, PBF `viscosity_eps=0`, toe vs Ritter (Linux x86_64):

| `dx` | PBF overshoot / `h` | tip fluid nbrs | WCSPH lag / `h` |
| --- | --- | --- | --- |
| 0.05 | ~13 | 1 | ~8 |
| 0.04 | ~11 | 5 | ~10 |
| 0.03 | ~9 | 4 | ~12 |

1. **Model mismatch:** Ritter is inviscid 1-D shallow-water on an infinite dry bed.
   PBF / DFSPH / WCSPH are 2-D particle methods with different dissipation.
2. **Front metric:** raw max-`x` is splash-outlier sensitive (e.g. PBF `dx=0.04`:
   raw 3.90 vs toe 2.26). Toe + density threshold is required.
3. **Kernel support:** discontinuous wet/dry front smoothed over `O(h)=O(2–2.4 dx)`.
4. **Sparse tip:** 1–5 fluid neighbours at the toe → truncated SPH support.
5. **Boundaries:** Akinci bed samples add impermeability / effective friction absent
   from frictionless Ritter; open-wall `ψ=dx²` (not closed-shell `1/ΣW`).
6. **Viscosity / XSPH / artificial pressure:** even mild damping changes tip speed;
   PBF systematically **overshoots**, WCSPH/DFSPH **undershoot** at practical `dx`.
7. **Finite reservoir:** `L_res=0.5` m ≠ infinite left state of the classical Riemann problem.

**Why &lt;15% vs Ritter is not claimed** at CI resolutions: items 1–7 above. Tolerances
are **not** loosened silently. DFSPH improves on coarse PBF error (~46% vs ~120% at
`dx=0.05`) but shares the same asymptotic model gap.

### Surface tension / droplet

Akinci (2013) cohesion + curvature (optional Becker–Teschner colour-field path).

Analytic Laplace: 2-D `Δp = σ/R`, 3-D `Δp = 2σ/R` (oracle bit-exact).

2-D free droplet WCSPH (`σ=0.2`, `R₀=0.15`, `dx=0.03`, strong AV/XSPH), measured:

| Quantity | Measured | CI lock |
| --- | --- | --- |
| Sphericity RMS `/R` | ~0.36 | **&lt; 0.40** |
| Radius drift | &lt; 50% | **&lt; 50%** |
| Laplace `Δp` vs `σ/R` | heavily smeared by Tait+AV | informational only |

A tight Laplace pressure lock is **deferred** (needs a dedicated CSF pressure
sampler without EOS contamination). Shape + analytic identity are locked.

### Other checks

| Check | Result |
| --- | --- |
| Mass conservation | bit-exact (constant particle mass) |
| Internal pair forces | Σ F = 0 (equal-mass pairs) |
| Viscosity-only pair kick | KE non-increasing |
| Determinism hashes | [`DETERMINISM.md`](DETERMINISM.md) (PBF + DFSPH) |

## Performance

Hot path: neighbour gather × pressure iterations. Optimisations:

- Cached neighbour lists across PBF iterations (`reuse_neighbours`, default on)
- DFSPH: `α` computed once per step, reused by both solvers; warm-started `κ`
- SoA particle state; reused scratch buffers
- No `HashMap` on the step path

**Criterion** (`cargo bench -p pe-fluid --bench fluid`), Linux x86_64 release:

| Workload | Measured |
| --- | --- |
| `fluid_tank_hash` (PBF 4³, 40 steps) | **~73.3 ms** |
| `pbf_1k_particles_10_steps` | **~204 ms** ≈ 4.9×10⁴ part·steps/s |
| `wcsph_256 × 20` | **~49.2 ms** |
| `dfsph_tank_hash` (4³, 40 steps) | **~186 ms** |

Neighbour-list reuse avoids one uniform-grid rebuild per PBF iteration.
