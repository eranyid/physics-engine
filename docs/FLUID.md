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
| Free-surface meshing | Out of scope |
| GPU / multi-thread | Deterministic single-thread default |
| Free-surface heave zero-crossings | WCSPH float is overdamped; oracle uses stiffness-derived period |
| Dynamic Archimedes long-term | Transient draft lock at fixed step window; FS bias grows later |

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

**Physics setup (asymptotic Ritter):** 2-D, `h₀ = 0.4` m, long reservoir
`L_res = 4.5 h₀ = 1.8` m (left rarefaction does not hit the wall before `τ=4`),
channel `≈ 7.4` m. DFSPH inviscid (`xsph=0`), Akinci bed samples **plus**
free-slip hard walls (`x≥0`, `y≥0`).

**Front metric:** binned **column height** `h(x) = (Σ area)/bin_w` with
`bin_w = dx`; front = rightmost bin with `h ≥ 0.02 h₀` (not the leading particle).

**Time samples:** dimensionless `τ = t √(g/h₀) ∈ {1, 2, 3, 4}`. Ritter is an
**asymptotic** dry-bed solution — early `τ∼1` lags the collapse transient;
agreement is judged at later `τ`.

#### DFSPH height-field front (measured, Linux x86_64)

| `dx` [m] | rel@τ=1 | rel@τ=2 | rel@τ=3 | rel@τ=4 | CI lock (τ=4) |
| --- | --- | --- | --- | --- | --- |
| 0.050 | 0.163 | 0.154 | 0.137 | **0.115** | **&lt; 0.15** |
| 0.045 | 0.126 | 0.107 | 0.084 | **0.0685** | **&lt; 0.10** |
| 0.040 | — | — | — | **0.060** | monotone series |
| 0.030 | — | — | — | **0.019** | monotone series |

Late-time error **decreases** through `dx=0.03` (τ=4: **0.115 → 0.0685 → 0.060 → 0.019**).
Offline probe: `cargo run -p pe-fluid --example probe_dam_ritter --release`.

**Splash-window root cause (fixed):** warm-started κ + uncapped pressure Δv
(`Δv ∝ 1/dt`) produced vertical bed splash below `dx≈0.04`. Fix: per-iteration
`max_velocity_correction = 0.8` on the Ritter/CI path (Jacobi reactions from the
capped Δv). Mid-time front lock remapped to **&lt; 0.12** after the clamp
(measured front rel ~0.107 at `dx=0.045`, τ=2).

#### Column height `h(x)` vs Ritter at τ=2

Mean `|h_sim − h_Ritter|/h₀` over wet bins (`h` or `h_Ritter` ≥ `0.02 h₀`),
`dx=0.045`: **~0.150** (CI lock **&lt; 0.25**); front rel **~0.107** (lock **&lt; 0.12**).
Mass conserved (constant particle mass).

#### Controlled experiment — what stopped finer convergence (pre-fix)

| Variant | Observation |
| --- | --- |
| Akinci walls only (no hard clamp) | Floor penetration; tip blows up as `dx` decreases |
| Hard walls only (no Akinci) | Stable but late-time lag **~20–22%**, flat in `dx` (truncated wall kernel) |
| Akinci + free-slip hard walls, uncapped Δv | Splash / `v_max` runaway for `dx ≲ 0.04` |
| Same + `max_velocity_correction=0.8` | Monotone τ=4 series through `dx=0.03` |

So the previous flat 0.46/0.42/0.39 series was an **invalid early-time toe metric**
(`t=0.22` ⇒ `τ≈1.09`, tip particle / density threshold), not evidence of a Ritter
model gap. With the height-field + late-`τ` protocol and Δv clamp, DFSPH converges
through `dx=0.03`.

Martin & Moyce (1952) / Koshizuka–Oka (1996) tables are **not** transcribed (no
verified primary source in this environment).

Fine `dx=0.0225` (`dx/2` below CI): `--ignored` test
`dfsph_dam_break_fine_dx_and_stability_window` — expected to leave the stable
window; numbers recorded when run, not silently greenwashed.

### Surface tension / droplet

Akinci (2013) cohesion + curvature (optional Becker–Teschner colour-field path).
**Sign:** cohesion is attractive (`F ∝ +σ m² C r̂` with `r̂ = (xⱼ−xᵢ)/r`).
**2-D scale:** discrete `C(r)` / normal pair-sum over-predicts the Laplace jump by
~20× vs SI `σ`; a documented `dim_scale = 1/20` is applied in 2-D so `σ` is
SI-interpretable for the droplet test (3-D keeps `1.0` pending calibration).

Analytic Laplace: 2-D `Δp = σ/R`, 3-D `Δp = 2σ/R` (oracle bit-exact).

**Pressure metric:** Shepard-corrected density → Tait pressure, interior mean
`d < 0.4 R` (excludes the free-surface boundary layer); differential
`Δp = p̄_int(σ) − p̄_int(0)` vs ambient-free `σ/R`.

2-D free droplet WCSPH (`σ=2`, `R₀=0.12`, `dx=0.02`, `c₀=12`, AV/XSPH), measured:

| Quantity | Measured | CI lock |
| --- | --- | --- |
| Sphericity RMS `/R` | ~0.30 | **&lt; 0.40** |
| Radius drift | ≪ 20% | **&lt; 20%** |
| Laplace `Δp` vs `σ/R` | **~0.056** rel | **&lt; 0.15** |

### Two-way fluid↔rigid coupling (Akinci 2012)

Boundary particles sampled on rigid surfaces; fluid→boundary reactions reduce to
force/torque about COM and inject into `pe_core` via [`ExternalLoads`]
(stable continuous-load interface). WCSPH preferred for float/sink oracles;
DFSPH used for penetration / coupling hash.

| Oracle | Measured (Linux x86_64) | CI lock |
| --- | --- | --- |
| Archimedes draft (`ρ_b/ρ_w`, WCSPH, `dx=0.05`, 2000×`2.5e-4`) | **rel ≈ 0.0653** | **&lt; 0.10** |
| Heave period via bottom-face stiffness `k=(F₊−F₋)/(2δ)` | **rel_T ≈ 0.504** (`T≈0.954` vs `0.635`) | **&lt; 0.60** |
| Sphere/disk sink peak `v` vs Cd=1 drag balance (2-D) | **rel ≈ 0.358** (`v_peak≈1.59` vs `2.48`); sank ≈0.39 m | **&lt; 0.55** + sank &gt; 0.25 m |
| Closed fluid+body momentum (no walls/gravity) | **‖ΔP‖ ≈ 2×10⁻¹⁴** | **&lt; 1×10⁻³** |
| Box penetration (WCSPH / DFSPH) | **max ≈ 0.080 / 0.046** on `dx=0.05` | **&lt; 2 dx** |
| Coupling hash | `e9a6a84fef4ce9d5` | cross-OS CI |

Heave note: free-surface WCSPH float is overdamped (no clean `y(t)` zero-crossings);
the oracle locks the stiffness-derived period against `2π√(m/(ρ g A))`. Do not
silently loosen locks.

### Other checks

| Check | Result |
| --- | --- |
| Mass conservation | bit-exact (constant particle mass) |
| Internal pair forces | Σ F = 0 (equal-mass pairs) |
| Viscosity-only pair kick | KE non-increasing |
| Determinism hashes | [`DETERMINISM.md`](DETERMINISM.md) (PBF + DFSPH + coupling) |

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
| `coupling_tank_hash` (2-D float, 80 steps) | see Criterion run |

Neighbour-list reuse avoids one uniform-grid rebuild per PBF iteration.
