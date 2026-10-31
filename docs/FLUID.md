# Particle fluids (`pe-fluid`)

Weakly compressible SPH (WCSPH) and Position Based Fluids (PBF) for 3-D liquids,
with a 2-D mode for analytic tests. Clean-room from published formulas — see
[`THIRD_PARTY.md`](../THIRD_PARTY.md). **No** surface tension (unless a later
milestone adds it), **no** free-surface mesh reconstruction, **no** DFSPH in this
crate yet, **no** fluid↔rigid two-way coupling (deferred; `pe-core` untouched).

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

Normalisation checked by rectangle-rule partition-of-unity tests
(`∫W ≈ 1`, error &lt; 1–5% at documented grid density). Gradient consistency:
`Σ V ∇W ≈ 0` (zeroth order) and `Σ V (xⱼ−xᵢ)⊗∇ᵢW ≈ I` (first order) on lattices.

## Neighbour search

Uniform grid with **sorted cell lists**: particles sorted by `(ix,iy,iz,index)`;
occupied cells stored in a dense sorted key array (binary search). **No**
`HashMap` iteration on the step path.

## WCSPH

- Density summation with Akinci (2012) boundary volume `ψ_b`.
- Tait / Cole EOS: `p = B[(ρ/ρ₀)^γ − 1]`, `B = ρ₀ c₀²/γ`, `γ = 7` (water).
- Sound-speed rule: **`c₀ ≥ 10 v_max`** (≈1% compressibility target).
- Monaghan artificial viscosity + optional XSPH.
- Symplectic Euler with CFL helper `dt ≤ λ h/(c+v_max)`.

## PBF (Macklin & Müller 2013)

- Constraint `Cᵢ = ρᵢ/ρ₀ − 1`, λ multipliers with CFM `ε ≈ 0.01`.
- Artificial pressure `s_corr` (tensile instability); ratio clamped ≤ 1; tank
  settles often set `k = 0`.
- Optional vorticity confinement (`vorticity_eps`, default 0).
- XSPH-like viscosity on velocities after the constraint solve.

## Deferred

| Item | Notes |
| --- | --- |
| DFSPH (Bender & Koschier 2015) | Divergence-free / density-invariant SPH |
| One-way fluid → rigid buoyancy/drag | Would sample particles onto `pe-core` bodies |
| Surface tension | Not implemented |
| Free-surface reconstruction | Marching cubes / splash maps out of scope |
| Large-scale scalability | Single-threaded deterministic; no GPU |

## Validation (honest tolerances)

| Check | Result / bound |
| --- | --- |
| Kernel ∫W (cubic 3-D, n_half=24) | error &lt; 1% |
| Gradient zeroth / first order (2-D lattice) | \|Σ∇W\| ~ 0; vs I &lt; 25% at h/8 |
| Tait hydrostatic linearisation `p≈ρ₀gh` | rel &lt; 5% at `c₀=50`, `h=1` m |
| WCSPH column pressure ordering | bottom mean p &gt; top mean p |
| PBF settled tank mean \|ρ−ρ₀\|/ρ₀ | **&lt; 1%** (2-D, `k=0`, 10 iters) |
| Dam-break front vs Ritter `x_f=2√(gh₀)t` | coarse PBF within **~55%** (lags ideal) |
| Mass conservation | bit-exact (constant particle mass) |
| Internal pair forces | Σ F = 0 (equal-mass pairs) |
| Viscosity-only pair kick | KE non-increasing |
| Determinism hash | locked in [`DETERMINISM.md`](DETERMINISM.md) |

Martin & Moyce (1952) experimental tables are **not** transcribed (no verified
primary table in-tree). Ritter (1892) shallow-water analytic dam-break is the
front oracle. Poiseuille / Couette closed forms are exposed as helpers; full 2-D
viscous channel soft-start is optional follow-up.

## Scalability

O(N) neighbour gather with compact support; cost dominated by neighbour loops ×
PBF iterations. Criterion baselines (particles/sec style workloads) live in
[`BENCHMARKS.md`](BENCHMARKS.md).
