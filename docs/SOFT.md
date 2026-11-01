# Soft bodies (`pe-soft`) — XPBD + materials

Clean-room XPBD soft bodies, cloth, ropes, tetrahedral FEM-style constraints, and
a small **nominal** material table. SI units; `f64`; portable `libm`; no FMA;
no `HashMap` iteration on the step path.

## Method

**XPBD** (Macklin, Müller, Chentanez 2016) with **substepping** (Macklin 2019
*Small Steps in Physics Simulation*):

1. Outer step `dt` is split into `N` equal substeps `h = dt/N`.
2. Each substep: predict `x ← x + h v` after `v ← v + h (g + F_ext/m)`, reset
   multipliers `λ`, project constraints for `iterations` passes, resolve
   static/self contacts, then `v ← (x − x_prev)/h`.
3. For a constraint `C(x)` with compliance `α` [SI depends on `C`]:
   - `α̃ = α / h²`
   - `Δλ = (−C − α̃ λ) / (∑ᵢ wᵢ |∇ᵢ C|² + α̃)`
   - `Δxᵢ = wᵢ ∇ᵢ C Δλ`, `λ ← λ + Δλ`

Compliance for a linear spring of stiffness `k` is `α = 1/k`. Zero compliance
is the rigid limit. Per-particle `ext_force` [N] is applied in prediction for
quasistatic loads (not cleared automatically).

### Constraints (v1)

| Kind | Use |
| --- | --- |
| Distance | Ropes, structural cloth springs, axial spring-chain |
| Distance bend | Cheap cloth bending (skip-one neighbors) |
| Dihedral bend | Angle-based cloth/shell bending |
| Volume (tet) | `C = 6(V − V₀)` per tetrahedron |
| Tetra FEM | StVK-like or stable neo-Hookean (Macklin & Müller 2021) |
| Shape matching | Optional cluster rigidification (Müller et al.) |
| Pressure | Closed-surface volume target (balloons) |

### Collisions

- Static **plane**, **sphere**, **AABB box** (positional push-out + friction/restitution on velocity).
- **Self-collision** via spatial hash into `Vec` buckets; pairs sorted by
  `(min_id, max_id)` before resolution (deterministic).

## Materials

`Material { density, youngs_modulus, poisson_ratio, friction, restitution,
yield_strength, hardening_modulus, thermal_conductivity, specific_heat,
thermal_expansion, fracture_stress, fracture_strain, verified, source_note }`.

Built-in table (`rubber_*`, `silicone`, `cloth_cotton`, `rope_nylon`, `jelly`,
`steel_nominal`, `copper_nominal`, …): **every entry has `verified: false`** —
nominal demo orders of magnitude only. Do not treat as handbook data.
Details and measured oracles: [`MATERIALS.md`](MATERIALS.md).

Conversions (unit-tested):

- Lamé: `μ = E / (2(1+ν))`, `λ = E ν / ((1+ν)(1−2ν))` (undefined as `ν → ½`)
- Stretch compliance: `α = L₀ / (E A)` for a uniaxial segment
- Volume compliance (helper): `α ≈ 1 / (K V₀)` with `K = λ + 2μ/3`

Plasticity (J2), fracture (max principal), and thermal conduction/expansion are
implemented as modules used by materials-track tests and optional callers; the
default XPBD `SoftWorld::step` remains elastic XPBD (fracture/thermal applied
when the caller invokes them).

## Analytic validation (measured)

| Oracle | Measured / bound | Notes |
| --- | --- | --- |
| Spring-mass period `T=2π√(m/k)`, `α=1/k` | **period rel err ≈ 0** at ≥4 substeps with outer `dt=T/400`; CI locks **&lt; 2%** for ≥32 substeps | Amplitude decays separately (XPBD numerical damping): ~0.17% over 10 peaks at 128 substeps; damping decreases as substeps increase |
| Zero-compliance distance | `|L − L₀| &lt; 1e-8` | Rigid limit |
| Hanging rope vs catenary | max `\|Δz\| &lt; 3 cm` on 1 m span | Fitted `a cosh(x/a)+c` |
| Axial bar `δL = F L/(E A)` (**spring-chain**) | **~0.21%** rel, mesh-independent (n=1…16); CI **&lt; 0.5%** | Persistent `ext_force` on free end; quasistatic settle `\|v\|→0` |
| Axial bar tet FEM vs `FL/(EA)` | **ν=0: ~0.2–0.8%** (4×1×1→12×3×3); **ν=0.3: ~1.0–1.4%** | StVK energy XPBD `C=√(2Ψ)`, `α=1/V₀`; λ-constraint skipped when `ν=0` (never rigid `J=1`). CI **&lt; 5%** on each mesh; coarse→mid refinement improves. |
| Near-incompressible tet volume (`ν=0.495`) | **~0.9%** Σ\|ΔV\|/Σ\|V₀\|; CI **&lt; 3%** | Soft cube drop onto plane |
| Cantilever tip vs EB `mg L³/(8EI)` | **8×2×2 ≈ 0.36; 16×2×2 ≈ 0.62** (ρ=20, small deflection); **32×4×4 ≈ 0.88** | Linear tets **shear-lock**. Prior ~1.0 EB match was the under-stiff FEM bug cancelling lock. CI: refinement increases tip ratio; 16×2×2 band **0.5–0.85** (not a &lt;5% continuum claim). |
| Cloth pinned corners | hangs; damped energy does not grow | — |
| Conservative spring energy (long run) | **no growth** (`E_max=E₀`); decay **≈12.7% at t≈21 s**, **≈23.8% at t≈42 s** (`α=0.01`, 64 substeps, `dt=1/240`) | XPBD numerical dissipation — not symplectic. CI locks decay bands. |
| Substep residual study | more substeps do not worsen loaded distance residual | — |
| Balloon overpressure | volume increases toward target | — |

### Axial / FEM notes

1. **Spring-chain** is the rigorous XPBD map of uniaxial Hooke (`α = L/(EA)`). With
   prediction-step `ext_force` it settles to static equilibrium within ~0.2%.
2. **Tet FEM (StVK)** uses continuum Green energy
   `Ψ = μ‖E‖²_F + (λ/2)(tr E)²` with XPBD `C = √(2Ψ)`, `α = 1/V₀` (split μ/λ).
   When `λ ≈ 0` the hydrostatic constraint is **omitted** — the old `α = 0` rigid
   `J = 1` path made isochoric stretch `I_C−3 = O(ε²)` and destroyed linear
   `FL/(EA)` stiffness (5–25× too soft). Gradients come from the first Piola
   `P = F(2μ E + λ tr(E) I)` (not separately normalized).
3. **Stable NH** (log form) is a single energy constraint
   `Ψ = (μ/2)(I_C−3) − μ ln J + (λ/2)(ln J)²`. Mesh-refinement table (release,
   40 substeps / 6 iters, settle `|v|→0`):

   | ν | mesh | rel vs `FL/(EA)` |
   | --- | --- | --- |
   | 0 | 4×1×1 | ~0.82% |
   | 0 | 8×2×2 | ~0.16% |
   | 0 | 12×3×3 | ~0.29% |
   | 0.3 | 4×1×1 | ~1.42% |
   | 0.3 | 8×2×2 | ~1.04% |
   | 0.3 | 12×3×3 | ~1.16% |

## Determinism

Canonical scene (rope + cloth + jelly above `z=0` plane), 120 steps @ 1/60 s,
10 substeps: see [`DETERMINISM.md`](DETERMINISM.md) and
`cargo run -p pe-soft --bin soft_hash --release`.

## Python

```python
import physics_engine_rs as pe
w = pe.SoftWorld.rope([-0.5,0,1], [0.5,0,1], segments=8)
w.set_params(substeps=20, damping=1.0)
w.run(1.0, 1/60)
print(w.x().shape)  # (n, 3) NumPy
```

## Materials track (this PR)

See [`MATERIALS.md`](MATERIALS.md): J2 plasticity (&lt;1% uniaxial), fracture
(&lt;2% threshold), thermal Fourier (&lt;2%), material table additions, locked
`materials_hash`.

## Deferred

- Full per-step FEM plastic coupling inside XPBD (constitutive map shipped)
- Cohesive fracture / crack paths; tearing UI
- Implicit thermal; radiation/convection BCs
- Two-way coupling to rigid `ContactWorld` / joints
- Anisotropic cloth (warp/weft), wet cloth, continuous collision for thin sheets
- GPU / SIMD particle loops
- `verified: true` handbook rows with primary sources

## References (formulas only — clean-room)

- Macklin, Müller, Chentanez (2016). XPBD.
- Müller et al. (2007). Position Based Dynamics.
- Macklin (2019). Small Steps in Physics Simulation.
- Macklin & Müller (2021). A Constraint-based Formulation of Stable Neo-Hookean Elasticity.
- Müller et al. (2005). Meshless deformations based on shape matching.
- Simo & Hughes — J2 return mapping / isotropic hardening (textbook identities).
- Fourier — 1-D heat equation fundamental mode (textbook).
