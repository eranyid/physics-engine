# Soft bodies (`pe-soft`) — XPBD + materials

Clean-room XPBD soft bodies, cloth, ropes, tetrahedral FEM-style constraints, and
a small **nominal** material table. SI units; `f64`; portable `libm`; no FMA;
no `HashMap` iteration on the step path.

## Method

**XPBD** (Macklin, Müller, Chentanez 2016) with **substepping** (Macklin 2019
*Small Steps in Physics Simulation*):

1. Outer step `dt` is split into `N` equal substeps `h = dt/N`.
2. Each substep: predict `x ← x + h v + h² a` (gravity), reset multipliers `λ`,
   project constraints for `iterations` passes, resolve static/self contacts,
   then `v ← (x − x_prev)/h`.
3. For a constraint `C(x)` with compliance `α` [SI depends on `C`]:
   - `α̃ = α / h²`
   - `Δλ = (−C − α̃ λ) / (∑ᵢ wᵢ |∇ᵢ C|² + α̃)`
   - `Δxᵢ = wᵢ ∇ᵢ C Δλ`, `λ ← λ + Δλ`

Compliance for a linear spring of stiffness `k` is `α = 1/k`. Zero compliance
is the rigid limit.

### Constraints (v1)

| Kind | Use |
| --- | --- |
| Distance | Ropes, structural cloth springs, axial edges |
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
yield_strength, thermal_conductivity, specific_heat, verified, source_note }`.

Built-in table (`rubber_*`, `silicone`, `cloth_cotton`, `rope_nylon`, `jelly`,
`steel_nominal`, …): **every entry has `verified: false`** — nominal demo
orders of magnitude only. Do not treat as handbook data.

Conversions (unit-tested):

- Lamé: `μ = E / (2(1+ν))`, `λ = E ν / ((1+ν)(1−2ν))` (undefined as `ν → ½`)
- Stretch compliance: `α = L₀ / (E A)` for a uniaxial segment
- Volume compliance (helper): `α ≈ 1 / (K V₀)` with `K = λ + 2μ/3`

Yield / thermal fields are stored for later plasticity / heat modules and are
**unused** by the XPBD stepper in v1.

## Analytic validation (honest tolerances)

| Oracle | Tolerance / note |
| --- | --- |
| Spring-mass period `T = 2π√(m/k)` via `α=1/k` | ≤ 8% relative (finite substeps) |
| Zero-compliance distance | `|L − L₀| < 1e-8` |
| Hanging rope vs catenary `a cosh(x/a)+c` | max `|Δz| < 3 cm` on 1 m span |
| Axial bar `δL = F L/(E A)` (tet + edge springs) | ≤ 40% relative (coarse mesh) |
| Near-incompressible tet volume (`ν→0.5`) | Σ\|ΔV\| / Σ\|V₀\| < 8% |
| Cantilever tip vs distributed EB `mg L³/(8EI)` | ratio in `[0.25, 2.5]` |
| Cloth pinned corners | hangs; damped energy does not grow unboundedly |
| Conservative spring energy | ≤ ~5% growth bound (XPBD not exactly symplectic) |
| Substep convergence | more substeps do not worsen distance residual |
| Balloon overpressure | volume increases toward target |

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

## Deferred (not in v1)

- Corotational FEM / full continuum strain measures beyond XPBD tet constraints
- Plasticity, fracture, tearing
- Thermal conduction / thermo-mechanical coupling
- Two-way coupling to rigid `ContactWorld` / joints
- Anisotropic cloth (warp/weft), wet cloth, continuous collision for thin sheets
- GPU / SIMD particle loops

## References (formulas only — clean-room)

- Macklin, Müller, Chentanez (2016). XPBD.
- Müller et al. (2007). Position Based Dynamics.
- Macklin (2019). Small Steps in Physics Simulation.
- Macklin & Müller (2021). A Constraint-based Formulation of Stable Neo-Hookean Elasticity.
- Müller et al. (2005). Meshless deformations based on shape matching.
