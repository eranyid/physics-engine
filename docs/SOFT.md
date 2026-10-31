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

## Analytic validation (measured)

| Oracle | Measured / bound | Notes |
| --- | --- | --- |
| Spring-mass period `T=2π√(m/k)`, `α=1/k` | **period rel err ≈ 0** at ≥4 substeps with outer `dt=T/400`; CI locks **&lt; 2%** for ≥32 substeps | Amplitude decays separately (XPBD numerical damping): ~0.17% over 10 peaks at 128 substeps; damping decreases as substeps increase |
| Zero-compliance distance | `|L − L₀| &lt; 1e-8` | Rigid limit |
| Hanging rope vs catenary | max `\|Δz\| &lt; 3 cm` on 1 m span | Fitted `a cosh(x/a)+c` |
| Axial bar `δL = F L/(E A)` (**spring-chain**) | **~0.21%** rel, mesh-independent (n=1…16); CI **&lt; 0.5%** | Persistent `ext_force` on free end; quasistatic settle `\|v\|→0` |
| Axial bar tet FEM vs `FL/(EA)` | **ν=0: ~5–9× too soft; ν=0.3 + fixed face: ~20–26× too soft** | Mesh refinement does **not** converge to continuum. Cause: XPBD tet compliance / gradient calibration under-stiff for this formulation (deferred). Fixed-face Poisson restraint alone cannot explain the gap. CI locks the documented under-stiff band (not a &lt;5% continuum claim). |
| Near-incompressible tet volume (`ν=0.495`) | **~0.9%** Σ\|ΔV\|/Σ\|V₀\|; CI **&lt; 3%** | Soft cube drop onto plane |
| Cantilever tip vs EB `mg L³/(8EI)` | **ratio ≈ 1.006** (tip_drop≈0.296 m, δ_EB≈0.294 m) | Timoshenko shear add-on `δ_s≈mg L/(2 k_s G A)` with `k_s=5/6` gives δ_Timo≈0.297 m, ratio≈0.996. Linear tets can shear-lock on thick beams; this slender mesh shows ~1% shear correction, no severe locking. CI **\|ratio−1\| &lt; 5%** vs EB. |
| Cloth pinned corners | hangs; damped energy does not grow | — |
| Conservative spring energy (long run) | **no growth** (`E_max=E₀`); decay **≈12.7% at t≈21 s**, **≈23.8% at t≈42 s** (`α=0.01`, 64 substeps, `dt=1/240`) | XPBD numerical dissipation — not symplectic. CI locks decay bands. |
| Substep residual study | more substeps do not worsen loaded distance residual | — |
| Balloon overpressure | volume increases toward target | — |

### Axial / FEM notes (why tet continuum &lt;5% was not claimed)

1. **Spring-chain** is the rigorous XPBD map of uniaxial Hooke (`α = L/(EA)`). With
   prediction-step `ext_force` it settles to static equilibrium within ~0.2%.
2. **Tet FEM** (StVK-style XPBD) remains under-stiff after removing the old
   edge-spring double-count and after small-strain energy/gradient corrections.
   Refining `n_x×n_y×n_z` does not drive error to zero — this is a **formulation
   calibration** limit, not a coarse-mesh artifact. Poisson blocking at the pinned
   face is a real Saint-Venant effect but is O((side/L)²) for aspect ratio 10, far
   smaller than the observed 5–25× gap.
3. When tet FEM continuum axial is fixed, replace the under-stiff band test with a
   &lt;5% continuum oracle and record the mesh-convergence table here.

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

- Corotational FEM / calibrated continuum tet moduli matching `FL/(EA)` under mesh refinement
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
