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

### Two-way rigid coupling (M28)

`SoftRigidWorld` (`pe_soft::coupling`) couples XPBD particles to dynamic
`pe-core` rigid bodies inside the same substep loop:

| Constraint | Model |
| --- | --- |
| **Attachment** | Soft particle pinned to a body-frame point: `C = x_p − (x + R local)` solved as a 3-vector XPBD system with rigid generalized inverse mass `W = m⁻¹I − [r×] I⁻¹ [r×]` at the attach point (Müller PBD / Deul–Bender style). Compliance α [m/N] per axis (`α = 0` hard). |
| **Contact** | Soft particle vs dynamic **OBB box** or **sphere**: unilateral positional XPBD along the contact normal; equal-and-opposite positional impulse on the rigid. Friction/restitution damp the particle relative velocity. |
| **Rigid–plane** | Dynamic box/sphere vs static planes (support point / sphere radius) so platforms can rest on the ground while soft bodies land on them. |

Gravity is taken from `SoftParams::gravity` and applied identically to soft
prediction and rigid prediction (no separate `ForceGenerator` on the hash path).
Velocities are reconstructed from positions / orientations after each substep.
Force diagnostics use Macklin XPBD `f ≈ λ / h²`.

**Limits (honest):**

- Convex shapes for coupling contacts: box + sphere only (no capsule/hull yet).
- Soft self-collision and soft–static paths remain one-way (statics infinite mass).
- Angular CCD / continuous soft–rigid collision for thin sheets deferred.
- Scene-schema serialization of soft worlds / attachments is **deferred** (API +
  Python only in M28; keep scene v3 backward compatible without a soft bump).
- Energy estimate for contact-rich stacks can chatter; the locked energy oracle
  uses a contact-free damped attachment oscillator.

**References (algorithms only — clean-room; see `THIRD_PARTY.md`):** Macklin
XPBD (2016) + Small Steps (2019); Müller et al. PBD (2007) rigid generalized
mass; Deul / Bender position-based rigid–soft coupling ideas. No FleX / PhysX /
PositionBasedDynamics source copied.

## Materials

`Material { density, youngs_modulus, poisson_ratio, friction, restitution,
yield_strength, hardening_modulus, thermal_conductivity, specific_heat,
thermal_expansion, fracture_stress, fracture_strain, verified, source_note }`.

Built-in table: prior demo rows (`rubber_*`, `steel_nominal`, …) stay
`verified: false`. Handbook rows (`steel_aisi1020`, `aluminum_6061_t6`,
`copper_c11000`, `titanium_ti6al4v`, `glass_soda_lime`, `concrete_normal_weight`,
`water_20c`) are `verified: true` with MatWeb/ASM/CRC citations — see
[`MATERIALS.md`](MATERIALS.md).

Conversions (unit-tested):

- Lamé: `μ = E / (2(1+ν))`, `λ = E ν / ((1+ν)(1−2ν))` (undefined as `ν → ½`)
- Stretch compliance: `α = L₀ / (E A)` for a uniaxial segment
- Volume compliance (helper): `α ≈ 1 / (K V₀)` with `K = λ + 2μ/3`

Plasticity (J2), fracture (max principal), and thermal conduction/expansion are
implemented as modules used by materials-track / M29 tests and optional callers;
the default XPBD `SoftWorld::step` remains elastic XPBD (fracture/thermal applied
when the caller invokes them). M29 adds unconditionally stable **backward Euler**
and **Crank–Nicolson** conduction (Thomas / CG), Robin/Neumann BCs, tet assembly,
and thermo-elastic rest updates (`α ΔT` on distances / tet volumes).

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
| Soft↔rigid linear momentum (isolated, no g) | abs drift **~2.8e-13**; CI **&lt; 1e-10** | Particle cluster vs free box |
| Soft on rigid box on ground (support = weight) | rel **~0.36%**; CI **&lt; 2%** | After settle; KE &lt; 0.05 J |
| Rope + free rigid freefall CoM | abs **~3.2e-4 m** at t=0.5 s; CI **&lt; 5e-4 m** | vs `z₀ − ½ g t²` |
| Pendulum heavy bob / stiff rope | period rel **~0.03%**; CI **&lt; 1%** | Small-angle `T=2π√(L/g)` |
| Damped attachment oscillator energy | **E_max = E₀**; decay **~52%** over 2000×1/240 s | No contacts; CI no growth + drop &gt; 1% |

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

Soft↔rigid coupling hash: `cargo run -p pe-soft --bin soft_coupling_hash --release`
(rope on dynamic box + jelly on dynamic platform, 180×1/120, 20 substeps) —
locked in [`DETERMINISM.md`](DETERMINISM.md). Existing soft / materials /
Godot / bake hashes are **unchanged** by M28.

M29 implicit thermal hash: `cargo run -p pe-soft --bin thermal_hash --release`
(`becda3e8805b5b90`). Soft / materials / soft_coupling hashes **unchanged**.

## Python

```python
import physics_engine_rs as pe
w = pe.SoftWorld.rope([-0.5,0,1], [0.5,0,1], segments=8)
w.set_params(substeps=20, damping=1.0)
w.run(1.0, 1/60)
print(w.x().shape)  # (n, 3) NumPy

# Two-way coupling
c = pe.SoftRigidWorld.rope_on_free_body(m_rope=1.0, m_body=3.0)
c.run(0.5, 1/240)
print(c.combined_com(), hex(pe.canonical_soft_coupling_hash()))

# M29 implicit thermal rod
rod = pe.ThermalRod(33, 1.0, 0.01, 8960.0, 385.0, 400.0, 16.5e-6)
rod.init_sine_mode(50.0)
rod.set_fixed([True] + [False]*31 + [True])
rod.step(25.0, "be")
print(rod.temperatures().shape, hex(pe.canonical_thermal_hash()))
```

## Materials track

See [`MATERIALS.md`](MATERIALS.md): J2 plasticity (&lt;1% uniaxial), fracture
(&lt;2% threshold), explicit thermal Fourier (&lt;2%), **M29 implicit** Fourier
(~5e-4), erfc / Robin / clamped-stress oracles, **M30 plastic XPBD** permanent
set (~0), **M31 finite-strain F_p / Hencky** unload (~0), **M32 spectral
Hencky / Kirchhoff on `b_e`**, **M33 bilinear cohesive fracture**, locked
`materials_hash` + `thermal_hash` + `plastic_hash` + `fs_plastic_hash` +
`be_plastic_hash` + `cohesive_hash`.

### Soft XPBD ↔ J2 plastic coupling (M30)

`PlasticSoftWorld` wraps `SoftWorld` and tracks selected distance / tet
constraints. Each `step(dt)` runs XPBD then a rest-update pass from the cold
geometry through the J2 return map (see [`MATERIALS.md`](MATERIALS.md)).

| Oracle | Measured | CI lock |
| --- | --- | --- |
| Plastic unload length vs `L₀(1+ε_p)` | **0** | **&lt; 1%** |
| Elastic springback | **0** | **&lt; 0.5%** |
| Hardening slope vs `E_t` | **~0** | **&lt; 1%** |
| Tet shear return-to-surface | **&lt; 1%** | **&lt; 1%** |

Locked `plastic_hash` = `73d9ebdd6677ec8c`
(`cargo run -p pe-soft --bin plastic_hash --release`). Soft / materials /
soft_coupling / thermal hashes unchanged.

```python
import physics_engine_rs as pe
bar = pe.PlasticBar(2e11, 0.3, 2.5e8, 2e9, l0=1.0, area=0.01)
length, eps_p, analytic = pe.soft_quasistatic_stretch_unload(
    2e11, 0.3, 2.5e8, 2e9, 1.0, 0.01, 3.25e-3
)
print(length, eps_p, analytic, hex(pe.canonical_plastic_hash()))
```

### Finite-strain multiplicative F_p (M31)

`FsPlasticSoftWorld` uses Hencky strain on bars and a multiplicative
`F = F_e F_p` update on tets (`exp(Δε_p)` + isochoric projection). See
[`MATERIALS.md`](MATERIALS.md).

| Oracle | Measured | CI lock |
| --- | --- | --- |
| Log unload vs `L₀ exp(ε_p)` | **~0** | **&lt; 1%** |
| Small-strain log≡eng | **≪ 0.5%** | **&lt; 0.5%** |
| Large-strain divergence at `λ=1.2` | **≫ 5%** | **&gt; 5%** |
| Tet `|det F_p − 1|` after shear | **~0** | **&lt; 1e-10** |

Locked `fs_plastic_hash` = `7c902cd304aea5ae`
(`cargo run -p pe-soft --bin fs_plastic_hash --release`). M30 `plastic_hash`
and soft / materials / soft_coupling / thermal hashes unchanged.

```python
import math
import physics_engine_rs as pe
lam = 1.06
length, eps_p, analytic = pe.soft_quasistatic_log_stretch_unload(
    1e7, 0.3, 5e4, 1e5, 1.0, 0.01, lam
)
print(length, eps_p, analytic, hex(pe.canonical_fs_plastic_hash()))
bar = pe.FsPlasticBar(1e7, 0.3, 5e4, 1e5, l0=1.0, area=0.01)
```

### Spectral Hencky / Kirchhoff on `b_e` (M32)

`BePlasticSoftWorld` replaces the M31 Green-of-`F_e` tet trial with the Simo
1992 spectral algorithm on `b_e = F_e F_eᵀ` (principal Hencky + Kirchhoff J2).
Bars reuse the M31 Hencky path. See [`MATERIALS.md`](MATERIALS.md).

| Oracle | Measured | CI lock |
| --- | --- | --- |
| Isochoric uniaxial `τ₁=2μ ln λ`, `σ_eq=3μ\|ln λ\|` | **~0** | **&lt; 1e-12** |
| Plastic Kirchhoff surface | **~0** | **&lt; 1e-10** |
| `det b_e = 1` after isochoric return | **~0** | **&lt; 1e-10** |
| Small-shear spectral≡Green | **≪ 2%** | **&lt; 2%** |
| Large-shear divergence | **≫ 5%** | **&gt; 5%** |
| Tet Kirchhoff surface after shear | **≪ 1%** | **&lt; 1%** |

Locked `be_plastic_hash` = `8562bc39f064698e`
(`cargo run -p pe-soft --bin be_plastic_hash --release`). M31 `fs_plastic_hash`
/ M30 `plastic_hash` and soft / materials / soft_coupling / thermal hashes
unchanged.

```python
import physics_engine_rs as pe
print(hex(pe.canonical_be_plastic_hash()))
seq = pe.soft_analytic_isochoric_uniaxial_seq(1e7, 0.3, 1.01)
bar = pe.BePlasticBar(1e7, 0.3, 5e4, 1e5, l0=1.0, area=0.01)
```

### Cohesive-zone fracture (M33)

`CohesiveSoftWorld` places irreversible bilinear Mode-I cohesive bonds on
distance constraints (`G_c = ½ σ_c δ_f`, secant damage, crack-path events).
See [`MATERIALS.md`](MATERIALS.md).

| Oracle | Measured | CI lock |
| --- | --- | --- |
| `G_c = ½ σ_c δ_f` + `∫ t dδ` | **0** / ~1e-8 | exact / **&lt; 1e-6** |
| Virgin peak `t(δ_c)=σ_c` | **0** | **&lt; 1e-12** |
| Irreversible secant unload | **~0** | **&lt; 1e-12** |
| XPBD fail past `δ_f` + islands | failed; 2 islands | — |
| Mid-softening damage ≡ analytic | **≪ 2%** | **&lt; 2%** |

Locked `cohesive_hash` = `28e662bfee66bfd0`
(`cargo run -p pe-soft --bin cohesive_hash --release`). M32 `be_plastic_hash`
and prior soft / materials / plastic / thermal hashes unchanged.

```python
import physics_engine_rs as pe
gc = pe.soft_analytic_gc(1e5, 2e-3, 1e-2)
print(gc, hex(pe.canonical_cohesive_hash()))
bar = pe.CohesiveBar(1e5, 2e-3, 1e-2, l0=1.0, area=0.01)
bar.set_free_end_x(1.0 + 1.05e-2)
bar.set_free_end_pinned(True)
for _ in range(80):
    bar.step(1.0 / 240.0)
print(bar.cohesive_state(), bar.crack_path())
```

## Deferred

- Mixed-mode / continuum-surface CZM; tet-face cohesive interfaces; tearing UI
- Radiation BCs; `k(T)` / `c_p(T)`
- Capsule/hull soft↔rigid contacts; soft scene-schema I/O
- Anisotropic cloth (warp/weft), wet cloth, continuous collision for thin sheets
- GPU / SIMD particle loops
- Additional handbook rows (wood species cells, elastomer compounds) with
  primary-table citations
- Corotational FEM

## References (formulas only — clean-room)

- Macklin, Müller, Chentanez (2016). XPBD.
- Müller et al. (2007). Position Based Dynamics.
- Macklin (2019). Small Steps in Physics Simulation.
- Macklin & Müller (2021). A Constraint-based Formulation of Stable Neo-Hookean Elasticity.
- Müller et al. (2005). Meshless deformations based on shape matching.
- Deul, Charrier, Bender — position-based rigid / soft coupling (algorithms only).
- Simo & Hughes — J2 return mapping / isotropic hardening (textbook identities).
- Fourier — 1-D heat equation fundamental mode (textbook).
- M30 plastic rest-update: Simo–Hughes + XPBD rest-from-plastic-flow pattern
  (algorithms only; see `THIRD_PARTY.md`).
- M31 finite-strain: Hencky strain; multiplicative `F=F_e F_p` with
  `exp(Δε_p)` map + isochoric projection (Simo & Hughes / continuum textbook;
  algorithms only — see `THIRD_PARTY.md`).
- M32 spectral Hencky / Kirchhoff on `b_e`: Simo (1992) / Simo & Hughes
  *Computational Inelasticity* spectral return (algorithms only — see
  `THIRD_PARTY.md`).
- M33 bilinear cohesive zone: Needleman / Camacho–Ortiz traction–separation
  identities; irreversible damage + secant unload (algorithms only — see
  `THIRD_PARTY.md`).
