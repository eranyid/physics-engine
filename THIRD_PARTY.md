# Third-party attributions

This file records copyright and license notices for any third-party source that
is copied or derived into this repository. Algorithmic inspiration alone does
not require an entry; **copied or substantially derived code does**.

Policy: see [docs/ROADMAP.md](docs/ROADMAP.md) § Reference project survey.
This project is distributed under the MIT License (`LICENSE`).

Normal Cargo/PyPI dependencies (declared in `Cargo.toml` / `pyproject.toml` and
fetched by the package manager) are **not** vendored here. Notable intentional
choice for determinism: the pure-Rust [`libm`](https://crates.io/crates/libm)
crate (MIT) supplies sin/cos/sqrt/exp on the step path — used as a dependency,
not copied into this tree.

## Entries (vendored / derived source)

### DOP853 Butcher tableau (SciPy / Hairer)

- **What:** Floating-point coefficient tables for the Dormand–Prince 8(5,3)
  method (`C`, `A`, `B`, `E3`, `E5`, dense-output `D`) used in
  `crates/pe-space/src/integrators/dop853.rs`.
- **Source:** SciPy 1.14.1
  [`scipy/integrate/_ivp/dop853_coefficients.py`](https://github.com/scipy/scipy/blob/v1.14.1/scipy/integrate/_ivp/dop853_coefficients.py)
  (BSD-3-Clause), which reproduces Hairer’s `dop853.f` tables from
  Hairer, Nørsett & Wanner, *Solving Ordinary Differential Equations I*.
- **License:** BSD-3-Clause (SciPy). Hairer’s original Fortran notice permits
  redistribution with attribution; we use SciPy’s BSD-licensed transcription.
- **Not used:** REBOUND / any GPL integrator source.

### G1 / G7 standard drag tables (JBM / BRL)

- **What:** Mach vs `C_D` knots for the G1 and G7 standard projectiles in
  `crates/pe-ballistics/src/g1g7_tables.rs`.
- **Source:** JBM Ballistics public downloads
  [`mcg1.txt`](https://jbmballistics.com/downloads/mcg1.txt) /
  [`mcg7.txt`](https://jbmballistics.com/downloads/mcg7.txt), obtained from the
  US Army Ballistic Research Laboratory (Aberdeen). Cross-checked against
  `py_ballisticcalc` `TableG1` / `TableG7` knot values (LGPL library **not**
  vendored).
- **License:** JBM Downloads page — “These programs and files are free for
  public use. I make no claims to them.”
  (<https://jbmballistics.com/downloads.html>).
- **Interpolation in-tree:** linear (default) and Fritsch–Carlson PCHIP
  (optional API).

## Algorithmic citations (clean-room; not derived source)

The following published descriptions informed clean-room reimplementations in
`crates/pe-collision` (M5a). **No Box2D / Solver2D / Bepu source was copied.**

| Topic | Citation | Where used |
| --- | --- | --- |
| Soft constraints (Hertz / ζ → bias, mass, impulse scales) | Erin Catto, [Solver2D](https://box2d.org/posts/2024/02/solver2d/) (2024-02-05); GDC 2011 “Soft Constraints”; mass-independent form attributed there to Ross Nordby (Bepu) | `softness.rs`, `soft_step.rs` |
| Soft Step / TGS_Soft pipeline (sub-steps, warm start, relax, restitution) | Erin Catto, Solver2D post + Box2D v3 release notes | `soft_step.rs`, `contact.rs` |
| Dynamic AABB tree (SAH insert, fat AABBs, rotations) | Published Box2D dynamic-tree / BVH descriptions (Erin Catto) | `aabb_tree.rs` |
| Island sleeping thresholds | Box2D-style defaults adapted to SI (documented in `islands.rs`) | `islands.rs` |
| Soft joints / motors / limits (maximal coordinates) | Erin Catto GDC soft-constraint talks; Box2D v3 joint descriptions (algorithms only) | `joints.rs`, `soft_step.rs` |
| Speculative contacts + bullet TOI CCD | Box2D v3 / Catto CCD notes (algorithms only); sphere–plane TOI is textbook | `ccd.rs` |
| GJK distance / intersection | Gino van den Bergen, *Collision Detection in Interactive 3D Environments* (2003); Erin Catto GDC GJK notes | `gjk.rs`, `support.rs` |
| EPA penetration | van den Bergen EPA; Catto contact notes | `epa.rs` |
| Quickhull convex hull | Barber, Dobkin, Huhdanpaa, Quickhull (ACM TOMS 1996) | `hull.rs` |
| Conservative advancement CCD | Mirtich; Catto CCD notes (algorithms only) | `ccd.rs` |
| Segment / triangle closest points | Ericson, *Real-Time Collision Detection* | `narrowphase.rs`, `gjk.rs` |

ISA constants and formulas follow ICAO Doc 7488 / ISO 2533 (public standards);
IMO IS Code criteria are from MSC.267(85) (regulatory text — criteria encoded as
pure thresholds, not copied source). Lanchester phugoid and elliptic-wing
formulas are textbook identities. No third-party simulation code was copied.

### M7 note (`pe-space`)

Algorithms implemented clean-room from published papers / textbooks (coefficients
and formulas only — **no** source translated from REBOUND, nyx-space, JSBSim, or
other GPL/AGPL/LGPL projects):

- Yoshida, H. (1990). Construction of higher order symplectic integrators.
  *Phys. Lett. A* **150**.
- Fehlberg, E. (1968). Classical fifth-, sixth-, seventh-, and eighth-order
  Runge–Kutta formulas with stepsize control. NASA TR R-287. (RKF78.)
- Hairer, E., Nørsett, S. P. & Wanner, G. *Solving ODEs I* — DOP853 method;
  coefficient tables via SciPy BSD transcription (see Entries above).
- Izzo, D. (2015). Revisiting Lambert's problem. *Celest. Mech. Dyn. Astr.* **121**.
- Battin / Danby / Vallado textbook identities for Stumpff functions and universal
  Kepler variables.
- Rein & Spiegel (2015). IAS15. *MNRAS* **446**, 1424 (arXiv:1409.4779) — Gauss–Radau
  nodes from `P_8+P_7=0`, predictor–corrector, `ε_b` step control, compensated
  summation. **No** REBOUND source.
- Rein & Tamayo (2015). WHFast. *MNRAS* **452**, 376 (arXiv:1506.01084) — Jacobi
  transforms (§3.1), DKD Wisdom–Holman split; Duncan, Levison & Lee (1998)
  democratic-heliocentric option. Universal Kepler drift via in-tree Stumpff
  helpers. **No** REBOUND source.
- Wisdom, J. (2006). Symplectic correctors for canonical heliocentric n-body
  maps. *AJ* **131**, 2294 — `α=√(7/40)`, `β=1/(48α)`, `X`/`Z` operators,
  orders 3/5/7/11. **No** REBOUND source.
- Wisdom & Holman (1991). Symplectic maps for the N-body problem. *AJ* **102**.
- Chenciner–Montgomery / Simó figure-eight initial conditions are published IC values.

### M9 note (`pe-hydro` waves / `pe-aero` compressibility & check-cases)

Algorithms implemented clean-room from published papers / textbooks (formulas and
coefficients only — **no** third-party simulation source translated):

- Airy (1845); Dean & Dalrymple — finite-depth dispersion `ω² = g k tanh(k h)`.
- Stokes (1847) / Fenton — second-order elevation.
- Gerstner (1802); Kinsman / Tessendorf — trochoidal waves with steepness `Q k A ≤ 1`.
- Pierson & Moskowitz (1964); Hasselmann et al. JONSWAP (1973); Longuet-Higgins /
  Mitsuyasu `cos^{2s}` directional spreading.
- Blackman & Vigna xoshiro256\*\* (deterministic PRNG; implemented in-tree, not
  vendored from an external crate).
- Prandtl–Glauert; Kármán–Tsien (1939); Ackeret linearized supersonic `C_L`.
- Perfect-gas normal-shock / isentropic identities (Anderson / NASA textbook).
- Dryden turbulence shaping (MIL-F-8785C / Hoblit) — 1st-order discrete surrogate.
- NASA/TM-2015-218675 scenarios 1–3 described qualitatively; **published trajectory
  tables are not used** (analytic / self-consistency oracles only).

### M11 note (`pe-hydro` Morison / seakeeping, `pe-aero` BEM)

Clean-room formulas only — **no** third-party simulation source translated:

- Morison, J. R., O’Brien, M. P., Johnson, J. W. & Schaaf, S. A. (1950). The
  force exerted by surface waves on piles. *Petrol. Trans. AIME* **189**.
- API RP 2A / DNV-RP-C205 — typical \(C_m\), \(C_d\) ranges cited as
  **nominal / unverified** defaults (not copied tables).
- Salvesen, N., Tuck, E. O. & Faltinsen, O. (1970). Ship motions and sea loads.
  *Trans. SNAME* **78** — strip heave/pitch + Haskind diffraction structure.
- Lewis, F. M. (1929). The inertia of water surrounding a pitching vessel —
  circular / Lewis-substitute sectional added-mass idea.
- Faltinsen (1990) *Sea Loads…*; Newman *Marine Hydrodynamics* — floating-circle
  \(a_{33}^\infty\), non-dimensional damping scaling (β=0.5 marked nominal).
- Cummins, W. E. (1962). The impulse response function and ship motions —
  retardation `K(t)` cosine transform + convolution time domain.
- Ogilvie, T. F. (1964). Recent progress toward the understanding and prediction
  of ship motions.
- Glauert, H. (1935). Airplane propellers (Durand). Momentum + blade element.
- Prandtl tip-loss factor as quoted in Leishman, *Principles of Helicopter
  Aerodynamics*.
- **Not used:** Vugts (1968) tabulated sections, WAMIT/NEMOH panel outputs,
  certified airfoil polars, commercial BEM codes.

### Blast / ballistics note (`pe-blast`, `pe-ballistics`)

Clean-room formulas only — **no** third-party simulation *code* translated:

- Hopkinson (1915) / Cranz (1926) cube-root scaling; Kinney & Graham (1985)
  *Explosive Shocks in Air* peak-overpressure closed form; Friedlander (1946)
  waveform.
- Swisdak, M. M., Jr. (1994). *Simplified Kingery Airblast Calculations*
  (NSWC IHTR / 26th DoD Explosives Safety Seminar; NTIS ADA526744) — Table 1
  metric coefficients transcribed for Kingery–Bulmash fits (valid-Z clamp).
- Rankine–Hugoniot / Anderson textbook gas dynamics; Sedov (1959) / Taylor (1950)
  point-blast similarity (energy integral for `ξ₀`).
- Toro (2009) HLLC / exact Riemann / MUSCL; Gottlieb–Shu SSP-RK2; Sod (1978).
- Gurney (1943) BRL 405 energy method (caller-supplied `√(2E)`); Mott (1947)
  fragment mass distribution.
- Exterior ballistics oracles / Coriolis–Eötvös (McCoy textbook forms); Poncelet
  penetration; Recht & Ipson (1963) residual velocity.

#### G1 / G7 standard drag tables (vendored data)

| Item | Detail |
| --- | --- |
| What | Mach vs `C_D` knots for the G1 and G7 standard projectiles |
| Where in tree | `crates/pe-ballistics/src/g1g7_tables.rs` |
| Primary source | JBM Ballistics public downloads, from US Army Ballistic Research Laboratory (Aberdeen): [`mcg1.txt`](https://jbmballistics.com/downloads/mcg1.txt), [`mcg7.txt`](https://jbmballistics.com/downloads/mcg7.txt) |
| Licence | JBM Downloads: “These programs and files are free for public use. I make no claims to them.” (<https://jbmballistics.com/downloads.html>) |
| Cross-check | Identical knot values in `py_ballisticcalc` `TableG1` / `TableG7` (LGPL-3.0 — **not** vendored; numbers only cross-checked) |
| Interpolation | **Linear** default for G1/G7 lookups; monotone cubic PCHIP in `DragTable::cd_monotone` |

No LGPL/GPL simulation source was copied — only the public Cd–Mach tables.

### M8 note (`pe-soft` XPBD soft bodies / materials)

Algorithms implemented clean-room from published papers (formulas only — **no**
third-party simulation source translated; no NVIDIA PhysX / FleX / Soft Body /
PositionBasedDynamics source copied):

- Macklin, M., Müller, M., Chentanez, N. (2016). XPBD: Position-Based Simulation
  of Compliant Constrained Dynamics. *Migraine / SCA* (compliance `α̃ = α/dt²`,
  multiplier accumulation).
- Müller, M. et al. (2007). Position Based Dynamics. *J. Vis. Commun. Image
  Represent.*
- Macklin, M. (2019). Small Steps in Physics Simulation. (substepping strategy.)
- Macklin, M. & Müller, M. (2021). A Constraint-based Formulation of Stable
  Neo-Hookean Elasticity. (XPBD tet FEM constraints.)
- Müller, M. et al. (2005). Meshless deformations based on shape matching.
- Simo, J. C. & Hughes, T. J. R. — J2 return mapping / linear isotropic hardening
  (textbook identities; plasticity module).
- Fourier / Carslaw–Jaeger textbook 1-D heat equation fundamental mode (thermal).
- Material table: prior demo rows remain `verified: false`. Handbook rows
  (`steel_aisi1020`, `aluminum_6061_t6`, `copper_c11000`, `titanium_ti6al4v`,
  `glass_soda_lime`, `concrete_normal_weight`, `water_20c`) are marked
  `verified: true` only with MatWeb / ASM MatWeb / CRC–IAPWS citations in
  `source_note` (retrieved 2026-11-01). See [`docs/MATERIALS.md`](docs/MATERIALS.md).

### Particle fluids note (`pe-fluid`)

Clean-room formulas only — **no** third-party SPH/PBF source translated
(no DualSPHysics, SPlisHSPlasH, NVIDIA Flex, LiquidFun, etc.):

- Monaghan, J. J. (1992). Smoothed particle hydrodynamics. *ARA&A* **30** —
  cubic spline kernel; artificial viscosity / XSPH practice.
- Wendland, H. (1995). Compactly supported radial functions. *Adv. Comput. Math.* **4**.
- Müller, M., Charypar, D. & Gross, M. (2003). Particle-based fluid simulation
  for interactive applications. *SCA* — poly6 / spiky kernels.
- Becker, M. & Teschner, M. (2007). Weakly compressible SPH. *SCA* — Tait EOS,
  `c₀ ≥ 10 v_max` stiffness rule.
- Akinci, N. et al. (2012). Versatile rigid–fluid coupling for incompressible
  SPH. *ACM TOG* **31** — boundary volume `ψ` (boundary particles only here).
- Macklin, M. & Müller, M. (2013). Position Based Fluids. *ACM TOG* **32** —
  λ density constraints, artificial pressure, vorticity confinement.
- Ritter, A. (1892). Dam-break shallow-water front `x_f = 2√(g h₀) t`.
- Bender, J. & Koschier, D. (2015). Divergence-Free SPH for Incompressible and
  Viscous Fluids. *SCA* / *IEEE TVCG* **23**(3) (2017) — DFSPH constant-density
  + divergence-free solvers, `α` factors, warm start, CFL (clean-room).
- Akinci, N., Akinci, G. & Teschner, M. (2013). Versatile surface tension and
  adhesion for SPH fluids. *ACM TOG* **32** — cohesion + curvature (optional).
- Bonet, J. & Lok, T.-S. L. (1999) / CSPM — Shepard filter and gradient
  correction (optional kernel renormalisation).
- **Not transcribed:** Martin & Moyce (1952) / Koshizuka–Oka (1996) experimental
  dam-break tables (no verified primary table in-tree). **Deferred:**
  fluid→rigid force sampling.

### M8 note (`pe-scene`, `pe-ffi`, Godot GDExtension)

- Scene / FFI / Godot binding code in this repository is **clean-room** (no
  third-party simulation source copied).
- Cargo dependencies used as packages (not vendored into this tree): `serde`,
  `serde_json`, `ron` (Apache-2.0 / MIT), `cbindgen` (tooling, MPL-2.0).
- Godot bindings: [`godot`](https://crates.io/crates/godot) (**godot-rust /
  gdext**, **MPL-2.0**). Used as a Cargo dependency from `godot/`; **no** gdext
  source is copied into this tree. Godot Engine itself is MIT
  (<https://godotengine.org/license/>). Editor/headless runs are **unverified**
  in CI — see [`docs/GODOT.md`](docs/GODOT.md).
