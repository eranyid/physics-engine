# Space / orbital module (`pe-space`)

M7 domain module: translational N-body gravity, Kepler / Lambert helpers, and
common orbital perturbations. Clean-room from published papers (see
[`THIRD_PARTY.md`](../THIRD_PARTY.md)); **not** derived from REBOUND, nyx-space,
or JSBSim. DOP853 coefficient tables are attributed to SciPy’s BSD-licensed
transcription of Hairer’s `dop853.f`.

## Units and frames

| Quantity | Unit | Notes |
| --- | --- | --- |
| Position | m | SI only inside the crate |
| Velocity | m/s | |
| Time | s | |
| Mass | kg | or dimensionless when `G=1` demos |
| μ / GM | m³/s² | prefer μ over evaluating `G·M` each step |
| Acceleration | m/s² | |

**Frame.** Inertial, right-handed. Origin is caller-defined (central body or
barycentre). No Earth-fixed / ECEF / sidereal-time transform yet; J₂ uses the
inertial `(x,y,z)` axes as the planet equatorial frame.

Earth defaults used in tests:

- `μ = 398600.4418 km³/s² = 3.986004418×10¹⁴ m³/s²`
- `R = 6378.137 km` (WGS84)
- `J₂ = 1.08262668×10⁻³`

## Models

### Gravity fields (`GravityField`)

- **`CentralMu`** — `a = −μ r / \|r\|³` for a single satellite about a fixed origin.
- **`NBodyPointMass`** — pairwise Newtonian; SI `G` or geometric `G=1`.
- Softening optional; pair loops are ordered `i < j` (deterministic).

### Integrators

| Name | Order | Symplectic | Adaptive | Status |
| --- | --- | --- | --- | --- |
| Leapfrog (velocity Verlet) | 2 | yes (pos.-dep.) | no | shipped |
| Yoshida 4 (Yoshida 1990) | 4 | yes | no | shipped |
| RK4 | 4 | no | no | shipped |
| RKF78 (Fehlberg 7(8)) | 7(8) | no | yes | shipped |
| DOP853 (Hairer 8(5,3)) | 8 | no | yes | shipped (SciPy/Hairer tables) |
| IAS15 (Rein & Spiegel 2015) | 15 | no | yes | shipped |
| WHFast-style Wisdom–Holman | 2 (+ correctors) | yes | no | shipped (Jacobi + democratic-heliocentric) |

**DOP853.** Exact Dormand–Prince 8(5,3) tableau from SciPy’s BSD-licensed
`dop853_coefficients.py` (Hairer–Nørsett–Wanner / `dop853.f`). Embedded 5th/3rd
error estimate for step control; optional dense output (7th-order interpolant).
Butcher checks: row sums `Σⱼ Aᵢⱼ = cᵢ` and `Σ Bᵢ = 1` to `<1e-14`.

**Measured oracles (Linux x86_64):**

| Check | Tolerance | Measured |
| --- | --- | --- |
| Butcher residual | 1e-14 | under |
| Convergence slope on `y'=y`, coarse `h∈{1,½,¼}` | 7–9.5 | **7.86** (ratios ~208 / 232) |
| Convergence slope on `y'=−½y³`, `h∈{½,¼,⅛}` | 7.5–9 | **7.82** (ratios ~156 / 226) |
| Kepler `e=0.5` one period `\|ΔE/E\|`, `\|ΔL\|/\|L\|` | 1e-10 | **~3.0e-12**, **~7.2e-13** |
| Arenstorf period closure | 1e-6 | **~8.3e-7** |

Slopes **~7.8–7.9** are a step/tolerance-regime effect (not a wrong tableau):
in the accessible fixed-step window before the `f64` floor, observed order sits
just below the pure asymptotic `h⁸` plateau (ideal ratio 256). Finer `y'=y`
steps hit round-off (~1e-15) before the ratio can climb further.

**IAS15.** Gauss–Radau nodes are derived from `P_8(x)+P_7(x)=0` on `[-1,1]`
(mapped to `[0,1]`), verified to `1e-15` against the locked table. Adaptive
control uses `ε_b` (default `1e-9`) and Neumaier-compensated position updates.

**WHFast.** Second-order DKD Wisdom–Holman map with universal-variable Kepler
drift (`propagate_universal` / Stumpff). Jacobi (Rein & Tamayo 2015) or
democratic-heliocentric (Duncan et al. 1998). **Symplectic correctors** of
orders 3/5/7/11 (Wisdom 2006 `Z=X∘X` operators; `α=√(7/40)`, `β=1/(48α)`).

**Measured corrector oracles** (synthetic 5-body, `ε~10⁻³`, `G=m₀=1`):

| Check | Tolerance | Measured |
| --- | --- | --- |
| Energy `\|ΔE/E\|` plain DKD vs order-11, `dt=1`, `t=500` | c11 < 0.5×plain | plain **3.5e-5**, c11 **4.6e-6** |
| Position err vs fine ref, plain vs order-5, `dt=0.5`, `t=20` | c5 < 0.8×plain | plain **5.0e-4**, c5 **1.7e-5** |
| Order-5 energy scaling `err(dt)/err(dt/2)` | > 2.5 (~4 for dt²) | **~4.95** |

Compensated (Neumaier) position updates and COM origin rebasing are optional on
[`NBodyState`](../crates/pe-space/src/state.rs).

### Kepler / Lambert

- Classical elements ↔ state (Vallado / Battin conventions).
- Universal-variable propagation with Stumpff `C,S` and Danby/Newton χ iteration.
- Lambert: 0-revolution Izzo (2015) geometry + Vallado-style ψ iteration.
  Multi-rev branches return `NoSolution` for now.

### Perturbations

- **J₂** first zonal (full acceleration including central term).
- **Cannonball SRP** (no eclipse).
- **Point-mass third body** (primary-centred form).
- **Drag:** exponential atmosphere; uses `pe-aero` ISA density for
  `0 ≤ h ≤ 20 km`, else the exponential model.

## Limits

- Translational DOF only (no attitude / SRP torque).
- No SPICE ephemerides, no Earth rotation, no higher geopotential (J₃…).
- WHFast correctors are Jacobi-oriented (helio path ignores corrector flag).
- ISA drag is invalid above 20 km (falls back to exponential).
- Determinism: `f64`, portable `libm`, no FMA, no `HashMap` iteration
  ([`DETERMINISM.md`](DETERMINISM.md)).

## Validation highlights

See `crates/pe-space/tests/space_analytic.rs` and Python `tests/test_space_rust.py`:

- LEO `r=7000 km`: `T≈5828.5 s`, `v≈7.5460 km/s`
- GEO `r=42164.17 km`: `T≈86164.09 s`
- `e=0.9` Yoshida4 over 1000 orbits: relative energy error `<1e-10` (per-orbit samples)
- `e=0.9` IAS15 over 1000 orbits: relative energy error `<1e-12` (machine-precision regime)
- J₂ nodal rate vs `−1.5 n J₂ (R/p)² cos i` (`a=6778 km`, `i=51.6°`):
  **~0.43 %** using ascending-node orbit averages over ~400 orbits (under 0.5 %).
- Sun-synchronous `+0.9856 deg/day` at ~800 km → `i≈98.6°`
- Chenciner–Montgomery figure-eight: IAS15 return `<1e-8` over one refined period
- Pythagorean 3-body (masses 3,4,5): IAS15 energy relative drift `<1e-10` through
  close encounters (`t=10`)
- WHFast synthetic 5-body: energy bounded over `1e6` time units; order-2
  convergence; reversibility; symplectic correctors as above
- DOP853: order-8 slope, Kepler drift, Arenstorf closure as above
