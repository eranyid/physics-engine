# Space / orbital module (`pe-space`)

M7 domain module: translational N-body gravity, Kepler / Lambert helpers, and
common orbital perturbations. Clean-room from published papers (see
[`THIRD_PARTY.md`](../THIRD_PARTY.md)); **not** derived from REBOUND, nyx-space,
or JSBSim.

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
| RKF78 (Fehlberg 7(8)) | 7(8) | no | yes | shipped (**DOP853 stand-in**) |
| IAS15 (Rein & Spiegel 2015) | 15 | no | yes | shipped |
| WHFast-style Wisdom–Holman | 2 | yes | no | shipped (Jacobi + democratic-heliocentric) |

**DOP853 deferred.** The Hairer–Nørsett–Wanner DOP853 tableau was **not**
reproduced and order-checked in this tree; `Dop853` remains a documented type
alias to RKF78. Revisit only with an exact published tableau + numerical order
tests.

**IAS15.** Gauss–Radau nodes are derived from `P_8(x)+P_7(x)=0` on `[-1,1]`
(mapped to `[0,1]`), verified to `1e-15` against the locked table. Adaptive
control uses `ε_b` (default `1e-9`) and Neumaier-compensated position updates.

**WHFast.** Second-order DKD Wisdom–Holman map with universal-variable Kepler
drift (`propagate_universal` / Stumpff). Jacobi (Rein & Tamayo 2015) or
democratic-heliocentric (Duncan et al. 1998). Symplectic correctors deferred.

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
- DOP853 tableau not shipped (RKF78 stand-in); WHFast symplectic correctors deferred.
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
  Osculating-only end-point Ω (previous M7) was ~5 % because of J₂ short-period
  terms; mean/orbit-averaged sampling removes most of that.
- Sun-synchronous `+0.9856 deg/day` at ~800 km → `i≈98.6°`
- Chenciner–Montgomery figure-eight: IAS15 return `<1e-8` over one refined period
  (`FIGURE_EIGHT_PERIOD_RETURN`; 8-digit IC + published `T` alone leaves ~4e-8)
- Pythagorean 3-body (masses 3,4,5): IAS15 energy relative drift `<1e-10` through
  close encounters (`t=10`)
- WHFast synthetic 5-body: energy bounded over `1e6` time units; order-2
  convergence; reversibility
