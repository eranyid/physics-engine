# Ballistics module (`pe-ballistics`)

Point-mass external ballistics with **per-bullet** simulation for games and
studio work. Uses `pe-aero` ISA density / speed of sound and Dryden wind,
`pe-core` impulses on impact, and read-only `pe-collision` shapes for hit-scan.

## Units and frames

| Quantity | Unit | Notes |
| --- | --- | --- |
| Position / range | m | ENU: +x east (or downrange), +y north/lateral, +z up |
| Velocity | m/s | |
| Mass | kg | |
| Force / impulse | N / N·s | |
| BC (SI) | kg/m² | mass / reference area; imperial G1/G7 BC [lb/in²] × `703.069580719 × 4/π` |
| Angles | rad | |
| Spin | rad/s or rev/s | from twist [m/rev] |

## Models

- **AmmoProfile** — versioned JSON/TOML; see [`AMMO.md`](AMMO.md). Bundled
  **generic_*** class profiles remain nominal (`verified: false`). Sourced
  cartridge rows set `verified: true` only with a citation for the locked
  numbers.
- **Drag** — published **G1 / G7** standard Cd–Mach tables (JBM / BRL; free for
  public use — see [`THIRD_PARTY.md`](../THIRD_PARTY.md)). Default interpolation
  is **linear**; monotone cubic PCHIP is available as `DragTable::cd_monotone`.
  BC scaling: `a_drag = −½ ρ v² Cd_std(M) / BC_si` (`A_ref = m / BC_si`).
  Legacy serde tags `G1StandIn` / `G7StandIn` deserialise to `G1` / `G7`.
  Analytic `C_D(M)` remains for demos (`DragModelRef::Analytic`).
- **Atmosphere / wind** — ISA `ρ`, `a`; steady + Dryden (`pe-aero`).
- **Earth** — Coriolis, Eötvös, `g(h)=g₀(R/(R+h))²`.
- **Spin** — documented spin-drift and Magnus engineering accelerations.
- **ProjectileSystem** — SoA, stepped in ascending `ShotId` order; per-shot RNG
  seeded from shot id; RK4; swept segment hit-tests (no tunneling).
- **Terminal** — KE/momentum, World impulse, Poncelet depth, Recht–Ipson
  residual velocity (caller material constants).
- **Oracles** — vacuum range/height, linear drag, quadratic terminal speed,
  Coriolis freefall east `(1/3)ω g t³ cos φ`, energy residual; G1 BC 0.5 @
  2800 fps drop/TOF vs py_ballisticcalc (see below).

## Limits

- Point-mass (not 6-DOF aeroballistics).
- Parallel stepping is not enabled (would require bit-identical reduction).

## Validation highlights

- Per-profile muzzle energy `½mv²`.
- Deterministic replay: same `ShotId` matches with 1 or 1000 other bullets.
- No tunneling: 1000 m/s vs 1 cm wall at `dt=1/60`.
- Cross-OS hash: 100-bullet mixed volley with wind (`docs/DETERMINISM.md`).
- **G1 BC 0.5, 2800 fps, ICAO, sight height 0** (linear Cd interp, spin off):
  drop / TOF / velocity locked against **py_ballisticcalc 2.2.0**
  (`DragModel(0.5, TableG1)`, `Atmo.icao()`) measured 2026-11-01:

  | Range (yd) | Drop (in) | TOF (s) | Velocity (fps) |
  | --- | --- | --- | --- |
  | 100 | −2.3199 | 0.110872 | 2615.05 |
  | 200 | −9.7308 | 0.229701 | 2437.52 |
  | 300 | −23.0023 | 0.357326 | 2266.97 |
  | 400 | −43.0505 | 0.494724 | 2103.16 |
  | 500 | −70.9732 | 0.643012 | 1946.36 |

  CI tolerances: drop ≤ max(5% rel, 1.0 in abs); TOF ≤ 3% rel; velocity ≤ 2% rel.

## Skipped / unverified

- Cartridge rows without a primary BC cell keep `bc.provenance.verified: false`
  (e.g. Federal AE9AP omits BC; M855 / .300 Win Mag / .50 BMG / 12 ga slug BC
  not locked to a fetched manufacturer table cell in this pass).
- Siacci / Army Standard Metro atmosphere tables (ICAO ISA used instead).
