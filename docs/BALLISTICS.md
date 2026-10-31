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
| BC (SI) | kg/m² | mass / reference area — **not** lb/in² G1 BC |
| Angles | rad | |
| Spin | rad/s or rev/s | from twist [m/rev] |

## Models

- **AmmoProfile** — versioned JSON/TOML; see [`AMMO.md`](AMMO.md).
- **Drag** — generic Mach–`C_D` table (linear / monotone API) + analytic
  `C_D(M)` stand-in. Tags `G1StandIn` / `G7StandIn` scale that model; they are
  **not** verified BRL G1/G7 tables (see skipped items below).
- **Atmosphere / wind** — ISA `ρ`, `a`; steady + Dryden (`pe-aero`).
- **Earth** — Coriolis, Eötvös, `g(h)=g₀(R/(R+h))²`.
- **Spin** — documented spin-drift and Magnus engineering accelerations.
- **ProjectileSystem** — SoA, stepped in ascending `ShotId` order; per-shot RNG
  seeded from shot id; RK4; swept segment hit-tests (no tunneling).
- **Terminal** — KE/momentum, World impulse, Poncelet depth, Recht–Ipson
  residual velocity (caller material constants).
- **Oracles** — vacuum range/height, linear drag, quadratic terminal speed,
  Coriolis freefall east `(1/3)ω g t³ cos φ`, energy residual.

## Limits

- Point-mass (not 6-DOF aeroballistics).
- G1/G7 standard drag functions **not** shipped from a verified primary table.
- Bundled ammo figures are nominal class ballpark values (`verified: false`).
- Parallel stepping is not enabled (would require bit-identical reduction).

## Validation highlights

- Per-profile muzzle energy `½mv²`.
- Deterministic replay: same `ShotId` matches with 1 or 1000 other bullets.
- No tunneling: 1000 m/s vs 1 cm wall at `dt=1/60`.
- Cross-OS hash: 100-bullet mixed volley with wind (`docs/DETERMINISM.md`).

## Skipped / unverified

- **G1/G7 drag coefficients** — not copied from McCoy / BRL primary tables in
  this milestone; analytic stand-in documented instead.
- Manufacturer-precise load data — intentionally omitted; only nominal class
  profiles.
