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

_None yet. No third-party simulation source has been vendored or derived into
this tree._

### M4 note

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
  Runge–Kutta formulas with stepsize control. NASA TR R-287. (RKF78; DOP853-class
  adaptive role.)
- Izzo, D. (2015). Revisiting Lambert's problem. *Celest. Mech. Dyn. Astr.* **121**.
- Battin / Danby / Vallado textbook identities for Stumpff functions and universal
  Kepler variables.
- Rein & Spiegel (2015) IAS15 and Wisdom & Holman (1991) WHFast are **cited as
  future work only** (stubs); no code from those papers' reference implementations
  was used.
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

### Blast / ballistics note (`pe-blast`, `pe-ballistics`)

Clean-room formulas only — **no** third-party simulation source translated:

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
- **Not used:** G1/G7 standard drag coefficient tables (no verified primary BRL
  transcription in this milestone — analytic `C_D(M)` stand-in instead).
