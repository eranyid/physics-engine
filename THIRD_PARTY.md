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

## Algorithmic citations (clean-room; not derived source)

The following published descriptions informed clean-room reimplementations in
`crates/pe-collision` (M5a). **No Box2D / Solver2D / Bepu source was copied.**

| Topic | Citation | Where used |
| --- | --- | --- |
| Soft constraints (Hertz / ζ → bias, mass, impulse scales) | Erin Catto, [Solver2D](https://box2d.org/posts/2024/02/solver2d/) (2024-02-05); GDC 2011 “Soft Constraints”; mass-independent form attributed there to Ross Nordby (Bepu) | `softness.rs`, `soft_step.rs` |
| Soft Step / TGS_Soft pipeline (sub-steps, warm start, relax, restitution) | Erin Catto, Solver2D post + Box2D v3 release notes | `soft_step.rs`, `contact.rs` |
| Dynamic AABB tree (SAH insert, fat AABBs, rotations) | Published Box2D dynamic-tree / BVH descriptions (Erin Catto) | `aabb_tree.rs` |
| Island sleeping thresholds | Box2D-style defaults adapted to SI (documented in `islands.rs`) | `islands.rs` |

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
