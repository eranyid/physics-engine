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
  Runge–Kutta formulas with stepsize control. NASA TR R-287. (RKF78; DOP853
  stand-in — Hairer tableau not reproduced.)
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
- Wisdom & Holman (1991). Symplectic maps for the N-body problem. *AJ* **102**.
- Chenciner–Montgomery / Simó figure-eight initial conditions are published IC values.
