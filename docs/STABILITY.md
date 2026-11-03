# Long-run stability suite (M22) + blast/hydro hardening (M27)

Opt-in / scheduled checks for secular drift on stacks, joints, orbits, and
aircraft trim. Default CI stays fast (`cargo test` skips `#[ignore]`).

M27 extends finite-input validation (`try_*` / typed errors) to `pe-blast` and
`pe-hydro`, with always-on proptests (no panics, finite outputs, monotonicity
where physics demands it). Holtrop–Mennen 1982 is verified against the published
ISP §5 numerical example (see oracle table below).

## How to run

```bash
# All ignored stability tests in the workspace (excluding pe-godot)
cargo test --workspace --exclude pe-godot --exclude pe-examples -- --ignored stability_ --nocapture

# Or per crate:
cargo test -p pe-collision --test stability --release -- --ignored --nocapture
cargo test -p pe-space --test stability --release -- --ignored --nocapture
cargo test -p pe-aero --test stability --release -- --ignored --nocapture
```

Larger property suites (same tests, more cases):

```bash
PROPTEST_CASES=256 cargo test -p pe-core --test properties --release
PROPTEST_CASES=256 cargo test -p pe-collision --test properties --release
PROPTEST_CASES=256 cargo test -p pe-scene --test roundtrip --release
PROPTEST_CASES=256 cargo test -p pe-blast --test properties --release
PROPTEST_CASES=256 cargo test -p pe-hydro --test properties --release
```

CI: job `rust-stability` runs on Monday cron and `workflow_dispatch`
(see `.github/workflows/ci.yml`).

## Measured drifts vs bounds

Recorded on the Cloud Agent host (Linux x86_64, `cargo test --release`,
Rust toolchain from `rust-toolchain.toml`). Re-measure after solver / integrator
changes; **do not loosen bounds silently**.

| Scenario | Steps / duration | Metric | Bound | Measured (M22) | Status |
| --- | --- | --- | --- | --- | --- |
| 5-box inelastic stack | 10⁵ × 1/240 s | max \|xy\| COM drift | `< 5 mm` | **0.372 mm** | pass |
| 5-box inelastic stack | 10⁵ × 1/240 s | kinetic energy | `< 1 J` | **0** (slept) | pass |
| 10-box pyramid | 10⁵ × 1/240 s | kinetic energy | `< 2 J` | **0** (slept) | pass |
| Rolling sphere μ=0.6 | 10⁵ × 1/240 s | \|z − r\| | `< 5 cm` | **2.2e-5 m** | pass |
| 6-link distance chain | 10⁵ × 1/240 s | tip \|r\| | `< 10 m` | **~0.25 m** | pass |
| Yoshida4 Kepler e=0.5 | 10⁴ orbits × 4096 steps | max \|ΔE/E₀\| | `< 1e-9` | **4.81e-12** | pass |
| Level-flight trim hold | 5 s × dt=1e-3 RK4 (Cm=0 polar) | \|Δz\| | `< 0.05 m` | **0** | pass |
| Level-flight trim hold | 5 s × dt=1e-3 RK4 (Cm=0 polar) | \|Δ\|v\|\|/\|v₀\| | `< 1e-3` | **0** | pass |

Fast invariant counterparts (always on in default CI) live in
`pe-core/tests/invariants.rs` and `pe-collision/tests/properties.rs`
(NaN/Inf escape, momentum, short inelastic KE bound, snapshot restore).
M27: `pe-blast/tests/properties.rs`, `pe-hydro/tests/properties.rs`.

## Holtrop–Mennen 1982 §5 oracle (M27)

Source: Holtrop & Mennen (1982), *Int. Shipbuild. Prog.* **29**(335), 166–170,
Section 5 numerical example (hypothetical single-screw ship, V = 25 knots).
Measured on Linux x86_64 release build; **do not loosen**.

| Quantity | Published | Measured | Abs error | Rel error | Bound |
| --- | ---: | ---: | ---: | ---: | --- |
| Fn | 0.2868 | 0.286841 | 4.1e-5 | 0.014% | `< 5e-4` |
| C_F | 0.001390 | 0.001390 | ~1e-8 | 0.001% | `< 5e-6` |
| S [m²] | 7381.45 | 7381.449 | 9.3e-4 | ~0% | `< 1` |
| 1+k₁ | 1.156 | 1.156444 | 4.4e-4 | 0.038% | `< 2e-3` |
| L_R [m] | 81.385 | 81.3849 | 1.3e-4 | 0.0002% | `< 0.05` |
| i_E [deg] | 12.08 | 12.0775 | 2.5e-3 | 0.021% | `< 0.15` |
| R_F [kN] | 869.63 | 869.787 | 0.157 | 0.018% | `< 2` |
| R_APP [kN] | 8.83 | 8.838 | 0.008 | 0.086% | `< 0.2` |
| R_W [kN] | 557.11 | 556.953 | 0.157 | 0.028% | `< 5` |
| R_B [kN] | 0.049 | 0.0492 | 1.9e-4 | 0.38% | `< 0.05` |
| R_TR [kN] | 0.00 | 0.00 | 0 | — | `< 0.05` |
| C_A | 0.000352 | 0.000352 | ~5e-7 | 0.14% | `< 5e-6` |
| R_A [kN] | 221.98 | 220.572 | 1.41 | 0.63% | `< 2` |
| R_T [kN] | 1793.26 | 1792.271 | 0.99 | 0.055% | `< 8` |
| P_E [kW] | 23063 | 23050.6 | 12.4 | 0.054% | `< 120` |

## Notes

- Sleep is enabled on stack/pyramid long runs (production-like); rolling and
  joint-chain keep sleep off so motion continues.
- Orbital bound is relative energy for **Yoshida4** only (RK4 secular drift is
  expected and covered elsewhere).
- Aircraft trim uses a **force-balance polar** (`C_m0 = C_mα = C_mq = 0`), matching
  the Python slow oracle — open-loop light-aircraft pitch derivatives are not
  held for 60 s without a controller (documented, not a silent loosen).
- Holtrop path is the **1982** decomposition (matches §5). The 1984 form-factor
  helper is exposed separately and is not the §5 oracle path.
