# Long-run stability suite (M22)

Opt-in / scheduled checks for secular drift on stacks, joints, orbits, and
aircraft trim. Default CI stays fast (`cargo test` skips `#[ignore]`).

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

## Notes

- Sleep is enabled on stack/pyramid long runs (production-like); rolling and
  joint-chain keep sleep off so motion continues.
- Orbital bound is relative energy for **Yoshida4** only (RK4 secular drift is
  expected and covered elsewhere).
- Aircraft trim uses a **force-balance polar** (`C_m0 = C_mα = C_mq = 0`), matching
  the Python slow oracle — open-loop light-aircraft pitch derivatives are not
  held for 60 s without a controller (documented, not a silent loosen).
