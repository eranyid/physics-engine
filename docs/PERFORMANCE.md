# Performance (M26)

Game-oriented throughput work with **byte-identical default-feature determinism**.
Locked cross-OS hashes must not change. Explicit SIMD / FMA stays off on the
oracle path (reassociation changes `f64` results). Optional island parallelism
remains behind `pe-collision/parallel` (off by default).

## How to measure

```bash
# Fast CI microbench (JSON medians, ms)
cargo run -p pe-examples --bin ci_bench --release > /tmp/ci_bench.json
python3 scripts/check_ci_bench.py --measured /tmp/ci_bench.json

# Criterion (local profiling; not in default test CI)
cargo bench -p pe-collision --bench scaling
cargo bench -p pe-collision --bench contact_pyramids
cargo bench -p pe-soft --bench soft
cargo bench -p pe-fluid --bench fluid

# Phase timers
cargo run -p pe-collision --example perf_profile --release
```

Baseline file: [`docs/baselines/ci_bench.json`](baselines/ci_bench.json).

## CI regression policy

Job `perf-bench` runs `ci_bench` in release with
`RUSTFLAGS=-C target-cpu=generic` and fails if **any** metric exceeds

\[
\text{measured} > \text{baseline} \times (1 + \text{margin})
\]

with default **margin = 1.0 (100%)** — i.e. fail on a **2×** slowdown. Shared
GitHub runners are noisy; this catches large regressions, not micro-jitter.
Re-record the baseline on this VM class after intentional speedups (do not
silently loosen margins to greenwash).

`perf-smoke` still checks locked hashes (+ optional parallel island bit-identity)
without wall-time asserts.

## Optimizations in M26 (default features)

Bit-identical path only:

| Area | Change |
| --- | --- |
| Broad phase | Write-into SAP / brute / dynamic-tree pair buffers; tree query stack reuse |
| Detect / step | No pair/contact clones on the hot path; active-contact filter into scratch |
| Islands / sleep | Union-find + sleep speed buffers reused across steps |
| Soft-step | `SoftStepScratch` reuses im/iinv/poses/rows/joint rows across steps |
| Soft self-collision | Persistent spatial-hash buckets/pairs on `SoftWorld` |
| DFSPH | Reused `dv_bound` + boundary-reaction weight scratch |

Not shipped in default features: FMA/`mul_add`, geometric manifold skip,
unordered maps, or any change that would move a locked hash.

## Measured on this Cloud Agent VM (Linux x86_64)

Toolchain: `rust-toolchain.toml` (1.99.1), release, `target-cpu=generic`.
Numbers below are **wall-clock** from `perf_profile` / `ci_bench` on this run —
not invented.

### Rigid pipeline (`perf_profile`, steps only)

| Workload | Before (main) | After (M26) | Ratio |
| --- | --- | --- | --- |
| pyramid 10 rows × 60 PGS | 29.213 ms | 28.702 ms | ~1.02× |
| pyramid 20 rows × 60 PGS | 131.285 ms | 129.977 ms | ~1.01× |
| pyramid 40 rows × 60 PGS | 609.339 ms | 606.913 ms | ~1.00× |
| falling 1k boxes, tree × 20 | 9.357 ms | 9.242 ms | ~1.01× |

Earlier rigid wins (scratch SoA PGS, incremental AABB tree, ~1.2–2× on pyramids /
1k falling) shipped in PR #26 and remain the larger historical step; M26 focuses
on remaining alloc churn + CI regression gates.

### `ci_bench` baseline (this run)

| Metric | Median [ms] |
| --- | --- |
| `pyramid_10_pgs_60` | 30.34 |
| `pyramid_10_softstep_60` | 33.81 |
| `pyramid_10_pgs_sleep_240` | 98.54 |
| `falling_100_tree_20` | 0.92 |
| `falling_1000_tree_20` | 9.57 |
| `falling_10000_tree_10` | 60.93 |
| `ragdoll_16_120` | 2.34 |
| `gjk_sphere_sphere_1k` | 0.15 |
| `epa_box_box_1k` | 5.90 |
| `soft_rope_hang_1s` | 0.66 |
| `soft_cloth_8x8_selfcol_60` | 26.65 |
| `pbf_512_10_steps` | 86.64 |
| `dfsph_216_10_steps` | 226.57 |

### Hashes that would have changed

None under default features. All locked CI hashes stayed byte-identical
(collision, soft-step, joints, convex, soft, fluid, DFSPH, …). Optimizations
that would have moved a hash were not shipped.

## Related docs

- Criterion history / broader suite: [`BENCHMARKS.md`](BENCHMARKS.md)
- Determinism rules: [`DETERMINISM.md`](DETERMINISM.md)
- Stability long-runs: [`STABILITY.md`](STABILITY.md)
