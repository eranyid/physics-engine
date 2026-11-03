#!/usr/bin/env python3
"""Compare ci_bench JSON medians against a committed baseline.

Noise margin: FAIL if any named metric exceeds baseline * (1 + margin).
Default margin is 1.0 (100% / 2× absolute) — shared GitHub runners are noisy;
this catches large regressions, not micro-jitter. Documented in docs/PERFORMANCE.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        type=Path,
        default=Path("docs/baselines/ci_bench.json"),
        help="Committed baseline JSON",
    )
    ap.add_argument(
        "--measured",
        type=Path,
        required=True,
        help="Measured JSON from pe-examples ci_bench",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Allowed relative regression (1.0 = 100%% / 2×)",
    )
    args = ap.parse_args()

    baseline = json.loads(args.baseline.read_text())
    measured = json.loads(args.measured.read_text())

    skip = {"host", "unit", "notes", "recorded_at"}
    base_keys = {k for k in baseline if k not in skip}
    meas_keys = {k for k in measured if k not in skip}
    missing = sorted(base_keys - meas_keys)
    if missing:
        print(f"::error::Measured run missing keys: {missing}", file=sys.stderr)
        return 1

    failed = []
    print("ci_bench regression check (margin={:.0%}):".format(args.margin))
    print(f"{'metric':38} {'baseline_ms':>12} {'measured_ms':>12} {'ratio':>8} status")
    for key in sorted(base_keys):
        b = float(baseline[key])
        m = float(measured[key])
        if b <= 0.0:
            print(f"::error::Baseline {key} is non-positive", file=sys.stderr)
            return 1
        ratio = m / b
        limit = 1.0 + args.margin
        ok = ratio <= limit
        status = "ok" if ok else "FAIL"
        print(f"{key:38} {b:12.3f} {m:12.3f} {ratio:8.3f} {status}")
        if not ok:
            failed.append((key, b, m, ratio, limit))

    if failed:
        print("::error::Performance regression(s) exceeded margin:", file=sys.stderr)
        for key, b, m, ratio, limit in failed:
            print(
                f"  {key}: measured {m:.3f} ms > baseline {b:.3f} ms × {limit:.2f} "
                f"(ratio {ratio:.3f})",
                file=sys.stderr,
            )
        return 1

    print("All ci_bench metrics within noise margin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
