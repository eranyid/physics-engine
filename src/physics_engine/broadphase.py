"""AABB broad phase: sweep-and-prune on x, plus an O(N^2) brute-force reference."""
from __future__ import annotations
import numpy as np


def _overlap(lo, hi, i, j) -> bool:
    return bool(np.all(lo[i] <= hi[j]) and np.all(lo[j] <= hi[i]))


def brute_force_pairs(lo: np.ndarray, hi: np.ndarray, skip=None) -> set[tuple[int, int]]:
    n = len(lo)
    return {(i, j) for i in range(n) for j in range(i + 1, n)
            if not (skip is not None and skip[i] and skip[j]) and _overlap(lo, hi, i, j)}


def sweep_and_prune(lo: np.ndarray, hi: np.ndarray, skip=None) -> set[tuple[int, int]]:
    """Pairs (i<j) whose closed AABBs overlap. `skip[i] and skip[j]` drops the pair (static-static)."""
    order = np.argsort(lo[:, 0], kind="stable")
    active: list[int] = []
    pairs = set()
    for i in order:
        xi = lo[i, 0]
        active = [j for j in active if hi[j, 0] >= xi]
        for j in active:
            if skip is not None and skip[i] and skip[j]:
                continue
            if np.all(lo[i, 1:] <= hi[j, 1:]) and np.all(lo[j, 1:] <= hi[i, 1:]):
                pairs.add((min(i, j), max(i, j)))
        active.append(i)
    return pairs
