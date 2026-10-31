//! AABB broad phase: sweep-and-prune on x, O(N²) brute-force, and (via
//! [`crate::aabb_tree::DynamicAabbTree`]) a dynamic AABB tree.
//!
//! Pair order is always sorted by `(min_id, max_id)` for determinism. The
//! [`BroadPhase`] trait lets backends swap without changing the contact pipeline.

use pe_math::Vec3;

use crate::shapes::Aabb;

/// Broad-phase interface. Implementations must return pairs sorted by `(i, j)`
/// with `i < j`, never iterating a hash map.
pub trait BroadPhase {
    /// Overlapping AABB pairs. When `skip_static[i] && skip_static[j]`, the pair
    /// is dropped (static–static).
    fn pairs(&self, aabbs: &[Aabb], skip_static: &[bool]) -> Vec<(usize, usize)>;
}

/// Sweep-and-prune along world x (default production broad phase).
#[derive(Clone, Copy, Debug, Default)]
pub struct SweepAndPrune;

impl BroadPhase for SweepAndPrune {
    fn pairs(&self, aabbs: &[Aabb], skip_static: &[bool]) -> Vec<(usize, usize)> {
        sweep_and_prune(aabbs, Some(skip_static))
    }
}

/// Brute-force O(N²) reference used in tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct BruteForce;

impl BroadPhase for BruteForce {
    fn pairs(&self, aabbs: &[Aabb], skip_static: &[bool]) -> Vec<(usize, usize)> {
        brute_force_pairs(aabbs, Some(skip_static))
    }
}

#[inline]
fn aabb_overlap(a: &Aabb, b: &Aabb) -> bool {
    a.lo.x <= b.hi.x
        && a.lo.y <= b.hi.y
        && a.lo.z <= b.hi.z
        && b.lo.x <= a.hi.x
        && b.lo.y <= a.hi.y
        && b.lo.z <= a.hi.z
}

#[inline]
fn yz_overlap(a: &Aabb, b: &Aabb) -> bool {
    a.lo.y <= b.hi.y && a.lo.z <= b.hi.z && b.lo.y <= a.hi.y && b.lo.z <= a.hi.z
}

/// All overlapping pairs via O(N²). Returns sorted `(i, j)` with `i < j`.
pub fn brute_force_pairs(aabbs: &[Aabb], skip_static: Option<&[bool]>) -> Vec<(usize, usize)> {
    let n = aabbs.len();
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if let Some(skip) = skip_static {
                if skip[i] && skip[j] {
                    continue;
                }
            }
            if aabb_overlap(&aabbs[i], &aabbs[j]) {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

/// Sweep-and-prune on x. Returns sorted `(i, j)` with `i < j` (same set as brute force).
///
/// Uses a **stable** sort of lower bounds (matches NumPy `argsort(..., kind="stable")`).
pub fn sweep_and_prune(aabbs: &[Aabb], skip_static: Option<&[bool]>) -> Vec<(usize, usize)> {
    let n = aabbs.len();
    let mut order: Vec<usize> = (0..n).collect();
    // Stable sort by lo.x
    order.sort_by(|&i, &j| {
        aabbs[i]
            .lo
            .x
            .partial_cmp(&aabbs[j].lo.x)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let mut active: Vec<usize> = Vec::new();
    let mut pairs = Vec::new();
    for &i in &order {
        let xi = aabbs[i].lo.x;
        active.retain(|&j| aabbs[j].hi.x >= xi);
        for &j in &active {
            if let Some(skip) = skip_static {
                if skip[i] && skip[j] {
                    continue;
                }
            }
            if yz_overlap(&aabbs[i], &aabbs[j]) {
                let a = i.min(j);
                let b = i.max(j);
                pairs.push((a, b));
            }
        }
        active.push(i);
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Convenience: build AABBs from parallel lo/hi arrays (tests / Python parity helpers).
pub fn aabbs_from_lo_hi(lo: &[Vec3], hi: &[Vec3]) -> Vec<Aabb> {
    lo.iter()
        .zip(hi.iter())
        .map(|(&l, &h)| Aabb { lo: l, hi: h })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pe_math::Vec3;

    #[test]
    fn sap_equals_brute_on_random_scenes() {
        for seed in 0..10u64 {
            // Fresh LCG per seed (avoid borrowing `state` across reassignment).
            let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let mut next = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                state
            };
            let uni = |next: &mut dyn FnMut() -> u64, a: f64, b: f64| -> f64 {
                let u = (next() >> 11) as f64 / ((1u64 << 53) as f64);
                a + (b - a) * u
            };
            let n = 200;
            let mut aabbs = Vec::with_capacity(n);
            let mut skip = vec![false; n];
            for i in 0..n {
                let cx = uni(&mut next, 0.0, 10.0);
                let cy = uni(&mut next, 0.0, 10.0);
                let cz = uni(&mut next, 0.0, 10.0);
                let ex = uni(&mut next, 0.05, 0.8);
                let ey = uni(&mut next, 0.05, 0.8);
                let ez = uni(&mut next, 0.05, 0.8);
                aabbs.push(Aabb {
                    lo: Vec3::new(cx - ex, cy - ey, cz - ez),
                    hi: Vec3::new(cx + ex, cy + ey, cz + ez),
                });
                skip[i] = uni(&mut next, 0.0, 1.0) < 0.2;
            }
            // Exact touching edge cases on x
            for i in 0..5 {
                aabbs[i].lo.x = aabbs[i + 5].hi.x;
                if aabbs[i].hi.x < aabbs[i].lo.x {
                    aabbs[i].hi.x = aabbs[i].lo.x;
                }
            }
            let expect = brute_force_pairs(&aabbs, Some(&skip));
            assert!(expect.len() > 50, "seed {seed}: too few pairs");
            assert_eq!(sweep_and_prune(&aabbs, Some(&skip)), expect, "seed {seed}");
        }
    }
}
