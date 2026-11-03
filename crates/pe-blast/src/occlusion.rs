//! Simple ray / AABB occlusion for blast shadowing.
//!
//! Engineering model: a face is shadowed if the segment from the charge origin
//! to the face centre first intersects another occluder AABB (excluding the
//! target body). AABB tests are conservative (may over-shadow). Diffraction,
//! clearing, and multiple reflections are **not** modelled.
//!
//! Occluders are walked in ascending `body` index order (deterministic; no
//! `HashMap`).

use pe_collision::Aabb;
use pe_math::libm_ops::fabs;
use pe_math::Vec3;

/// Occluder AABB paired with a body index (for self-exclusion).
#[derive(Clone, Copy, Debug)]
pub struct OccluderAabb {
    /// Body index in the world / contact set.
    pub body: usize,
    /// World-space AABB [m].
    pub aabb: Aabb,
}

/// Ray–AABB intersection: earliest `t ≥ tmin` for `origin + t·dir` with
/// `t ≤ tmax`. `dir` need not be unit. Returns `None` if no hit in range.
pub fn ray_aabb(origin: Vec3, dir: Vec3, aabb: Aabb, tmin: f64, tmax: f64) -> Option<f64> {
    let mut t0 = tmin;
    let mut t1 = tmax;
    for (o, d, lo, hi) in [
        (origin.x, dir.x, aabb.lo.x, aabb.hi.x),
        (origin.y, dir.y, aabb.lo.y, aabb.hi.y),
        (origin.z, dir.z, aabb.lo.z, aabb.hi.z),
    ] {
        if fabs(d) < 1e-15 {
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d;
        let mut ta = (lo - o) * inv;
        let mut tb = (hi - o) * inv;
        if ta > tb {
            core::mem::swap(&mut ta, &mut tb);
        }
        t0 = t0.max(ta);
        t1 = t1.min(tb);
        if t0 > t1 {
            return None;
        }
    }
    Some(t0)
}

/// `true` if the open segment `(origin → target)` is blocked by any occluder
/// other than `exclude_body`.
///
/// Parameter `t` is in `[0,1]` along `target − origin`. Hits with
/// `t ≤ 1e-6` (at the origin) or `t ≥ 1 − 1e-6` (at the target face) are
/// ignored so the target's own AABB does not self-shadow when it contains the
/// face centre.
pub fn is_shadowed(
    origin: Vec3,
    target: Vec3,
    exclude_body: usize,
    occluders: &[OccluderAabb],
) -> bool {
    let dir = target - origin;
    let len_sq = dir.norm_squared();
    if len_sq < 1e-30 {
        return false;
    }
    let mut best_t = 1.0;
    let mut hit = false;
    // Ascending body index — caller should pre-sort; we still scan in given order
    // for bit-stability of the boolean (any-hit, not nearest-hit selection).
    for o in occluders {
        if o.body == exclude_body {
            continue;
        }
        if let Some(t) = ray_aabb(origin, dir, o.aabb, 0.0, 1.0) {
            if t > 1e-6 && t < 1.0 - 1e-6 && t < best_t {
                best_t = t;
                hit = true;
            }
        }
    }
    hit
}

/// Sort occluders by ascending body index (deterministic prep for callers).
pub fn sort_occluders(occluders: &mut [OccluderAabb]) {
    occluders.sort_by_key(|a| a.body);
}

#[cfg(test)]
mod tests {
    use super::*;
    use pe_math::Vec3;

    #[test]
    fn empty_never_shadowed() {
        assert!(!is_shadowed(Vec3::zero(), Vec3::new(1.0, 0.0, 0.0), 0, &[]));
    }

    #[test]
    fn box_between_shadows() {
        let occ = [OccluderAabb {
            body: 1,
            aabb: Aabb {
                lo: Vec3::new(0.4, -0.5, -0.5),
                hi: Vec3::new(0.6, 0.5, 0.5),
            },
        }];
        assert!(is_shadowed(Vec3::zero(), Vec3::new(1.0, 0.0, 0.0), 0, &occ));
    }

    #[test]
    fn exclude_self() {
        let occ = [OccluderAabb {
            body: 0,
            aabb: Aabb {
                lo: Vec3::new(0.4, -0.5, -0.5),
                hi: Vec3::new(0.6, 0.5, 0.5),
            },
        }];
        assert!(!is_shadowed(
            Vec3::zero(),
            Vec3::new(1.0, 0.0, 0.0),
            0,
            &occ
        ));
    }
}
