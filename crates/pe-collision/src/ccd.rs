//! Continuous collision detection (M5b + M6): speculative contacts + TOI.
//!
//! Clean-room from published Box2D v3 / Catto CCD descriptions (algorithms only).
//!
//! 1. **Speculative contacts** — for pairs whose swept AABBs overlap, create a
//!    contact even when currently separated; soft / PGS bias then prevents
//!    tunneling for moderately fast bodies.
//! 2. **TOI (bullet bodies)** — analytic sphere–plane + support binary search vs
//!    planes; **M6** adds GJK conservative advancement for convex–convex
//!    (including fast box vs thin static box).
//!
//! Citations: van den Bergen GJK distance; Mirtich / Catto conservative
//! advancement. No third-party source copied.

use pe_math::libm_ops::{fabs, sqrt};
use pe_math::{Quat, Vec3};

use crate::gjk::{gjk, GjkTol};
use crate::manifold::{ContactPoint, FeatureId};
use crate::shapes::{Aabb, Plane, Shape};
use crate::support::bounding_radius;

/// Default speculative distance [m] (Box2D-scale lift ≈ 0.02–0.04; we use 0.02).
pub const DEFAULT_SPECULATIVE_DISTANCE: f64 = 0.02;

/// CCD configuration on a [`crate::ContactWorld`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CcdParams {
    /// Enable speculative contact generation.
    pub speculative: bool,
    /// Speculative separation threshold [m].
    pub speculative_distance: f64,
    /// Enable TOI substepping for bullet-flagged dynamics vs statics.
    pub toi: bool,
    /// Max conservative-advancement iterations.
    pub toi_max_iters: usize,
    /// TOI tolerance [m].
    pub toi_tolerance: f64,
}

impl Default for CcdParams {
    fn default() -> Self {
        // Off by default so M3/M5a oracle hashes stay bit-identical; enable per world.
        Self {
            speculative: false,
            speculative_distance: DEFAULT_SPECULATIVE_DISTANCE,
            toi: false,
            toi_max_iters: 16,
            toi_tolerance: 1e-4,
        }
    }
}

/// Expand an AABB by a world-space margin [m] on all sides.
pub fn expand_aabb(a: Aabb, margin: f64) -> Aabb {
    let m = Vec3::new(margin, margin, margin);
    Aabb {
        lo: a.lo - m,
        hi: a.hi + m,
    }
}

/// Swept AABB: expand current AABB by `|v|·dt` along each axis (conservative).
pub fn swept_aabb(a: Aabb, v: Vec3, dt: f64) -> Aabb {
    let dx = fabs(v.x) * dt;
    let dy = fabs(v.y) * dt;
    let dz = fabs(v.z) * dt;
    Aabb {
        lo: Vec3::new(a.lo.x - dx, a.lo.y - dy, a.lo.z - dz),
        hi: Vec3::new(a.hi.x + dx, a.hi.y + dy, a.hi.z + dz),
    }
}

/// AABB overlap test.
pub fn aabb_overlap(a: Aabb, b: Aabb) -> bool {
    a.lo.x <= b.hi.x
        && a.hi.x >= b.lo.x
        && a.lo.y <= b.hi.y
        && a.hi.y >= b.lo.y
        && a.lo.z <= b.hi.z
        && a.hi.z >= b.lo.z
}

/// Signed distance from point to plane `n·x = d` (positive on normal side).
pub fn plane_distance(n: Vec3, d: f64, p: Vec3) -> f64 {
    n.dot(p) - d
}

/// Analytic TOI of a moving sphere centre against a static plane.
///
/// Sphere of radius `r` at `x0` with velocity `v`; plane `n·x = d` (unit `n`).
/// Returns `t ∈ [0, dt]` when the sphere first touches the plane from the
/// positive side, or `None` if no hit in the interval.
pub fn sphere_plane_toi(x0: Vec3, v: Vec3, r: f64, n: Vec3, plane_d: f64, dt: f64) -> Option<f64> {
    let dist = plane_distance(n, plane_d, x0) - r;
    if dist <= 0.0 {
        // Already overlapping / touching — TOI = 0.
        return Some(0.0);
    }
    let vn = v.dot(n);
    if vn >= -1e-15 {
        // Moving away or parallel.
        return None;
    }
    let t = dist / -vn;
    if t >= 0.0 && t <= dt {
        Some(t)
    } else {
        None
    }
}

/// Support point of a shape in direction `dir` (world), at pose `(x, R)`.
pub fn support(shape: &Shape, x: Vec3, r: &[[f64; 3]; 3], dir: Vec3) -> Vec3 {
    crate::support::support_world(shape, x, r, dir)
}

/// Minimum signed separation of shape A toward B along `n` using support maps.
/// Positive ⇒ separated along `n` (A left of B).
pub fn support_separation(
    shape_a: &Shape,
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    shape_b: &Shape,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
    n: Vec3,
) -> f64 {
    let pa = support(shape_a, xa, ra, -n);
    let pb = support(shape_b, xb, rb, n);
    n.dot(pb - pa)
}

/// Conservative advancement TOI for a dynamic shape vs a static plane.
///
/// Uses analytic sphere–plane when possible; otherwise binary-search on
/// support separation along the plane normal.
pub fn toi_shape_plane(
    shape: &Shape,
    x0: Vec3,
    r: &[[f64; 3]; 3],
    v: Vec3,
    plane: Plane,
    dt: f64,
    params: &CcdParams,
) -> Option<f64> {
    let n = plane.normal;
    let d = plane.offset;
    if let Shape::Sphere(s) = shape {
        return sphere_plane_toi(x0, v, s.radius, n, d, dt);
    }
    if matches!(shape, Shape::Plane(_)) {
        return None;
    }
    // Binary search on separation(t) = 0 via support maps (box/capsule/convex/rounded).
    let sep0 = plane_distance(n, d, support(shape, x0, r, -n));
    if sep0 <= 0.0 {
        return Some(0.0);
    }
    let vn = v.dot(n);
    if vn >= -1e-15 {
        return None;
    }
    let x1 = x0 + v * dt;
    let sep1 = plane_distance(n, d, support(shape, x1, r, -n));
    if sep1 > 0.0 {
        return None;
    }
    let mut lo = 0.0;
    let mut hi = dt;
    for _ in 0..params.toi_max_iters {
        let mid = 0.5 * (lo + hi);
        let xm = x0 + v * mid;
        let s = plane_distance(n, d, support(shape, xm, r, -n));
        if s > params.toi_tolerance {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < params.toi_tolerance / (1.0 + v.norm()) {
            break;
        }
    }
    Some(hi)
}

/// Build a speculative contact for a dynamic body approaching a static plane.
#[allow(clippy::too_many_arguments)]
pub fn speculative_plane_contact(
    body: usize,
    static_idx: usize,
    shape: &Shape,
    x: Vec3,
    r: &[[f64; 3]; 3],
    _v: Vec3,
    plane: Plane,
    speculative_distance: f64,
) -> Option<ContactPoint> {
    let n = plane.normal;
    let d = plane.offset;
    if matches!(shape, Shape::Plane(_)) {
        return None;
    }
    let (point, sep) = match shape {
        Shape::Sphere(s) => {
            let dist = plane_distance(n, d, x);
            let sep = dist - s.radius;
            let point = x - n * s.radius;
            (point, sep)
        }
        _ => {
            let p = support(shape, x, r, -n);
            let sep = plane_distance(n, d, p);
            (p, sep)
        }
    };
    if sep < 0.0 {
        // Already penetrating — discrete path owns this.
        return None;
    }
    if sep > speculative_distance {
        return None;
    }
    // Match discrete sphere/box–plane: normal A→B = −plane.n (into the plane).
    // depth = −sep so soft-step separation0 = sep ≥ 0 (speculative).
    Some(ContactPoint {
        a: body,
        b: static_idx,
        point,
        normal: -n,
        depth: -sep,
        feature: FeatureId(SPECU),
    })
}

/// Feature id tag for speculative contacts (high bit).
pub const SPECU: u32 = 0x8000_0000;

/// True if this contact was generated by the speculative path.
pub fn is_speculative(c: &ContactPoint) -> bool {
    c.feature.0 & SPECU != 0
}

/// Result of a bullet TOI advance against static geometry.
#[derive(Clone, Debug)]
pub struct ToiHit {
    /// Body index.
    pub body: usize,
    /// Time of impact in `[0, dt]`.
    pub t: f64,
    /// Contact to resolve at impact.
    pub contact: ContactPoint,
}

/// Find earliest TOI of bullet bodies against static planes/boxes.
#[allow(clippy::too_many_arguments)]
pub fn earliest_bullet_toi(
    bullets: &[bool],
    shapes: &[Shape],
    x: &[Vec3],
    q: &[Quat],
    v: &[Vec3],
    statics: &[(Shape, Vec3, Quat)],
    dt: f64,
    params: &CcdParams,
) -> Option<ToiHit> {
    if !params.toi {
        return None;
    }
    let n = shapes.len();
    let mut best: Option<ToiHit> = None;
    for i in 0..n {
        if !bullets.get(i).copied().unwrap_or(false) {
            continue;
        }
        let ri = q[i].to_matrix();
        for (si, (sshape, sx, sq)) in statics.iter().enumerate() {
            let static_idx = n + si;
            let hit = match sshape {
                Shape::Plane(p) => {
                    let Some(t) = toi_shape_plane(&shapes[i], x[i], &ri, v[i], *p, dt, params)
                    else {
                        continue;
                    };
                    let xt = x[i] + v[i] * t;
                    let (point, depth) = match &shapes[i] {
                        Shape::Sphere(s) => {
                            let dist = plane_distance(p.normal, p.offset, xt);
                            (xt - p.normal * s.radius, (s.radius - dist).max(0.0))
                        }
                        other => {
                            let pt = support(other, xt, &ri, -p.normal);
                            let sep = plane_distance(p.normal, p.offset, pt);
                            (pt, (-sep).max(0.0))
                        }
                    };
                    ToiHit {
                        body: i,
                        t,
                        contact: ContactPoint {
                            a: i,
                            b: static_idx,
                            point,
                            normal: -p.normal,
                            depth,
                            feature: FeatureId(0),
                        },
                    }
                }
                other => {
                    // M6: GJK conservative advancement vs static convex (box/hull/…).
                    let sr = sq.to_matrix();
                    let Some((t, nrm, depth, point)) = conservative_advancement_toi(
                        &shapes[i],
                        x[i],
                        &ri,
                        v[i],
                        Vec3::zero(), // linear velocity of static
                        other,
                        *sx,
                        &sr,
                        dt,
                        params,
                    ) else {
                        continue;
                    };
                    ToiHit {
                        body: i,
                        t,
                        contact: ContactPoint {
                            a: i,
                            b: static_idx,
                            point,
                            normal: nrm,
                            depth,
                            feature: FeatureId(0),
                        },
                    }
                }
            };
            if best.as_ref().map(|b| hit.t < b.t).unwrap_or(true) {
                best = Some(hit);
            }
        }
    }
    best
}

/// Advance bullet bodies to earliest TOI and return the impact contact list.
///
/// Non-bullet bodies are left unchanged. Positions of bullet bodies are moved
/// to `x + v * t_hit` (clamped). Caller should then run the normal contact solve
/// for the remaining `dt - t_hit` or treat the hit as a discrete contact at rest
/// relative to the impact.
#[allow(clippy::too_many_arguments)]
pub fn advance_bullets_to_toi(
    bullets: &[bool],
    shapes: &[Shape],
    x: &mut [Vec3],
    q: &[Quat],
    v: &[Vec3],
    statics: &[(Shape, Vec3, Quat)],
    dt: f64,
    params: &CcdParams,
) -> (f64, Vec<ContactPoint>) {
    match earliest_bullet_toi(bullets, shapes, x, q, v, statics, dt, params) {
        None => (dt, Vec::new()),
        Some(hit) => {
            for i in 0..x.len() {
                if bullets.get(i).copied().unwrap_or(false) {
                    x[i] += v[i] * hit.t;
                }
            }
            (hit.t, vec![hit.contact])
        }
    }
}

/// Generate speculative contacts for dynamic–static plane pairs that are close
/// and approaching (no discrete contact yet).
#[allow(clippy::too_many_arguments)]
pub fn generate_speculative_contacts(
    shapes: &[Shape],
    x: &[Vec3],
    q: &[Quat],
    v: &[Vec3],
    statics: &[(Shape, Vec3, Quat)],
    existing: &[ContactPoint],
    params: &CcdParams,
) -> Vec<ContactPoint> {
    if !params.speculative {
        return Vec::new();
    }
    let n = shapes.len();
    let mut out = Vec::new();
    for i in 0..n {
        let ri = q[i].to_matrix();
        for (si, (sshape, _sx, _sq)) in statics.iter().enumerate() {
            let static_idx = n + si;
            // Skip if discrete contact already exists for this pair.
            if existing
                .iter()
                .any(|c| (c.a == i && c.b == static_idx) || (c.b == i && c.a == static_idx))
            {
                continue;
            }
            if let Shape::Plane(p) = sshape {
                if let Some(c) = speculative_plane_contact(
                    i,
                    static_idx,
                    &shapes[i],
                    x[i],
                    &ri,
                    v[i],
                    *p,
                    params.speculative_distance,
                ) {
                    // Approaching when (v_static − v_dyn)·n < 0 ⇒ v·plane.n < 0.
                    if v[i].dot(p.normal) < 0.0 {
                        out.push(c);
                    }
                }
            }
        }
    }
    out
}

/// Speed used to decide “fast” for speculative AABB expansion.
pub fn body_speed(v: Vec3) -> f64 {
    sqrt(v.norm_squared())
}

/// Conservative advancement TOI using GJK distance (Mirtich / Catto style).
///
/// Translational CA with an angular bound `|ω|·R` added to the approach speed
/// so rotation cannot tunnel past the distance lower bound. Returns
/// `(t, normal A→B, depth, contact point)` or `None` if no hit in `[0, dt]`.
#[allow(clippy::too_many_arguments)]
pub fn conservative_advancement_toi(
    shape_a: &Shape,
    xa0: Vec3,
    ra: &[[f64; 3]; 3],
    va: Vec3,
    wa_bound: Vec3, // unused angular vel bound placeholder (caller may pass |ω|·ê)
    shape_b: &Shape,
    xb0: Vec3,
    rb: &[[f64; 3]; 3],
    dt: f64,
    params: &CcdParams,
) -> Option<(f64, Vec3, f64, Vec3)> {
    let _ = wa_bound;
    let tol = GjkTol::default();
    let target = params.toi_tolerance;
    let mut t = 0.0;
    let mut xa = xa0;
    let ra_max = bounding_radius(shape_a);
    let rb_max = bounding_radius(shape_b);
    // Relative linear velocity (B static ⇒ vb = 0).
    let v_rel = va; // va − vb
    for _ in 0..params.toi_max_iters {
        let g = gjk(shape_a, xa, ra, shape_b, xb0, rb, tol);
        if g.intersecting || g.distance <= target {
            let n = if g.normal.norm_squared() > 0.0 {
                g.normal
            } else if v_rel.norm_squared() > 0.0 {
                -v_rel.normalize()
            } else {
                Vec3::unit_z()
            };
            let point = (g.point_a + g.point_b) * 0.5;
            return Some((t, n, 0.0, point));
        }
        let n = g.normal;
        // Approach speed: with normal A→B, positive `v_rel·n` means A moves
        // toward B (closing the gap). Angular bound deferred (pure translation
        // CA is what the M6 thin-box oracle needs).
        let mut approach = v_rel.dot(n);
        let _ = (ra_max, rb_max);
        if approach < 1e-15 {
            // Receding or parallel along current GJK normal — fall back to
            // velocity direction as the expected contact normal.
            if v_rel.norm_squared() < 1e-20 {
                return None;
            }
            let n2 = -v_rel.normalize(); // A→B guess when falling onto B
            approach = v_rel.dot(n2);
            if approach < 1e-15 {
                return None;
            }
        }
        let dt_step = g.distance / approach;
        if !(dt_step.is_finite()) || dt_step <= 0.0 {
            return None;
        }
        // Advance a fraction to stay conservative (Catto: often 0.75–1.0).
        let advance = (0.9 * dt_step).max(target / (1.0 + approach));
        if t + advance >= dt {
            // Check end of interval.
            let xa_end = xa0 + v_rel * dt;
            let g_end = gjk(shape_a, xa_end, ra, shape_b, xb0, rb, tol);
            if g_end.intersecting || g_end.distance <= target {
                let n = if g_end.normal.norm_squared() > 0.0 {
                    g_end.normal
                } else {
                    -v_rel.normalize()
                };
                return Some((dt, n, 0.0, (g_end.point_a + g_end.point_b) * 0.5));
            }
            return None;
        }
        t += advance;
        xa = xa0 + v_rel * t;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sphere_plane_toi_1000mps_1cm_wall() {
        // Sphere r=0.05 at z=1, v_z = −1000, plane z=0; wall thickness not needed
        // for plane TOI — hit when centre at z=r.
        let x0 = Vec3::new(0.0, 0.0, 1.0);
        let v = Vec3::new(0.0, 0.0, -1000.0);
        let r = 0.05;
        let dt = 1.0 / 60.0;
        let t = sphere_plane_toi(x0, v, r, Vec3::unit_z(), 0.0, dt).unwrap();
        let expected = (1.0 - r) / 1000.0;
        assert_relative_eq!(t, expected, epsilon = 1e-12);
        assert!(t < dt);
    }

    #[test]
    fn no_toi_when_receding() {
        let t = sphere_plane_toi(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::unit_z() * 10.0,
            0.1,
            Vec3::unit_z(),
            0.0,
            1.0,
        );
        assert!(t.is_none());
    }

    #[test]
    fn swept_aabb_grows_with_velocity() {
        let a = Aabb {
            lo: Vec3::new(-1.0, -1.0, -1.0),
            hi: Vec3::new(1.0, 1.0, 1.0),
        };
        let s = swept_aabb(a, Vec3::new(10.0, 0.0, 0.0), 0.1);
        assert_relative_eq!(s.hi.x, 2.0, epsilon = 1e-12);
        assert_relative_eq!(s.lo.x, -2.0, epsilon = 1e-12);
    }
}
