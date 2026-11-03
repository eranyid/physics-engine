//! Scene queries against a [`ContactWorld`]: raycast, AABB overlap, sphere cast.
//!
//! Sphere / plane ray hits are exact. Box / capsule / convex ray hits use the
//! world AABB (normals are AABB-face approximations — documented as unverified
//! for exact OBB). Sphere cast vs sphere/plane is analytic; other shapes use
//! a conservative expanded-AABB sweep.

use pe_math::libm_ops::{fabs, sqrt};
use pe_math::{Quat, Vec3};

use crate::shapes::{Aabb, Shape};
use crate::ContactWorld;

/// Closest raycast hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// Body index; static colliders are encoded as `n + static_index`.
    pub body: usize,
    /// Hit distance along the unit ray [m] (`origin + t * dir`).
    pub t: f64,
    /// World hit point [m].
    pub point: Vec3,
    /// Outward surface normal at the hit.
    pub normal: Vec3,
}

/// Closest sphere-cast hit (swept sphere of radius `radius`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeCastHit {
    /// Body index (`n + static_index` for statics).
    pub body: usize,
    /// Fraction along the cast segment in `[0, 1]` (0 = already overlapping).
    pub toi: f64,
    /// World hit point on the obstacle surface [m].
    pub point: Vec3,
    /// Outward normal on the obstacle.
    pub normal: Vec3,
}

fn ray_sphere(
    origin: Vec3,
    dir: Vec3,
    center: Vec3,
    radius: f64,
    tmin: f64,
    tmax: f64,
) -> Option<(f64, Vec3)> {
    let oc = origin - center;
    let b = oc.dot(dir);
    let c = oc.norm_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = sqrt(disc);
    let mut t = -b - s;
    if t < tmin || t > tmax {
        t = -b + s;
        if t < tmin || t > tmax {
            return None;
        }
    }
    let p = origin + dir * t;
    let n = (p - center) / radius.max(1e-30);
    Some((t, n))
}

fn ray_plane(
    origin: Vec3,
    dir: Vec3,
    n: Vec3,
    offset: f64,
    tmin: f64,
    tmax: f64,
) -> Option<(f64, Vec3)> {
    let len = n.norm();
    if len == 0.0 {
        return None;
    }
    let nn = n / len;
    let denom = nn.dot(dir);
    if fabs(denom) < 1e-15 {
        return None;
    }
    let t = (offset - nn.dot(origin)) / denom;
    if t < tmin || t > tmax {
        return None;
    }
    let normal = if denom < 0.0 { nn } else { -nn };
    Some((t, normal))
}

fn ray_aabb(origin: Vec3, dir: Vec3, aabb: Aabb, tmin: f64, tmax: f64) -> Option<f64> {
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

fn aabb_face_normal(aabb: Aabb, p: Vec3) -> Vec3 {
    let cx = (aabb.lo.x + aabb.hi.x) * 0.5;
    let cy = (aabb.lo.y + aabb.hi.y) * 0.5;
    let cz = (aabb.lo.z + aabb.hi.z) * 0.5;
    let dx = (p.x - cx) / (aabb.hi.x - aabb.lo.x).max(1e-30);
    let dy = (p.y - cy) / (aabb.hi.y - aabb.lo.y).max(1e-30);
    let dz = (p.z - cz) / (aabb.hi.z - aabb.lo.z).max(1e-30);
    if fabs(dx) >= fabs(dy) && fabs(dx) >= fabs(dz) {
        Vec3::new(dx.signum(), 0.0, 0.0)
    } else if fabs(dy) >= fabs(dz) {
        Vec3::new(0.0, dy.signum(), 0.0)
    } else {
        Vec3::new(0.0, 0.0, dz.signum())
    }
}

fn ray_shape(
    origin: Vec3,
    dir: Vec3,
    shape: &Shape,
    x: Vec3,
    q: Quat,
    tmin: f64,
    tmax: f64,
) -> Option<(f64, Vec3)> {
    let r = q.to_matrix();
    match shape {
        Shape::Sphere(s) => ray_sphere(origin, dir, x, s.radius, tmin, tmax),
        Shape::Plane(p) => ray_plane(origin, dir, p.normal, p.offset, tmin, tmax),
        Shape::Box(_) | Shape::Capsule(_) | Shape::Convex(_) | Shape::Rounded(_) => {
            let aabb = shape.aabb(x, &r);
            let t = ray_aabb(origin, dir, aabb, tmin, tmax)?;
            let p = origin + dir * t;
            Some((t, aabb_face_normal(aabb, p)))
        }
    }
}

/// Closest ray hit against dynamics + statics. `dir` is normalized internally.
pub fn raycast(cw: &ContactWorld, origin: Vec3, dir: Vec3, tmax: f64) -> Option<RayHit> {
    let len = dir.norm();
    if !(len.is_finite() && len > 0.0) || !(tmax.is_finite() && tmax > 0.0) {
        return None;
    }
    let dir = dir / len;
    let mut best: Option<RayHit> = None;
    let update = |best: &mut Option<RayHit>, body: usize, t: f64, n: Vec3| {
        if t < 0.0 || t > tmax {
            return;
        }
        if best.as_ref().map(|h| t < h.t).unwrap_or(true) {
            *best = Some(RayHit {
                body,
                t,
                point: origin + dir * t,
                normal: n,
            });
        }
    };
    for i in 0..cw.world.n {
        if let Some((t, n)) = ray_shape(
            origin,
            dir,
            &cw.shapes[i],
            cw.world.state.x[i],
            cw.world.state.q[i],
            0.0,
            tmax,
        ) {
            update(&mut best, i, t, n);
        }
    }
    for (si, s) in cw.statics.iter().enumerate() {
        if let Some((t, n)) = ray_shape(origin, dir, &s.shape, s.position, s.orientation, 0.0, tmax)
        {
            update(&mut best, cw.world.n + si, t, n);
        }
    }
    best
}

fn aabb_overlap(a: Aabb, b: Aabb) -> bool {
    a.lo.x <= b.hi.x
        && a.hi.x >= b.lo.x
        && a.lo.y <= b.hi.y
        && a.hi.y >= b.lo.y
        && a.lo.z <= b.hi.z
        && a.hi.z >= b.lo.z
}

/// AABB overlap: body indices whose world AABB overlaps `query` (statics as `n+si`).
pub fn overlap_aabb(cw: &ContactWorld, query: Aabb, out: &mut Vec<usize>) {
    out.clear();
    for i in 0..cw.world.n {
        let r = cw.world.state.q[i].to_matrix();
        let a = cw.shapes[i].aabb(cw.world.state.x[i], &r);
        if aabb_overlap(a, query) {
            out.push(i);
        }
    }
    for (si, s) in cw.statics.iter().enumerate() {
        let r = s.orientation.to_matrix();
        let a = s.shape.aabb(s.position, &r);
        if aabb_overlap(a, query) {
            out.push(cw.world.n + si);
        }
    }
}

fn expand_aabb(a: Aabb, radius: f64) -> Aabb {
    let r = Vec3::new(radius, radius, radius);
    Aabb {
        lo: a.lo - r,
        hi: a.hi + r,
    }
}

fn sphere_cast_sphere(
    origin: Vec3,
    dir: Vec3,
    cast_radius: f64,
    center: Vec3,
    radius: f64,
    tmax: f64,
) -> Option<(f64, Vec3)> {
    // Equivalent to ray vs sphere of radius r_cast + r_target.
    let (t, n) = ray_sphere(origin, dir, center, cast_radius + radius, 0.0, tmax)?;
    Some((t, n))
}

fn sphere_cast_plane(
    origin: Vec3,
    dir: Vec3,
    cast_radius: f64,
    n: Vec3,
    offset: f64,
    tmax: f64,
) -> Option<(f64, Vec3)> {
    let len = n.norm();
    if len == 0.0 {
        return None;
    }
    let nn = n / len;
    // Inflate the solid half-space by cast_radius along the outward normal.
    let signed = nn.dot(origin) - offset;
    let side = if signed >= 0.0 { 1.0 } else { -1.0 };
    let target = offset + side * cast_radius;
    let denom = nn.dot(dir);
    if fabs(denom) < 1e-15 {
        return None;
    }
    let t = (target - nn.dot(origin)) / denom;
    if t < 0.0 || t > tmax {
        return None;
    }
    Some((t, nn * side))
}

fn sphere_cast_shape(
    origin: Vec3,
    dir: Vec3,
    cast_radius: f64,
    shape: &Shape,
    x: Vec3,
    q: Quat,
    tmax: f64,
) -> Option<(f64, Vec3)> {
    let r = q.to_matrix();
    match shape {
        Shape::Sphere(s) => sphere_cast_sphere(origin, dir, cast_radius, x, s.radius, tmax),
        Shape::Plane(p) => sphere_cast_plane(origin, dir, cast_radius, p.normal, p.offset, tmax),
        Shape::Box(_) | Shape::Capsule(_) | Shape::Convex(_) | Shape::Rounded(_) => {
            let aabb = expand_aabb(shape.aabb(x, &r), cast_radius);
            let t = ray_aabb(origin, dir, aabb, 0.0, tmax)?;
            let p = origin + dir * t;
            Some((t, aabb_face_normal(aabb, p)))
        }
    }
}

/// Sweep a sphere of `radius` from `origin` along `dir` up to length `|dir|` or `tmax`.
///
/// `dir` need not be unit; the cast distance is `min(tmax, |dir|)` along the unit
/// direction of `dir`. Returns the closest hit.
pub fn sphere_cast(
    cw: &ContactWorld,
    origin: Vec3,
    dir: Vec3,
    radius: f64,
    tmax: f64,
) -> Option<ShapeCastHit> {
    if !(radius.is_finite() && radius >= 0.0) || !(tmax.is_finite() && tmax > 0.0) {
        return None;
    }
    let len = dir.norm();
    if !(len.is_finite() && len > 0.0) {
        return None;
    }
    let dir_u = dir / len;
    let max_t = tmax.min(len);
    let mut best: Option<ShapeCastHit> = None;
    let update = |best: &mut Option<ShapeCastHit>, body: usize, t: f64, n: Vec3| {
        if t < 0.0 || t > max_t {
            return;
        }
        let better = match best {
            Some(h) => t < h.toi * max_t,
            None => true,
        };
        if better {
            let point = origin + dir_u * t - n * radius;
            *best = Some(ShapeCastHit {
                body,
                toi: if max_t > 0.0 { t / max_t } else { 0.0 },
                point,
                normal: n,
            });
        }
    };
    for i in 0..cw.world.n {
        if let Some((t, n)) = sphere_cast_shape(
            origin,
            dir_u,
            radius,
            &cw.shapes[i],
            cw.world.state.x[i],
            cw.world.state.q[i],
            max_t,
        ) {
            update(&mut best, i, t, n);
        }
    }
    for (si, s) in cw.statics.iter().enumerate() {
        if let Some((t, n)) = sphere_cast_shape(
            origin,
            dir_u,
            radius,
            &s.shape,
            s.position,
            s.orientation,
            max_t,
        ) {
            update(&mut best, cw.world.n + si, t, n);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::{Material, Plane, Sphere};
    use crate::StaticCollider;
    use pe_core::forces::UniformGravity;
    use pe_core::{RigidBody, World};
    use pe_math::STANDARD_GRAVITY;

    fn empty_y_up() -> ContactWorld {
        ContactWorld::new(
            World::new(
                vec![],
                vec![Box::new(UniformGravity::new(Vec3::new(
                    0.0,
                    -STANDARD_GRAVITY,
                    0.0,
                )))],
            ),
            vec![],
            vec![],
            Some(vec![]),
        )
    }

    #[test]
    fn ray_sphere_analytic_distance() {
        let mut cw = empty_y_up();
        let r = 0.5;
        let body = RigidBody::new(1.0)
            .with_inertia(Sphere::new(r).inertia(1.0))
            .with_position(Vec3::new(0.0, 0.0, 5.0));
        cw.add_dynamic_body(body, Shape::Sphere(Sphere::new(r)), Material::default());
        let hit = raycast(&cw, Vec3::new(0.0, 0.0, 0.0), Vec3::unit_z(), 100.0).expect("hit");
        // Center at z=5, radius 0.5 → first hit at t=4.5
        assert!((hit.t - 4.5).abs() < 1e-12, "t={}", hit.t);
        assert_eq!(hit.body, 0);
        assert!((hit.normal.z + 1.0).abs() < 1e-12 || (hit.normal.z - (-1.0)).abs() < 1e-12);
        assert!((hit.point.z - 4.5).abs() < 1e-12);
    }

    #[test]
    fn ray_plane_y_up_analytic() {
        let mut cw = empty_y_up();
        cw.add_static(
            StaticCollider::new(Shape::Plane(Plane::new(Vec3::unit_y(), 0.0)))
                .with_material(Material::default()),
        );
        let hit = raycast(
            &cw,
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            10.0,
        )
        .expect("hit");
        assert!((hit.t - 2.0).abs() < 1e-12, "t={}", hit.t);
        assert!((hit.normal.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn overlap_aabb_finds_sphere() {
        let mut cw = empty_y_up();
        let body = RigidBody::new(1.0)
            .with_inertia(Sphere::new(0.25).inertia(1.0))
            .with_position(Vec3::new(1.0, 0.0, 0.0));
        cw.add_dynamic_body(body, Shape::Sphere(Sphere::new(0.25)), Material::default());
        let mut out = Vec::new();
        overlap_aabb(
            &cw,
            Aabb {
                lo: Vec3::new(0.5, -1.0, -1.0),
                hi: Vec3::new(1.5, 1.0, 1.0),
            },
            &mut out,
        );
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn sphere_cast_vs_sphere_analytic() {
        let mut cw = empty_y_up();
        let target_r = 0.5;
        let body = RigidBody::new(1.0)
            .with_inertia(Sphere::new(target_r).inertia(1.0))
            .with_position(Vec3::new(0.0, 0.0, 4.0));
        cw.add_dynamic_body(
            body,
            Shape::Sphere(Sphere::new(target_r)),
            Material::default(),
        );
        let cast_r = 0.5;
        // Centers 4 m apart; contact when gap closes to cast_r+target_r=1 → travel 3 m.
        let hit =
            sphere_cast(&cw, Vec3::zero(), Vec3::new(0.0, 0.0, 10.0), cast_r, 10.0).expect("hit");
        let travel = hit.toi * 10.0;
        assert!(
            (travel - 3.0).abs() < 1e-12,
            "travel={travel} toi={}",
            hit.toi
        );
    }

    #[test]
    fn raycast_miss_returns_none() {
        let cw = empty_y_up();
        assert!(raycast(&cw, Vec3::zero(), Vec3::unit_y(), 10.0).is_none());
    }
}
