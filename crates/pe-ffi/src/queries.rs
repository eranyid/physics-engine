//! Raycast and overlap queries against a [`pe_collision::ContactWorld`].

use pe_collision::shapes::{Aabb, Shape};
use pe_collision::ContactWorld;
use pe_math::libm_ops::{fabs, sqrt};
use pe_math::{Quat, Vec3};

/// Raycast hit (closest).
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    /// Body index (`usize::MAX` = static collider index encoded as `n + si`).
    pub body: usize,
    /// Hit distance along ray [m] (`origin + t * dir_unit`).
    pub t: f64,
    /// World hit point [m].
    pub point: Vec3,
    /// Outward normal at hit.
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
    let nn = {
        let len = n.norm();
        if len == 0.0 {
            return None;
        }
        n / len
    };
    let denom = nn.dot(dir);
    if fabs(denom) < 1e-15 {
        return None;
    }
    let t = (offset - nn.dot(origin)) / denom;
    if t < tmin || t > tmax {
        return None;
    }
    // Hit the solid side only when approaching from outside (dir · n < 0).
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
            // Conservative AABB hit; normal from AABB face (unverified exact OBB).
            let aabb = shape.aabb(x, &r);
            let t = ray_aabb(origin, dir, aabb, tmin, tmax)?;
            let p = origin + dir * t;
            // Pick dominant axis normal toward outside of AABB.
            let cx = (aabb.lo.x + aabb.hi.x) * 0.5;
            let cy = (aabb.lo.y + aabb.hi.y) * 0.5;
            let cz = (aabb.lo.z + aabb.hi.z) * 0.5;
            let dx = (p.x - cx) / (aabb.hi.x - aabb.lo.x).max(1e-30);
            let dy = (p.y - cy) / (aabb.hi.y - aabb.lo.y).max(1e-30);
            let dz = (p.z - cz) / (aabb.hi.z - aabb.lo.z).max(1e-30);
            let normal = if fabs(dx) >= fabs(dy) && fabs(dx) >= fabs(dz) {
                Vec3::new(dx.signum(), 0.0, 0.0)
            } else if fabs(dy) >= fabs(dz) {
                Vec3::new(0.0, dy.signum(), 0.0)
            } else {
                Vec3::new(0.0, 0.0, dz.signum())
            };
            Some((t, normal))
        }
    }
}

/// Closest ray hit against dynamics + statics. `dir` need not be unit (normalized here).
pub(crate) fn raycast(cw: &ContactWorld, origin: Vec3, dir: Vec3, tmax: f64) -> Option<RayHit> {
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
        if let Some((t, n)) =
            ray_shape(origin, dir, &s.shape, s.position, s.orientation, 0.0, tmax)
        {
            update(&mut best, cw.world.n + si, t, n);
        }
    }
    best
}

/// AABB overlap query: returns body indices whose world AABB overlaps `query`.
pub(crate) fn overlap_aabb(cw: &ContactWorld, query: Aabb, out: &mut Vec<usize>) {
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

fn aabb_overlap(a: Aabb, b: Aabb) -> bool {
    a.lo.x <= b.hi.x
        && a.hi.x >= b.lo.x
        && a.lo.y <= b.hi.y
        && a.hi.y >= b.lo.y
        && a.lo.z <= b.hi.z
        && a.hi.z >= b.lo.z
}
