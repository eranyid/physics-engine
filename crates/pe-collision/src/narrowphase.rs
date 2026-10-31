//! Narrow phase. Every routine returns contacts with normal A→B and depth > 0
//! meaning penetration [m].
//!
//! [`NarrowPhase`] is the extension hook for GJK/EPA or SAT-with-feature-caching;
//! [`DedicatedNarrowPhase`] ports the Python dedicated primitive pairs.

use pe_math::libm_ops::fabs;
use pe_math::{Quat, Vec3};

use crate::manifold::{Contact, FeatureId};
use crate::shapes::{clamp, col, sign_np, support_extent, BoxShape, Capsule, Plane, Shape};

/// Pose of a collider in world frame.
#[derive(Clone, Copy, Debug)]
pub struct ColliderPose {
    /// Shape.
    pub shape: Shape,
    /// COM / origin [m].
    pub position: Vec3,
    /// Rotation matrix body→world (row-major). Identity for planes.
    pub rotation: [[f64; 3]; 3],
}

impl ColliderPose {
    /// Construct a pose; planes ignore rotation.
    pub fn new(shape: Shape, position: Vec3, rotation: [[f64; 3]; 3]) -> Self {
        Self {
            shape,
            position,
            rotation,
        }
    }

    /// Identity-orientation pose.
    pub fn at(shape: Shape, position: Vec3) -> Self {
        Self::new(shape, position, Quat::identity().to_matrix())
    }
}

/// Narrow-phase interface. Implementations must be deterministic (no hash maps).
pub trait NarrowPhase {
    /// Contacts for the ordered pair `(a, b)` with normals A→B.
    fn collide(&self, a: &ColliderPose, b: &ColliderPose) -> Vec<Contact>;
}

/// Dedicated primitive pairs matching the Python oracle (sphere/box/capsule/plane + box SAT).
#[derive(Clone, Copy, Debug, Default)]
pub struct DedicatedNarrowPhase;

impl NarrowPhase for DedicatedNarrowPhase {
    fn collide(&self, a: &ColliderPose, b: &ColliderPose) -> Vec<Contact> {
        collide(
            a.shape,
            a.position,
            &a.rotation,
            b.shape,
            b.position,
            &b.rotation,
        )
    }
}

fn mk(point: Vec3, normal: Vec3, depth: f64) -> Contact {
    Contact::new(point, normal, depth)
}

/// Sphere–sphere.
pub fn sphere_sphere(xa: Vec3, ra: f64, xb: Vec3, rb: f64) -> Vec<Contact> {
    let d = xb - xa;
    let dist = d.norm();
    if dist >= ra + rb {
        return Vec::new();
    }
    let n = if dist > 1e-12 {
        d / dist
    } else {
        Vec3::unit_z()
    };
    let depth = ra + rb - dist;
    vec![mk(xa + n * (ra - 0.5 * depth), n, depth)]
}

/// Sphere–plane. Plane pose is unused (plane is world-frame).
pub fn sphere_plane(xa: Vec3, r: f64, plane: Plane) -> Vec<Contact> {
    let n = plane.n();
    let dist = n.dot(xa) - plane.offset;
    if dist >= r {
        return Vec::new();
    }
    let depth = r - dist;
    vec![mk(xa - n * (r - 0.5 * depth), -n, depth)]
}

/// Box–plane: one contact per penetrating vertex.
pub fn box_plane(xa: Vec3, ra: &[[f64; 3]; 3], box_s: BoxShape, plane: Plane) -> Vec<Contact> {
    let n = plane.n();
    let mut out = Vec::new();
    for v in box_s.vertices(xa, ra) {
        let dist = n.dot(v) - plane.offset;
        if dist < 0.0 {
            out.push(mk(v - n * (0.5 * dist), -n, -dist));
        }
    }
    out
}

/// Sphere–box.
pub fn sphere_box(xa: Vec3, r: f64, xb: Vec3, rb: &[[f64; 3]; 3], box_s: BoxShape) -> Vec<Contact> {
    let h = box_s.h();
    // p = Rᵀ (xa - xb)
    let d_world = xa - xb;
    let p = Vec3::new(
        rb[0][0] * d_world.x + rb[1][0] * d_world.y + rb[2][0] * d_world.z,
        rb[0][1] * d_world.x + rb[1][1] * d_world.y + rb[2][1] * d_world.z,
        rb[0][2] * d_world.x + rb[1][2] * d_world.y + rb[2][2] * d_world.z,
    );
    let q = Vec3::new(
        clamp(p.x, -h.x, h.x),
        clamp(p.y, -h.y, h.y),
        clamp(p.z, -h.z, h.z),
    );
    let d = p - q;
    let dist = d.norm();
    if dist > 1e-12 {
        if dist >= r {
            return Vec::new();
        }
        let n_local = -d / dist; // sphere -> box
        let depth = r - dist;
        let q_world = xb
            + Vec3::new(
                rb[0][0] * q.x + rb[0][1] * q.y + rb[0][2] * q.z,
                rb[1][0] * q.x + rb[1][1] * q.y + rb[1][2] * q.z,
                rb[2][0] * q.x + rb[2][1] * q.y + rb[2][2] * q.z,
            );
        let n_world = Vec3::new(
            rb[0][0] * n_local.x + rb[0][1] * n_local.y + rb[0][2] * n_local.z,
            rb[1][0] * n_local.x + rb[1][1] * n_local.y + rb[1][2] * n_local.z,
            rb[2][0] * n_local.x + rb[2][1] * n_local.y + rb[2][2] * n_local.z,
        );
        return vec![mk(q_world + n_world * (0.5 * depth), n_world, depth)];
    }
    // Centre inside the box: exit through the nearest face
    let margins = [h.x - fabs(p.x), h.y - fabs(p.y), h.z - fabs(p.z)];
    let mut k = 0;
    if margins[1] < margins[k] {
        k = 1;
    }
    if margins[2] < margins[k] {
        k = 2;
    }
    let pk = [p.x, p.y, p.z][k];
    let s = if pk >= 0.0 { 1.0 } else { -1.0 };
    let mut n_local = Vec3::zero();
    match k {
        0 => n_local.x = -s,
        1 => n_local.y = -s,
        _ => n_local.z = -s,
    }
    let depth = r + [h.x, h.y, h.z][k] - fabs(pk);
    let n_world = Vec3::new(
        rb[0][0] * n_local.x + rb[0][1] * n_local.y + rb[0][2] * n_local.z,
        rb[1][0] * n_local.x + rb[1][1] * n_local.y + rb[1][2] * n_local.z,
        rb[2][0] * n_local.x + rb[2][1] * n_local.y + rb[2][2] * n_local.z,
    );
    vec![mk(xa, n_world, depth)]
}

/// Closest points between segments `[p1,q1]` and `[p2,q2]` (Ericson RTCD 5.1.9).
pub fn closest_seg_seg(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    let eps = 1e-14;
    if a <= eps && e <= eps {
        return (p1, p2);
    }
    let (s, t) = if a <= eps {
        (0.0, clamp(f / e, 0.0, 1.0))
    } else {
        let c = d1.dot(r);
        if e <= eps {
            (clamp(-c / a, 0.0, 1.0), 0.0)
        } else {
            let b = d1.dot(d2);
            let den = a * e - b * b;
            let mut s = if den > eps {
                clamp((b * f - c * e) / den, 0.0, 1.0)
            } else {
                0.0
            };
            let mut t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            }
            (s, t)
        }
    };
    (p1 + d1 * s, p2 + d2 * t)
}

fn closest_on_segment(p: Vec3, q: Vec3, x: Vec3) -> Vec3 {
    let d = q - p;
    let dd = d.dot(d);
    let t = if dd > 0.0 {
        clamp((x - p).dot(d) / dd, 0.0, 1.0)
    } else {
        0.0
    };
    p + d * t
}

/// Capsule–plane: sphere–plane at each endpoint.
pub fn capsule_plane(xa: Vec3, ra: &[[f64; 3]; 3], cap: Capsule, plane: Plane) -> Vec<Contact> {
    let (p, q) = cap.segment(xa, ra);
    let mut out = sphere_plane(p, cap.radius, plane);
    out.extend(sphere_plane(q, cap.radius, plane));
    out
}

/// Capsule–sphere.
pub fn capsule_sphere(
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    cap: Capsule,
    xb: Vec3,
    rb: f64,
) -> Vec<Contact> {
    let (p, q) = cap.segment(xa, ra);
    let c = closest_on_segment(p, q, xb);
    sphere_sphere(c, cap.radius, xb, rb)
}

/// Capsule–capsule.
pub fn capsule_capsule(
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    ca: Capsule,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
    cb: Capsule,
) -> Vec<Contact> {
    let (pa0, pa1) = ca.segment(xa, ra);
    let (pb0, pb1) = cb.segment(xb, rb);
    let (pa, pb) = closest_seg_seg(pa0, pa1, pb0, pb1);
    sphere_sphere(pa, ca.radius, pb, cb.radius)
}

fn reduce_points(points: &[Vec3], depths: &[f64], nmax: usize) -> Vec<usize> {
    if points.len() <= nmax {
        return (0..points.len()).collect();
    }
    let mut i0 = 0;
    for i in 1..depths.len() {
        if depths[i] > depths[i0] {
            i0 = i;
        }
    }
    let mut i1 = 0;
    let mut best_d = -1.0;
    for i in 0..points.len() {
        let d = (points[i] - points[i0]).norm();
        if d > best_d {
            best_d = d;
            i1 = i;
        }
    }
    let ax = points[i1] - points[i0];
    let mut cr: Vec<Vec3> = points.iter().map(|&p| ax.cross(p - points[i0])).collect();
    let mut iref = 0;
    let mut best_n = -1.0;
    for (i, c) in cr.iter().enumerate() {
        let n = c.norm();
        if n > best_n {
            best_n = n;
            iref = i;
        }
    }
    let refr = cr[iref];
    let mut s: Vec<f64> = cr.iter().map(|c| c.dot(refr)).collect();
    let mut imax = 0;
    let mut imin = 0;
    for i in 1..s.len() {
        if s[i] > s[imax] {
            imax = i;
        }
        if s[i] < s[imin] {
            imin = i;
        }
    }
    // dict.fromkeys preserves order and dedupes
    let mut out = Vec::new();
    for idx in [i0, i1, imax, imin] {
        if !out.contains(&idx) {
            out.push(idx);
        }
    }
    let _ = (&mut cr, &mut s); // silence unused mut if optimized
    out
}

/// Box–box SAT over 15 axes; face contacts → clipped manifold (≤4), edge–edge → 1 point.
pub fn box_box(
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    a: BoxShape,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
    b: BoxShape,
) -> Vec<Contact> {
    let ha = a.h();
    let hb = b.h();
    let t = xb - xa;
    let mut best_face_ov = f64::INFINITY;
    let mut best_face_n = Vec3::zero();
    let mut best_face_kind = b'A';
    let mut best_face_idx = 0usize;

    for (kind, axes_r) in [(b'A', ra), (b'B', rb)] {
        for idx in 0..3 {
            let l = col(axes_r, idx);
            let d = t.dot(l);
            let ov = support_extent(ha, ra, l) + support_extent(hb, rb, l) - fabs(d);
            if ov < 0.0 {
                return Vec::new();
            }
            if ov < best_face_ov {
                best_face_ov = ov;
                best_face_n = l * if d >= 0.0 { 1.0 } else { -1.0 };
                best_face_kind = kind;
                best_face_idx = idx;
            }
        }
    }

    let mut best_edge_ov = f64::INFINITY;
    let mut best_edge_n = Vec3::zero();
    let mut best_edge_i = 0usize;
    let mut best_edge_j = 0usize;
    for i in 0..3 {
        for j in 0..3 {
            let l_raw = col(ra, i).cross(col(rb, j));
            let nl = l_raw.norm();
            if nl < 1e-6 {
                continue;
            }
            let l = l_raw / nl;
            let d = t.dot(l);
            let ov = support_extent(ha, ra, l) + support_extent(hb, rb, l) - fabs(d);
            if ov < 0.0 {
                return Vec::new();
            }
            if ov < best_edge_ov {
                best_edge_ov = ov;
                best_edge_n = l * if d >= 0.0 { 1.0 } else { -1.0 };
                best_edge_i = i;
                best_edge_j = j;
            }
        }
    }

    let (ov, n, kind, idx_i, idx_j) = if best_edge_ov < 0.95 * best_face_ov - 1e-5 {
        (best_edge_ov, best_edge_n, b'E', best_edge_i, best_edge_j)
    } else {
        (best_face_ov, best_face_n, best_face_kind, best_face_idx, 0)
    };

    if kind == b'E' {
        let i = idx_i;
        let j = idx_j;
        let mut pa = xa;
        for k in 0..3 {
            if k != i {
                pa += col(ra, k) * (ha.to_array()[k] * sign_np(col(ra, k).dot(n)));
            }
        }
        let mut pb = xb;
        for k in 0..3 {
            if k != j {
                pb -= col(rb, k) * (hb.to_array()[k] * sign_np(col(rb, k).dot(n)));
            }
        }
        let (ca, cb) = closest_seg_seg(
            pa - col(ra, i) * ha.to_array()[i],
            pa + col(ra, i) * ha.to_array()[i],
            pb - col(rb, j) * hb.to_array()[j],
            pb + col(rb, j) * hb.to_array()[j],
        );
        return vec![mk((ca + cb) * 0.5, n, ov)];
    }

    let (xr, rr, hr, xi, ri, hi, nref, k) = if kind == b'A' {
        (xa, ra, ha, xb, rb, hb, n, idx_i)
    } else {
        (xb, rb, hb, xa, ra, ha, -n, idx_i)
    };

    // Incident face: face of the other box most anti-parallel to nref
    let dots = [
        col(ri, 0).dot(nref),
        col(ri, 1).dot(nref),
        col(ri, 2).dot(nref),
    ];
    let mut m = 0usize;
    if fabs(dots[1]) > fabs(dots[m]) {
        m = 1;
    }
    if fabs(dots[2]) > fabs(dots[m]) {
        m = 2;
    }
    // -sign(dots[m]) or -1.0  (NumPy: 0 is falsy → -1)
    let mut s = -sign_np(dots[m]);
    if s == 0.0 {
        s = -1.0;
    }
    let fc = xi + col(ri, m) * (s * hi.to_array()[m]);
    let (u_, v_) = match m {
        0 => (1usize, 2usize),
        1 => (0usize, 2usize),
        _ => (0usize, 1usize),
    };
    let u = col(ri, u_) * hi.to_array()[u_];
    let v = col(ri, v_) * hi.to_array()[v_];
    let mut poly = vec![fc + u + v, fc - u + v, fc - u - v, fc + u - v];

    // Clip against the 4 side planes of the reference face
    for a_idx in 0..3 {
        if a_idx == k {
            continue;
        }
        for sign in [1.0, -1.0] {
            let nrm = col(rr, a_idx) * sign;
            let off = sign * (col(rr, a_idx).dot(xr)) + hr.to_array()[a_idx];
            let mut out = Vec::new();
            let npoly = poly.len();
            if npoly == 0 {
                return Vec::new();
            }
            for i in 0..npoly {
                let p = poly[i];
                let q = poly[(i + 1) % npoly];
                let dp = nrm.dot(p) - off;
                let dq = nrm.dot(q) - off;
                if dp <= 0.0 {
                    out.push(p);
                }
                if dp * dq < 0.0 {
                    out.push(p + (q - p) * (dp / (dp - dq)));
                }
            }
            poly = out;
            if poly.is_empty() {
                return Vec::new();
            }
        }
    }

    let face_off = nref.dot(xr) + hr.to_array()[k];
    let mut pts = Vec::new();
    let mut deps = Vec::new();
    for p in &poly {
        let dep = face_off - nref.dot(*p);
        if dep >= 0.0 {
            pts.push(*p + nref * (0.5 * dep));
            deps.push(dep);
        }
    }
    let keep = reduce_points(&pts, &deps, 4);
    keep.into_iter().map(|i| mk(pts[i], n, deps[i])).collect()
}

/// Dispatch on shape types; normal always A→B. Capsule–box is not implemented.
pub fn collide(
    sa: Shape,
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    sb: Shape,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
) -> Vec<Contact> {
    if sa.order_index() > sb.order_index() {
        return collide(sb, xb, rb, sa, xa, ra)
            .into_iter()
            .map(Contact::flipped)
            .collect();
    }
    match (sa, sb) {
        (Shape::Sphere(a), Shape::Sphere(b)) => sphere_sphere(xa, a.radius, xb, b.radius),
        (Shape::Sphere(a), Shape::Capsule(b)) => capsule_sphere(xb, rb, b, xa, a.radius)
            .into_iter()
            .map(Contact::flipped)
            .collect(),
        (Shape::Sphere(a), Shape::Box(b)) => sphere_box(xa, a.radius, xb, rb, b),
        (Shape::Sphere(a), Shape::Plane(b)) => sphere_plane(xa, a.radius, b),
        (Shape::Capsule(a), Shape::Capsule(b)) => capsule_capsule(xa, ra, a, xb, rb, b),
        (Shape::Capsule(a), Shape::Plane(b)) => capsule_plane(xa, ra, a, b),
        (Shape::Capsule(_), Shape::Box(_)) => {
            panic!("capsule-box collision is not implemented");
        }
        (Shape::Box(a), Shape::Box(b)) => box_box(xa, ra, a, xb, rb, b),
        (Shape::Box(a), Shape::Plane(b)) => box_plane(xa, ra, a, b),
        (Shape::Plane(_), Shape::Plane(_)) => Vec::new(),
        // Remaining orderings are handled by the swap above.
        _ => unreachable!("shape order swap should have handled this pair"),
    }
}

/// Assign sequential feature ids within a contact list (warm-start key component).
pub fn assign_feature_ids(contacts: &mut [Contact]) {
    for (i, c) in contacts.iter_mut().enumerate() {
        c.feature = FeatureId(i as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::Sphere;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    use pe_math::Quat;
    use proptest::prelude::*;

    fn eye() -> [[f64; 3]; 3] {
        Quat::identity().to_matrix()
    }

    fn rot(axis: Vec3, ang: f64) -> [[f64; 3]; 3] {
        Quat::from_axis_angle(axis, ang).to_matrix()
    }

    #[test]
    fn sphere_sphere_depth_and_normal() {
        let c = collide(
            Shape::Sphere(Sphere::new(1.0)),
            Vec3::zero(),
            &eye(),
            Shape::Sphere(Sphere::new(0.5)),
            Vec3::new(1.2, 0.0, 0.0),
            &eye(),
        );
        assert_eq!(c.len(), 1);
        assert_abs_diff_eq!(c[0].depth, 0.3, epsilon = 1e-15);
        assert_abs_diff_eq!(c[0].normal.x, 1.0, epsilon = 1e-15);
        assert!(collide(
            Shape::Sphere(Sphere::new(1.0)),
            Vec3::zero(),
            &eye(),
            Shape::Sphere(Sphere::new(0.5)),
            Vec3::new(1.51, 0.0, 0.0),
            &eye(),
        )
        .is_empty());
    }

    #[test]
    fn sphere_plane_and_symmetry() {
        let pl = Plane::new(Vec3::unit_z(), 0.0);
        let c = collide(
            Shape::Sphere(Sphere::new(0.5)),
            Vec3::new(0.0, 0.0, 0.4),
            &eye(),
            Shape::Plane(pl),
            Vec3::zero(),
            &eye(),
        );
        assert_abs_diff_eq!(c[0].depth, 0.1, epsilon = 1e-15);
        assert_abs_diff_eq!(c[0].normal.z, -1.0, epsilon = 1e-15);
        let c2 = collide(
            Shape::Plane(pl),
            Vec3::zero(),
            &eye(),
            Shape::Sphere(Sphere::new(0.5)),
            Vec3::new(0.0, 0.0, 0.4),
            &eye(),
        );
        assert_abs_diff_eq!(c2[0].normal.z, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn box_plane_four_corners_and_tilted() {
        let pl = Plane::new(Vec3::unit_z(), 0.0);
        let c = collide(
            Shape::Box(BoxShape::new(1.0, 1.0, 1.0)),
            Vec3::new(0.0, 0.0, 0.99),
            &eye(),
            Shape::Plane(pl),
            Vec3::zero(),
            &eye(),
        );
        assert_eq!(c.len(), 4);
        for k in &c {
            assert_abs_diff_eq!(k.depth, 0.01, epsilon = 1e-12);
        }
        let r = rot(Vec3::unit_y(), PI / 4.0);
        let c = collide(
            Shape::Box(BoxShape::new(1.0, 1.0, 1.0)),
            Vec3::new(0.0, 0.0, 1.4),
            &r,
            Shape::Plane(pl),
            Vec3::zero(),
            &eye(),
        );
        assert_eq!(c.len(), 2);
        assert_abs_diff_eq!(c[0].depth, 2.0_f64.sqrt() - 1.4, epsilon = 1e-12);
    }

    #[test]
    fn sphere_box_outside_edge_and_inside() {
        let b = BoxShape::new(1.0, 1.0, 1.0);
        let x = Vec3::new(1.3, 1.3, 0.0);
        let c = collide(
            Shape::Sphere(Sphere::new(0.5)),
            x,
            &eye(),
            Shape::Box(b),
            Vec3::zero(),
            &eye(),
        );
        assert_abs_diff_eq!(c[0].depth, 0.5 - 0.3 * 2.0_f64.sqrt(), epsilon = 1e-12);
        let n_exp = -Vec3::new(1.0, 1.0, 0.0) / 2.0_f64.sqrt();
        assert_abs_diff_eq!(c[0].normal.x, n_exp.x, epsilon = 1e-12);
        assert_abs_diff_eq!(c[0].normal.y, n_exp.y, epsilon = 1e-12);
        let c = collide(
            Shape::Sphere(Sphere::new(0.2)),
            Vec3::new(0.9, 0.0, 0.0),
            &eye(),
            Shape::Box(b),
            Vec3::zero(),
            &eye(),
        );
        assert_abs_diff_eq!(c[0].depth, 0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(c[0].normal.x, -1.0, epsilon = 1e-12);
        assert!(collide(
            Shape::Sphere(Sphere::new(0.2)),
            Vec3::new(1.3, 0.0, 0.0),
            &eye(),
            Shape::Box(b),
            Vec3::zero(),
            &eye(),
        )
        .is_empty());
    }

    #[test]
    fn box_box_face_manifold() {
        let a = BoxShape::new(1.0, 1.0, 1.0);
        let b = BoxShape::new(0.5, 0.5, 0.5);
        let c = collide(
            Shape::Box(a),
            Vec3::zero(),
            &eye(),
            Shape::Box(b),
            Vec3::new(0.2, 0.1, 1.45),
            &rot(Vec3::unit_z(), 0.3),
        );
        assert_eq!(c.len(), 4);
        for k in &c {
            assert_abs_diff_eq!(k.normal.x, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(k.normal.y, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(k.normal.z, 1.0, epsilon = 1e-12);
            assert_abs_diff_eq!(k.depth, 0.05, epsilon = 1e-12);
            assert!(k.point.x.abs() <= 1.0 && k.point.y.abs() <= 1.0);
        }
    }

    #[test]
    fn box_box_offset_manifold_is_clipped() {
        let a = BoxShape::new(1.0, 1.0, 1.0);
        let c = collide(
            Shape::Box(a),
            Vec3::zero(),
            &eye(),
            Shape::Box(a),
            Vec3::new(1.5, 0.0, 1.9),
            &eye(),
        );
        let mut xs: Vec<f64> = c.iter().map(|k| k.point.x).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(c.len(), 4);
        assert_abs_diff_eq!(xs[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(*xs.last().unwrap(), 1.0, epsilon = 1e-12);
        for k in &c {
            assert_abs_diff_eq!(k.depth, 0.1, epsilon = 1e-12);
        }
    }

    #[test]
    fn box_box_edge_edge() {
        let a = BoxShape::new(1.0, 1.0, 1.0);
        let ra = rot(Vec3::unit_x(), FRAC_PI_4);
        let rb = rot(Vec3::unit_y(), FRAC_PI_4);
        let h = 2.0 * 2.0_f64.sqrt() - 0.1;
        let c = collide(
            Shape::Box(a),
            Vec3::zero(),
            &ra,
            Shape::Box(a),
            Vec3::new(0.0, 0.0, h),
            &rb,
        );
        assert_eq!(c.len(), 1);
        assert_relative_eq!(c[0].depth, 0.1, max_relative = 1e-9);
        assert_abs_diff_eq!(c[0].normal.z, 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(c[0].point.x, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(c[0].point.y, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(c[0].point.z, h / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn box_box_separated_by_edge_axis() {
        let a = BoxShape::new(1.0, 1.0, 1.0);
        let ra = rot(Vec3::unit_x(), FRAC_PI_4);
        let rb = rot(Vec3::unit_y(), FRAC_PI_4);
        assert!(collide(
            Shape::Box(a),
            Vec3::zero(),
            &ra,
            Shape::Box(a),
            Vec3::new(0.0, 0.0, 2.0 * 2.0_f64.sqrt() + 0.01),
            &rb,
        )
        .is_empty());
    }

    #[test]
    fn capsule_contacts() {
        let cap = Capsule::new(0.2, 0.5);
        let r = rot(Vec3::unit_y(), FRAC_PI_2);
        let c = collide(
            Shape::Capsule(cap),
            Vec3::new(0.0, 0.0, 0.19),
            &r,
            Shape::Plane(Plane::new(Vec3::unit_z(), 0.0)),
            Vec3::zero(),
            &eye(),
        );
        assert_eq!(c.len(), 2);
        for k in &c {
            assert_abs_diff_eq!(k.depth, 0.01, epsilon = 1e-12);
        }
        let c = collide(
            Shape::Capsule(cap),
            Vec3::zero(),
            &r,
            Shape::Sphere(Sphere::new(0.3)),
            Vec3::new(0.3, 0.0, 0.45),
            &eye(),
        );
        assert_abs_diff_eq!(c[0].depth, 0.05, epsilon = 1e-12);
        let c = collide(
            Shape::Capsule(cap),
            Vec3::zero(),
            &r,
            Shape::Capsule(cap),
            Vec3::new(0.1, 0.0, 0.85),
            &eye(),
        );
        assert_abs_diff_eq!(c[0].depth, 0.05, epsilon = 1e-12);
        assert_abs_diff_eq!(c[0].normal.z, 1.0, epsilon = 1e-12);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn pair_results_symmetric(
            ax in -2.0f64..2.0, ay in -2.0f64..2.0, az in 0.5f64..3.0,
            bx in -2.0f64..2.0, by in -2.0f64..2.0, bz in 0.5f64..3.0,
            ra in 0.1f64..0.8, rb in 0.1f64..0.8,
        ) {
            let sa = Shape::Sphere(Sphere::new(ra));
            let sb = Shape::Sphere(Sphere::new(rb));
            let xa = Vec3::new(ax, ay, az);
            let xb = Vec3::new(bx, by, bz);
            let ab = collide(sa, xa, &eye(), sb, xb, &eye());
            let ba = collide(sb, xb, &eye(), sa, xa, &eye());
            prop_assert_eq!(ab.len(), ba.len());
            for (a, b) in ab.iter().zip(ba.iter()) {
                prop_assert!((a.depth - b.depth).abs() < 1e-12);
                prop_assert!((a.normal + b.normal).norm() < 1e-12);
                prop_assert!((a.point - b.point).norm() < 1e-12);
                prop_assert!(a.normal.norm().is_finite());
                prop_assert!((a.normal.norm() - 1.0).abs() < 1e-12 || a.depth == 0.0);
                prop_assert!(!a.depth.is_nan());
                prop_assert!(!a.point.x.is_nan());
            }
        }

        #[test]
        fn manifold_normal_unit_no_nan_box_plane(
            zx in 0.5f64..1.5,
            ang in -0.4f64..0.4,
        ) {
            let pl = Plane::new(Vec3::unit_z(), 0.0);
            let r = rot(Vec3::new(1.0, 0.2, 0.0).normalize(), ang);
            let c = collide(
                Shape::Box(BoxShape::new(0.3, 0.3, 0.3)),
                Vec3::new(0.0, 0.0, zx),
                &r,
                Shape::Plane(pl),
                Vec3::zero(),
                &eye(),
            );
            for k in &c {
                prop_assert!(k.depth.is_finite() && !k.depth.is_nan());
                prop_assert!(k.normal.norm().is_finite());
                prop_assert!((k.normal.norm() - 1.0).abs() < 1e-12);
                prop_assert!(!k.point.x.is_nan() && !k.point.y.is_nan() && !k.point.z.is_nan());
            }
        }
    }
}
