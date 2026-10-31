//! Contact manifold generation for capsule–box, capsule–capsule, convex–convex.
//!
//! Face clipping follows the Sutherland–Hodgman / reference–incident pattern
//! used by the existing box–box SAT path (Catto / Ericson style). Up to 4
//! contact points in 3D. Feature ids encode reference face + incident vertex
//! for warm starting.

use pe_math::libm_ops::fabs;
use pe_math::Vec3;

use crate::epa::{epa_from_overlap, EpaTol};
use crate::gjk::{gjk, GjkTol};
use crate::hull::ConvexHull;
use crate::manifold::{Contact, FeatureId};
use crate::narrowphase::{box_box, capsule_capsule, closest_seg_seg, sphere_box};
use crate::shapes::{clamp, col, sign_np, BoxShape, Capsule, Shape};
use crate::support::support_world;

/// Encode a geometric feature id: high 16 bits = reference feature, low 16 =
/// incident feature (vertex or edge index).
#[inline]
pub fn feature_pair(ref_id: u16, inc_id: u16) -> FeatureId {
    FeatureId(((ref_id as u32) << 16) | (inc_id as u32))
}

/// Capsule–box manifold (dedicated). Uses segment–OBB closest points, then a
/// sphere–box contact at the capsule sample; falls back to GJK/EPA+clip when
/// the capsule axis is deeply inside the box.
pub fn capsule_box(
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    cap: Capsule,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
    box_s: BoxShape,
) -> Vec<Contact> {
    let (p0, p1) = cap.segment(xa, ra);
    // Sample both endpoints and the closest point of the segment to the box COM
    // projected onto the OBB — generate up to a few sphere–box contacts and
    // keep the deepest unique ones.
    let closest_on_seg = {
        // Closest point of segment to box centre (good for shallow contacts).
        let d = p1 - p0;
        let dd = d.dot(d);
        let t = if dd > 0.0 {
            clamp((xb - p0).dot(d) / dd, 0.0, 1.0)
        } else {
            0.0
        };
        p0 + d * t
    };
    let samples = [p0, p1, closest_on_seg];
    let mut contacts = Vec::new();
    for (si, sample) in samples.iter().enumerate() {
        let mut cs = sphere_box(*sample, cap.radius, xb, rb, box_s);
        for (k, c) in cs.iter_mut().enumerate() {
            c.feature = feature_pair(si as u16, k as u16);
            // Dedup by proximity.
            let dup = contacts.iter().any(|e: &Contact| {
                (e.point - c.point).norm_squared() < 1e-8 && e.normal.dot(c.normal) > 0.99
            });
            if !dup {
                contacts.push(*c);
            }
        }
    }
    if !contacts.is_empty() {
        // Keep ≤ 4 deepest.
        contacts.sort_by(|a, b| b.depth.total_cmp(&a.depth));
        contacts.truncate(4);
        return contacts;
    }
    // Deep / axis-inside: GJK/EPA single point.
    let sa = Shape::Capsule(cap);
    let sb = Shape::Box(box_s);
    convex_convex_contacts(&sa, xa, ra, &sb, xb, rb)
}

/// Capsule–capsule with feature ids (wraps dedicated closest-segment path).
pub fn capsule_capsule_manifold(
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    ca: Capsule,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
    cb: Capsule,
) -> Vec<Contact> {
    let mut cs = capsule_capsule(xa, ra, ca, xb, rb, cb);
    for (i, c) in cs.iter_mut().enumerate() {
        c.feature = feature_pair(0, i as u16);
    }
    // Also emit endpoint-sphere contacts when nearly parallel and overlapping
    // along a segment (improves warm-start stability for side-by-side capsules).
    let (pa0, pa1) = ca.segment(xa, ra);
    let (pb0, pb1) = cb.segment(xb, rb);
    let da = (pa1 - pa0).normalize();
    let db = (pb1 - pb0).normalize();
    if fabs(da.dot(db)) > 0.98 {
        for (ei, (pa, pb)) in [(pa0, pb0), (pa1, pb1)].into_iter().enumerate() {
            let d = pb - pa;
            let dist = d.norm();
            let rsum = ca.radius + cb.radius;
            if dist < rsum && dist > 1e-12 {
                let n = d / dist;
                let depth = rsum - dist;
                let point = pa + n * (ca.radius - 0.5 * depth);
                let c = Contact {
                    point,
                    normal: n,
                    depth,
                    feature: feature_pair(1, ei as u16),
                };
                let dup = cs.iter().any(|e| (e.point - c.point).norm_squared() < 1e-8);
                if !dup {
                    cs.push(c);
                }
            }
        }
        cs.sort_by(|a, b| b.depth.total_cmp(&a.depth));
        cs.truncate(4);
    }
    let _ = closest_seg_seg;
    cs
}

/// General convex–convex contacts via GJK/EPA + reference/incident face clipping.
///
/// When both shapes are boxes, delegates to the existing SAT clipper (oracle
/// parity). Otherwise: EPA gives the separating normal / depth; if a hull face
/// is available as reference, clip the incident face to build ≤4 points.
pub fn convex_convex_contacts(
    sa: &Shape,
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    sb: &Shape,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
) -> Vec<Contact> {
    if let (Shape::Box(a), Shape::Box(b)) = (sa, sb) {
        return box_box(xa, ra, *a, xb, rb, *b);
    }

    let gjk_tol = GjkTol::default();
    let epa_tol = EpaTol::default();
    let g = gjk(sa, xa, ra, sb, xb, rb, gjk_tol);
    if !g.intersecting {
        return Vec::new();
    }
    let Some(pen) = epa_from_overlap(sa, xa, ra, sb, xb, rb, epa_tol) else {
        // Shallow: single contact from GJK witnesses.
        let depth = (g.point_a - g.point_b).norm();
        if depth < 1e-15 {
            return Vec::new();
        }
        let n = if g.normal.norm_squared() > 0.0 {
            g.normal
        } else {
            (g.point_b - g.point_a).normalize()
        };
        return vec![Contact {
            point: (g.point_a + g.point_b) * 0.5,
            normal: n,
            depth: depth.max(1e-12),
            feature: feature_pair(0, 0),
        }];
    };

    // Try face clipping when we have a convex hull reference face.
    if let Some(manifold) = try_clip_manifold(sa, xa, ra, sb, xb, rb, pen.normal) {
        return manifold;
    }
    if let Some(mut manifold) = try_clip_manifold(sb, xb, rb, sa, xa, ra, -pen.normal) {
        for c in &mut manifold {
            *c = c.flipped();
        }
        return manifold;
    }

    // Single-point EPA contact.
    vec![Contact {
        point: (pen.point_a + pen.point_b) * 0.5,
        normal: pen.normal,
        depth: pen.depth,
        feature: feature_pair(0, 0),
    }]
}

fn try_clip_manifold(
    ref_shape: &Shape,
    xr: Vec3,
    rr: &[[f64; 3]; 3],
    inc_shape: &Shape,
    xi: Vec3,
    ri: &[[f64; 3]; 3],
    n_ref: Vec3,
) -> Option<Vec<Contact>> {
    let (ref_face, ref_plane_d, ref_fid) = match ref_shape {
        Shape::Box(b) => box_reference_face(b, xr, rr, n_ref)?,
        Shape::Convex(h) => hull_reference_face(h, xr, rr, n_ref)?,
        Shape::Rounded(rd) => {
            // Clip against the core; inflate depths by radius later.
            return try_clip_manifold(&rd.inner, xr, rr, inc_shape, xi, ri, n_ref);
        }
        _ => return None,
    };

    let (inc_poly, inc_fid_base) = match inc_shape {
        Shape::Box(b) => box_incident_face(b, xi, ri, n_ref)?,
        Shape::Convex(h) => hull_incident_face(h, xi, ri, n_ref)?,
        Shape::Rounded(rd) => {
            let (mut poly, base) = match rd.inner.as_ref() {
                Shape::Box(b) => box_incident_face(b, xi, ri, n_ref)?,
                Shape::Convex(h) => hull_incident_face(h, xi, ri, n_ref)?,
                _ => return None,
            };
            // Push incident poly along −n by radius for the swept surface.
            for p in &mut poly {
                *p -= n_ref * rd.radius;
            }
            (poly, base)
        }
        Shape::Sphere(_s) => {
            // Degenerate: single support point.
            let p = support_world(inc_shape, xi, ri, -n_ref);
            return Some(vec![Contact {
                point: p + n_ref * (0.5 * (ref_plane_d - n_ref.dot(p)).max(0.0)),
                normal: n_ref,
                depth: (ref_plane_d - n_ref.dot(p)).max(0.0),
                feature: feature_pair(ref_fid, 0),
            }]);
        }
        Shape::Capsule(c) => {
            let (p0, p1) = c.segment(xi, ri);
            let poly = vec![p0 - n_ref * c.radius, p1 - n_ref * c.radius];
            (poly, 0u16)
        }
        _ => return None,
    };

    // Side planes from reference face edges.
    let mut poly = inc_poly;
    let m = ref_face.len();
    if m < 3 {
        return None;
    }
    for i in 0..m {
        let a = ref_face[i];
        let b = ref_face[(i + 1) % m];
        let edge = b - a;
        let mut side_n = edge.cross(n_ref);
        let sn = side_n.norm();
        if sn < 1e-14 {
            continue;
        }
        side_n /= sn;
        // Orient so reference face centroid is inside.
        let centroid = ref_face.iter().fold(Vec3::zero(), |s, &p| s + p) / (m as f64);
        if side_n.dot(centroid - a) > 0.0 {
            side_n = -side_n;
        }
        let off = side_n.dot(a);
        poly = clip_poly(&poly, side_n, off);
        if poly.is_empty() {
            return None;
        }
    }

    let mut out = Vec::new();
    for (k, p) in poly.iter().enumerate() {
        let dep = ref_plane_d - n_ref.dot(*p);
        if dep >= 0.0 {
            out.push(Contact {
                point: *p + n_ref * (0.5 * dep),
                normal: n_ref,
                depth: dep,
                feature: feature_pair(ref_fid, inc_fid_base.wrapping_add(k as u16)),
            });
        }
    }
    if out.is_empty() {
        return None;
    }
    out.sort_by(|a, b| b.depth.total_cmp(&a.depth));
    // Reduce to ≤ 4 (same heuristic as box–box).
    if out.len() > 4 {
        out = reduce_manifold(&out, 4);
    }
    Some(out)
}

fn box_reference_face(
    b: &BoxShape,
    x: Vec3,
    r: &[[f64; 3]; 3],
    n_ref: Vec3,
) -> Option<(Vec<Vec3>, f64, u16)> {
    let h = b.h();
    let dots = [
        col(r, 0).dot(n_ref),
        col(r, 1).dot(n_ref),
        col(r, 2).dot(n_ref),
    ];
    let mut k = 0usize;
    if fabs(dots[1]) > fabs(dots[k]) {
        k = 1;
    }
    if fabs(dots[2]) > fabs(dots[k]) {
        k = 2;
    }
    let mut s = sign_np(dots[k]);
    if s == 0.0 {
        s = 1.0;
    }
    let fc = x + col(r, k) * (s * h.to_array()[k]);
    let (u_, v_) = match k {
        0 => (1usize, 2usize),
        1 => (0usize, 2usize),
        _ => (0usize, 1usize),
    };
    let u = col(r, u_) * h.to_array()[u_];
    let v = col(r, v_) * h.to_array()[v_];
    let face = vec![fc + u + v, fc - u + v, fc - u - v, fc + u - v];
    let plane_d = n_ref.dot(fc);
    let fid = (k as u16) + if s > 0.0 { 0 } else { 3 };
    Some((face, plane_d, fid))
}

fn box_incident_face(
    b: &BoxShape,
    x: Vec3,
    r: &[[f64; 3]; 3],
    n_ref: Vec3,
) -> Option<(Vec<Vec3>, u16)> {
    let h = b.h();
    let dots = [
        col(r, 0).dot(n_ref),
        col(r, 1).dot(n_ref),
        col(r, 2).dot(n_ref),
    ];
    let mut m = 0usize;
    if fabs(dots[1]) > fabs(dots[m]) {
        m = 1;
    }
    if fabs(dots[2]) > fabs(dots[m]) {
        m = 2;
    }
    let mut s = -sign_np(dots[m]);
    if s == 0.0 {
        s = -1.0;
    }
    let fc = x + col(r, m) * (s * h.to_array()[m]);
    let (u_, v_) = match m {
        0 => (1usize, 2usize),
        1 => (0usize, 2usize),
        _ => (0usize, 1usize),
    };
    let u = col(r, u_) * h.to_array()[u_];
    let v = col(r, v_) * h.to_array()[v_];
    Some((
        vec![fc + u + v, fc - u + v, fc - u - v, fc + u - v],
        (m as u16) + 10,
    ))
}

fn hull_reference_face(
    h: &ConvexHull,
    x: Vec3,
    r: &[[f64; 3]; 3],
    n_ref: Vec3,
) -> Option<(Vec<Vec3>, f64, u16)> {
    let mut best = 0usize;
    let mut best_dot = f64::NEG_INFINITY;
    for (i, _) in h.faces.iter().enumerate() {
        let (n, _, _) = h.face_world_plane(i, x, r)?;
        let d = n.dot(n_ref);
        if d > best_dot {
            best_dot = d;
            best = i;
        }
    }
    if best_dot < 0.1 {
        return None;
    }
    let (n, d, tri) = h.face_world_plane(best, x, r)?;
    let _ = n;
    // Inflate plane by hull radius.
    let d = d + h.radius;
    Some((tri.to_vec(), d, best as u16))
}

fn hull_incident_face(
    h: &ConvexHull,
    x: Vec3,
    r: &[[f64; 3]; 3],
    n_ref: Vec3,
) -> Option<(Vec<Vec3>, u16)> {
    let mut best = 0usize;
    let mut best_dot = f64::INFINITY;
    for (i, _) in h.faces.iter().enumerate() {
        let (n, _, _) = h.face_world_plane(i, x, r)?;
        let d = n.dot(n_ref);
        if d < best_dot {
            best_dot = d;
            best = i;
        }
    }
    let (_, _, tri) = h.face_world_plane(best, x, r)?;
    let mut poly = tri.to_vec();
    if h.radius > 0.0 {
        for p in &mut poly {
            *p -= n_ref * h.radius;
        }
    }
    Some((poly, best as u16))
}

fn clip_poly(poly: &[Vec3], nrm: Vec3, off: f64) -> Vec<Vec3> {
    let mut out = Vec::new();
    let n = poly.len();
    if n == 0 {
        return out;
    }
    for i in 0..n {
        let p = poly[i];
        let q = poly[(i + 1) % n];
        let dp = nrm.dot(p) - off;
        let dq = nrm.dot(q) - off;
        if dp <= 0.0 {
            out.push(p);
        }
        if dp * dq < 0.0 {
            out.push(p + (q - p) * (dp / (dp - dq)));
        }
    }
    out
}

fn reduce_manifold(points: &[Contact], nmax: usize) -> Vec<Contact> {
    if points.len() <= nmax {
        return points.to_vec();
    }
    let mut i0 = 0;
    for i in 1..points.len() {
        if points[i].depth > points[i0].depth {
            i0 = i;
        }
    }
    let mut i1 = 0;
    let mut best_d = -1.0;
    for i in 0..points.len() {
        let d = (points[i].point - points[i0].point).norm();
        if d > best_d {
            best_d = d;
            i1 = i;
        }
    }
    let ax = points[i1].point - points[i0].point;
    let mut imax = 0;
    let mut imin = 0;
    let mut smax = f64::NEG_INFINITY;
    let mut smin = f64::INFINITY;
    for i in 0..points.len() {
        let cr = ax.cross(points[i].point - points[i0].point);
        // Use a stable scalar via dot with a reference cross.
        let s = cr.dot(ax.cross(points[0].point - points[i0].point + Vec3::unit_x() * 1e-9));
        if s > smax {
            smax = s;
            imax = i;
        }
        if s < smin {
            smin = s;
            imin = i;
        }
    }
    let mut out = Vec::new();
    for idx in [i0, i1, imax, imin] {
        if !out
            .iter()
            .any(|c: &Contact| (c.point - points[idx].point).norm_squared() < 1e-20)
        {
            out.push(points[idx]);
        }
        if out.len() >= nmax {
            break;
        }
    }
    out
}
