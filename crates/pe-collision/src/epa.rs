//! Expanding Polytope Algorithm (EPA) for penetration depth / normal.
//!
//! Clean-room from Gino van den Bergen (EPA) and Erin Catto GDC notes.
//! Builds an origin-containing tetrahedron via GJK-style supports, then expands
//! until the closest face is within [`EpaTol`].

use pe_math::libm_ops::fabs;
use pe_math::Vec3;

use crate::gjk::{gjk, GjkTol};
use crate::shapes::Shape;
use crate::support::support_world;

/// EPA tolerances [SI]. Documented honestly — do not silent-loosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpaTol {
    /// Absolute face-distance convergence [m].
    pub abs: f64,
    /// Relative convergence vs support scale.
    pub rel: f64,
    /// Degenerate face area threshold [m²].
    pub area: f64,
    /// Maximum EPA iterations.
    pub max_iters: usize,
    /// Maximum faces retained in the polytope.
    pub max_faces: usize,
}

impl Default for EpaTol {
    fn default() -> Self {
        Self {
            abs: 1e-8,
            rel: 1e-10,
            area: 1e-24,
            max_iters: 64,
            max_faces: 64,
        }
    }
}

/// EPA penetration result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpaResult {
    /// Penetration depth [m] (≥ 0).
    pub depth: f64,
    /// Unit normal A→B (direction to separate B from A).
    pub normal: Vec3,
    /// Contact point on A [m].
    pub point_a: Vec3,
    /// Contact point on B [m].
    pub point_b: Vec3,
    /// Iterations used.
    pub iterations: usize,
}

#[derive(Clone, Copy, Debug)]
struct Vert {
    p: Vec3,
    a: Vec3,
    b: Vec3,
}

#[derive(Clone, Copy, Debug)]
struct Face {
    idx: [usize; 3],
    normal: Vec3,
    dist: f64,
}

/// Run EPA when shapes overlap. Returns `None` if not intersecting.
#[allow(clippy::too_many_arguments)]
pub fn epa(
    sa: &Shape,
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    sb: &Shape,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
    gjk_tol: GjkTol,
    epa_tol: EpaTol,
) -> Option<EpaResult> {
    let g = gjk(sa, xa, ra, sb, xb, rb, gjk_tol);
    if !g.intersecting {
        return None;
    }
    epa_from_overlap(sa, xa, ra, sb, xb, rb, epa_tol)
}

/// EPA assuming the caller already knows the shapes overlap.
pub fn epa_from_overlap(
    sa: &Shape,
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    sb: &Shape,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
    tol: EpaTol,
) -> Option<EpaResult> {
    // Fast path: sphere–sphere has a closed form (also validates EPA).
    if let (Shape::Sphere(a), Shape::Sphere(b)) = (sa, sb) {
        let d = xb - xa;
        let dist = d.norm();
        let n = if dist > 1e-15 {
            d / dist
        } else {
            Vec3::unit_x()
        };
        let depth = a.radius + b.radius - dist;
        if depth < 0.0 {
            return None;
        }
        let pa = xa + n * a.radius;
        let pb = xb - n * b.radius;
        // Midpoint contact; normal A→B.
        return Some(EpaResult {
            depth,
            normal: n,
            point_a: pa - n * (0.5 * depth),
            point_b: pb + n * (0.5 * depth),
            iterations: 0,
        });
    }

    let mut verts = build_containing_tet(sa, xa, ra, sb, xb, rb, tol)?;
    let mut faces = Vec::new();
    for tri in [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]] {
        if let Some(f) = make_face(&verts, tri[0], tri[1], tri[2], tol) {
            faces.push(f);
        }
    }
    if faces.len() < 4 {
        return shallow_fallback(xa, xb, tol);
    }

    let mut iterations = 0usize;
    loop {
        iterations += 1;
        if iterations > tol.max_iters || faces.len() > tol.max_faces {
            break;
        }
        // Closest face with positive distance (skip degenerate origin-on-plane faces).
        let mut best = None;
        for (i, face) in faces.iter().enumerate() {
            if face.dist <= tol.abs {
                continue;
            }
            match best {
                None => best = Some(i),
                Some(b) if face.dist < faces[b].dist => best = Some(i),
                _ => {}
            }
        }
        let best = best.or_else(|| {
            // All faces degenerate — pick minimum absolute dist.
            (0..faces.len()).min_by(|&i, &j| faces[i].dist.total_cmp(&faces[j].dist))
        })?;
        let f = faces[best];
        let v = support_vert(sa, xa, ra, sb, xb, rb, f.normal);
        let d = v.p.dot(f.normal);
        let eps = tol.abs + tol.rel * (1.0 + v.p.norm());
        if d - f.dist <= eps {
            return Some(face_result(&verts, &f, iterations));
        }

        // Remove faces visible from v; collect horizon.
        let mut visible = vec![false; faces.len()];
        let mut any = false;
        for (i, face) in faces.iter().enumerate() {
            if face.normal.dot(v.p) - face.dist > tol.abs {
                visible[i] = true;
                any = true;
            }
        }
        if !any {
            return Some(face_result(&verts, &f, iterations));
        }

        let mut horizon: Vec<(usize, usize)> = Vec::new();
        for (i, face) in faces.iter().enumerate() {
            if !visible[i] {
                continue;
            }
            for &(a, b) in &[
                (face.idx[0], face.idx[1]),
                (face.idx[1], face.idx[2]),
                (face.idx[2], face.idx[0]),
            ] {
                let mut shared_vis = false;
                for (j, other) in faces.iter().enumerate() {
                    if !visible[j] {
                        continue;
                    }
                    let o = other.idx;
                    if (o[0] == b && o[1] == a)
                        || (o[1] == b && o[2] == a)
                        || (o[2] == b && o[0] == a)
                    {
                        shared_vis = true;
                        break;
                    }
                }
                if !shared_vis {
                    horizon.push((a, b));
                }
            }
        }
        if horizon.is_empty() {
            return Some(face_result(&verts, &f, iterations));
        }

        let new_idx = verts.len();
        verts.push(v);
        let mut new_faces = Vec::new();
        for (i, face) in faces.iter().enumerate() {
            if !visible[i] {
                new_faces.push(*face);
            }
        }
        for &(a, b) in &horizon {
            if let Some(nf) = make_face(&verts, a, b, new_idx, tol) {
                new_faces.push(nf);
            }
        }
        if new_faces.is_empty() {
            return Some(face_result(&verts, &f, iterations));
        }
        faces = new_faces;
    }

    let mut best = 0usize;
    for i in 1..faces.len() {
        if faces[i].dist < faces[best].dist {
            best = i;
        }
    }
    Some(face_result(&verts, &faces[best], iterations))
}

fn shallow_fallback(xa: Vec3, xb: Vec3, tol: EpaTol) -> Option<EpaResult> {
    let mut n = xb - xa;
    if n.norm_squared() < 1e-20 {
        n = Vec3::unit_z();
    } else {
        n = n.normalize();
    }
    Some(EpaResult {
        depth: tol.abs,
        normal: n,
        point_a: xa,
        point_b: xb,
        iterations: 0,
    })
}

/// Build a tetrahedron of Minkowski supports that contains the origin
/// **strictly** (all face plane distances > 0). Avoids the common failure
/// where ±(xb−xa) supports put the origin on an edge.
fn build_containing_tet(
    sa: &Shape,
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    sb: &Shape,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
    tol: EpaTol,
) -> Option<Vec<Vert>> {
    // Prefer diagonal directions so axis-aligned boxes don't produce an
    // origin-on-edge tetrahedron.
    let seed_dirs = [
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::unit_x(),
        Vec3::unit_y(),
        Vec3::unit_z(),
        -Vec3::unit_x(),
        -Vec3::unit_y(),
        -Vec3::unit_z(),
    ];
    let mut verts: Vec<Vert> = Vec::new();
    for d in seed_dirs {
        let v = support_vert(sa, xa, ra, sb, xb, rb, d);
        if verts.iter().all(|e| (e.p - v.p).norm_squared() > 1e-18) {
            verts.push(v);
        }
        if verts.len() >= 4 {
            // Try every combination of 4 until we get a strict containing tet.
            let n = verts.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        for l in (k + 1)..n {
                            let mut cand = vec![verts[i], verts[j], verts[k], verts[l]];
                            let vol = tet_vol(cand[0].p, cand[1].p, cand[2].p, cand[3].p);
                            if fabs(vol) < 1e-20 {
                                continue;
                            }
                            if vol < 0.0 {
                                cand.swap(2, 3);
                            }
                            if origin_in_tet_strict(&cand, tol) {
                                return Some(cand);
                            }
                        }
                    }
                }
            }
        }
    }
    // Last resort: any containing tet (possibly with origin on a face).
    if verts.len() >= 4 {
        let mut cand = verts[..4].to_vec();
        let vol = tet_vol(cand[0].p, cand[1].p, cand[2].p, cand[3].p);
        if fabs(vol) >= 1e-20 {
            if vol < 0.0 {
                cand.swap(2, 3);
            }
            return Some(cand);
        }
    }
    None
}

fn origin_in_tet_strict(verts: &[Vert], tol: EpaTol) -> bool {
    if !origin_in_tet(verts) {
        return false;
    }
    // All face distances must be strictly positive.
    for tri in [[0usize, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]] {
        let Some(f) = make_face(verts, tri[0], tri[1], tri[2], tol) else {
            return false;
        };
        if f.dist <= tol.abs * 10.0 {
            return false;
        }
    }
    true
}

fn origin_in_tet(verts: &[Vert]) -> bool {
    let a = verts[0].p;
    let b = verts[1].p;
    let c = verts[2].p;
    let d = verts[3].p;
    let Some(bary) = barycentric_tet(a, b, c, d, Vec3::zero()) else {
        return false;
    };
    bary.iter().all(|&w| w >= -1e-8)
}

fn barycentric_tet(a: Vec3, b: Vec3, c: Vec3, d: Vec3, p: Vec3) -> Option<[f64; 4]> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = d - a;
    let vp = p - a;
    let den = v0.dot(v1.cross(v2));
    if fabs(den) < 1e-30 {
        return None;
    }
    let w1 = vp.dot(v1.cross(v2)) / den;
    let w2 = v0.dot(vp.cross(v2)) / den;
    let w3 = v0.dot(v1.cross(vp)) / den;
    let w0 = 1.0 - w1 - w2 - w3;
    Some([w0, w1, w2, w3])
}

fn support_vert(
    sa: &Shape,
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    sb: &Shape,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
    dir: Vec3,
) -> Vert {
    let a = support_world(sa, xa, ra, dir);
    let b = support_world(sb, xb, rb, -dir);
    Vert { p: a - b, a, b }
}

fn tet_vol(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    (b - a).dot((c - a).cross(d - a))
}

fn make_face(verts: &[Vert], i: usize, j: usize, k: usize, tol: EpaTol) -> Option<Face> {
    let a = verts[i].p;
    let b = verts[j].p;
    let c = verts[k].p;
    let n = (b - a).cross(c - a);
    let nn = n.norm();
    if nn * nn < tol.area {
        return None;
    }
    let mut normal = n / nn;
    let mut dist = normal.dot(a);
    // Outward = away from origin when origin is inside ⇒ dist ≥ 0.
    if dist < 0.0 {
        normal = -normal;
        dist = -dist;
        return Some(Face {
            idx: [i, k, j],
            normal,
            dist,
        });
    }
    Some(Face {
        idx: [i, j, k],
        normal,
        dist,
    })
}

fn face_result(verts: &[Vert], face: &Face, iterations: usize) -> EpaResult {
    let a = verts[face.idx[0]];
    let b = verts[face.idx[1]];
    let c = verts[face.idx[2]];
    let p0 = face.normal * face.dist;
    let bary = triangle_bary(a.p, b.p, c.p, p0);
    let pa = a.a * bary[0] + b.a * bary[1] + c.a * bary[2];
    let pb = a.b * bary[0] + b.b * bary[1] + c.b * bary[2];
    EpaResult {
        depth: face.dist.max(0.0),
        normal: face.normal,
        point_a: pa,
        point_b: pb,
        iterations,
    }
}

fn triangle_bary(a: Vec3, b: Vec3, c: Vec3, p: Vec3) -> [f64; 3] {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let den = d00 * d11 - d01 * d01;
    if fabs(den) < 1e-30 {
        return [1.0, 0.0, 0.0];
    }
    let v = (d11 * d20 - d01 * d21) / den;
    let w = (d00 * d21 - d01 * d20) / den;
    let u = 1.0 - v - w;
    [u, v, w]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::{BoxShape, Sphere};
    use approx::assert_abs_diff_eq;
    use pe_math::Quat;

    fn eye() -> [[f64; 3]; 3] {
        Quat::identity().to_matrix()
    }

    #[test]
    fn sphere_sphere_penetration() {
        let a = Shape::Sphere(Sphere::new(1.0));
        let b = Shape::Sphere(Sphere::new(1.0));
        let r = epa_from_overlap(
            &a,
            Vec3::zero(),
            &eye(),
            &b,
            Vec3::new(1.0, 0.0, 0.0),
            &eye(),
            EpaTol::default(),
        )
        .unwrap();
        assert_abs_diff_eq!(r.depth, 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(r.normal.x, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn box_box_face_penetration() {
        let a = Shape::Box(BoxShape::new(1.0, 1.0, 1.0));
        let b = Shape::Box(BoxShape::new(1.0, 1.0, 1.0));
        let r = epa_from_overlap(
            &a,
            Vec3::zero(),
            &eye(),
            &b,
            Vec3::new(1.5, 0.0, 0.0),
            &eye(),
            EpaTol::default(),
        )
        .unwrap();
        assert_abs_diff_eq!(r.depth, 0.5, epsilon = 1e-4);
        assert!(r.normal.x > 0.9, "normal={:?} depth={}", r.normal, r.depth);
    }
}
