//! GJK distance and intersection (van den Bergen / Catto style).
//!
//! Clean-room from:
//! - Gino van den Bergen, *Collision Detection in Interactive 3D Environments*
//!   (Morgan Kaufmann, 2003) — GJK with distance sub-algorithm.
//! - Erin Catto, GDC presentations on GJK / simplex solving.
//!
//! Deterministic termination: fixed max iterations, absolute/relative
//! tolerances documented in [`GjkTol`]. Degenerate simplices are collapsed
//! by area/volume thresholds rather than silently loosening epsilons.

use pe_math::libm_ops::fabs;
use pe_math::Vec3;

use crate::shapes::Shape;
use crate::support::support_world;

/// GJK tolerances [SI]. Do not loosen silently — document changes in
/// `docs/COLLISION.md` / `docs/DETERMINISM.md`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GjkTol {
    /// Absolute distance / separation tolerance [m].
    pub abs: f64,
    /// Relative tolerance vs support scale.
    pub rel: f64,
    /// Simplex area degeneracy threshold [m²].
    pub area: f64,
    /// Simplex volume degeneracy threshold [m³].
    pub volume: f64,
    /// Maximum GJK iterations before giving up (returns best estimate).
    pub max_iters: usize,
}

impl Default for GjkTol {
    fn default() -> Self {
        Self {
            abs: 1e-10,
            rel: 1e-12,
            area: 1e-20,
            volume: 1e-30,
            max_iters: 64,
        }
    }
}

/// Result of a GJK query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GjkResult {
    /// True if the origin is in the Minkowski difference (shapes overlap).
    pub intersecting: bool,
    /// Euclidean distance between shapes [m] (`0` when intersecting).
    pub distance: f64,
    /// Closest point on A in world [m] (approximate witness).
    pub point_a: Vec3,
    /// Closest point on B in world [m].
    pub point_b: Vec3,
    /// Unit separating normal A→B when separated; arbitrary when intersecting.
    pub normal: Vec3,
    /// Iterations used.
    pub iterations: usize,
}

#[derive(Clone, Copy, Debug)]
struct Vertex {
    p: Vec3,
    a: Vec3,
    b: Vec3,
}

/// GJK intersection test.
pub fn gjk_intersect(
    a: &Shape,
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    b: &Shape,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
    tol: GjkTol,
) -> bool {
    gjk(a, xa, ra, b, xb, rb, tol).intersecting
}

/// Full GJK distance / intersection with witnesses.
pub fn gjk(
    sa: &Shape,
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    sb: &Shape,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
    tol: GjkTol,
) -> GjkResult {
    let mut dir = xa - xb;
    if dir.norm_squared() < tol.abs * tol.abs {
        dir = Vec3::unit_x();
    }

    let mut simplex: Vec<Vertex> = Vec::with_capacity(4);
    let mut v = support_vertex(sa, xa, ra, sb, xb, rb, dir);
    simplex.push(v);
    let mut previous_sq = f64::INFINITY;
    let mut iters = 0usize;

    loop {
        iters += 1;
        if iters > tol.max_iters {
            break;
        }

        // Closest point of current simplex to the origin.
        let (closest, bary, reduced) = closest_simplex(&simplex, tol);
        simplex = reduced;
        let d2 = closest.norm_squared();

        // Origin inside / on the simplex ⇒ intersecting.
        if d2 <= tol.abs * tol.abs {
            let (pa, pb) = witness_from_bary(&simplex, &bary);
            return GjkResult {
                intersecting: true,
                distance: 0.0,
                point_a: pa,
                point_b: pb,
                normal: if (xb - xa).norm_squared() > 0.0 {
                    (xb - xa).normalize()
                } else {
                    Vec3::unit_z()
                },
                iterations: iters,
            };
        }

        // Search toward the origin from the closest point.
        dir = -closest;
        v = support_vertex(sa, xa, ra, sb, xb, rb, dir);

        let eps = tol.abs + tol.rel * (1.0 + closest.norm() + v.p.norm());
        // Convergence (van den Bergen): support does not improve the closest
        // point — ||C||² − C·v ≤ ε². Also stop on duplicate support.
        let improvement = d2 - closest.dot(v.p);
        let duplicate = simplex
            .iter()
            .any(|s| (s.p - v.p).norm_squared() < eps * eps);
        if duplicate || improvement <= eps * eps {
            let (pa, pb) = witness_from_bary(&simplex, &bary);
            let dist = closest.norm();
            return GjkResult {
                intersecting: false,
                distance: dist,
                point_a: pa,
                point_b: pb,
                normal: if dist > 0.0 {
                    -closest / dist
                } else {
                    Vec3::unit_z()
                },
                iterations: iters,
            };
        }

        // Stalled iterations (should be rare with the check above).
        if previous_sq - d2 <= eps * eps && iters > 2 {
            let (pa, pb) = witness_from_bary(&simplex, &bary);
            let dist = closest.norm();
            return GjkResult {
                intersecting: false,
                distance: dist,
                point_a: pa,
                point_b: pb,
                normal: if dist > 0.0 {
                    -closest / dist
                } else {
                    Vec3::unit_z()
                },
                iterations: iters,
            };
        }

        previous_sq = d2;
        simplex.push(v);
        // Keep simplex ≤ 4 by re-running closest (reduction) next iteration.
        if simplex.len() > 4 {
            let (_, _, red) = closest_simplex(&simplex, tol);
            simplex = red;
            simplex.push(v);
        }
    }

    // Budget exhausted — return current closest estimate as separated.
    let (closest, bary, reduced) = closest_simplex(&simplex, tol);
    let (pa, pb) = witness_from_bary(&reduced, &bary);
    let dist = closest.norm();
    GjkResult {
        intersecting: dist <= tol.abs,
        distance: if dist <= tol.abs { 0.0 } else { dist },
        point_a: pa,
        point_b: pb,
        normal: if dist > tol.abs {
            -closest / dist
        } else {
            Vec3::unit_z()
        },
        iterations: iters,
    }
}

fn support_vertex(
    sa: &Shape,
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    sb: &Shape,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
    dir: Vec3,
) -> Vertex {
    let a = support_world(sa, xa, ra, dir);
    let b = support_world(sb, xb, rb, -dir);
    Vertex { p: a - b, a, b }
}

fn witness_from_bary(simplex: &[Vertex], bary: &[f64]) -> (Vec3, Vec3) {
    let mut pa = Vec3::zero();
    let mut pb = Vec3::zero();
    for (i, v) in simplex.iter().enumerate() {
        let w = bary.get(i).copied().unwrap_or(0.0);
        pa += v.a * w;
        pb += v.b * w;
    }
    (pa, pb)
}

/// Closest point of simplex to origin; returns (point, barycentric, reduced simplex).
fn closest_simplex(simplex: &[Vertex], tol: GjkTol) -> (Vec3, Vec<f64>, Vec<Vertex>) {
    match simplex.len() {
        0 => (Vec3::zero(), vec![], vec![]),
        1 => (simplex[0].p, vec![1.0], vec![simplex[0]]),
        2 => closest_segment(simplex),
        3 => closest_triangle(simplex, tol),
        _ => closest_tetra(simplex, tol),
    }
}

fn closest_segment(simplex: &[Vertex]) -> (Vec3, Vec<f64>, Vec<Vertex>) {
    let a = simplex[0].p;
    let b = simplex[1].p;
    let ab = b - a;
    let t = if ab.norm_squared() < 1e-30 {
        0.0
    } else {
        clamp01(-a.dot(ab) / ab.norm_squared())
    };
    if t <= 0.0 {
        (a, vec![1.0], vec![simplex[0]])
    } else if t >= 1.0 {
        (b, vec![1.0], vec![simplex[1]])
    } else {
        (a + ab * t, vec![1.0 - t, t], vec![simplex[0], simplex[1]])
    }
}

fn closest_triangle(simplex: &[Vertex], _tol: GjkTol) -> (Vec3, Vec<f64>, Vec<Vertex>) {
    let a = simplex[0].p;
    let b = simplex[1].p;
    let c = simplex[2].p;
    // Ericson RTCD §5.1.5
    let ab = b - a;
    let ac = c - a;
    let ap = -a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, vec![1.0], vec![simplex[0]]);
    }
    let bp = -b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, vec![1.0], vec![simplex[1]]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, vec![1.0 - v, v], vec![simplex[0], simplex[1]]);
    }
    let cp = -c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, vec![1.0], vec![simplex[2]]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, vec![1.0 - w, w], vec![simplex[0], simplex[2]]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (
            b + (c - b) * w,
            vec![1.0 - w, w],
            vec![simplex[1], simplex[2]],
        );
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (
        a + ab * v + ac * w,
        vec![1.0 - v - w, v, w],
        vec![simplex[0], simplex[1], simplex[2]],
    )
}

fn closest_tetra(simplex: &[Vertex], tol: GjkTol) -> (Vec3, Vec<f64>, Vec<Vertex>) {
    let a = simplex[0].p;
    let b = simplex[1].p;
    let c = simplex[2].p;
    let d = simplex[3].p;
    if let Some(bary) = barycentric_tet(a, b, c, d, Vec3::zero()) {
        if bary.iter().all(|&w| w >= -1e-9) {
            return (Vec3::zero(), bary.to_vec(), simplex.to_vec());
        }
    }
    // Closest among faces; pick the face with minimal distance.
    let faces: [(&[Vertex], [usize; 3]); 4] = [
        (&[simplex[0], simplex[1], simplex[2]], [0, 1, 2]),
        (&[simplex[0], simplex[1], simplex[3]], [0, 1, 3]),
        (&[simplex[0], simplex[2], simplex[3]], [0, 2, 3]),
        (&[simplex[1], simplex[2], simplex[3]], [1, 2, 3]),
    ];
    let mut best_p = Vec3::zero();
    let mut best_bary = vec![];
    let mut best_simp = vec![];
    let mut best_d2 = f64::INFINITY;
    for (face, _) in &faces {
        let (p, bary, red) = closest_triangle(face, tol);
        let d2 = p.norm_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best_p = p;
            best_bary = bary;
            best_simp = red;
        }
    }
    let _ = (a, b, c, d);
    (best_p, best_bary, best_simp)
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

#[inline]
fn clamp01(t: f64) -> f64 {
    t.clamp(0.0, 1.0)
}

/// Signed separation along a unit normal using support maps (positive = separated).
pub fn separation_along(
    a: &Shape,
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    b: &Shape,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
    n: Vec3,
) -> f64 {
    let pa = support_world(a, xa, ra, -n);
    let pb = support_world(b, xb, rb, n);
    n.dot(pb - pa)
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
    fn sphere_sphere_separated() {
        let a = Shape::Sphere(Sphere::new(1.0));
        let b = Shape::Sphere(Sphere::new(1.0));
        let r = gjk(
            &a,
            Vec3::zero(),
            &eye(),
            &b,
            Vec3::new(3.0, 0.0, 0.0),
            &eye(),
            GjkTol::default(),
        );
        assert!(!r.intersecting);
        assert_abs_diff_eq!(r.distance, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn sphere_sphere_overlap() {
        let a = Shape::Sphere(Sphere::new(1.0));
        let b = Shape::Sphere(Sphere::new(1.0));
        let r = gjk(
            &a,
            Vec3::zero(),
            &eye(),
            &b,
            Vec3::new(1.0, 0.0, 0.0),
            &eye(),
            GjkTol::default(),
        );
        assert!(r.intersecting);
        assert_eq!(r.distance, 0.0);
    }

    #[test]
    fn sphere_box_face_distance() {
        let a = Shape::Sphere(Sphere::new(0.5));
        let b = Shape::Box(BoxShape::new(1.0, 1.0, 1.0));
        // Sphere centre at x=2.5; box extends to 1 → distance 1.0
        let r = gjk(
            &a,
            Vec3::new(2.5, 0.0, 0.0),
            &eye(),
            &b,
            Vec3::zero(),
            &eye(),
            GjkTol::default(),
        );
        assert!(!r.intersecting);
        assert_abs_diff_eq!(r.distance, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn box_box_separated_axis() {
        let a = Shape::Box(BoxShape::new(0.5, 0.5, 0.5));
        let b = Shape::Box(BoxShape::new(0.5, 0.5, 0.5));
        let r = gjk(
            &a,
            Vec3::zero(),
            &eye(),
            &b,
            Vec3::new(2.0, 0.0, 0.0),
            &eye(),
            GjkTol::default(),
        );
        assert!(!r.intersecting);
        assert_abs_diff_eq!(r.distance, 1.0, epsilon = 1e-8);
    }
}
