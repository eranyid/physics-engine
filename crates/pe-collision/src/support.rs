//! Support maps for GJK/EPA (sphere, box, capsule, convex, rounded).
//!
//! Clean-room from Gino van den Bergen, *Collision Detection in Interactive
//! 3D Environments* (2003), and Erin Catto GDC notes. No third-party source.

use pe_math::Vec3;

use crate::hull::ConvexHull;
use crate::shapes::{BoxShape, Capsule, Plane, Shape, Sphere};

/// World-space support: point of the shape extremal in direction `dir`
/// (direction need not be unit; zero dir returns the origin pose).
pub fn support_world(shape: &Shape, x: Vec3, r: &[[f64; 3]; 3], dir: Vec3) -> Vec3 {
    if dir.norm_squared() < 1e-30 {
        return x;
    }
    match shape {
        Shape::Sphere(s) => support_sphere(s, x, dir),
        Shape::Box(b) => support_box(b, x, r, dir),
        Shape::Capsule(c) => support_capsule(c, x, r, dir),
        Shape::Plane(p) => support_plane(p, dir),
        Shape::Convex(h) => h.support(x, r, dir),
        Shape::Rounded(rd) => {
            let core = support_world(&rd.inner, x, r, dir);
            core + dir.normalize() * rd.radius
        }
    }
}

/// Support of sphere centred at `x`.
#[inline]
pub fn support_sphere(s: &Sphere, x: Vec3, dir: Vec3) -> Vec3 {
    x + dir.normalize() * s.radius
}

/// Support of an OBB.
pub fn support_box(b: &BoxShape, x: Vec3, r: &[[f64; 3]; 3], dir: Vec3) -> Vec3 {
    let lx = r[0][0] * dir.x + r[1][0] * dir.y + r[2][0] * dir.z;
    let ly = r[0][1] * dir.x + r[1][1] * dir.y + r[2][1] * dir.z;
    let lz = r[0][2] * dir.x + r[1][2] * dir.y + r[2][2] * dir.z;
    let hx = if lx >= 0.0 {
        b.half_extents.x
    } else {
        -b.half_extents.x
    };
    let hy = if ly >= 0.0 {
        b.half_extents.y
    } else {
        -b.half_extents.y
    };
    let hz = if lz >= 0.0 {
        b.half_extents.z
    } else {
        -b.half_extents.z
    };
    let local = Vec3::new(hx, hy, hz);
    Vec3::new(
        x.x + r[0][0] * local.x + r[0][1] * local.y + r[0][2] * local.z,
        x.y + r[1][0] * local.x + r[1][1] * local.y + r[1][2] * local.z,
        x.z + r[2][0] * local.x + r[2][1] * local.y + r[2][2] * local.z,
    )
}

/// Capsule = segment ⊕ sphere.
pub fn support_capsule(c: &Capsule, x: Vec3, r: &[[f64; 3]; 3], dir: Vec3) -> Vec3 {
    let axis = Vec3::new(r[0][2], r[1][2], r[2][2]);
    let p0 = x - axis * c.half_height;
    let p1 = x + axis * c.half_height;
    let p = if dir.dot(p0) > dir.dot(p1) { p0 } else { p1 };
    p + dir.normalize() * c.radius
}

/// Plane support: unbounded in the tangent plane; for GJK vs planes we do not
/// use this path (dedicated plane pairs). Returns a far point along −n when
/// querying into the solid.
fn support_plane(p: &Plane, dir: Vec3) -> Vec3 {
    let n = p.n();
    // Project onto plane then push far in tangent directions — practically
    // unused; callers should keep plane pairs on the dedicated path.
    let t = dir - n * dir.dot(n);
    let origin = n * p.offset;
    if t.norm_squared() < 1e-30 {
        origin - n * 1.0e6
    } else {
        origin + t.normalize() * 1.0e6
    }
}

/// Minkowski difference support: `S_A(d) − S_B(−d)`.
#[inline]
pub fn support_minkowski(
    a: &Shape,
    xa: Vec3,
    ra: &[[f64; 3]; 3],
    b: &Shape,
    xb: Vec3,
    rb: &[[f64; 3]; 3],
    dir: Vec3,
) -> Vec3 {
    support_world(a, xa, ra, dir) - support_world(b, xb, rb, -dir)
}

/// Maximum radial extent from COM (for CA angular bound) [m].
pub fn bounding_radius(shape: &Shape) -> f64 {
    match shape {
        Shape::Sphere(s) => s.radius,
        Shape::Box(b) => b.half_extents.norm(),
        Shape::Capsule(c) => c.half_height + c.radius,
        Shape::Plane(_) => f64::INFINITY,
        Shape::Convex(h) => {
            let mut m: f64 = 0.0;
            for v in &h.vertices {
                m = m.max(v.norm());
            }
            m + h.radius
        }
        Shape::Rounded(rd) => bounding_radius(&rd.inner) + rd.radius,
    }
}

/// Helper to wrap a convex hull as a shape with optional extra radius.
pub fn rounded_hull(hull: ConvexHull, radius: f64) -> Shape {
    if radius == 0.0 && hull.radius == 0.0 {
        Shape::Convex(hull)
    } else if hull.radius == 0.0 {
        Shape::Rounded(RoundedShape {
            inner: Box::new(Shape::Convex(hull)),
            radius,
        })
    } else {
        let mut h = hull;
        h.radius += radius;
        Shape::Convex(h)
    }
}

/// Minkowski sum of an inner shape with a sphere of `radius` [m].
#[derive(Clone, Debug, PartialEq)]
pub struct RoundedShape {
    /// Core shape (must itself be convex; not a plane).
    pub inner: Box<Shape>,
    /// Sweep radius [m].
    pub radius: f64,
}

impl RoundedShape {
    /// Construct a rounded shape; `radius` must be ≥ 0.
    pub fn new(inner: Shape, radius: f64) -> Self {
        assert!(radius >= 0.0 && radius.is_finite());
        Self {
            inner: Box::new(inner),
            radius,
        }
    }
}
