//! Hit-scan and swept segment tests against pe-collision shapes (read-only).
//!
//! Tunneling-free segment tests: a finite segment (or swept point) is tested
//! against spheres, OBBs, and capsules. Implemented in this crate so
//! `pe-collision` need not change.

use pe_collision::{BoxShape, Capsule, Sphere};
use pe_math::libm_ops;
use pe_math::Vec3;

/// Hit result for a segment / ray query.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    /// Hit position [m].
    pub point: Vec3,
    /// Outward unit normal at the hit.
    pub normal: Vec3,
    /// Parametric distance along the segment ∈ [0,1] (or ray t ≥ 0).
    pub t: f64,
}

/// Ray–sphere. Ray: `o + t d` with `t ≥ 0`, `d` not necessarily unit.
/// Sphere center `c` [m], radius `r` [m].
pub fn ray_sphere(o: Vec3, d: Vec3, c: Vec3, r: f64) -> Option<Hit> {
    let oc = o - c;
    let a = d.norm_squared();
    if a < 1e-30 {
        return None;
    }
    let b = 2.0 * oc.dot(d);
    let cc = oc.norm_squared() - r * r;
    let disc = b * b - 4.0 * a * cc;
    if disc < 0.0 {
        return None;
    }
    let s = libm_ops::sqrt(disc);
    let t0 = (-b - s) / (2.0 * a);
    let t1 = (-b + s) / (2.0 * a);
    let t = if t0 >= 0.0 {
        t0
    } else if t1 >= 0.0 {
        t1
    } else {
        return None;
    };
    let point = o + d * t;
    let normal = (point - c).normalize();
    Some(Hit { point, normal, t })
}

/// Segment–sphere: segment `a → b` (inclusive). Returns earliest hit.
pub fn segment_sphere(a: Vec3, b: Vec3, c: Vec3, r: f64) -> Option<Hit> {
    let d = b - a;
    let hit = ray_sphere(a, d, c, r)?;
    if hit.t < 0.0 || hit.t > 1.0 {
        return None;
    }
    // Reparametrize note: t is already in segment parameter if d = b−a.
    Some(hit)
}

/// Segment vs [`Sphere`] shape at world center `x`.
pub fn segment_vs_sphere(a: Vec3, b: Vec3, x: Vec3, sphere: Sphere) -> Option<Hit> {
    segment_sphere(a, b, x, sphere.radius)
}

/// Segment vs OBB (`BoxShape` at `x` with rotation `r` row-major).
pub fn segment_vs_box(
    a: Vec3,
    b: Vec3,
    x: Vec3,
    r: &[[f64; 3]; 3],
    box_s: BoxShape,
) -> Option<Hit> {
    // Transform segment into box local frame.
    let rt = transpose(r);
    let al = mul(rt, a - x);
    let bl = mul(rt, b - x);
    let h = box_s.half_extents;
    let (tmin, tmax, n_local) = clip_aabb(al, bl, -h, h)?;
    if tmax < 0.0 || tmin > 1.0 {
        return None;
    }
    let t = if tmin >= 0.0 { tmin } else { tmax };
    if !(0.0..=1.0).contains(&t) {
        return None;
    }
    let point = a + (b - a) * t;
    let normal = mul(*r, n_local).normalize();
    Some(Hit { point, normal, t })
}

/// Segment vs capsule (world pose).
pub fn segment_vs_capsule(
    a: Vec3,
    b: Vec3,
    x: Vec3,
    r: &[[f64; 3]; 3],
    cap: Capsule,
) -> Option<Hit> {
    let (p0, p1) = cap.segment(x, r);
    // Capsule = segment swept by radius: find closest approach of two segments,
    // then sphere test at the capsule point. For robustness use dense samples
    // along the capsule axis + end spheres (deterministic, tunneling-free for
    // the segment query when combined with endpoint spheres).
    let mut best: Option<Hit> = None;
    let n = 16usize;
    for i in 0..=n {
        let s = i as f64 / n as f64;
        let c = p0 + (p1 - p0) * s;
        if let Some(h) = segment_sphere(a, b, c, cap.radius) {
            if best.map(|b| h.t < b.t).unwrap_or(true) {
                best = Some(h);
            }
        }
    }
    best
}

/// Swept point (segment) against a list of colliders; earliest hit wins.
#[derive(Clone, Debug)]
pub enum ShapeRef {
    /// Sphere.
    Sphere {
        /// World center [m].
        x: Vec3,
        /// Shape.
        s: Sphere,
    },
    /// Oriented box.
    Box {
        /// World center [m].
        x: Vec3,
        /// Rotation row-major.
        r: [[f64; 3]; 3],
        /// Shape.
        s: BoxShape,
    },
    /// Capsule.
    Capsule {
        /// World center [m].
        x: Vec3,
        /// Rotation row-major.
        r: [[f64; 3]; 3],
        /// Shape.
        s: Capsule,
    },
}

/// Segment vs a single [`ShapeRef`].
pub fn segment_vs_shape(a: Vec3, b: Vec3, shape: &ShapeRef) -> Option<Hit> {
    match shape {
        ShapeRef::Sphere { x, s } => segment_vs_sphere(a, b, *x, *s),
        ShapeRef::Box { x, r, s } => segment_vs_box(a, b, *x, r, *s),
        ShapeRef::Capsule { x, r, s } => segment_vs_capsule(a, b, *x, r, *s),
    }
}

/// Earliest segment hit among shapes, skipping shape index `exclude`.
pub fn segment_vs_shapes_excluding(
    a: Vec3,
    b: Vec3,
    shapes: &[ShapeRef],
    exclude: usize,
) -> Option<(usize, Hit)> {
    let mut best: Option<(usize, Hit)> = None;
    for (i, sh) in shapes.iter().enumerate() {
        if i == exclude {
            continue;
        }
        if let Some(hit) = segment_vs_shape(a, b, sh) {
            if best.map(|(_, b)| hit.t < b.t).unwrap_or(true) {
                best = Some((i, hit));
            }
        }
    }
    best
}

/// Exit point of a straight path through `shape`.
///
/// Starting at `entry` (on or inside the shape) and travelling along unit
/// direction `dir`, returns `(exit_point, outward_normal_at_exit, chord_m)`.
/// `max_len` [m] bounds the search (use something larger than the shape).
/// Implemented as a reverse cast from `entry + dir·max_len` back to `entry`.
pub fn shape_exit(
    entry: Vec3,
    dir: Vec3,
    shape: &ShapeRef,
    max_len: f64,
) -> Option<(Vec3, Vec3, f64)> {
    let d = dir.normalize();
    let far = entry + d * max_len;
    let hit = segment_vs_shape(far, entry, shape)?;
    let chord = (hit.point - entry).dot(d).max(0.0);
    Some((hit.point, hit.normal, chord))
}

/// Conservative bounding radius [m] of a [`ShapeRef`] (for exit searches).
pub fn shape_bounding_radius(shape: &ShapeRef) -> f64 {
    match shape {
        ShapeRef::Sphere { s, .. } => s.radius,
        ShapeRef::Box { s, .. } => s.half_extents.norm(),
        ShapeRef::Capsule { s, .. } => s.radius + s.half_height,
    }
}

/// Earliest segment hit among shapes; also returns shape index.
pub fn segment_vs_shapes(a: Vec3, b: Vec3, shapes: &[ShapeRef]) -> Option<(usize, Hit)> {
    let mut best: Option<(usize, Hit)> = None;
    for (i, sh) in shapes.iter().enumerate() {
        let h = segment_vs_shape(a, b, sh);
        if let Some(hit) = h {
            if best.map(|(_, b)| hit.t < b.t).unwrap_or(true) {
                best = Some((i, hit));
            }
        }
    }
    best
}

fn transpose(r: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [r[0][0], r[1][0], r[2][0]],
        [r[0][1], r[1][1], r[2][1]],
        [r[0][2], r[1][2], r[2][2]],
    ]
}

fn mul(r: [[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
        r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
        r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
    )
}

/// Liang–Barsky style AABB clip in local coordinates; returns (t_enter, t_exit, normal_at_enter).
fn clip_aabb(a: Vec3, b: Vec3, lo: Vec3, hi: Vec3) -> Option<(f64, f64, Vec3)> {
    let d = b - a;
    let mut t0 = 0.0;
    let mut t1 = 1.0;
    let mut n_enter = Vec3::zero();
    let planes = [
        (0, -1.0, a.x - lo.x, -d.x), // x = lo
        (0, 1.0, hi.x - a.x, d.x),
        (1, -1.0, a.y - lo.y, -d.y),
        (1, 1.0, hi.y - a.y, d.y),
        (2, -1.0, a.z - lo.z, -d.z),
        (2, 1.0, hi.z - a.z, d.z),
    ];
    for (axis, sign, p, q) in planes {
        // Entering when moving toward the plane from outside.
        if q.abs() < 1e-15 {
            if p < 0.0 {
                return None;
            }
            continue;
        }
        let t = p / q;
        if q > 0.0 {
            // exiting
            if t < t1 {
                t1 = t;
            }
        } else {
            // entering
            if t > t0 {
                t0 = t;
                n_enter = match axis {
                    0 => Vec3::new(sign, 0.0, 0.0),
                    1 => Vec3::new(0.0, sign, 0.0),
                    _ => Vec3::new(0.0, 0.0, sign),
                };
            }
        }
        if t0 > t1 {
            return None;
        }
    }
    Some((t0, t1, n_enter))
}
