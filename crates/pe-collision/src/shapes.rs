//! Collision shapes (body frame, centred on the COM) and mass properties (SI).

use core::f64::consts::PI;

use pe_math::libm_ops::fabs;
use pe_math::Vec3;

use crate::hull::ConvexHull;
use crate::support::RoundedShape;

/// Axis-aligned bounding box in world frame [m].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Lower corner [m].
    pub lo: Vec3,
    /// Upper corner [m].
    pub hi: Vec3,
}

/// Sphere of radius `radius` [m], centred on the body COM.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    /// Radius [m].
    pub radius: f64,
}

impl Sphere {
    /// Construct a sphere [m].
    pub fn new(radius: f64) -> Self {
        Self { radius }
    }

    /// Volume [m³].
    pub fn volume(self) -> f64 {
        4.0 / 3.0 * PI * self.radius * self.radius * self.radius
    }

    /// Inertia about COM [kg m²]: `(2/5) m r² I`.
    pub fn inertia(self, mass: f64) -> [[f64; 3]; 3] {
        let i = 0.4 * mass * self.radius * self.radius;
        [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]]
    }

    /// World AABB at position `x` [m] (orientation unused).
    pub fn aabb(self, x: Vec3, _r: &[[f64; 3]; 3]) -> Aabb {
        let r = Vec3::new(self.radius, self.radius, self.radius);
        Aabb {
            lo: x - r,
            hi: x + r,
        }
    }
}

/// Oriented box with half-extents `(hx, hy, hz)` [m] about the COM.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxShape {
    /// Half-extents [m].
    pub half_extents: Vec3,
}

impl BoxShape {
    /// Construct from half-extents [m].
    pub fn new(hx: f64, hy: f64, hz: f64) -> Self {
        Self {
            half_extents: Vec3::new(hx, hy, hz),
        }
    }

    /// Half-extents as [`Vec3`].
    pub fn h(self) -> Vec3 {
        self.half_extents
    }

    /// Volume [m³].
    pub fn volume(self) -> f64 {
        8.0 * self.half_extents.x * self.half_extents.y * self.half_extents.z
    }

    /// Inertia about COM [kg m²] for a solid box of full edges `2h`.
    pub fn inertia(self, mass: f64) -> [[f64; 3]; 3] {
        let a = 2.0 * self.half_extents.x;
        let b = 2.0 * self.half_extents.y;
        let c = 2.0 * self.half_extents.z;
        [
            [mass * (b * b + c * c) / 12.0, 0.0, 0.0],
            [0.0, mass * (a * a + c * c) / 12.0, 0.0],
            [0.0, 0.0, mass * (a * a + b * b) / 12.0],
        ]
    }

    /// World AABB of the OBB at `x` with rotation `r` (row-major).
    pub fn aabb(self, x: Vec3, r: &[[f64; 3]; 3]) -> Aabb {
        let h = self.half_extents;
        // e = |R| @ h
        let e = Vec3::new(
            fabs(r[0][0]) * h.x + fabs(r[0][1]) * h.y + fabs(r[0][2]) * h.z,
            fabs(r[1][0]) * h.x + fabs(r[1][1]) * h.y + fabs(r[1][2]) * h.z,
            fabs(r[2][0]) * h.x + fabs(r[2][1]) * h.y + fabs(r[2][2]) * h.z,
        );
        Aabb {
            lo: x - e,
            hi: x + e,
        }
    }

    /// Eight corner vertices in world frame [m].
    pub fn vertices(self, x: Vec3, r: &[[f64; 3]; 3]) -> [Vec3; 8] {
        let h = self.half_extents;
        let mut out = [Vec3::zero(); 8];
        let mut k = 0;
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    let local = Vec3::new(sx * h.x, sy * h.y, sz * h.z);
                    // x + R @ local
                    out[k] = x + Vec3::new(
                        r[0][0] * local.x + r[0][1] * local.y + r[0][2] * local.z,
                        r[1][0] * local.x + r[1][1] * local.y + r[1][2] * local.z,
                        r[2][0] * local.x + r[2][1] * local.y + r[2][2] * local.z,
                    );
                    k += 1;
                }
            }
        }
        out
    }
}

/// Capsule: segment of half-length `half_height` along body z, swept by `radius` [m].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule {
    /// Sweep radius [m].
    pub radius: f64,
    /// Half segment length along body +z [m].
    pub half_height: f64,
}

impl Capsule {
    /// Construct a capsule [m].
    pub fn new(radius: f64, half_height: f64) -> Self {
        Self {
            radius,
            half_height,
        }
    }

    /// Volume [m³]: cylinder + two hemispheres.
    pub fn volume(self) -> f64 {
        let r = self.radius;
        let h = 2.0 * self.half_height;
        PI * r * r * h + 4.0 / 3.0 * PI * r * r * r
    }

    /// Inertia about COM [kg m²] (cylinder + hemisphere formula matching the Python oracle).
    pub fn inertia(self, mass: f64) -> [[f64; 3]; 3] {
        let r = self.radius;
        let h = 2.0 * self.half_height;
        let vc = PI * r * r * h;
        let vs = 4.0 / 3.0 * PI * r * r * r;
        let mc = mass * vc / (vc + vs);
        let ms = mass * vs / (vc + vs);
        let iz = mc * r * r / 2.0 + ms * 2.0 * r * r / 5.0;
        let ix = mc * (h * h / 12.0 + r * r / 4.0)
            + ms * (2.0 * r * r / 5.0 + h * h / 4.0 + 3.0 * h * r / 8.0);
        [[ix, 0.0, 0.0], [0.0, ix, 0.0], [0.0, 0.0, iz]]
    }

    /// World-space segment endpoints `(p, q)` [m].
    pub fn segment(self, x: Vec3, r: &[[f64; 3]; 3]) -> (Vec3, Vec3) {
        // R[:, 2] * half_height
        let a = Vec3::new(r[0][2], r[1][2], r[2][2]) * self.half_height;
        (x - a, x + a)
    }

    /// World AABB.
    pub fn aabb(self, x: Vec3, r: &[[f64; 3]; 3]) -> Aabb {
        let (p, q) = self.segment(x, r);
        let rad = Vec3::new(self.radius, self.radius, self.radius);
        Aabb {
            lo: Vec3::new(p.x.min(q.x), p.y.min(q.y), p.z.min(q.z)) - rad,
            hi: Vec3::new(p.x.max(q.x), p.y.max(q.y), p.z.max(q.z)) + rad,
        }
    }
}

/// Static half-space `{p : n·p <= offset}`; `n` points out of the solid (world frame).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// Outward unit normal (stored unnormalized; accessors renormalize).
    pub normal: Vec3,
    /// Plane offset [m]: `n · p = offset` on the surface.
    pub offset: f64,
}

impl Plane {
    /// Construct from a (possibly non-unit) normal and offset [m].
    pub fn new(normal: Vec3, offset: f64) -> Self {
        Self { normal, offset }
    }

    /// Unit outward normal.
    pub fn n(self) -> Vec3 {
        let v = self.normal;
        let len = v.norm();
        if len == 0.0 {
            Vec3::unit_z()
        } else {
            v / len
        }
    }

    /// Infinite AABB (planes participate in every broad-phase pair with dynamics).
    pub fn aabb(self, _x: Vec3, _r: &[[f64; 3]; 3]) -> Aabb {
        Aabb {
            lo: Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
            hi: Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
        }
    }
}

/// Contact material: restitution `e` and Coulomb friction `μ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// Coefficient of restitution (combined with `max`).
    pub restitution: f64,
    /// Coulomb friction (combined with `√(μa μb)`).
    pub friction: f64,
}

impl Material {
    /// Construct material parameters.
    pub fn new(restitution: f64, friction: f64) -> Self {
        Self {
            restitution,
            friction,
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            restitution: 0.0,
            friction: 0.5,
        }
    }
}

/// Tagged collision shape.
///
/// New M6 variants ([`Shape::Convex`], [`Shape::Rounded`]) are append-only so
/// existing `order_index` values for sphere/capsule/box/plane stay unchanged
/// (locked collision hashes).
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    /// Sphere.
    Sphere(Sphere),
    /// Oriented box.
    Box(BoxShape),
    /// Capsule.
    Capsule(Capsule),
    /// Infinite plane (typically static).
    Plane(Plane),
    /// Convex polyhedron (optional Minkowski radius on the hull).
    Convex(ConvexHull),
    /// Minkowski sum of an inner convex shape with a sphere.
    Rounded(RoundedShape),
}

impl Shape {
    /// Volume [m³]; planes return 0.
    pub fn volume(&self) -> f64 {
        match self {
            Self::Sphere(s) => s.volume(),
            Self::Box(b) => b.volume(),
            Self::Capsule(c) => c.volume(),
            Self::Plane(_) => 0.0,
            Self::Convex(h) => {
                // Sharp volume; rounded margin ignored (same note as hull inertia).
                h.volume()
            }
            Self::Rounded(rd) => {
                // Lower bound: core volume (exact sphere-swept volume not required for M6).
                rd.inner.volume()
            }
        }
    }

    /// Inertia about COM [kg m²] for the given mass; planes return zero.
    pub fn inertia(&self, mass: f64) -> [[f64; 3]; 3] {
        match self {
            Self::Sphere(s) => s.inertia(mass),
            Self::Box(b) => b.inertia(mass),
            Self::Capsule(c) => c.inertia(mass),
            Self::Plane(_) => [[0.0; 3]; 3],
            Self::Convex(h) => h.inertia(mass),
            Self::Rounded(rd) => rd.inner.inertia(mass),
        }
    }

    /// World AABB at pose `(x, R)`.
    pub fn aabb(&self, x: Vec3, r: &[[f64; 3]; 3]) -> Aabb {
        match self {
            Self::Sphere(s) => s.aabb(x, r),
            Self::Box(b) => b.aabb(x, r),
            Self::Capsule(c) => c.aabb(x, r),
            Self::Plane(p) => p.aabb(x, r),
            Self::Convex(h) => h.aabb(x, r),
            Self::Rounded(rd) => {
                let mut a = rd.inner.aabb(x, r);
                let m = Vec3::new(rd.radius, rd.radius, rd.radius);
                a.lo -= m;
                a.hi += m;
                a
            }
        }
    }

    /// Dispatch order index matching the Python oracle
    /// (`Sphere < Capsule < Box < Plane < Convex < Rounded`).
    pub fn order_index(&self) -> u8 {
        match self {
            Self::Sphere(_) => 0,
            Self::Capsule(_) => 1,
            Self::Box(_) => 2,
            Self::Plane(_) => 3,
            Self::Convex(_) => 4,
            Self::Rounded(_) => 5,
        }
    }

    /// True if this shape should use the GJK/EPA convex path against another convex.
    pub fn is_general_convex(&self) -> bool {
        matches!(self, Self::Convex(_) | Self::Rounded(_))
    }
}

/// `v · |Rᵀ L|` support radius term for an OBB half-extent vector.
#[inline]
pub(crate) fn support_extent(h: Vec3, r: &[[f64; 3]; 3], axis: Vec3) -> f64 {
    // h @ |Rᵀ @ L|
    let lx = fabs(r[0][0] * axis.x + r[1][0] * axis.y + r[2][0] * axis.z);
    let ly = fabs(r[0][1] * axis.x + r[1][1] * axis.y + r[2][1] * axis.z);
    let lz = fabs(r[0][2] * axis.x + r[1][2] * axis.y + r[2][2] * axis.z);
    h.x * lx + h.y * ly + h.z * lz
}

/// Column `i` of a row-major 3×3 as a [`Vec3`].
#[inline]
pub(crate) fn col(r: &[[f64; 3]; 3], i: usize) -> Vec3 {
    Vec3::new(r[0][i], r[1][i], r[2][i])
}

/// Clip scalar into `[lo, hi]`.
#[inline]
pub(crate) fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Sign matching NumPy: `sign(0) = 0`.
#[inline]
pub(crate) fn sign_np(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use pe_math::Quat;

    #[test]
    fn box_sphere_mass_properties() {
        let b = BoxShape::new(0.5, 1.0, 1.5);
        let i = b.inertia(3.0);
        // full edges 1,2,3
        assert_abs_diff_eq!(i[0][0], 3.0 * (4.0 + 9.0) / 12.0, epsilon = 1e-15);
        assert_eq!(b.volume(), 6.0);
        assert_abs_diff_eq!(Sphere::new(2.0).volume(), 32.0 / 3.0 * PI, epsilon = 1e-14);
    }

    #[test]
    fn rotated_box_aabb_is_tight() {
        let b = BoxShape::new(1.0, 0.5, 0.25);
        let r = Quat::from_axis_angle(Vec3::unit_z(), PI / 4.0).to_matrix();
        let aabb = b.aabb(Vec3::zero(), &r);
        let verts = b.vertices(Vec3::zero(), &r);
        let mut lo = verts[0];
        let mut hi = verts[0];
        for v in &verts[1..] {
            lo = Vec3::new(lo.x.min(v.x), lo.y.min(v.y), lo.z.min(v.z));
            hi = Vec3::new(hi.x.max(v.x), hi.y.max(v.y), hi.z.max(v.z));
        }
        assert_abs_diff_eq!(aabb.lo.x, lo.x, epsilon = 1e-12);
        assert_abs_diff_eq!(aabb.lo.y, lo.y, epsilon = 1e-12);
        assert_abs_diff_eq!(aabb.lo.z, lo.z, epsilon = 1e-12);
        assert_abs_diff_eq!(aabb.hi.x, hi.x, epsilon = 1e-12);
        assert_abs_diff_eq!(aabb.hi.y, hi.y, epsilon = 1e-12);
        assert_abs_diff_eq!(aabb.hi.z, hi.z, epsilon = 1e-12);
    }
}
