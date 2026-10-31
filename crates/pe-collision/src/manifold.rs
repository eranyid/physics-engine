//! Contact / manifold point types with stable feature ids for warm starting.

use pe_math::Vec3;

/// Stable geometric feature identifier on a manifold point.
///
/// M3 warm-start matching uses the sequential contact index within a body pair
/// (matching the Python oracle key `(a, b, k)`). Dedicated geometric feature
/// encoding (face/edge/vertex ids for GJK/SAT caching) is the extension hook for
/// later milestones — keep this type on every manifold point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct FeatureId(pub u32);

/// Narrow-phase contact: world point, unit normal A→B, penetration depth [m].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Contact location in world frame [m].
    pub point: Vec3,
    /// Unit normal pointing from shape A toward shape B.
    pub normal: Vec3,
    /// Penetration depth [m] (`> 0` means overlapping).
    pub depth: f64,
    /// Stable feature id for warm starting / caching.
    pub feature: FeatureId,
}

impl Contact {
    /// Construct a contact without a feature id (assigned later by the solver/detect).
    pub fn new(point: Vec3, normal: Vec3, depth: f64) -> Self {
        Self {
            point,
            normal,
            depth,
            feature: FeatureId(0),
        }
    }

    /// Flip the normal (A↔B swap).
    pub fn flipped(self) -> Self {
        Self {
            point: self.point,
            normal: -self.normal,
            depth: self.depth,
            feature: self.feature,
        }
    }
}

/// Contact attached to a body pair (after broad+narrow).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactPoint {
    /// Body / collider index A (dynamic bodies first, then statics).
    pub a: usize,
    /// Body / collider index B.
    pub b: usize,
    /// World contact point [m].
    pub point: Vec3,
    /// Unit normal A→B.
    pub normal: Vec3,
    /// Penetration depth [m].
    pub depth: f64,
    /// Feature id (warm-start key component).
    pub feature: FeatureId,
}

/// Manifold point with warm-start cache fields (design surface for future solvers).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManifoldPoint {
    /// World contact point [m].
    pub point: Vec3,
    /// Unit normal A→B.
    pub normal: Vec3,
    /// Penetration depth [m].
    pub depth: f64,
    /// Stable feature id.
    pub feature: FeatureId,
    /// Accumulated normal impulse (warm start).
    pub normal_impulse: f64,
    /// Accumulated tangent impulse vector (warm start).
    pub tangent_impulse: Vec3,
}
