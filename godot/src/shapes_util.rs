//! Build pe-collision shapes from Godot exported parameters.

use godot::prelude::*;
use pe_collision::hull::ConvexHull;
use pe_collision::shapes::{BoxShape, Capsule, Shape, Sphere};
use pe_math::Vec3 as PeVec3;

use crate::convert::{shape_kind, to_pe_vec3};

/// Try to build a collision shape from exported kind + parameters.
pub(crate) fn build_shape(
    kind: i64,
    radius: f64,
    hx: f64,
    hy: f64,
    hz: f64,
    capsule_half_height: f64,
    convex_points: &PackedVector3Array,
) -> Option<Shape> {
    match kind {
        shape_kind::SPHERE => {
            if !(radius.is_finite() && radius >= 0.0) {
                return None;
            }
            Some(Shape::Sphere(Sphere::new(radius)))
        }
        shape_kind::BOX => {
            if !(hx.is_finite()
                && hy.is_finite()
                && hz.is_finite()
                && hx >= 0.0
                && hy >= 0.0
                && hz >= 0.0)
            {
                return None;
            }
            Some(Shape::Box(BoxShape::new(hx, hy, hz)))
        }
        shape_kind::CAPSULE => {
            if !(radius.is_finite()
                && capsule_half_height.is_finite()
                && radius >= 0.0
                && capsule_half_height >= 0.0)
            {
                return None;
            }
            Some(Shape::Capsule(Capsule::new(radius, capsule_half_height)))
        }
        shape_kind::CONVEX => {
            let n = convex_points.len();
            if n < 4 {
                return None;
            }
            let mut pts = Vec::with_capacity(n);
            for i in 0..n {
                pts.push(to_pe_vec3(convex_points.get(i).unwrap_or(Vector3::ZERO)));
            }
            ConvexHull::from_points(&pts).ok().map(Shape::Convex)
        }
        _ => None,
    }
}

/// Default unit box half-extents helper for demos.
#[allow(dead_code)]
pub(crate) fn unit_box() -> Shape {
    Shape::Box(BoxShape::new(0.5, 0.5, 0.5))
}

#[allow(dead_code)]
pub(crate) fn pe_zero() -> PeVec3 {
    PeVec3::zero()
}
