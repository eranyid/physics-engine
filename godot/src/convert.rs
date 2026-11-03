//! Conversions between Godot and pe-math types (f32 boundary ↔ f64 engine).

use godot::prelude::*;
use pe_math::{Quat as PeQuat, Vec3 as PeVec3};

#[inline]
pub(crate) fn to_pe_vec3(v: Vector3) -> PeVec3 {
    PeVec3::new(v.x as f64, v.y as f64, v.z as f64)
}

#[inline]
pub(crate) fn to_godot_vec3(v: PeVec3) -> Vector3 {
    Vector3::new(v.x as f32, v.y as f32, v.z as f32)
}

#[inline]
pub(crate) fn to_pe_quat(q: Quaternion) -> PeQuat {
    // Godot: (x,y,z,w); pe: (w,x,y,z)
    PeQuat::new(q.w as f64, q.x as f64, q.y as f64, q.z as f64).normalize()
}

#[inline]
pub(crate) fn to_godot_quat(q: PeQuat) -> Quaternion {
    Quaternion::new(q.x as f32, q.y as f32, q.z as f32, q.w as f32)
}

/// Shape kind exported to GDScript.
pub(crate) mod shape_kind {
    pub const SPHERE: i64 = 0;
    pub const BOX: i64 = 1;
    pub const CAPSULE: i64 = 2;
    pub const CONVEX: i64 = 3;
}

/// Body type exported to GDScript.
pub(crate) mod body_type {
    pub const DYNAMIC: i64 = 0;
    pub const STATIC: i64 = 1;
    pub const KINEMATIC: i64 = 2;
}
