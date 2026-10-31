//! Hamilton quaternions, scalar-first `[w, x, y, z]`; active body→world rotation.

use crate::libm_ops;
use crate::vec3::Vec3;

/// Unit quaternion for orientation. Components: `w` (scalar), `x`, `y`, `z`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    /// Scalar (real) part
    pub w: f64,
    /// i component
    pub x: f64,
    /// j component
    pub y: f64,
    /// k component
    pub z: f64,
}

impl Quat {
    /// Construct from components `[w, x, y, z]` (not necessarily unit).
    #[inline]
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Identity rotation.
    #[inline]
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0)
    }

    /// From a length-4 slice `[w, x, y, z]`.
    #[inline]
    pub fn from_slice(q: &[f64; 4]) -> Self {
        Self::new(q[0], q[1], q[2], q[3])
    }

    /// Write components into `[w, x, y, z]`.
    #[inline]
    pub fn to_array(self) -> [f64; 4] {
        [self.w, self.x, self.y, self.z]
    }

    /// Euclidean norm of the 4-tuple (dimensionless). Uses portable [`libm_ops::sqrt`].
    #[inline]
    pub fn norm(self) -> f64 {
        libm_ops::sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)
    }

    /// Renormalize to unit length. Panics if the quaternion is near-zero.
    #[inline]
    pub fn normalize(self) -> Self {
        let n = self.norm();
        debug_assert!(n > 0.0, "cannot normalize zero quaternion");
        Self::new(self.w / n, self.x / n, self.y / n, self.z / n)
    }

    /// Hamilton product `self ⊗ other` (compose rotations: apply `other` then `self`).
    #[inline]
    pub fn multiply(self, other: Self) -> Self {
        let (aw, ax, ay, az) = (self.w, self.x, self.y, self.z);
        let (bw, bx, by, bz) = (other.w, other.x, other.y, other.z);
        Self::new(
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    }

    /// Conjugate `q*` (inverse for unit quaternions).
    #[inline]
    pub fn conjugate(self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Active rotation of a body-frame vector into the world frame: `R(q) v`.
    ///
    /// `q` is assumed unit (or nearly); call [`normalize`](Self::normalize) first if needed.
    #[inline]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let r = self.to_matrix();
        Vec3::new(
            r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
            r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
            r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
        )
    }

    /// World→body: `R(q)ᵀ v` (unit `q`).
    #[inline]
    pub fn inverse_rotate(self, v: Vec3) -> Vec3 {
        self.conjugate().rotate(v)
    }

    /// Rotation matrix `R(q)` such that `v_world = R v_body` (row-major 3×3).
    #[inline]
    pub fn to_matrix(self) -> [[f64; 3]; 3] {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }

    /// Quaternion for a right-handed rotation of `angle` [rad] about a unit `axis`.
    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        let axis = axis.normalize();
        let half = 0.5 * angle;
        let s = libm_ops::sin(half);
        Self::new(libm_ops::cos(half), s * axis.x, s * axis.y, s * axis.z)
    }

    /// Quaternion for a rotation vector `rv = angle * axis` [rad].
    ///
    /// Matches the Python oracle: `sin(θ/2)/θ` with a series for tiny `|rv|`.
    /// Uses portable libm for sin/cos (deterministic across OS).
    pub fn exp_rotvec(rv: Vec3) -> Self {
        let th = rv.norm();
        let half = 0.5 * th;
        // sin(th/2)/th with a series for tiny angles (same threshold as Python: 1e-8)
        let s = if th > 1e-8 {
            libm_ops::sin(half) / th
        } else {
            0.5 - th * th / 48.0
        };
        Self::new(libm_ops::cos(half), s * rv.x, s * rv.y, s * rv.z)
    }

    /// Pure-vector quaternion `(0, v)` used in `q̇ = ½ ω ⊗ q`.
    #[inline]
    pub fn from_vec(v: Vec3) -> Self {
        Self::new(0.0, v.x, v.y, v.z)
    }

    /// Vector part as [`Vec3`].
    #[inline]
    pub fn vec(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Scale all four components.
    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self::new(self.w * s, self.x * s, self.y * s, self.z * s)
    }
}

impl Default for Quat {
    fn default() -> Self {
        Self::identity()
    }
}

impl core::ops::Add for Quat {
    type Output = Self;

    /// Component-wise add (used by RK4; result is renormalized by the stepper).
    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use proptest::prelude::*;

    #[test]
    fn z90_rotates_x_to_y() {
        let q = Quat::from_axis_angle(Vec3::unit_z(), core::f64::consts::FRAC_PI_2);
        let v = q.rotate(Vec3::unit_x());
        assert_abs_diff_eq!(v.x, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(v.y, 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(v.z, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn matrix_is_rotation() {
        let q = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), 0.7);
        let r = q.to_matrix();
        // R R^T ≈ I
        for i in 0..3 {
            for j in 0..3 {
                let mut d = 0.0;
                for k in 0..3 {
                    d += r[i][k] * r[j][k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(d, expected, epsilon = 1e-14);
            }
        }
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        assert_relative_eq!(det, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn composition_matches_matrices() {
        let a = Quat::from_axis_angle(Vec3::unit_x(), 0.3);
        let b = Quat::from_axis_angle(Vec3::unit_y(), -0.5);
        let ab = a.multiply(b);
        let ra = a.to_matrix();
        let rb = b.to_matrix();
        let rab = ab.to_matrix();
        for i in 0..3 {
            for j in 0..3 {
                let mut expected = 0.0;
                for k in 0..3 {
                    expected += ra[i][k] * rb[k][j];
                }
                assert_abs_diff_eq!(rab[i][j], expected, epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn exp_rotvec_matches_axis_angle() {
        let axis = Vec3::new(1.0, 2.0, -0.5).normalize();
        let q1 = Quat::exp_rotvec(axis * 0.7);
        let q2 = Quat::from_axis_angle(axis, 0.7);
        assert_abs_diff_eq!(q1.w, q2.w, epsilon = 1e-15);
        assert_abs_diff_eq!(q1.x, q2.x, epsilon = 1e-15);
        assert_abs_diff_eq!(q1.y, q2.y, epsilon = 1e-15);
        assert_abs_diff_eq!(q1.z, q2.z, epsilon = 1e-15);
        let q0 = Quat::exp_rotvec(Vec3::zero());
        assert_eq!(q0, Quat::identity());
    }

    proptest! {
        #[test]
        fn normalize_has_unit_norm(
            w in -2.0f64..2.0,
            x in -2.0f64..2.0,
            y in -2.0f64..2.0,
            z in -2.0f64..2.0
        ) {
            let q = Quat::new(w, x, y, z);
            prop_assume!(q.norm() > 1e-9);
            let n = q.normalize().norm();
            prop_assert!((n - 1.0).abs() < 1e-12);
        }
    }
}
