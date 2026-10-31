//! Rigid body description and inertia helpers (SI).

use pe_math::{Quat, Vec3};

/// Stable index into a [`crate::World`]'s body arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BodyId(pub usize);

/// Initial description of a rigid body.
///
/// Positions/velocities are world-frame, at the centre of mass.
/// - `mass` [kg]
/// - `inertia_body` [kg m²] body-frame about COM (row-major 3×3)
/// - `position` [m], `velocity` [m/s]
/// - `orientation` unit quaternion body→world
/// - `angular_velocity` [rad/s] world frame
#[derive(Clone, Debug)]
pub struct RigidBody {
    /// Mass [kg], must be positive.
    pub mass: f64,
    /// Body-frame inertia about COM [kg m²], 3×3 row-major.
    pub inertia_body: [[f64; 3]; 3],
    /// COM position in world frame [m].
    pub position: Vec3,
    /// Linear velocity of COM [m/s].
    pub velocity: Vec3,
    /// Orientation quaternion (body→world).
    pub orientation: Quat,
    /// Angular velocity in world frame [rad/s].
    pub angular_velocity: Vec3,
    /// Optional debug name.
    pub name: String,
}

impl RigidBody {
    /// Create a body with the given mass [kg] and identity inertia / zero state.
    pub fn new(mass: f64) -> Self {
        assert!(mass > 0.0, "mass must be positive");
        Self {
            mass,
            inertia_body: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            position: Vec3::zero(),
            velocity: Vec3::zero(),
            orientation: Quat::identity(),
            angular_velocity: Vec3::zero(),
            name: String::new(),
        }
    }

    /// Builder: set world position [m].
    pub fn with_position(mut self, p: Vec3) -> Self {
        self.position = p;
        self
    }

    /// Builder: set linear velocity [m/s].
    pub fn with_velocity(mut self, v: Vec3) -> Self {
        self.velocity = v;
        self
    }

    /// Builder: set body-frame inertia [kg m²].
    pub fn with_inertia(mut self, i: [[f64; 3]; 3]) -> Self {
        Self::validate_inertia(&i);
        self.inertia_body = i;
        self
    }

    /// Builder: diagonal body-frame inertia [kg m²].
    pub fn with_inertia_diag(self, ixx: f64, iyy: f64, izz: f64) -> Self {
        self.with_inertia([[ixx, 0.0, 0.0], [0.0, iyy, 0.0], [0.0, 0.0, izz]])
    }

    /// Builder: orientation (normalized).
    pub fn with_orientation(mut self, q: Quat) -> Self {
        self.orientation = q.normalize();
        self
    }

    /// Builder: world-frame angular velocity [rad/s].
    pub fn with_angular_velocity(mut self, w: Vec3) -> Self {
        self.angular_velocity = w;
        self
    }

    fn validate_inertia(i: &[[f64; 3]; 3]) {
        // Symmetry
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (i[r][c] - i[c][r]).abs() < 1e-12,
                    "inertia tensor must be symmetric"
                );
            }
        }
        // Positive-definite via leading minors (Sylvester) for 3×3
        let d1 = i[0][0];
        let d2 = i[0][0] * i[1][1] - i[0][1] * i[1][0];
        let d3 = i[0][0] * (i[1][1] * i[2][2] - i[1][2] * i[2][1])
            - i[0][1] * (i[1][0] * i[2][2] - i[1][2] * i[2][0])
            + i[0][2] * (i[1][0] * i[2][1] - i[1][1] * i[2][0]);
        assert!(
            d1 > 0.0 && d2 > 0.0 && d3 > 0.0,
            "inertia must be positive definite"
        );
    }
}

/// Solid sphere inertia about COM: `(2/5) m r² I` [kg m²].
pub fn inertia_solid_sphere(mass: f64, radius: f64) -> [[f64; 3]; 3] {
    let i = (2.0 / 5.0) * mass * radius * radius;
    [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]]
}

/// Solid box with full edge lengths `a` (x), `b` (y), `c` (z) [m]; inertia [kg m²].
pub fn inertia_solid_box(mass: f64, a: f64, b: f64, c: f64) -> [[f64; 3]; 3] {
    [
        [mass * (b * b + c * c) / 12.0, 0.0, 0.0],
        [0.0, mass * (a * a + c * c) / 12.0, 0.0],
        [0.0, 0.0, mass * (a * a + b * b) / 12.0],
    ]
}

/// Solid cylinder with axis along body z; radius `r` [m], height `h` [m]; inertia [kg m²].
pub fn inertia_solid_cylinder(mass: f64, r: f64, h: f64) -> [[f64; 3]; 3] {
    let i = mass * (3.0 * r * r + h * h) / 12.0;
    let izz = 0.5 * mass * r * r;
    [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, izz]]
}

/// Invert a 3×3 matrix (for body inertia). Panics if singular.
pub(crate) fn invert3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    assert!(det.abs() > 0.0, "singular matrix");
    let inv_det = 1.0 / det;
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]
}

/// Matrix-vector product `M v` for 3×3 row-major.
#[inline]
pub(crate) fn mat_vec(m: &[[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

/// `A @ B` for 3×3.
#[inline]
pub(crate) fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Transpose of 3×3.
#[inline]
pub(crate) fn mat_transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn cylinder_inertia_formula() {
        let i = inertia_solid_cylinder(2.0, 0.5, 1.0);
        assert_abs_diff_eq!(i[0][0], 0.29166666666666663, epsilon = 1e-15);
        assert_abs_diff_eq!(i[1][1], 0.29166666666666663, epsilon = 1e-15);
        assert_abs_diff_eq!(i[2][2], 0.25, epsilon = 1e-15);
    }

    #[test]
    fn box_cube_inertia() {
        // cube side 1, mass 6: I = m a² / 6 = 1
        let i = inertia_solid_box(6.0, 1.0, 1.0, 1.0);
        assert_abs_diff_eq!(i[0][0], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(i[1][1], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(i[2][2], 1.0, epsilon = 1e-15);
    }
}
