//! 3-vectors in SI (component units depend on context: m, m/s, N, …).

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::libm_ops;

/// Cartesian 3-vector with `f64` components.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// x component
    pub x: f64,
    /// y component
    pub y: f64,
    /// z component
    pub z: f64,
}

impl Vec3 {
    /// Construct from components.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Zero vector.
    #[inline]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Unit vector along +x.
    #[inline]
    pub const fn unit_x() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }

    /// Unit vector along +y.
    #[inline]
    pub const fn unit_y() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }

    /// Unit vector along +z.
    #[inline]
    pub const fn unit_z() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }

    /// From a length-3 slice `[x, y, z]`.
    #[inline]
    pub fn from_slice(v: &[f64; 3]) -> Self {
        Self::new(v[0], v[1], v[2])
    }

    /// Write components into `[x, y, z]`.
    #[inline]
    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Euclidean norm [same units as components]. Uses portable [`libm_ops::sqrt`].
    #[inline]
    pub fn norm(self) -> f64 {
        libm_ops::sqrt(self.norm_squared())
    }

    /// Squared Euclidean norm.
    #[inline]
    pub fn norm_squared(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product `self × other`.
    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Normalize to unit length. Returns zero if norm is zero.
    #[inline]
    pub fn normalize(self) -> Self {
        let n = self.norm();
        if n == 0.0 {
            Self::zero()
        } else {
            self / n
        }
    }

    /// Component-wise multiply.
    #[inline]
    pub fn mul_elementwise(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl MulAssign<f64> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, s: f64) {
        *self = *self * s;
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, s: f64) -> Self {
        Self::new(self.x / s, self.y / s, self.z / s)
    }
}

impl DivAssign<f64> for Vec3 {
    #[inline]
    fn div_assign(&mut self, s: f64) {
        *self = *self / s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    #[test]
    fn cross_orthogonal_to_inputs() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(-4.0, 0.5, 1.0);
        let c = a.cross(b);
        assert_abs_diff_eq!(c.dot(a), 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(c.dot(b), 0.0, epsilon = 1e-14);
    }

    proptest! {
        #[test]
        fn norm_nonnegative(x in -1e6f64..1e6, y in -1e6f64..1e6, z in -1e6f64..1e6) {
            prop_assert!(Vec3::new(x, y, z).norm() >= 0.0);
        }

        #[test]
        fn add_commutative(ax in -1e3f64..1e3, ay in -1e3f64..1e3, az in -1e3f64..1e3,
                           bx in -1e3f64..1e3, by in -1e3f64..1e3, bz in -1e3f64..1e3) {
            let a = Vec3::new(ax, ay, az);
            let b = Vec3::new(bx, by, bz);
            prop_assert_eq!(a + b, b + a);
        }
    }
}
