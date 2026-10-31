//! Soft-body particles (XPBD positions / velocities / inverse masses).

use pe_math::Vec3;

/// Soft particle with inverse mass `w = 1/m` (`w = 0` ⇒ kinematic / pinned).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    /// World position [m].
    pub x: Vec3,
    /// World velocity [m/s].
    pub v: Vec3,
    /// Position at the start of the current substep (for velocity update) [m].
    pub x_prev: Vec3,
    /// Inverse mass [1/kg]; 0 = infinite mass (pinned).
    pub inv_mass: f64,
    /// Persistent external force [N] applied during XPBD prediction (`a += F/m`).
    ///
    /// Not cleared automatically — callers set / clear as needed for quasistatic loads.
    pub ext_force: Vec3,
}

impl Particle {
    /// Dynamic particle with finite mass `mass` [kg].
    pub fn dynamic(x: Vec3, mass: f64) -> Self {
        let inv = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            x,
            v: Vec3::zero(),
            x_prev: x,
            inv_mass: inv,
            ext_force: Vec3::zero(),
        }
    }

    /// Kinematic / pinned particle (infinite mass).
    pub fn pinned(x: Vec3) -> Self {
        Self {
            x,
            v: Vec3::zero(),
            x_prev: x,
            inv_mass: 0.0,
            ext_force: Vec3::zero(),
        }
    }

    /// Mass [kg], or `+∞` when pinned.
    pub fn mass(&self) -> f64 {
        if self.inv_mass == 0.0 {
            f64::INFINITY
        } else {
            1.0 / self.inv_mass
        }
    }
}
