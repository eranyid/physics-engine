//! Force generators. Each `apply` adds into `F` [N] and `T` [N m].
//!
//! Generators are stored and applied in a fixed `Vec` order (no hash maps).

use pe_math::libm_ops;
use pe_math::{Vec3, G as G_SI, STANDARD_GRAVITY};

use crate::world::{State, World};

/// Trait for force/torque generators. Implementations must be deterministic
/// given the same world/state/`t` and must not iterate hash maps.
pub trait ForceGenerator: Send + Sync {
    /// Add contributions into `f` [N] and `torque` [N m] (about each COM).
    fn apply(&self, world: &World, s: &State, t: f64, f: &mut [Vec3], torque: &mut [Vec3]);

    /// Optional potential energy [J] for conservative forces. Default: none.
    fn potential(&self, _world: &World, _s: &State) -> Option<f64> {
        None
    }
}

/// Uniform gravity `F = m g`. `g` [m/s²], default `(0, 0, -g₀)`.
#[derive(Clone, Debug)]
pub struct UniformGravity {
    /// Gravity acceleration vector [m/s²].
    pub g: Vec3,
}

impl UniformGravity {
    /// Earth-surface gravity along −z.
    pub fn earth() -> Self {
        Self {
            g: Vec3::new(0.0, 0.0, -STANDARD_GRAVITY),
        }
    }

    /// Custom `g` [m/s²].
    pub fn new(g: Vec3) -> Self {
        Self { g }
    }
}

impl ForceGenerator for UniformGravity {
    fn apply(&self, world: &World, _s: &State, _t: f64, f: &mut [Vec3], _torque: &mut [Vec3]) {
        for i in 0..world.n {
            f[i] += world.mass[i] * self.g;
        }
    }

    fn potential(&self, world: &World, s: &State) -> Option<f64> {
        let mut u = 0.0;
        for i in 0..world.n {
            u -= world.mass[i] * s.x[i].dot(self.g);
        }
        Some(u)
    }
}

/// Pairwise point-mass gravity
/// `F_ij = G m_i m_j (x_j − x_i) / (|r|² + ε²)^(3/2)`.
///
/// Antisymmetric pair loop → Newton's third law (linear momentum conserved).
#[derive(Clone, Debug)]
pub struct NewtonianGravity {
    /// Gravitational constant [m³ kg⁻¹ s⁻²].
    pub g_const: f64,
    /// Softening length squared [m²].
    pub eps2: f64,
}

impl NewtonianGravity {
    /// Use SI `G`, no softening.
    pub fn new() -> Self {
        Self {
            g_const: G_SI,
            eps2: 0.0,
        }
    }

    /// Custom `G` [m³ kg⁻¹ s⁻²] and softening length [m].
    pub fn with_params(g_const: f64, softening: f64) -> Self {
        Self {
            g_const,
            eps2: softening * softening,
        }
    }
}

impl Default for NewtonianGravity {
    fn default() -> Self {
        Self::new()
    }
}

impl ForceGenerator for NewtonianGravity {
    fn apply(&self, world: &World, s: &State, _t: f64, f: &mut [Vec3], _torque: &mut [Vec3]) {
        // Fixed nested loop i < j — deterministic, no hash maps.
        for i in 0..world.n {
            for j in (i + 1)..world.n {
                let d = s.x[j] - s.x[i]; // x_j - x_i
                let r2 = d.norm_squared() + self.eps2;
                if r2 == 0.0 {
                    continue;
                }
                let inv_r = 1.0 / libm_ops::sqrt(r2);
                let inv_r3 = inv_r / r2; // 1 / r^3
                let scale = self.g_const * world.mass[i] * world.mass[j] * inv_r3;
                let force = d * scale; // on i, toward j
                f[i] += force;
                f[j] -= force;
            }
        }
    }

    fn potential(&self, world: &World, s: &State) -> Option<f64> {
        let mut u = 0.0;
        for i in 0..world.n {
            for j in (i + 1)..world.n {
                let d = s.x[j] - s.x[i];
                let r2 = d.norm_squared() + self.eps2;
                if r2 == 0.0 {
                    continue;
                }
                u -= self.g_const * world.mass[i] * world.mass[j] / libm_ops::sqrt(r2);
            }
        }
        Some(u)
    }
}

/// Stokes drag `F = −b (v − wind)`, `b` [kg/s] (scalar or per-body).
#[derive(Clone, Debug)]
pub struct LinearDrag {
    /// Drag coefficients [kg/s], length 1 (broadcast) or N.
    pub b: Vec<f64>,
    /// Wind velocity [m/s].
    pub wind: Vec3,
}

impl LinearDrag {
    /// Uniform `b` [kg/s] for every body, optional wind [m/s].
    pub fn new(b: f64, wind: Vec3) -> Self {
        Self { b: vec![b], wind }
    }

    /// Per-body `b` [kg/s].
    pub fn per_body(b: Vec<f64>, wind: Vec3) -> Self {
        Self { b, wind }
    }

    fn coeff(&self, i: usize) -> f64 {
        if self.b.len() == 1 {
            self.b[0]
        } else {
            self.b[i]
        }
    }
}

impl ForceGenerator for LinearDrag {
    fn apply(&self, world: &World, s: &State, _t: f64, f: &mut [Vec3], _torque: &mut [Vec3]) {
        for i in 0..world.n {
            f[i] -= (s.v[i] - self.wind) * self.coeff(i);
        }
    }
}

/// Newton drag `F = −k |u| u`, `u = v − wind`, `k` [kg/m] (= ½ ρ C_d A).
#[derive(Clone, Debug)]
pub struct QuadraticDrag {
    /// Drag coefficients [kg/m], length 1 (broadcast) or N.
    pub k: Vec<f64>,
    /// Wind velocity [m/s].
    pub wind: Vec3,
}

impl QuadraticDrag {
    /// Uniform `k` [kg/m].
    pub fn new(k: f64, wind: Vec3) -> Self {
        Self { k: vec![k], wind }
    }

    /// `k = ½ ρ C_d A` [kg/m].
    pub fn coefficient(rho: f64, cd: f64, area: f64) -> f64 {
        0.5 * rho * cd * area
    }

    fn coeff(&self, i: usize) -> f64 {
        if self.k.len() == 1 {
            self.k[0]
        } else {
            self.k[i]
        }
    }
}

impl ForceGenerator for QuadraticDrag {
    fn apply(&self, world: &World, s: &State, _t: f64, f: &mut [Vec3], _torque: &mut [Vec3]) {
        for i in 0..world.n {
            let u = s.v[i] - self.wind;
            let speed = u.norm(); // libm sqrt
            f[i] -= u * (self.coeff(i) * speed);
        }
    }
}

/// Hookean spring of stiffness `k` [N/m] pulling body `i`'s COM to a fixed anchor
/// (rest length 0).
#[derive(Clone, Debug)]
pub struct Spring {
    /// Body index.
    pub body: usize,
    /// Anchor position [m].
    pub anchor: Vec3,
    /// Stiffness [N/m].
    pub k: f64,
}

impl Spring {
    /// Spring on `body` to `anchor` [m] with stiffness `k` [N/m].
    pub fn new(body: usize, anchor: Vec3, k: f64) -> Self {
        Self { body, anchor, k }
    }
}

impl ForceGenerator for Spring {
    fn apply(&self, _world: &World, s: &State, _t: f64, f: &mut [Vec3], _torque: &mut [Vec3]) {
        f[self.body] -= (s.x[self.body] - self.anchor) * self.k;
    }

    fn potential(&self, _world: &World, s: &State) -> Option<f64> {
        let d = s.x[self.body] - self.anchor;
        Some(0.5 * self.k * d.dot(d))
    }
}

/// Constant world-frame torque [N m] on body `i`.
#[derive(Clone, Debug)]
pub struct ConstantTorque {
    /// Body index.
    pub body: usize,
    /// Torque [N m].
    pub tau: Vec3,
}

impl ConstantTorque {
    /// Constant torque [N m] on `body`.
    pub fn new(body: usize, tau: Vec3) -> Self {
        Self { body, tau }
    }
}

impl ForceGenerator for ConstantTorque {
    fn apply(&self, _world: &World, _s: &State, _t: f64, _f: &mut [Vec3], torque: &mut [Vec3]) {
        torque[self.body] += self.tau;
    }
}
