//! NASA TM-2015-218675 6-DOF check-case scenarios 1–3 in **flat-Earth** form.
//!
//! The published trajectory tables are **not** available offline in this tree.
//! Oracles are therefore analytic / self-consistency identities only — we do
//! **not** invent published numeric trajectory points.
//!
//! # Reference
//! - Murman, S. M., et al. / NASA. *Six-Degree-of-Freedom Trajectory Simulation
//!   Check Cases.* NASA/TM-2015-218675. Scenarios 1–3 described qualitatively:
//!   (1) dragless sphere, (2) tumbling brick, (3) sphere with quadratic drag.
//!
//! Flat-Earth: uniform `g = −g₀ ê_z`, no Coriolis / curvature.

use pe_core::forces::{ForceGenerator, QuadraticDrag, UniformGravity};
use pe_core::integrators::rk4;
use pe_core::world::{State, World};
use pe_core::{inertia_solid_box, inertia_solid_sphere, RigidBody};
use pe_math::libm_ops;
use pe_math::{Quat, Vec3, STANDARD_GRAVITY};

/// Scenario identifier matching NASA TM-2015-218675 check cases 1–3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NasaCheckCase {
    /// Case 1: dragless sphere (ballistic under uniform g).
    DraglessSphere = 1,
    /// Case 2: tumbling brick (torque-free rigid body + translation under g).
    TumblingBrick = 2,
    /// Case 3: sphere with quadratic drag.
    SphereWithDrag = 3,
}

/// Shared IC-ish parameters (flat-Earth engineering stand-in, not copied tables).
#[derive(Clone, Debug)]
pub struct CheckCaseConfig {
    /// Mass [kg].
    pub mass: f64,
    /// Sphere radius [m] (cases 1 & 3).
    pub radius: f64,
    /// Brick side lengths [m] (case 2).
    pub brick: (f64, f64, f64),
    /// Initial position [m].
    pub x0: Vec3,
    /// Initial velocity [m/s].
    pub v0: Vec3,
    /// Initial angular velocity [rad/s] (case 2).
    pub omega0: Vec3,
    /// Quadratic drag coefficient k [kg/m] for case 3 (`F = −k |u| u`).
    pub drag_k: f64,
    /// Gravity magnitude [m/s²].
    pub g: f64,
}

impl Default for CheckCaseConfig {
    fn default() -> Self {
        Self {
            mass: 1.0,
            radius: 0.5,
            brick: (2.0, 1.0, 0.5),
            x0: Vec3::new(0.0, 0.0, 1000.0),
            v0: Vec3::new(100.0, 0.0, 50.0),
            omega0: Vec3::new(0.3, 0.5, 0.1),
            drag_k: 0.01,
            g: STANDARD_GRAVITY,
        }
    }
}

/// Build a [`World`] for the given check case (flat Earth).
pub fn build_check_case(case: NasaCheckCase, cfg: &CheckCaseConfig) -> World {
    let gravity = UniformGravity::new(Vec3::new(0.0, 0.0, -cfg.g));
    match case {
        NasaCheckCase::DraglessSphere => {
            let body = RigidBody::new(cfg.mass)
                .with_inertia(inertia_solid_sphere(cfg.mass, cfg.radius))
                .with_position(cfg.x0)
                .with_velocity(cfg.v0);
            World::new(vec![body], vec![Box::new(gravity)])
        }
        NasaCheckCase::TumblingBrick => {
            let (a, b, c) = cfg.brick;
            let body = RigidBody::new(cfg.mass)
                .with_inertia(inertia_solid_box(cfg.mass, a, b, c))
                .with_position(cfg.x0)
                .with_velocity(cfg.v0)
                .with_angular_velocity(cfg.omega0);
            World::new(vec![body], vec![Box::new(gravity)])
        }
        NasaCheckCase::SphereWithDrag => {
            let body = RigidBody::new(cfg.mass)
                .with_inertia(inertia_solid_sphere(cfg.mass, cfg.radius))
                .with_position(cfg.x0)
                .with_velocity(cfg.v0);
            World::new(
                vec![body],
                vec![
                    Box::new(gravity),
                    Box::new(QuadraticDrag::new(cfg.drag_k, Vec3::zero())),
                ],
            )
        }
    }
}

/// Analytic flat-Earth dragless projectile: `x(t) = x0 + v0 t + ½ g t²`.
pub fn dragless_ballistic(x0: Vec3, v0: Vec3, g: f64, t: f64) -> (Vec3, Vec3) {
    let gg = Vec3::new(0.0, 0.0, -g);
    let v = v0 + gg * t;
    let x = x0 + v0 * t + gg * (0.5 * t * t);
    (x, v)
}

/// Mechanical energy for uniform gravity: `½ m |v|² + m g z` (+ rotational KE).
pub fn mechanical_energy(world: &World, s: &State, g: f64) -> f64 {
    let mut e = 0.0;
    for i in 0..world.n {
        let m = world.mass[i];
        e += 0.5 * m * s.v[i].norm_squared();
        e += m * g * s.x[i].z;
        // Rotational KE from world L: T = ½ ω · L with ω = R I⁻¹ Rᵀ L
        let omega = world.omega_i(s, i);
        e += 0.5 * omega.dot(s.l[i]);
    }
    e
}

/// Total world-frame angular momentum about the origin (incl. orbital).
pub fn angular_momentum_about_origin(world: &World, s: &State) -> Vec3 {
    let mut l = Vec3::zero();
    for i in 0..world.n {
        l += s.x[i].cross(s.v[i] * world.mass[i]) + s.l[i];
    }
    l
}

/// Closed-form 1-D vertical motion with quadratic drag opposing velocity.
///
/// For general 3-D the closed form is awkward; vertical ascent/descent with
/// `F = −k |v| v − m g` has known `tanh` / `tan` solutions. Here we expose the
/// **horizontal** drag-free invariant and a vertical terminal speed
/// `v_term = √(m g / k)` for case-3 self-consistency.
pub fn quadratic_drag_terminal_speed(mass: f64, k: f64, g: f64) -> f64 {
    libm_ops::sqrt(mass * g / k)
}

/// Integrate a check case for `t_end` [s] with RK4 step `dt` [s].
pub fn run_check_case(case: NasaCheckCase, cfg: &CheckCaseConfig, t_end: f64, dt: f64) -> World {
    let mut w = build_check_case(case, cfg);
    w.run_for(t_end, dt, rk4);
    w
}

/// Marker force generator documenting that case-2 has no aero torque (torque-free).
#[derive(Clone, Debug, Default)]
pub struct NoAeroTorque;

impl ForceGenerator for NoAeroTorque {
    fn apply(&self, _world: &World, _s: &State, _t: f64, _f: &mut [Vec3], _torque: &mut [Vec3]) {}
}

/// Quaternion norm helper for attitude checks.
pub fn quat_norm(q: Quat) -> f64 {
    q.norm()
}
