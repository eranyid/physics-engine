//! Stage-1 rigid-body dynamics: world, forces, integrators, diagnostics.
//!
//! Conventions match the Python reference oracle (`src/physics_engine`):
//! - SI units only (m, kg, s, N, rad)
//! - Quaternion `[w, x, y, z]`, active body→world
//! - Rotational state stored as **world-frame angular momentum** `L`
//! - Deterministic step: fixed body/generator order, no hash maps

#![deny(missing_docs)]
#![forbid(unsafe_code)]
// SoA body loops intentionally index parallel arrays; iterator rewrites obscure the physics.
#![allow(clippy::needless_range_loop)]

pub mod body;
pub mod diagnostics;
pub mod forces;
pub mod integrators;
pub mod world;

pub use body::{
    inertia_solid_box, inertia_solid_cylinder, inertia_solid_sphere, BodyId, RigidBody,
};
pub use forces::{
    ConstantTorque, ForceGenerator, LinearDrag, NewtonianGravity, QuadraticDrag, Spring,
    UniformGravity,
};
pub use integrators::{
    explicit_euler, rk4, symplectic_euler, velocity_verlet, Integrator, IntegratorKind,
};
pub use world::{State, World};
