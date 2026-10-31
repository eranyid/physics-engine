//! Print the canonical stage-1 state hash for cross-OS CI.
//!
//! Usage: `cargo run -p pe-core --bin canonical_hash --release`
//! Scenario must stay in lockstep with `docs/DETERMINISM.md` and the Python
//! `physics_engine_rs.canonical_determinism_hash()` helper.

use pe_core::diagnostics::state_hash64;
use pe_core::forces::{ConstantTorque, LinearDrag, Spring, UniformGravity};
use pe_core::integrators::rk4;
use pe_core::{RigidBody, World};
use pe_math::Vec3;

fn main() {
    let mut w = World::new(
        vec![
            RigidBody::new(1.0)
                .with_position(Vec3::new(0.0, 0.0, 10.0))
                .with_velocity(Vec3::new(1.0, 0.0, 0.0)),
            RigidBody::new(2.0)
                .with_position(Vec3::new(1.0, 0.0, 0.0))
                .with_velocity(Vec3::new(0.0, 0.5, 0.0))
                .with_inertia_diag(0.4, 0.5, 0.6)
                .with_angular_velocity(Vec3::new(0.1, 0.0, 0.2)),
        ],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(LinearDrag::new(0.01, Vec3::zero())),
            Box::new(Spring::new(0, Vec3::new(0.0, 0.0, 10.0), 5.0)),
            Box::new(ConstantTorque::new(1, Vec3::new(0.0, 0.0, 0.05))),
        ],
    );
    w.run_for(2.0, 1e-3, rk4);
    let h = state_hash64(&w.state);
    println!("{h:016x}");
}
