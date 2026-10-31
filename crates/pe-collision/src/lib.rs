//! Stage-2 collision and contact dynamics.
//!
//! Port of the Python reference (`shapes` / `broadphase` / `narrowphase` / `contact`)
//! with behavioral parity as the first goal (same algorithms, sorted pair order).
//!
//! Extension hooks (not implemented here — M5+ / M6):
//! - [`BroadPhase`] so a dynamic AABB tree can replace sweep-and-prune
//! - [`NarrowPhase`] so GJK/EPA or SAT-with-feature-caching can replace dedicated pairs
//! - Stable [`FeatureId`] on manifold points for warm starting across frames
//!
//! Out of scope for M3: soft-step, CCD, joints, islands/sleeping.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod broadphase;
pub mod contact;
pub mod manifold;
pub mod narrowphase;
pub mod shapes;

pub use broadphase::{brute_force_pairs, sweep_and_prune, BroadPhase, BruteForce, SweepAndPrune};
pub use contact::{ContactWorld, StaticCollider};
pub use manifold::{Contact, ContactPoint, FeatureId, ManifoldPoint};
pub use narrowphase::{collide, DedicatedNarrowPhase, NarrowPhase};
pub use shapes::{Aabb, BoxShape, Capsule, Material, Plane, Shape, Sphere};

use pe_core::diagnostics::state_hash64;
use pe_core::forces::UniformGravity;
use pe_core::{RigidBody, World};
use pe_math::Vec3;

/// Build the locked 10-box pyramid contact world used by cross-OS collision hashing.
pub fn canonical_collision_world() -> ContactWorld {
    let h = 0.5;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    let mut mats = Vec::new();
    let mut row = 0usize;
    let mut placed = 0usize;
    while placed < 10 {
        let n = row + 1;
        let x0 = -(n as f64 - 1.0) * h;
        for i in 0..n {
            if placed >= 10 {
                break;
            }
            let x = x0 + 2.0 * h * (i as f64);
            let z = h + 2.0 * h * (row as f64);
            bodies.push(
                RigidBody::new(1.0)
                    .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
                    .with_position(Vec3::new(x, 0.0, z)),
            );
            shapes.push(Shape::Box(BoxShape::new(h, h, h)));
            mats.push(Material::new(0.0, 0.5));
            placed += 1;
        }
        row += 1;
    }
    let mut cw = ContactWorld::new(
        World::new(bodies, vec![Box::new(UniformGravity::earth())]),
        shapes,
        vec![
            StaticCollider::new(Shape::Plane(Plane::new(Vec3::unit_z(), 0.0)))
                .with_material(Material::new(0.0, 1.0)),
        ],
        Some(mats),
    );
    cw.iterations = 20;
    cw
}

/// Run the canonical 10-box pyramid for 600 steps @ dt=1/240; return FNV-1a state hash.
pub fn canonical_collision_hash() -> u64 {
    let mut cw = canonical_collision_world();
    for _ in 0..600 {
        cw.step(1.0 / 240.0);
    }
    state_hash64(&cw.world.state)
}

/// Locked cross-OS collision hash (Linux x86_64 release). See `docs/DETERMINISM.md`.
pub const CANONICAL_COLLISION_HASH: u64 = 0x50d5_2b21_93aa_528c;

#[cfg(test)]
mod hash_tests {
    use super::*;

    #[test]
    fn collision_hash_matches_locked_value() {
        assert_eq!(canonical_collision_hash(), CANONICAL_COLLISION_HASH);
    }

    #[test]
    fn collision_hash_is_bit_stable_on_repeat() {
        assert_eq!(canonical_collision_hash(), canonical_collision_hash());
    }
}
