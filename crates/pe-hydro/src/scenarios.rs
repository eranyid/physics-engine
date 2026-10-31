//! Canonical scenarios for cross-OS determinism hashes.

use pe_core::diagnostics::state_hash64;
use pe_core::forces::UniformGravity;
use pe_core::integrators::rk4;
use pe_core::{inertia_solid_box, RigidBody, World};
use pe_math::Vec3;

use crate::force::Buoyancy;
use crate::hull::{BoxHull, Hull, RHO_FRESHWATER};
use crate::waves::{AiryWave, WaveField};

/// Wave-driven floating box: Airy η=0.25 m, T=8 s, h=50 m; RK4 2 s @ 1e-2.
///
/// Must stay in sync with `docs/DETERMINISM.md`, the `wave_body_hash` binary, and
/// `physics_engine_rs.canonical_wave_body_hash()`.
pub fn canonical_wave_body_hash() -> u64 {
    let hull = BoxHull::new(12.0, 4.0, 2.0);
    let mass = 24_000.0;
    let draft = hull.upright_draft_for_mass(mass, RHO_FRESHWATER);
    let z_c = -draft + 0.5 * hull.depth;
    let wave = AiryWave::earth(0.25, 8.0, 50.0, 0.0);
    let body = RigidBody::new(mass)
        .with_inertia(inertia_solid_box(mass, 12.0, 4.0, 2.0))
        .with_position(Vec3::new(0.0, 0.0, z_c));
    let mut w = World::new(
        vec![body],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(
                Buoyancy::freshwater(0, Hull::Box(hull), 0.0)
                    .with_wave(WaveField::Airy(wave))
                    .with_froude_krylov(true),
            ),
        ],
    );
    w.run_for(2.0, 1e-2, rk4);
    state_hash64(&w.state)
}
