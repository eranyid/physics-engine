//! Canonical fluid determinism scenario (cross-OS hash).
//!
//! Small 3-D PBF tank settle then a few steps — must stay in sync with
//! `docs/DETERMINISM.md`, the `fluid_hash` binary, and Python bindings.

use pe_math::{Vec3, STANDARD_GRAVITY};

use crate::diagnostics::particles_hash64;
use crate::particles::{fill_box, fill_box_boundary, mass_from_spacing, Particles};
use crate::pbf::{PbfParams, PbfWorld};
use crate::Dim;

/// Steps in the canonical fluid hash scenario.
pub const FLUID_HASH_STEPS: usize = 40;
/// Timestep [s].
pub const FLUID_HASH_DT: f64 = 1.0 / 120.0;

/// Build the canonical PBF tank (fluid block in a closed box).
pub fn fluid_hash_world() -> PbfWorld {
    let dim = Dim::D3;
    let dx = 0.05;
    let h = 2.0 * dx;
    let rho0 = 1000.0;
    let mass = mass_from_spacing(rho0, dx, dim);
    let mut parts = Particles::new(dim, rho0, mass, h);
    // Fluid: 4×4×4 block
    fill_box(
        &mut parts,
        Vec3::new(0.1, 0.1, 0.1),
        4,
        4,
        4,
        dx,
        Vec3::zero(),
    );
    // Boundary shell
    fill_box_boundary(
        &mut parts,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.4, 0.4, 0.45),
        dx,
        1,
    );
    let params = PbfParams {
        gravity: Vec3::new(0.0, 0.0, -STANDARD_GRAVITY),
        iterations: 4,
        art_pressure_k: 0.0,
        vorticity_eps: 0.0,
        ..PbfParams::default()
    };
    PbfWorld::new(parts, params)
}

/// Run the canonical scenario and return the FNV-1a particle hash.
pub fn fluid_tank_hash() -> u64 {
    let mut world = fluid_hash_world();
    for _ in 0..FLUID_HASH_STEPS {
        world.step(FLUID_HASH_DT);
    }
    particles_hash64(&world.particles)
}
