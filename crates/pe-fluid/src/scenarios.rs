//! Canonical fluid determinism scenarios (cross-OS hashes).
//!
//! Must stay in sync with `docs/DETERMINISM.md`, the `fluid_hash` binary, and
//! Python bindings.

use pe_math::{Vec3, STANDARD_GRAVITY};

use crate::dfsph::{DfsphParams, DfsphWorld};
use crate::diagnostics::particles_hash64;
use crate::kernels::KernelKind;
use crate::particles::{
    fill_box, fill_box_boundary, mass_from_spacing, recomputed_akinci_psi, Particles,
};
use crate::pbf::{PbfParams, PbfWorld};
use crate::Dim;

/// Steps in the canonical PBF fluid hash scenario.
pub const FLUID_HASH_STEPS: usize = 40;
/// Timestep [s] for the PBF hash scenario.
pub const FLUID_HASH_DT: f64 = 1.0 / 120.0;

/// Steps in the canonical DFSPH fluid hash scenario.
pub const DFSPH_HASH_STEPS: usize = 40;
/// Timestep [s] for the DFSPH hash scenario.
pub const DFSPH_HASH_DT: f64 = 1.0 / 120.0;

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

/// Run the canonical PBF scenario and return the FNV-1a particle hash.
pub fn fluid_tank_hash() -> u64 {
    let mut world = fluid_hash_world();
    for _ in 0..FLUID_HASH_STEPS {
        world.step(FLUID_HASH_DT);
    }
    particles_hash64(&world.particles)
}

/// Build the canonical DFSPH tank (same geometry as the PBF hash world).
pub fn dfsph_hash_world() -> DfsphWorld {
    let dim = Dim::D3;
    let dx = 0.05;
    let h = 2.0 * dx;
    let rho0 = 1000.0;
    let mass = crate::particles::mass_calibrated_to_rest(
        rho0,
        dx,
        h,
        dim,
        crate::kernels::KernelKind::CubicSpline,
    );
    let mut parts = Particles::new(dim, rho0, mass, h);
    fill_box(
        &mut parts,
        Vec3::new(0.1, 0.1, 0.1),
        4,
        4,
        4,
        dx,
        Vec3::zero(),
    );
    fill_box_boundary(
        &mut parts,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.4, 0.4, 0.45),
        dx,
        1,
    );
    recomputed_akinci_psi(&mut parts, KernelKind::CubicSpline);
    let params = DfsphParams {
        gravity: Vec3::new(0.0, 0.0, -STANDARD_GRAVITY),
        particle_diameter: dx,
        density_error_tol: 0.01,
        divergence_error_tol: 0.01,
        max_density_iters: 8,
        max_divergence_iters: 8,
        xsph: crate::viscosity::Xsph { epsilon: 0.05 },
        adaptive_dt: false,
        warm_start: true,
        ..DfsphParams::default()
    };
    DfsphWorld::new(parts, params)
}

/// Run the canonical DFSPH scenario and return the FNV-1a particle hash.
pub fn dfsph_tank_hash() -> u64 {
    let mut world = dfsph_hash_world();
    for _ in 0..DFSPH_HASH_STEPS {
        world.step(DFSPH_HASH_DT);
    }
    particles_hash64(&world.particles)
}
