//! Particle fluids: SPH (WCSPH), Position Based Fluids (PBF), and DFSPH.
//!
//! # Scope
//!
//! 3-D particle liquids with a 2-D mode for analytic tests. Clean-room from
//! published SPH / PBF / DFSPH literature (see [`docs/FLUID.md`](../../docs/FLUID.md)
//! and [`THIRD_PARTY.md`](../../THIRD_PARTY.md)).
//!
//! Two-way fluid↔rigid coupling (Akinci 2012 boundary samples) lives in
//! [`coupling`] and feeds [`pe_core::ExternalLoads`]. Free-surface mesh
//! reconstruction remains deferred.
//!
//! # Determinism
//!
//! `f64`, portable [`pe_math::libm_ops`], no FMA, sorted uniform-grid neighbour
//! lists (no `HashMap` iteration on the step path), in-tree PRNG unused on the
//! default step (available via `pe-math` when stochastic forcing is added).

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod cfl;
pub mod coupling;
pub mod dfsph;
pub mod diagnostics;
pub mod eos;
pub mod grid;
pub mod kernels;
pub mod oracles;
pub mod particles;
pub mod pbf;
pub mod scenarios;
pub mod surface_tension;
pub mod validation;
pub mod viscosity;
pub mod wcsph;

/// Spatial dimension for kernels / neighbour stencils.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    /// Two-dimensional simulations (z forced to 0).
    D2,
    /// Three-dimensional simulations.
    D3,
}

pub use cfl::{max_accel, suggested_dt, CflParams};
pub use coupling::{
    archimedes_box_draft, barge_heave_added_mass_2d, coupled_linear_momentum, coupling_hash_world,
    coupling_tank_hash, disk_terminal_velocity_2d, estimate_heave_period, floating_box_draft_error,
    floating_box_draft_error_wcsph, heave_period_analytic, heave_period_from_stiffness,
    heave_period_with_added_mass, kinematic_bottom_force_at_draft, make_floating_box_2d,
    make_floating_box_wcsph_2d, max_box_penetration, max_box_penetration_wcsph,
    sphere_freefall_accel_added_mass, sphere_terminal_velocity, CoupledDfsphWorld,
    CoupledWcsphWorld, FluidRigidCoupler, RigidSample, SampleShape, COUPLING_HASH_DT,
    COUPLING_HASH_STEPS,
};
pub use dfsph::{DfsphParams, DfsphWorld};
pub use diagnostics::{
    gravitational_potential, kinetic_energy, linear_momentum, max_density_error,
    mean_density_error, particles_hash64, total_mass,
};
pub use eos::{ensure_sound_speed, TaitEos, SOUND_SPEED_FACTOR, TAIT_GAMMA_WATER};
pub use grid::{CellKey, UniformGrid};
pub use kernels::{
    apply_gradient_correction, grad_w, gradient_consistency_error,
    gradient_consistency_error_corrected, gradient_consistency_error_truncated,
    gradient_correction_matrix, gradient_sum_norm, partition_of_unity_error, shepard_factor,
    shepard_w, support_radius, w, KernelCorrection, KernelKind,
};
pub use oracles::{
    couette_u, hydrostatic_pressure, poiseuille_u, ritter_c0, ritter_front, ritter_height,
    ritter_velocity,
};
pub use particles::{
    fill_box, fill_box_boundary, mass_calibrated_to_rest, mass_from_spacing, recomputed_akinci_psi,
    ParticleKind, Particles,
};
pub use pbf::{PbfParams, PbfWorld};
pub use scenarios::{
    dfsph_hash_world, dfsph_tank_hash, fluid_hash_world, fluid_tank_hash, DFSPH_HASH_DT,
    DFSPH_HASH_STEPS, FLUID_HASH_DT, FLUID_HASH_STEPS,
};
pub use surface_tension::{
    cohesion_c, droplet_com_radius, droplet_pressure_jump, droplet_pressure_jump_shepard,
    droplet_sphericity_error, fill_droplet_2d, laplace_pressure, shepard_density,
    surface_tension_accel, SurfaceTensionModel, SurfaceTensionParams,
};
pub use validation::{
    column_heights, dam_break_convergence, dam_break_front_at, dam_break_front_curve_dfsph,
    dam_break_front_curve_pbf, dam_break_ritter_geometry, dam_break_times_from_tau,
    dfsph_dam_break_convergence, dfsph_dam_break_front_and_height_at, dfsph_dam_break_front_at,
    dfsph_dam_break_late_front_by_dx, fluid_front_x, fluid_front_x_height, fluid_front_x_toe,
    front_x_from_height_field, make_dam_break_2d, make_dfsph_dam_break_2d,
    make_dfsph_dam_break_akinci_only, ritter_front_rel_error, ritter_height_profile_rel_error,
    tait_hydrostatic_profile_error, wcsph_dam_break_front_at, DAM_BREAK_TAU_SAMPLES,
};
pub use viscosity::{max_speed, pi_ij, MonaghanViscosity, Xsph};
pub use wcsph::{WcsphParams, WcsphWorld};
