//! Particle fluids: SPH (WCSPH) and Position Based Fluids (PBF).
//!
//! # Scope
//!
//! 3-D particle liquids with a 2-D mode for analytic tests. Clean-room from
//! published SPH / PBF literature (see [`docs/FLUID.md`](../../docs/FLUID.md)
//! and [`THIRD_PARTY.md`](../../THIRD_PARTY.md)).
//!
//! **Deferred:** DFSPH (Bender & Koschier 2015); one-way fluid→rigid buoyancy /
//! drag coupling (would touch `pe-core`); surface tension; free-surface
//! reconstruction.
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
pub mod diagnostics;
pub mod eos;
pub mod grid;
pub mod kernels;
pub mod oracles;
pub mod particles;
pub mod pbf;
pub mod scenarios;
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
pub use diagnostics::{
    gravitational_potential, kinetic_energy, linear_momentum, max_density_error,
    mean_density_error, particles_hash64, total_mass,
};
pub use eos::{ensure_sound_speed, TaitEos, SOUND_SPEED_FACTOR, TAIT_GAMMA_WATER};
pub use grid::{CellKey, UniformGrid};
pub use kernels::{
    grad_w, gradient_consistency_error, gradient_sum_norm, partition_of_unity_error,
    support_radius, w, KernelKind,
};
pub use oracles::{
    couette_u, hydrostatic_pressure, poiseuille_u, ritter_c0, ritter_front, ritter_height,
    ritter_velocity,
};
pub use particles::{fill_box, fill_box_boundary, mass_from_spacing, ParticleKind, Particles};
pub use pbf::{PbfParams, PbfWorld};
pub use scenarios::{fluid_hash_world, fluid_tank_hash, FLUID_HASH_DT, FLUID_HASH_STEPS};
pub use viscosity::{max_speed, pi_ij, MonaghanViscosity, Xsph};
pub use wcsph::{WcsphParams, WcsphWorld};
