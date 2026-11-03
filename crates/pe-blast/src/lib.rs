//! Engineering blast / shock models for games and studio simulation.
//!
//! # Scope
//!
//! Physical simulation of blast **loading**, shock waves, fragment kinematics,
//! and pre-fractured rigid destruction: Hopkinson–Cranz scaling, Kinney–Graham
//! overpressure, Friedlander waveforms, Rankine–Hugoniot jumps, Sedov–Taylor
//! point blasts, a 1-D Euler solver, Kingery–Bulmash / RH rigid-body loads with
//! AABB shadowing and [`pe_core::ExternalLoads`] coupling, Gurney/Mott
//! fragments as ballistic bodies, and Voronoi cohesive compounds.
//!
//! **Not in scope:** explosive chemistry, recipes, weapon design, or manufacture.
//! TNT-equivalence and Gurney energy are caller-supplied parameters.
//!
//! # Determinism
//!
//! Matches [`docs/DETERMINISM.md`](../../docs/DETERMINISM.md): `f64`, portable
//! [`pe_math::libm_ops`], no FMA, fixed iteration order, no `HashMap` on the
//! step path. Stochastic fragment / Voronoi sampling uses seeded
//! [`pe_math::Xoshiro256StarStar`].
//!
//! # Units
//!
//! SI throughout. Every public function documents argument units in its docs.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod compound;
pub mod error;
pub mod euler;
pub mod fragments;
pub mod friedlander;
pub mod kingery;
pub mod kinney;
pub mod loading;
pub mod occlusion;
pub mod rankine_hugoniot;
pub mod scaling;
pub mod scenarios;
pub mod sedov;

pub use compound::{
    bond_break_force, bond_breaks_on_impulse, break_bonds_by_impulse, chunks_to_bodies,
    chunks_to_box_shapes, cohesive_joints_from_partition, voronoi_box_partition, CohesiveBondSpec,
    VoronoiChunk, VoronoiPartition,
};
pub use error::BlastError;
pub use euler::{
    hllc, sod_l1_density_error, Euler1d, ExactRiemann, Geometry1d, Prim, GAMMA_DEFAULT,
};
pub use fragments::{
    build_fragment_specs, detonate_cased_charge, fibonacci_sphere, fragment_ammo_profile,
    fragments_kinetic_energy, fragments_momentum, gurney_constant_by_name, gurney_constants,
    gurney_metal_ke, gurney_velocity, mott_mean_mass, mott_sample_many, mott_sample_mass,
    spawn_fragments_ballistic, try_gurney_velocity, FragmentSpec, GurneyConstant, GurneyGeometry,
};
pub use friedlander::{
    positive_impulse, pressure as friedlander_pressure, sample as friedlander_sample,
    try_positive_impulse, try_pressure as try_friedlander_pressure,
    try_sample as try_friedlander_sample,
};
pub use kingery::{
    arrival_time, incident_impulse, incident_overpressure, positive_duration,
    reflected_overpressure as kingery_reflected_overpressure, scaled_arrival_time,
    scaled_incident_impulse, scaled_positive_duration, shock_front_velocity, KingeryResult,
};
pub use kinney::{
    peak_overpressure, peak_overpressure_at, peak_overpressure_ratio, try_peak_overpressure,
    try_peak_overpressure_at, try_peak_overpressure_ratio, P0_DEFAULT,
};
pub use loading::{
    accumulate_blast_external, accumulate_blast_external_all, apply_blast_impulse,
    apply_blast_impulse_params, dynamic_pressure_pa, face_positive_impulse,
    face_positive_impulse_params, face_pressure, face_pressure_params, incidence_factor,
    peak_face_overpressure, peak_incident_pa, rigid_wall_reflection_coefficient, standoff_factor,
    BlastLoadParams, BlastSource, ExposedFace, OverpressureModel, PressureImpulseCurve,
    ReflectionMix,
};
pub use occlusion::{is_shadowed, ray_aabb, sort_occluders, OccluderAabb};
pub use rankine_hugoniot::{
    dynamic_pressure, from_overpressure_ratio, reflected_overpressure, ShockState, GAMMA_AIR,
};
pub use scaling::{
    equivalent_tnt_mass, range_from_scaled, scaled_distance, try_equivalent_tnt_mass,
    try_range_from_scaled, try_scaled_distance,
};
pub use scenarios::{
    blast_destruction_hash, blast_external_loads_hash, blast_fragments_hash, blast_pyramid_hash,
    blast_pyramid_world, BLAST_HASH_DT, BLAST_HASH_STEPS,
};
pub use sedov::{
    shock_radius, shock_speed, solve_xi0, time_at_radius, xi0_gamma_14_spherical, SedovGeometry,
    SedovSolution,
};
