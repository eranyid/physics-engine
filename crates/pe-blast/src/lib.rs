//! Engineering blast / shock models for games and studio simulation.
//!
//! # Scope
//!
//! Physical simulation of blast **loading**, shock waves, and fragment
//! kinematics: Hopkinson–Cranz scaling, Kinney–Graham overpressure, Friedlander
//! waveforms, Rankine–Hugoniot jumps, Sedov–Taylor point blasts, a 1-D Euler
//! solver, a 3-D Euler voxel solver ([`euler3d`], [`blast3d`]) for blast
//! waves in rooms, corridors and behind cover, rigid-body blast impulses, and
//! Gurney/Mott fragment helpers.
//!
//! **Not in scope:** explosive chemistry, recipes, weapon design, or manufacture.
//! TNT-equivalence and Gurney energy are caller-supplied parameters.
//!
//! # Determinism
//!
//! Matches [`docs/DETERMINISM.md`](../../docs/DETERMINISM.md): `f64`, portable
//! [`pe_math::libm_ops`], no FMA, fixed iteration order, no `HashMap` on the
//! step path.
//!
//! # Units
//!
//! SI throughout. Every public function documents argument units in its docs.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod blast3d;
pub mod euler;
pub mod euler3d;
pub mod fragments;
pub mod friedlander;
pub mod kingery;
pub mod kinney;
pub mod loading;
pub mod rankine_hugoniot;
pub mod scaling;
pub mod scenarios;
pub mod sedov;

pub use blast3d::{
    grid_hash, BlastScene, Detonation, DetonationReport, PressureField, ProbeRecord, TNT_DENSITY,
    TNT_ENERGY_J_PER_KG,
};
pub use euler::{
    hllc, sod_l1_density_error, Euler1d, ExactRiemann, Geometry1d, Prim, GAMMA_DEFAULT,
};
pub use euler3d::{hllc3, Boundary, Grid3d, Limiter, Prim3, P_AIR, RHO_AIR};
pub use fragments::{gurney_velocity, mott_sample_many, mott_sample_mass, GurneyGeometry};
pub use friedlander::{
    positive_impulse, pressure as friedlander_pressure, sample as friedlander_sample,
};
pub use kingery::{
    arrival_time, incident_impulse, incident_overpressure, positive_duration,
    reflected_overpressure as kingery_reflected_overpressure, scaled_arrival_time,
    scaled_incident_impulse, scaled_positive_duration, shock_front_velocity, KingeryResult,
};
pub use kinney::{peak_overpressure, peak_overpressure_at, peak_overpressure_ratio, P0_DEFAULT};
pub use loading::{
    apply_blast_impulse, face_positive_impulse, face_pressure, incidence_factor, standoff_factor,
    BlastSource, ExposedFace, PressureImpulseCurve,
};
pub use rankine_hugoniot::{
    dynamic_pressure, from_overpressure_ratio, reflected_overpressure, ShockState, GAMMA_AIR,
};
pub use scaling::{equivalent_tnt_mass, range_from_scaled, scaled_distance};
pub use scenarios::{
    blast3d_corridor_hash, blast3d_corridor_scene, blast_pyramid_hash, blast_pyramid_world,
    BLAST3D_HASH_DIMS, BLAST3D_HASH_DX, BLAST3D_HASH_STEPS, BLAST_HASH_DT, BLAST_HASH_STEPS,
};
pub use sedov::{
    shock_radius, shock_speed, solve_xi0, time_at_radius, xi0_gamma_14_spherical, SedovGeometry,
    SedovSolution,
};
