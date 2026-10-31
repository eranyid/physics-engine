//! Engineering blast / shock models for games and studio simulation.
//!
//! # Scope
//!
//! Physical simulation of blast **loading**, shock waves, and fragment
//! kinematics: Hopkinson–Cranz scaling, Kinney–Graham overpressure, Friedlander
//! waveforms, Rankine–Hugoniot jumps, Sedov–Taylor point blasts, a 1-D Euler
//! solver, rigid-body blast impulses, and Gurney/Mott fragment helpers.
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

pub mod euler;
pub mod fragments;
pub mod friedlander;
pub mod kingery;
pub mod kinney;
pub mod loading;
pub mod rankine_hugoniot;
pub mod scaling;
pub mod scenarios;
pub mod sedov;

pub use euler::{
    hllc, sod_l1_density_error, Euler1d, ExactRiemann, Geometry1d, Prim, GAMMA_DEFAULT,
};
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
pub use scenarios::{blast_pyramid_hash, blast_pyramid_world, BLAST_HASH_DT, BLAST_HASH_STEPS};
pub use sedov::{
    shock_radius, shock_speed, solve_xi0, time_at_radius, xi0_gamma_14_spherical, SedovGeometry,
    SedovSolution,
};
