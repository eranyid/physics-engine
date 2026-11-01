//! Point-mass external ballistics, ammo profiles, and per-bullet simulation.
//!
//! # Scope
//!
//! Game / studio simulation of individual projectiles: `AmmoProfile` (serde),
//! SoA `ProjectileSystem`, ISA atmosphere + Dryden wind from `pe-aero`, Coriolis /
//! Eötvös, spin-drift / Magnus engineering models, terminal impact helpers, and
//! tunneling-free hit-scan against `pe-collision` shapes (read-only).
//!
//! **Drag:** published G1/G7 Cd–Mach tables (JBM/BRL transcription) with linear
//! interpolation and classical BC scaling; see [`drag`], [`g1g7_tables`],
//! [`docs/BALLISTICS.md`](../../docs/BALLISTICS.md), [`docs/AMMO.md`](../../docs/AMMO.md).
//!
//! # Determinism
//!
//! `f64`, portable `libm`, no FMA, projectiles stepped in ascending [`ShotId`]
//! order, per-shot RNG seeded from the shot id.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod ammo;
pub mod drag;
pub mod earth;
#[allow(clippy::approx_constant)] // Cd table knots, not f64::consts
pub mod g1g7_tables;
pub mod hitscan;
pub mod impact;
pub mod oracles;
pub mod projectile;
pub mod scenarios;
pub mod spin;
pub mod table;

pub use ammo::{
    bundled_by_name, bundled_profiles, AmmoProfile, BallisticCoefficient, FieldProvenance,
    AMMO_PROFILE_VERSION,
};
pub use drag::{
    analytic_cd, analytic_cd_params, area_from_bc_si, bc_imperial_to_si, bc_si_to_imperial,
    caliber_area, drag_force_magnitude, g1_table, g7_table, standard_cd, DragModelRef, DragTable,
    BC_LB_IN2_TO_KG_M2,
};
pub use earth::{
    freefall_eastward_deflection, gravity_magnitude, LocalFrame, EARTH_OMEGA, EARTH_RADIUS_M,
};
pub use hitscan::{
    ray_sphere, segment_sphere, segment_vs_box, segment_vs_capsule, segment_vs_shape,
    segment_vs_shapes, segment_vs_shapes_excluding, segment_vs_sphere, shape_bounding_radius,
    shape_exit, Hit, ShapeRef,
};
pub use impact::{
    apply_impact_impulse, kinetic_energy, momentum, poncelet_depth, recht_ipson_residual,
};
pub use oracles::{
    coriolis_freefall_east, energy_residual, g0, linear_drag_vx, linear_drag_vz,
    quadratic_terminal_speed, vacuum_flight_time, vacuum_max_height, vacuum_range,
};
pub use projectile::{
    apply_hit_impulses, HitEvent, HitOutcome, HitResolver, MuzzlePose, Projectile,
    ProjectileSystem, RecoilImpulse, ShotId, StopOnHit, MAX_INTERACTIONS_PER_SUBSTEP,
};
pub use scenarios::{
    ballistics_volley_hash, ballistics_volley_system, BALLISTICS_HASH_DT, BALLISTICS_HASH_N,
    BALLISTICS_HASH_STEPS,
};
pub use spin::{magnus_accel, spin_drift_accel, DEFAULT_MAGNUS_K, DEFAULT_SPIN_DRIFT_K};
pub use table::{generate_table, TableRow};
