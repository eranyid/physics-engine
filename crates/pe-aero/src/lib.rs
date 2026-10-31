//! Stage-3 aerodynamics: ISA atmosphere, wing polar / stall, thrust, aircraft forces.
//!
//! Conventions match the Python reference oracle (`src/physics_engine/atmosphere.py`,
//! `aero.py`): SI units, body +x nose / +y right / +z up, α = atan2(−v_z, v_x).
//! Deterministic step path uses portable [`pe_math::libm_ops`] only.
//!
//! **Not modelled:** sideslip / lateral-directional derivatives, ground effect,
//! compressibility (Mach ≈ 1 singularities), advance-ratio propeller maps.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod atmosphere;
pub mod force;
pub mod helpers;
pub mod polar;
pub mod thrust;

pub use atmosphere::{
    density, pressure, speed_of_sound, state, temperature, AtmState, DensityModel,
    IsaAltitudeError, GAMMA_AIR, H_STRATO_TOP, H_TROPOPAUSE, LAPSE, P0, RHO0, R_AIR, T0,
};
pub use force::{AirData, AircraftAero};
pub use helpers::{
    lanchester_phugoid_period, level_flight_trim, level_flight_trim_earth,
    terminal_velocity_vertical, LevelFlightTrim,
};
pub use polar::{dynamic_pressure, PolarError, WingPolar};
pub use thrust::{FixedThrust, JetThrust, PropellerThrust};
