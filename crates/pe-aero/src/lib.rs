//! Atmosphere, wing polar / stall, thrust, compressibility, Dryden wind, NASA check cases,
//! and blade-element / momentum (BEM) rotors.
//!
//! Conventions match the Python reference oracle (`src/physics_engine/atmosphere.py`,
//! `aero.py`): SI units, body +x nose / +y right / +z up, α = atan2(−v_z, v_x).
//! Deterministic step path uses portable [`pe_math::libm_ops`] only; turbulence /
//! spectra share [`pe_math::Xoshiro256StarStar`].
//!
//! **M11:** [`bem`] Glauert BEMT + Prandtl tip loss — see `docs/ROTOR.md`.
//!
//! **Deferred:** sideslip / lateral-directional derivatives, ground effect,
//! swirl induction `a'`, full 2nd-order Dryden v/w filters.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod atmosphere;
pub mod bem;
pub mod check_cases;
pub mod compressibility;
pub mod force;
pub mod helpers;
pub mod polar;
pub mod thrust;
pub mod turbulence;

pub use atmosphere::{
    density, pressure, speed_of_sound, state, temperature, AtmState, DensityModel,
    IsaAltitudeError, GAMMA_AIR, H_STRATO_TOP, H_TROPOPAUSE, LAPSE, P0, RHO0, R_AIR, T0,
};
pub use bem::{
    figure_of_merit, hover_induced_velocity, ideal_hover_power, ideal_rotor_polar,
    induced_velocity_momentum, nominal_rotor_polar, prandtl_tip_loss, rho_sl, BemResult, BemRotor,
    BemRotorForce, BladeStation, RotorGeometry,
};
pub use check_cases::{
    angular_momentum_about_origin, build_check_case, dragless_ballistic, mechanical_energy,
    quadratic_drag_terminal_speed, quat_norm, run_check_case, CheckCaseConfig, NasaCheckCase,
    NoAeroTorque,
};
pub use compressibility::{
    ackeret_cl, isentropic, karman_tsien, normal_shock, prandtl_glauert, prandtl_glauert_factor,
    Isentropic, NormalShock, PG_MACH_CLAMP,
};
pub use force::{AirData, AircraftAero};
pub use helpers::{
    lanchester_phugoid_period, level_flight_trim, level_flight_trim_earth,
    terminal_velocity_vertical, LevelFlightTrim,
};
pub use polar::{dynamic_pressure, PolarError, WingPolar};
pub use thrust::{FixedThrust, JetThrust, PropellerThrust};
pub use turbulence::{DrydenParams, DrydenTurbulence, SteadyWind, WindField};
