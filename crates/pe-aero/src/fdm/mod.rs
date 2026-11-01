//! Six-degree-of-freedom rigid-body flight dynamics.
//!
//! Body-frame forces and moments from nondimensional stability derivatives
//! (`C_L`, `C_D`, `C_Y`, `C_l`, `C_m`, `C_n` vs α, β, p̂, q̂, r̂, and control
//! surfaces), thrust with optional COM offset, ISA density, and optional
//! steady/Dryden wind. Attitude is the engine quaternion (body→world).
//!
//! Includes a steady level-flight trim solver, numerical state-space
//! linearisation, and eigenmode extraction for longitudinal dynamics.
//!
//! **Data provenance:** aircraft tables that ship with citations are marked
//! `verified: true`; anything else is `verified: false` / unverified.

pub mod derivatives;
pub mod eigen;
pub mod forces;
pub mod linearise;
pub mod navion;
pub mod trim;

pub use derivatives::{ControlInputs, StabilityDerivatives};
pub use eigen::{classify_longitudinal_modes, ComplexEigen, LongitudinalModes};
pub use forces::{
    air_angles, body_rates_hat, eval_aero_body, SixDofAero, SixDofForces, SixDofModel,
};
pub use linearise::{
    coordinated_turn_load_factor, coordinated_turn_rate, linearise_longitudinal,
    phugoid_approximation, LongitudinalStateSpace,
};
pub use navion::{navion_nelson, NavionData};
pub use trim::{trim_steady_level, TrimGuess, TrimResult, TrimStatus};
