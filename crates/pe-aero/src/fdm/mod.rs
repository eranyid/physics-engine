//! Six-degree-of-freedom rigid-body flight dynamics.
//!
//! Body-frame forces and moments from nondimensional stability derivatives
//! (`C_L`, `C_D`, `C_Y`, `C_l`, `C_m`, `C_n` vs α, β, α̇, p̂, q̂, r̂, and control
//! surfaces), thrust with optional COM offset, ISA density, optional
//! steady/Dryden wind, and Wieselsberger/McCormick ground effect. Attitude is
//! the engine quaternion (body→world).
//!
//! Includes a steady level-flight trim solver, numerical state-space
//! linearisation (longitudinal + lateral-directional), and eigenmode
//! extraction.
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
pub use eigen::{
    classify_lateral_modes, classify_longitudinal_modes, ComplexEigen, LateralModes,
    LongitudinalModes,
};
pub use forces::{
    air_angles, alpha_dot_hat, alpha_dot_kinematic, body_rates_hat, eval_aero_body,
    eval_aero_body_with_alphadot, SixDofAero, SixDofForces, SixDofModel,
};
pub use linearise::{
    coordinated_turn_load_factor, coordinated_turn_rate, linearise_lateral, linearise_longitudinal,
    linearise_longitudinal_ex, phugoid_approximation, short_period_approximation,
    short_period_approximation_with_alphadot, LateralStateSpace, LongitudinalStateSpace,
};
pub use navion::{navion_nelson, NavionData};
pub use trim::{trim_steady_level, TrimGuess, TrimResult, TrimStatus};
