//! Space / orbital dynamics (M7): N-body gravity fields, symplectic and adaptive
//! integrators, Kepler / Lambert helpers, and common orbital perturbations.
//!
//! # Units and frames
//!
//! - **SI only** internally: positions [m], velocities [m/s], times [s], masses [kg],
//!   accelerations [m/s²], GM / μ [m³/s²]. Helpers accept documented km-scale
//!   Earth constants and convert at the boundary.
//! - Prefer storing gravitational parameters **μ = GM** directly rather than
//!   multiplying `G * M` at every force evaluation (reduces round-off and matches
//!   published ephemeris practice).
//! - Default frame: inertial, right-handed, origin chosen by the caller (often the
//!   central body or system barycentre). No Earth-fixed / ECEF transform yet.
//!
//! # Determinism
//!
//! Matches [`docs/DETERMINISM.md`](../../docs/DETERMINISM.md): `f64`, portable
//! [`pe_math::libm_ops`], no `mul_add` / FMA, fixed body order, no `HashMap`
//! iteration on the step path.
//!
//! # Clean-room policy
//!
//! Algorithms are derived from published papers / textbooks cited in module
//! comments and [`THIRD_PARTY.md`](../../THIRD_PARTY.md). **Do not** copy or
//! translate code from GPL/AGPL/LGPL projects (REBOUND, nyx-space, JSBSim, …).
//!
//! # Deferred
//!
//! - Full IAS15 (Rein & Spiegel 2015) — interface stub + tracked TODO.
//! - Full WHFast Wisdom–Holman with symplectic correctors — stub + tracked TODO
//!   (universal-variable Kepler drift helpers used by the stub are implemented).

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod compensated;
pub mod constants;
pub mod diagnostics;
pub mod field;
pub mod integrator;
pub mod integrators;
pub mod kepler;
pub mod lambert;
pub mod perturbations;
pub mod scenarios;
pub mod state;

pub use compensated::{kahan_sum, neumaier_sum, CompensatedVec3};
pub use constants::{
    EARTH_J2, EARTH_MU, EARTH_MU_KM3_S2, EARTH_RADIUS, EARTH_RADIUS_KM, MU_SUN, TWO_PI,
};
pub use diagnostics::{angular_momentum, kinetic_energy, potential_energy, total_energy};
pub use field::{CentralMu, GravityField, NBodyPointMass};
pub use integrator::{AdaptiveStep, Integrator, StepResult};
pub use integrators::{
    dormand_prince_8_7, ias15, leapfrog, rk4, rkf78, whfast, yoshida4, Dop853, Ias15, Leapfrog,
    Rk4, Rkf78, Whfast, Yoshida4,
};
pub use kepler::{
    elements_to_state, mean_motion, orbital_period, propagate_universal, state_to_elements,
    stumpff_c, stumpff_s, KeplerElements, KeplerError,
};
pub use lambert::{lambert_izzo, LambertError, LambertSolution};
pub use perturbations::{
    accel_cannonball_srp, accel_drag_exponential, accel_j2, accel_third_body, DragModel, J2Model,
    SrpModel, ThirdBody,
};
pub use scenarios::{
    figure_eight_hash, figure_eight_state, FIGURE_EIGHT_HASH_STEPS, FIGURE_EIGHT_PERIOD,
};
pub use state::{NBodyState, OriginRebase};
