//! Stage-3 hydrostatics + waves + Morison + strip-theory seakeeping.
//!
//! Conventions match the Python reference (`src/physics_engine/hydro.py`) for
//! still-water hydrostatics: SI units, body +x bow / +y starboard / +z up,
//! waterplane `z = water_level`. Wave conventions: see [`waves`].
//! Deterministic mesh clipping and wave sums use portable [`pe_math::libm_ops`]
//! (no `HashMap` in the step; spectral phases from seeded xoshiro256**).
//!
//! **M11:** [`morison`] (slender-member wave loads) and [`seakeeping`] (STF
//! heave/pitch strip theory). **Cummins** retardation memory: [`cummins`].
//! See `docs/SEAKEEPING.md`.
//!
//! **Deferred:** 6-DOF / coupled Cummins, MacCamy–Fuchs, tabulated Vugts
//! sections, forward-speed strip theory.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod clip;
pub mod cummins;
pub mod force;
pub mod helpers;
pub mod hull;
pub mod morison;
pub mod scenarios;
pub mod seakeeping;
pub mod waves;

pub use cummins::{
    box_heave_b33_samples, box_heave_retardation, canonical_cummins_hash, damping_from_retardation,
    free_decay_cummins_vs_fd, relative_l2, retardation_from_damping, CumminsHeave,
    RetardationKernel,
};
pub use force::{Buoyancy, HullDrag, WaveDrag};
pub use helpers::{
    angle_of_vanishing_stability, bm_from_inertia, box_added_mass_heave, box_added_mass_surge,
    box_righting_arm_curve, check_imo_is_code, heave_period, heave_period_earth,
    heave_period_with_added_mass, metacentric_height, righting_arm, righting_moment, roll_period,
    roll_period_earth, IsCodeResult,
};
pub use hull::{
    sphere_cap_volume, BoxHull, ConvexMeshHull, CylinderHull, Hull, SphereHull, SubmergedState,
    RHO_FRESHWATER, RHO_SEAWATER,
};
pub use morison::{
    airy_drag_closed_form_no_reversal, airy_inertia_closed_form, integrate_morison,
    integrate_morison_fixed, normal_component, MorisonForce, MorisonLoad, MorisonMember,
    NOMINAL_CD_CIRCULAR, NOMINAL_CM_CIRCULAR,
};
pub use scenarios::{canonical_m11_hash, canonical_wave_body_hash};
pub use seakeeping::{
    assemble_strip_matrices, box_barge_stations, box_heave_natural_omega,
    box_heave_natural_omega_earth, free_decay_heave, heave_pitch_rao, roll_natural_period,
    sectional_a33, sectional_a33_inf, sectional_b33, HeavePitchRao, HeavePitchTd, StripMatrices,
    StripStation,
};
pub use waves::{
    deep_water_wavelength, dispersion_omega, group_speed, jonswap_spectrum, phase_speed,
    pierson_moskowitz_hs_approx, pierson_moskowitz_spectrum, solve_wave_number, AiryWave,
    GerstnerError, GerstnerWave, SpectralOcean, SpectrumKind, SpreadingParams, StokesSecondOrder,
    Wave, WaveComponent, WaveField,
};
