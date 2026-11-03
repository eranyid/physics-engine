//! Stage-3 hydrostatics + waves + Morison + strip-theory seakeeping + M21 ships/sea.
//!
//! Conventions match the Python reference (`src/physics_engine/hydro.py`) for
//! still-water hydrostatics: SI units, body +x bow / +y starboard / +z up,
//! waterplane `z = water_level`. Wave conventions: see [`waves`].
//! Deterministic mesh clipping and wave sums use portable [`pe_math::libm_ops`]
//! (no `HashMap` in the step; spectral phases from seeded xoshiro256**).
//!
//! **M11:** [`morison`] (slender-member wave loads) and [`seakeeping`] (STF
//! heave/pitch strip theory). **Cummins** retardation memory: [`cummins`].
//!
//! **M21:** six-DOF Cummins [`cummins6`], forward-speed STF / encounter
//! frequency [`forward_speed`], ITTC-57 resistance [`resistance`], Nomoto /
//! linear MMG maneuvering [`maneuvering`]. See `docs/SEAKEEPING.md`.
//!
//! **Deferred:** MacCamy–Fuchs; tabulated Vugts sections; verified Holtrop–
//! Mennen coefficient tables; KVLCC2/SIMMAN derivative tables.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod clip;
pub mod cummins;
pub mod cummins6;
pub mod force;
pub mod forward_speed;
pub mod helpers;
pub mod hull;
pub mod maneuvering;
pub mod morison;
pub mod resistance;
pub mod scenarios;
pub mod seakeeping;
pub mod waves;

pub use cummins::{
    added_mass_delta_from_retardation, box_heave_b33_samples, box_heave_retardation,
    canonical_cummins_hash, cummins_char_residual, damping_from_retardation, fit_free_decay_pole,
    free_decay_cummins_vs_pole, free_decay_frozen_b_mismatch, free_decay_pole, heave_energy_proxy,
    laplace_retardation, relative_l2, retardation_from_damping, CumminsHeave, RetardationKernel,
};
pub use cummins6::{
    box_a_inf_matrix, box_hydrostatic_restoring, box_retardation_6dof, canonical_cummins6_hash,
    excitation_head_seas_6, excitation_head_seas_time, heave_pitch_natural_omegas,
    roll_decay_analytic, roll_decay_rk4, sectional_a22_inf, sectional_b22, solve6,
    truncation_tail_l1, Cummins6Dof, MatrixRetardationKernel, RigidMass6, N_DOF,
};
pub use force::{Buoyancy, FrictionModel, HullDrag, WaveDrag};
pub use forward_speed::{
    assemble_strip_matrices_forward_speed, encounter_frequency, encounter_frequency_deep,
    encounter_frequency_dispersive, ship_froude,
};
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
pub use maneuvering::{
    nominal_linear_mmg_unverified, nomoto_from_linear_mmg, propeller_thrust_unverified,
    rudder_force_unverified, zigzag_10_10, LinearMmgDerivatives, MmgPlanar, NomotoKT, NomotoState,
};
pub use morison::{
    airy_drag_closed_form_no_reversal, airy_inertia_closed_form, integrate_morison,
    integrate_morison_fixed, normal_component, MorisonForce, MorisonLoad, MorisonMember,
    NOMINAL_CD_CIRCULAR, NOMINAL_CM_CIRCULAR,
};
pub use resistance::{
    calm_water_resistance, froude_number, ittc57_cf, ittc57_cf_from_speed, ittc57_friction_force,
    residual_cf_linear_froude_unverified, reynolds_number, NU_FRESHWATER_20C_NOMINAL,
    NU_SEAWATER_15C_NOMINAL,
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
