//! Stage-3 hydrostatics + waves: hull buoyancy, GM/GZ, Airy/Stokes/Gerstner/spectra.
//!
//! Conventions match the Python reference (`src/physics_engine/hydro.py`) for
//! still-water hydrostatics: SI units, body +x bow / +y starboard / +z up,
//! waterplane `z = water_level`. Wave conventions: see [`waves`].
//! Deterministic mesh clipping and wave sums use portable [`pe_math::libm_ops`]
//! (no `HashMap` in the step; spectral phases from seeded xoshiro256**).
//!
//! **Deferred:** Morison forces, strip-theory / panel seakeeping, radiation damping.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod clip;
pub mod force;
pub mod helpers;
pub mod hull;
pub mod scenarios;
pub mod waves;

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
pub use scenarios::canonical_wave_body_hash;
pub use waves::{
    deep_water_wavelength, dispersion_omega, group_speed, jonswap_spectrum, phase_speed,
    pierson_moskowitz_hs_approx, pierson_moskowitz_spectrum, solve_wave_number, AiryWave,
    GerstnerError, GerstnerWave, SpectralOcean, SpectrumKind, SpreadingParams, StokesSecondOrder,
    Wave, WaveComponent, WaveField,
};
