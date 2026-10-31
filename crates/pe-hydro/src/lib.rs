//! Stage-3 hydrostatics: hull buoyancy, GM/GZ, periods, hull drag, IMO IS Code.
//!
//! Conventions match the Python reference (`src/physics_engine/hydro.py`): SI units,
//! body +x bow / +y starboard / +z up, waterplane `z = water_level`. Deterministic
//! mesh clipping uses portable [`pe_math::libm_ops`] (no `HashMap` in the step).
//!
//! **Not modelled (later milestones):** Airy / spectral waves, 6-DOF seakeeping,
//! radiation damping. Deep-water wave helpers are omitted because the Python
//! oracle does not define them.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod clip;
pub mod force;
pub mod helpers;
pub mod hull;

pub use force::{Buoyancy, HullDrag, WaveDrag};
pub use helpers::{
    angle_of_vanishing_stability, bm_from_inertia, box_added_mass_heave, box_added_mass_surge,
    box_righting_arm_curve, check_imo_is_code, heave_period, heave_period_earth,
    metacentric_height, righting_arm, righting_moment, roll_period, roll_period_earth,
    IsCodeResult,
};
pub use hull::{
    sphere_cap_volume, BoxHull, ConvexMeshHull, CylinderHull, Hull, SphereHull, SubmergedState,
    RHO_FRESHWATER, RHO_SEAWATER,
};
