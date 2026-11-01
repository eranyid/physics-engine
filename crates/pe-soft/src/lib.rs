//! XPBD soft bodies, cloth, ropes, continuum materials, plasticity, fracture, thermal.
//!
//! # Scope
//!
//! Particles with inverse mass, XPBD constraints with compliance
//! (`α̃ = α/dt²`), substep-based solving (Macklin 2019 *Small Steps*), and
//! Lagrange multiplier accumulation (Macklin, Müller, Chentanez 2016).
//!
//! Constraints: distance, distance-based / dihedral bending, tetra volume,
//! tetrahedral StVK / stable neo-Hookean FEM (Macklin & Müller 2021), optional
//! shape matching, pressure/volume balloons, static plane/sphere/box contacts,
//! and self-collision via a deterministic spatial hash.
//!
//! Materials track: J2 von Mises plasticity with linear isotropic hardening,
//! max-principal stress/strain fracture (distance + tet) with deterministic
//! ordering and island labels, and lumped heat conduction + thermal expansion.
//!
//! # Determinism
//!
//! Matches [`docs/DETERMINISM.md`](../../docs/DETERMINISM.md): `f64`, portable
//! [`pe_math::libm_ops`], no FMA, fixed iteration order, no `HashMap` on the
//! step path, self-collision pairs sorted by `(min_id, max_id)`, fracture
//! candidates sorted by `(kind, index)`.
//!
//! # Units
//!
//! SI throughout. Material table: demo rows are **nominal** (`verified: false`);
//! handbook rows may be `verified: true` with citations in `source_note`
//! unless a primary source was fetched — see [`docs/MATERIALS.md`](../../docs/MATERIALS.md).
//!
//! # Deferred
//!
//! Corotational FEM, anisotropic cloth, two-way rigid coupling, GPU/SIMD —
//! see [`docs/SOFT.md`](../../docs/SOFT.md).

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod builders;
pub mod collision;
pub mod constraint;
pub mod energy;
pub mod fracture;
pub mod material;
pub mod particle;
pub mod plasticity;
pub mod scenarios;
pub mod solver;
pub mod thermal;

pub use builders::{
    add_shape_match_all, make_axial_bar_grid, make_axial_bar_springs, make_axial_bar_tets,
    make_balloon, make_cantilever, make_cloth, make_rope, make_soft_cube,
};
pub use collision::{SelfCollisionParams, StaticBox, StaticColliders, StaticPlane, StaticSphere};
pub use constraint::{
    mesh_volume, tetra_dm_inv, tetra_volume, ConstraintSet, DihedralBendConstraint,
    DistanceBendConstraint, DistanceConstraint, Multiplier, PressureConstraint,
    ShapeMatchConstraint, TetraFemConstraint, VolumeConstraint,
};
pub use energy::{
    gravitational_potential, kinetic_energy, stretch_potential, total_energy, volume_error_sum,
};
pub use fracture::{
    active_constraint_count, apply_fracture, constraint_islands, distance_axial_strain,
    distance_axial_stress, fracture_released_spring_energy, soft_state_has_nan, FractureCriterion,
    FractureEvent, FractureKind,
};
pub use material::{
    alpha_tilde, bar_wave_speed, builtin_materials, bulk_modulus, lame_lambda, lame_mu,
    material_by_name, stiffness_from_compliance, stretch_compliance, stretch_stiffness, Material,
};
pub use particle::Particle;
pub use plasticity::{
    return_map_j2_3d, return_map_uniaxial, uniaxial_tension_curve, J2Params, J2PlasticState3,
    J2Return3, SymTensor3, UniaxialPlasticState, UniaxialReturn,
};
pub use scenarios::{
    materials_canonical_hash, soft_canonical_hash, soft_canonical_world, MATERIALS_HASH_DT,
    MATERIALS_HASH_STEPS, SOFT_HASH_DT, SOFT_HASH_STEPS,
};
pub use solver::{soft_state_hash64, SoftParams, SoftWorld};
pub use thermal::{
    apply_thermal_expansion_from_cold, expanded_rest_length, free_expansion_delta_l,
    init_rod_sine_mode, rod_fourier_mode1, steady_linear_profile, ThermalEdge, ThermalMesh,
    ThermalProps,
};
