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
//! self-collision via a deterministic spatial hash, and **two-way coupling** to
//! dynamic `pe-core` rigid bodies (attachments + particle↔convex contacts).
//!
//! Materials track: J2 von Mises plasticity with linear isotropic hardening,
//! max-principal stress/strain fracture (distance + tet) with deterministic
//! ordering and island labels, lumped / **implicit** heat conduction +
//! thermal expansion (M29), **XPBD ↔ J2 plastic rest coupling** (M30),
//! **finite-strain multiplicative F_p / Hencky bar plasticity** (M31),
//! **spectral Hencky / Kirchhoff return on `b_e`** (M32, Simo 1992), and
//! **bilinear cohesive-zone fracture / crack-path tracking** (M33).
//!
//! # Determinism
//!
//! Matches [`docs/DETERMINISM.md`](../../docs/DETERMINISM.md): `f64`, portable
//! [`pe_math::libm_ops`], no FMA, fixed iteration order, no `HashMap` on the
//! step path, self-collision pairs sorted by `(min_id, max_id)`, fracture
//! candidates sorted by `(kind, index)`, cohesive crack events in ascending
//! bond-index discovery order.
//!
//! # Units
//!
//! SI throughout. Material table: demo rows are **nominal** (`verified: false`);
//! handbook rows may be `verified: true` with citations in `source_note`
//! unless a primary source was fetched — see [`docs/MATERIALS.md`](../../docs/MATERIALS.md).
//!
//! # Deferred
//!
//! Corotational FEM, anisotropic cloth, radiation / `k(T)`, GPU/SIMD — see
//! [`docs/SOFT.md`](../../docs/SOFT.md).

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod builders;
pub mod cohesive_fracture;
pub mod collision;
pub mod constraint;
pub mod coupling;
pub mod energy;
pub mod finite_strain_plastic;
pub mod fracture;
pub mod material;
pub mod particle;
pub mod plastic_coupling;
pub mod plasticity;
pub mod scenarios;
pub mod solver;
pub mod spectral_hencky_plastic;
pub mod thermal;

pub use builders::{
    add_shape_match_all, make_axial_bar_grid, make_axial_bar_springs, make_axial_bar_tets,
    make_balloon, make_cantilever, make_cloth, make_rope, make_soft_cube,
};
pub use cohesive_fracture::{
    analytic_gc, cohesive_canonical_hash, cohesive_hash_params, cohesive_hash_world,
    damage_from_delta_max, discrete_work_to_failure, eval_bilinear_czm, integrate_virgin_envelope,
    make_cohesive_bar, make_cohesive_chain, path_work_unit_area, quasistatic_cohesive_open,
    traction_virgin_envelope, update_cohesive_bonds, BilinearCzmParams, CohesiveBond,
    CohesiveSoftWorld, CohesiveUpdateStats, CrackEvent, CzmEval, COHESIVE_HASH_DT,
    COHESIVE_HASH_STEPS,
};
pub use collision::{
    SelfCollisionParams, SelfCollisionScratch, StaticBox, StaticColliders, StaticPlane,
    StaticSphere,
};
pub use constraint::{
    mesh_volume, tetra_dm_inv, tetra_volume, ConstraintSet, DihedralBendConstraint,
    DistanceBendConstraint, DistanceConstraint, Multiplier, PressureConstraint,
    ShapeMatchConstraint, TetraFemConstraint, VolumeConstraint,
};
pub use coupling::{
    make_damped_attachment_oscillator, make_momentum_exchange_scene, make_pendulum_bob,
    make_rope_on_free_body, make_soft_on_box_on_ground, soft_coupling_canonical_hash,
    soft_coupling_hash_world, soft_coupling_state_hash64, AttachmentConstraint,
    CouplingDiagnostics, RigidCollider, RigidShape, SoftRigidWorld, COUPLING_HASH_DT,
    COUPLING_HASH_STEPS,
};
pub use energy::{
    gravitational_potential, kinetic_energy, stretch_potential, total_energy, volume_error_sum,
};
pub use finite_strain_plastic::{
    analytic_log_permanent_strain, analytic_log_uniaxial_stress, bar_log_strain,
    fs_bar_axial_force, fs_plastic_canonical_hash, fs_plastic_hash_params, fs_plastic_hash_world,
    isochoric_project, make_fs_plastic_bar, quasistatic_log_stretch_unload, sym_eigenvalues_3,
    sym_exp3, update_fs_plastic_distances, update_fs_plastic_tets, FsPlasticDistance,
    FsPlasticSoftWorld, FsPlasticTet, FsPlasticUpdateStats, FS_PLASTIC_HASH_DT,
    FS_PLASTIC_HASH_STEPS,
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
pub use plastic_coupling::{
    analytic_permanent_strain, analytic_uniaxial_stress, bar_axial_force, bar_engineering_strain,
    make_plastic_bar, plastic_canonical_hash, plastic_hash_params, plastic_hash_world,
    quasistatic_stretch_unload, update_plastic_distances, update_plastic_tets, PlasticDistance,
    PlasticSoftWorld, PlasticTet, PlasticUpdateStats, PLASTIC_HASH_DT, PLASTIC_HASH_STEPS,
};
pub use plasticity::{
    return_map_j2_3d, return_map_uniaxial, uniaxial_tension_curve, J2Params, J2PlasticState3,
    J2Return3, SymTensor3, UniaxialPlasticState, UniaxialReturn,
};
pub use scenarios::{
    materials_canonical_hash, soft_canonical_hash, soft_canonical_world, thermal_canonical_hash,
    MATERIALS_HASH_DT, MATERIALS_HASH_STEPS, SOFT_HASH_DT, SOFT_HASH_STEPS, THERMAL_HASH_DT,
    THERMAL_HASH_STEPS,
};
pub use solver::{soft_state_hash64, SoftParams, SoftWorld};
pub use spectral_hencky_plastic::{
    analytic_isochoric_uniaxial_kirchhoff, analytic_isochoric_uniaxial_seq,
    assemble_from_principal, be_bar_update_stats_from_fs, be_plastic_canonical_hash,
    be_plastic_hash_params, be_plastic_hash_world, left_cauchy_green, make_be_plastic_bar,
    quasistatic_be_log_stretch_unload, return_map_spectral_hencky_be, sym_spectral_decompose,
    update_be_plastic_tets, BePlasticSoftWorld, BePlasticTet, BePlasticUpdateStats,
    SpectralHenckyReturn, BE_PLASTIC_HASH_DT, BE_PLASTIC_HASH_STEPS,
};
pub use thermal::{
    apply_thermal_expansion_from_cold, apply_thermal_expansion_tets_from_cold,
    apply_thermal_expansion_volumes_from_cold, clamped_thermal_stress, expanded_rest_length,
    free_expansion_delta_l, init_rod_sine_mode, robin_lumped_exp, rod_fourier_mode1,
    semi_infinite_erfc, steady_linear_profile, ThermalEdge, ThermalMesh, ThermalProps,
    ThermalScheme,
};
