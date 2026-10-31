//! XPBD soft bodies, cloth, ropes, and basic continuum material models.
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
//! # Determinism
//!
//! Matches [`docs/DETERMINISM.md`](../../docs/DETERMINISM.md): `f64`, portable
//! [`pe_math::libm_ops`], no FMA, fixed iteration order, no `HashMap` on the
//! step path, self-collision pairs sorted by `(min_id, max_id)`.
//!
//! # Units
//!
//! SI throughout. Material table values are **nominal** and `verified: false`
//! unless a primary source was fetched.
//!
//! # Deferred
//!
//! Corotational FEM, plasticity, fracture, and thermal coupling — see
//! [`docs/SOFT.md`](../../docs/SOFT.md).

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod builders;
pub mod collision;
pub mod constraint;
pub mod energy;
pub mod material;
pub mod particle;
pub mod scenarios;
pub mod solver;

pub use builders::{
    add_shape_match_all, make_axial_bar_tets, make_balloon, make_cantilever, make_cloth, make_rope,
    make_soft_cube,
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
pub use material::{
    alpha_tilde, bar_wave_speed, builtin_materials, bulk_modulus, lame_lambda, lame_mu,
    material_by_name, stiffness_from_compliance, stretch_compliance, stretch_stiffness, Material,
};
pub use particle::Particle;
pub use scenarios::{soft_canonical_hash, soft_canonical_world, SOFT_HASH_DT, SOFT_HASH_STEPS};
pub use solver::{soft_state_hash64, SoftParams, SoftWorld};
