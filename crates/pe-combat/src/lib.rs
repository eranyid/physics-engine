//! Combat physics for games and studio simulation.
//!
//! # Scope
//!
//! What happens **after** a projectile reaches a surface, built on top of
//! [`pe_ballistics`] flight and [`pe_blast`] loading:
//!
//! - [`armor`] — versioned armor / medium materials and layered stacks.
//! - [`penetration`] — Lambert–Jonas plate perforation, Poncelet media,
//!   ricochet, post-perforation yaw, spall, and the [`CombatResolver`] that
//!   plugs into [`pe_ballistics::ProjectileSystem::step_with_resolver`].
//! - [`hitzones`] — characters, hit zones, vulnerability, health state.
//! - [`damage`] — target table, damage events, wounds and bleeding.
//! - [`blast_damage`] — blast pressure / impulse injury per zone.
//! - [`fragments`] — spall and blast fragments as projectiles.
//!
//! All material constants are **caller-supplied** or bundled as nominal
//! `generic_*` rows (`verified: false`). These are engineering approximations
//! for games — not armor or weapon design tools.
//!
//! # Determinism
//!
//! `f64`, portable `libm`, no FMA, no `HashMap` on the step path. Every
//! stochastic choice draws from the hit projectile's own per-shot RNG.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]
// `!(x > 0.0)` deliberately rejects NaN in validation paths.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod armor;
pub mod blast_damage;
pub mod damage;
pub mod fragments;
pub mod hitzones;
pub mod penetration;

pub use armor::{
    bundled_material_by_name, bundled_materials, ArmorLayer, ArmorMaterial, ArmorStack,
    PenetrationModel, RicochetParams, SpallParams, ARMOR_MATERIAL_VERSION,
};
pub use blast_damage::{blast_on_character, total_push, ZoneBlast};
pub use damage::{apply_damage_events, DamageEvent, DamageOutcome, Target, TargetTable};
pub use fragments::{
    fragment_drag_multiplier, fragment_profile, fragment_shot_id, spawn_all_spall,
    spawn_fragment_burst, spawn_spall, FRAGMENT_FORM_FACTOR, FRAGMENT_ID_BIT,
};
pub use hitzones::{
    default_vulnerability, BlastTolerance, Character, Condition, HealthModifiers, HealthParams,
    HealthState, HitZone, Limb, Wound, ZoneKind, ZoneShape, ZoneVulnerability,
};
pub use penetration::{
    lambert_jonas_v50, lambert_z, poncelet_exit_speed, resolve_stack, ricochet_probability,
    ricochet_velocity, CombatResolver, ImpactKind, ImpactRecord, SpallFragment, StackInput,
    StackResult,
};
