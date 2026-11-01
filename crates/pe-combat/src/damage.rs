//! From impacts to wounds: target table, damage events, health application.
//!
//! The [`TargetTable`] says what each shape index in the projectile system's
//! `shapes` slice is. Character zones resolve `zone armor + tissue`; the
//! energy dissipated in the tissue layer drives damage, bleeding, pain and
//! limb impairment, and energy absorbed by worn armor becomes blunt trauma.
//!
//! ```text
//! blunt  = blunt_factor · E_armor
//! damage = damage_factor · (E_tissue + w_blunt · blunt) / j_per_hp      [HP]
//! bleed  = bleed_l_per_s_per_j · E_tissue,  r(t) = r0 e^{−t/τ}          [L/s]
//! pain  += (E_tissue + blunt) / pain_j
//! limb   *= e^{−g (E_tissue + w_blunt · blunt)}
//! ```

use core::ops::Range;

use pe_ballistics::projectile::ShotId;
use pe_math::libm_ops;
use pe_math::Vec3;

use crate::armor::{ArmorLayer, ArmorStack};
use crate::hitzones::{Character, HealthState, ZoneKind, ZoneVulnerability};

/// What a shape is, for the [`crate::CombatResolver`].
#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    /// Stops projectiles; no physics beyond the hit record.
    Inert,
    /// Armor or medium (walls, plates, vehicles, terrain).
    Armor(ArmorStack),
    /// A character hit zone. `stack` = zone armor + tissue (last layer).
    Zone {
        /// Character index (into the slice given to [`apply_damage_events`]).
        character: usize,
        /// Zone index within the character.
        zone: usize,
        /// Armor layers followed by the tissue layer.
        stack: ArmorStack,
    },
}

/// Per-shape targets, indexed like the `shapes` slice.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TargetTable {
    /// Targets.
    pub targets: Vec<Target>,
}

impl TargetTable {
    /// Empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// From per-shape armor (`None` → [`Target::Inert`]).
    pub fn from_armor(shape_armor: Vec<Option<ArmorStack>>) -> Self {
        Self {
            targets: shape_armor
                .into_iter()
                .map(|a| a.map(Target::Armor).unwrap_or(Target::Inert))
                .collect(),
        }
    }

    /// Append a target; returns its shape index.
    pub fn push(&mut self, t: Target) -> usize {
        self.targets.push(t);
        self.targets.len() - 1
    }

    /// Target for shape index `i`.
    pub fn get(&self, i: usize) -> Option<&Target> {
        self.targets.get(i)
    }

    /// Number of targets.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Append one [`Target::Zone`] per zone of `ch` (zone order, matching
    /// [`Character::zone_shapes`]). Returns the shape-index range used.
    pub fn push_character(&mut self, character_index: usize, ch: &Character) -> Range<usize> {
        let start = self.targets.len();
        for (zi, z) in ch.zones.iter().enumerate() {
            let mut layers = z.armor.layers.clone();
            layers.push(ArmorLayer {
                material: z.tissue_material,
                thickness_m: 0.0,
            });
            self.targets.push(Target::Zone {
                character: character_index,
                zone: zi,
                stack: ArmorStack { layers },
            });
        }
        start..self.targets.len()
    }
}

/// One hit on a character zone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamageEvent {
    /// Character index.
    pub character: usize,
    /// Zone index.
    pub zone: usize,
    /// Shot id.
    pub shot_id: ShotId,
    /// Time [s].
    pub time_s: f64,
    /// Entry point [m].
    pub position: Vec3,
    /// Incident unit direction.
    pub direction: Vec3,
    /// Incident kinetic energy [J].
    pub ke_in_j: f64,
    /// Energy deposited in tissue [J] (incl. spall from breached worn armor).
    pub e_tissue_j: f64,
    /// Energy absorbed by worn armor [J].
    pub e_armor_j: f64,
    /// Bullet got through worn armor into tissue.
    pub penetrated_armor: bool,
    /// Bullet exited the zone.
    pub through_and_through: bool,
    /// Bullet glanced off.
    pub ricochet: bool,
    /// Wound track length [m].
    pub wound_track_m: f64,
    /// Incident momentum [kg·m/s].
    pub momentum_in: Vec3,
    /// Outgoing momentum [kg·m/s].
    pub momentum_out: Vec3,
}

impl DamageEvent {
    /// Momentum transferred to the body [kg·m/s] (for hit reactions).
    pub fn momentum_transfer(&self) -> Vec3 {
        self.momentum_in - self.momentum_out
    }
}

/// What one event did to a character.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamageOutcome {
    /// Character index.
    pub character: usize,
    /// Zone kind.
    pub zone: ZoneKind,
    /// Damage added [HP].
    pub damage_hp: f64,
    /// Initial bleed rate of the new wound [L/s].
    pub bleed_l_s: f64,
    /// Blunt energy [J].
    pub blunt_j: f64,
    /// Pain added.
    pub pain: f64,
    /// Instant incapacitation triggered.
    pub instant_incap: bool,
}

impl HealthState {
    /// Apply one hit on a zone of kind `kind` with vulnerability `v`.
    pub fn apply_hit(
        &mut self,
        ev: &DamageEvent,
        kind: ZoneKind,
        v: &ZoneVulnerability,
    ) -> DamageOutcome {
        let p = self.params;
        let blunt = v.blunt_factor * ev.e_armor_j.max(0.0);
        let e_t = ev.e_tissue_j.max(0.0);
        let weighted = e_t + p.blunt_weight * blunt;
        let damage = v.damage_factor * weighted / p.j_per_hp;
        let bleed = v.bleed_l_per_s_per_j * e_t;
        let pain = (e_t + blunt) / p.pain_j;
        if let Some(l) = kind.limb() {
            self.limb_function[l as usize] *= libm_ops::exp(-v.impairment_per_j * weighted);
        }
        self.add_wound(kind, bleed, v.clot_tau_s);
        self.add_damage(damage, pain);
        let instant = v.instant_incap_j.map(|th| e_t >= th).unwrap_or(false);
        if instant {
            self.incapacitate();
        }
        DamageOutcome {
            character: ev.character,
            zone: kind,
            damage_hp: damage,
            bleed_l_s: bleed,
            blunt_j: blunt,
            pain,
            instant_incap: instant,
        }
    }
}

/// Apply events to characters (`event.character` indexes `characters`).
/// Events for unknown characters / zones are skipped.
pub fn apply_damage_events(
    characters: &mut [Character],
    events: &[DamageEvent],
) -> Vec<DamageOutcome> {
    let mut out = Vec::with_capacity(events.len());
    for ev in events {
        let Some(ch) = characters.get_mut(ev.character) else {
            continue;
        };
        let Some(z) = ch.zones.get(ev.zone) else {
            continue;
        };
        let (kind, vuln) = (z.kind, z.vulnerability);
        let mut o = ch.health.apply_hit(ev, kind, &vuln);
        o.character = ev.character;
        out.push(o);
    }
    out
}
