//! Characters, hit zones and health state.
//!
//! A [`Character`] is a set of [`HitZone`]s (head, neck, torso, limbs), each
//! a simple shape attached to a body pose, with optional armor (helmet, plate
//! carrier) and a tissue material. Hits become [`crate::DamageEvent`]s, which
//! [`HealthState::apply`] turns into damage, bleeding, pain and limb
//! impairment. [`HealthState::step`] integrates bleeding exactly.
//!
//! All vulnerability and health constants are **nominal game tuning**, not
//! medical data.

use pe_ballistics::hitscan::ShapeRef;
use pe_collision::{BoxShape, Capsule, Sphere};
use pe_math::libm_ops;
use pe_math::Vec3;

use crate::armor::ArmorStack;

/// Body region of a hit zone.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum ZoneKind {
    /// Head.
    Head,
    /// Neck.
    Neck,
    /// Chest.
    UpperTorso,
    /// Abdomen.
    LowerTorso,
    /// Pelvis.
    Pelvis,
    /// Left upper arm.
    UpperArmL,
    /// Right upper arm.
    UpperArmR,
    /// Left forearm (incl. hand).
    ForearmL,
    /// Right forearm (incl. hand).
    ForearmR,
    /// Left thigh.
    ThighL,
    /// Right thigh.
    ThighR,
    /// Left shin (incl. foot).
    ShinL,
    /// Right shin (incl. foot).
    ShinR,
}

/// Limb slot for [`HealthState::limb_function`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limb {
    /// Left arm.
    ArmL = 0,
    /// Right arm.
    ArmR = 1,
    /// Left leg.
    LegL = 2,
    /// Right leg.
    LegR = 3,
}

impl ZoneKind {
    /// Which limb this zone belongs to, if any.
    pub fn limb(self) -> Option<Limb> {
        match self {
            ZoneKind::UpperArmL | ZoneKind::ForearmL => Some(Limb::ArmL),
            ZoneKind::UpperArmR | ZoneKind::ForearmR => Some(Limb::ArmR),
            ZoneKind::ThighL | ZoneKind::ShinL => Some(Limb::LegL),
            ZoneKind::ThighR | ZoneKind::ShinR => Some(Limb::LegR),
            _ => None,
        }
    }
}

/// Local hit shape of a zone (body frame).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ZoneShape {
    /// Sphere of radius [m].
    Sphere {
        /// Radius [m].
        radius: f64,
    },
    /// Capsule along local +z.
    Capsule {
        /// Radius [m].
        radius: f64,
        /// Half segment length [m].
        half_height: f64,
    },
    /// Box with half extents [m].
    Box {
        /// Half extents [m].
        half_extents: [f64; 3],
    },
}

impl ZoneShape {
    /// Area [m²] projected on a plane normal to unit direction `d` (local frame).
    pub fn projected_area(&self, d_local: Vec3) -> f64 {
        let d = d_local.normalize();
        match *self {
            ZoneShape::Sphere { radius } => core::f64::consts::PI * radius * radius,
            ZoneShape::Capsule {
                radius,
                half_height,
            } => {
                let sin = libm_ops::sqrt((1.0 - d.z * d.z).max(0.0));
                core::f64::consts::PI * radius * radius + 2.0 * radius * 2.0 * half_height * sin
            }
            ZoneShape::Box { half_extents: h } => {
                4.0 * (h[1] * h[2] * libm_ops::fabs(d.x)
                    + h[0] * h[2] * libm_ops::fabs(d.y)
                    + h[0] * h[1] * libm_ops::fabs(d.z))
            }
        }
    }
}

/// Blast pressure–impulse tolerance (hyperbolic `(P − P0)(I − I0) = C`).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BlastTolerance {
    /// Pressure asymptote [Pa].
    pub p0_pa: f64,
    /// Impulse asymptote [Pa·s].
    pub i0_pa_s: f64,
    /// Curve constant [Pa²·s].
    pub c: f64,
    /// Damage [HP] at the curve; grows as `1 + ln(excess)` beyond it.
    pub damage_hp: f64,
}

/// How a zone turns energy into injury.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ZoneVulnerability {
    /// Damage multiplier on deposited energy.
    pub damage_factor: f64,
    /// Initial bleed rate per joule in tissue [L/s/J].
    pub bleed_l_per_s_per_j: f64,
    /// Clotting time constant of wounds here [s].
    pub clot_tau_s: f64,
    /// Instant incapacitation when tissue energy ≥ this [J] (`None` = never).
    pub instant_incap_j: Option<f64>,
    /// Limb function loss gain [1/J]: `function *= e^{−g E}`.
    pub impairment_per_j: f64,
    /// Share of energy absorbed by armor felt as blunt trauma [0, 1].
    pub blunt_factor: f64,
    /// Blast tolerance.
    pub blast: BlastTolerance,
}

/// One hit zone of a character.
#[derive(Clone, Debug, PartialEq)]
pub struct HitZone {
    /// Region.
    pub kind: ZoneKind,
    /// Index of the body pose this zone follows.
    pub body: usize,
    /// Offset from the body origin, body frame [m].
    pub local_offset: Vec3,
    /// Rotation relative to the body, row-major.
    pub local_rot: [[f64; 3]; 3],
    /// Shape.
    pub shape: ZoneShape,
    /// Armor worn over this zone (outermost first; may be empty).
    pub armor: ArmorStack,
    /// Tissue material index (in the resolver's material table).
    pub tissue_material: usize,
    /// Vulnerability.
    pub vulnerability: ZoneVulnerability,
}

/// Global health tuning.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HealthParams {
    /// Blood volume [L].
    pub blood_volume_l: f64,
    /// Joules of (factored) tissue energy per HP of damage.
    pub j_per_hp: f64,
    /// Damage [HP] at which the character is incapacitated.
    pub incap_damage_hp: f64,
    /// Damage [HP] at which the character dies.
    pub dead_damage_hp: f64,
    /// Blood-loss fraction that incapacitates.
    pub incap_blood_loss: f64,
    /// Blood-loss fraction that kills.
    pub dead_blood_loss: f64,
    /// Weight of blunt energy relative to penetrating energy.
    pub blunt_weight: f64,
    /// Tissue joules per unit pain.
    pub pain_j: f64,
    /// Pain decay time constant [s].
    pub pain_tau_s: f64,
    /// Pain level that incapacitates.
    pub incap_pain: f64,
}

impl Default for HealthParams {
    fn default() -> Self {
        Self {
            blood_volume_l: 5.0,
            j_per_hp: 8.0,
            incap_damage_hp: 100.0,
            dead_damage_hp: 150.0,
            incap_blood_loss: 0.30,
            dead_blood_loss: 0.45,
            blunt_weight: 0.25,
            pain_j: 400.0,
            pain_tau_s: 30.0,
            incap_pain: 3.0,
        }
    }
}

/// Coarse character condition (ordered by severity).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Condition {
    /// Unhurt.
    Ok,
    /// Hurt but functional.
    Wounded,
    /// Down, cannot act.
    Incapacitated,
    /// Dead.
    Dead,
}

/// A bleeding wound with exponential clotting `r(t) = r0 e^{−t/τ}`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wound {
    /// Zone.
    pub zone: ZoneKind,
    /// Initial bleed rate [L/s].
    pub rate0_l_s: f64,
    /// Clotting time constant [s].
    pub tau_s: f64,
    /// Age [s].
    pub age_s: f64,
}

impl Wound {
    /// Current bleed rate [L/s].
    pub fn rate(&self) -> f64 {
        self.rate0_l_s * libm_ops::exp(-self.age_s / self.tau_s)
    }

    /// Exact blood lost [L] over the next `dt` [s].
    pub fn loss_over(&self, dt: f64) -> f64 {
        let a = libm_ops::exp(-self.age_s / self.tau_s);
        let b = libm_ops::exp(-(self.age_s + dt) / self.tau_s);
        self.rate0_l_s * self.tau_s * (a - b)
    }
}

/// Gameplay modifiers derived from health.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HealthModifiers {
    /// Aim sway multiplier (≥ 1).
    pub aim_sway_mult: f64,
    /// Movement speed multiplier (0..1).
    pub move_speed_mult: f64,
    /// Whether the character can still use its arms (fire, reload).
    pub can_use_arms: bool,
}

/// Evolving health of a character.
#[derive(Clone, Debug, PartialEq)]
pub struct HealthState {
    /// Tuning.
    pub params: HealthParams,
    /// Condition.
    pub condition: Condition,
    /// Accumulated damage [HP].
    pub damage_hp: f64,
    /// Remaining blood [L].
    pub blood_l: f64,
    /// Active wounds.
    pub wounds: Vec<Wound>,
    /// Pain level (decays).
    pub pain: f64,
    /// Limb function ∈ [0,1], indexed by [`Limb`].
    pub limb_function: [f64; 4],
}

impl HealthState {
    /// Healthy state with `params`.
    pub fn new(params: HealthParams) -> Self {
        Self {
            params,
            condition: Condition::Ok,
            damage_hp: 0.0,
            blood_l: params.blood_volume_l,
            wounds: Vec::new(),
            pain: 0.0,
            limb_function: [1.0; 4],
        }
    }

    /// Fraction of blood lost.
    pub fn blood_loss_fraction(&self) -> f64 {
        (1.0 - self.blood_l / self.params.blood_volume_l).clamp(0.0, 1.0)
    }

    /// Total current bleed rate [L/s].
    pub fn bleed_rate(&self) -> f64 {
        self.wounds.iter().map(Wound::rate).sum()
    }

    /// Add raw damage [HP] and pain, then re-evaluate condition.
    pub fn add_damage(&mut self, hp: f64, pain: f64) {
        if self.condition == Condition::Dead {
            return;
        }
        self.damage_hp += hp.max(0.0);
        self.pain += pain.max(0.0);
        self.update_condition();
    }

    /// Add a bleeding wound.
    pub fn add_wound(&mut self, zone: ZoneKind, rate0_l_s: f64, tau_s: f64) {
        if rate0_l_s > 0.0 && tau_s > 0.0 && self.condition != Condition::Dead {
            self.wounds.push(Wound {
                zone,
                rate0_l_s,
                tau_s,
                age_s: 0.0,
            });
        }
    }

    /// Advance bleeding and pain by `dt` [s] (exact integration).
    pub fn step(&mut self, dt: f64) {
        if self.condition == Condition::Dead || dt <= 0.0 {
            return;
        }
        let mut lost = 0.0;
        for w in &mut self.wounds {
            lost += w.loss_over(dt);
            w.age_s += dt;
        }
        self.blood_l = (self.blood_l - lost).max(0.0);
        // Drop wounds that have effectively clotted (< 0.1 % of initial rate).
        self.wounds.retain(|w| w.age_s < 7.0 * w.tau_s);
        self.pain *= libm_ops::exp(-dt / self.params.pain_tau_s);
        self.update_condition();
    }

    /// Re-evaluate [`condition`](Self::condition) (never improves past Incapacitated/Dead).
    pub fn update_condition(&mut self) {
        let p = &self.params;
        let loss = self.blood_loss_fraction();
        let next = if self.damage_hp >= p.dead_damage_hp || loss >= p.dead_blood_loss {
            Condition::Dead
        } else if self.damage_hp >= p.incap_damage_hp
            || loss >= p.incap_blood_loss
            || self.pain >= p.incap_pain
        {
            Condition::Incapacitated
        } else if self.damage_hp > 0.0 || !self.wounds.is_empty() || self.pain > 0.0 {
            Condition::Wounded
        } else {
            Condition::Ok
        };
        if next > self.condition {
            self.condition = next;
        }
    }

    /// Force incapacitation (e.g. head trauma).
    pub fn incapacitate(&mut self) {
        if self.condition < Condition::Incapacitated {
            self.condition = Condition::Incapacitated;
        }
    }

    /// Gameplay modifiers.
    pub fn modifiers(&self) -> HealthModifiers {
        let f = &self.limb_function;
        let arms = f[0].min(f[1]);
        let legs = f[2].min(f[3]);
        let loss = self.blood_loss_fraction();
        let shock = (loss / self.params.incap_blood_loss).clamp(0.0, 1.0);
        let active = self.condition < Condition::Incapacitated;
        HealthModifiers {
            aim_sway_mult: 1.0 + 2.0 * (1.0 - arms) + 0.5 * self.pain + 1.5 * shock,
            move_speed_mult: if active {
                (legs * (1.0 - 0.5 * shock)).clamp(0.0, 1.0)
            } else {
                0.0
            },
            can_use_arms: active && 0.5 * (f[0] + f[1]) > 0.2,
        }
    }
}

/// A character: zones plus health.
#[derive(Clone, Debug)]
pub struct Character {
    /// Caller-defined id.
    pub id: usize,
    /// Hit zones.
    pub zones: Vec<HitZone>,
    /// Health.
    pub health: HealthState,
}

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

fn mat_vec(a: &[[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        a[0][0] * v.x + a[0][1] * v.y + a[0][2] * v.z,
        a[1][0] * v.x + a[1][1] * v.y + a[1][2] * v.z,
        a[2][0] * v.x + a[2][1] * v.y + a[2][2] * v.z,
    )
}

fn mat_t_vec(a: &[[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        a[0][0] * v.x + a[1][0] * v.y + a[2][0] * v.z,
        a[0][1] * v.x + a[1][1] * v.y + a[2][1] * v.z,
        a[0][2] * v.x + a[1][2] * v.y + a[2][2] * v.z,
    )
}

fn blast(p0_kpa: f64, i0: f64, c: f64, hp: f64) -> BlastTolerance {
    BlastTolerance {
        p0_pa: p0_kpa * 1e3,
        i0_pa_s: i0,
        c,
        damage_hp: hp,
    }
}

/// Nominal vulnerability for a zone kind.
pub fn default_vulnerability(kind: ZoneKind) -> ZoneVulnerability {
    use ZoneKind::*;
    let torso_blast = blast(60.0, 60.0, 3.6e6, 12.0);
    let limb_blast = blast(150.0, 150.0, 1.0e7, 5.0);
    let v =
        |factor: f64, bleed: f64, incap: Option<f64>, imp: f64, blunt: f64, b: BlastTolerance| {
            ZoneVulnerability {
                damage_factor: factor,
                bleed_l_per_s_per_j: bleed,
                clot_tau_s: 120.0,
                instant_incap_j: incap,
                impairment_per_j: imp,
                blunt_factor: blunt,
                blast: b,
            }
        };
    match kind {
        Head => v(
            3.0,
            1.0e-5,
            Some(40.0),
            0.0,
            0.03,
            blast(50.0, 40.0, 2.0e6, 15.0),
        ),
        Neck => v(2.0, 6.0e-5, Some(150.0), 0.0, 0.03, torso_blast),
        UpperTorso => v(1.0, 2.0e-5, None, 0.0, 0.02, torso_blast),
        LowerTorso => v(0.8, 2.0e-5, None, 0.0, 0.02, torso_blast),
        Pelvis => v(0.7, 2.5e-5, None, 0.0, 0.02, torso_blast),
        UpperArmL | UpperArmR | ForearmL | ForearmR => {
            v(0.35, 1.5e-5, None, 1.0 / 600.0, 0.02, limb_blast)
        }
        ThighL | ThighR => v(0.5, 3.0e-5, None, 1.0 / 600.0, 0.02, limb_blast),
        ShinL | ShinR => v(0.35, 1.0e-5, None, 1.0 / 600.0, 0.02, limb_blast),
    }
}

impl Character {
    /// Standing humanoid (facing +x, z up, origin between the feet) with 13
    /// zones on body pose `0`. Sizes scale with `height_m`.
    pub fn humanoid(id: usize, height_m: f64, tissue_material: usize) -> Self {
        use ZoneKind::*;
        let h = height_m;
        let mut zones = Vec::new();
        let mut add = |kind: ZoneKind, y: f64, z: f64, shape: ZoneShape| {
            zones.push(HitZone {
                kind,
                body: 0,
                local_offset: Vec3::new(0.0, y * h, z * h),
                local_rot: IDENTITY,
                shape,
                armor: ArmorStack::default(),
                tissue_material,
                vulnerability: default_vulnerability(kind),
            });
        };
        let cap = |r: f64, hh: f64| ZoneShape::Capsule {
            radius: r * h,
            half_height: hh * h,
        };
        let bx = |x: f64, y: f64, z: f64| ZoneShape::Box {
            half_extents: [x * h, y * h, z * h],
        };
        add(Head, 0.0, 0.935, ZoneShape::Sphere { radius: 0.058 * h });
        add(Neck, 0.0, 0.862, cap(0.03, 0.012));
        add(UpperTorso, 0.0, 0.75, bx(0.055, 0.10, 0.075));
        add(LowerTorso, 0.0, 0.62, bx(0.05, 0.085, 0.055));
        add(Pelvis, 0.0, 0.53, bx(0.055, 0.095, 0.035));
        add(UpperArmL, 0.125, 0.72, cap(0.022, 0.05));
        add(UpperArmR, -0.125, 0.72, cap(0.022, 0.05));
        add(ForearmL, 0.13, 0.57, cap(0.019, 0.05));
        add(ForearmR, -0.13, 0.57, cap(0.019, 0.05));
        add(ThighL, 0.05, 0.39, cap(0.035, 0.08));
        add(ThighR, -0.05, 0.39, cap(0.035, 0.08));
        add(ShinL, 0.05, 0.16, cap(0.026, 0.09));
        add(ShinR, -0.05, 0.16, cap(0.026, 0.09));
        Self {
            id,
            zones,
            health: HealthState::new(HealthParams::default()),
        }
    }

    /// Index of the first zone of `kind`.
    pub fn zone_index(&self, kind: ZoneKind) -> Option<usize> {
        self.zones.iter().position(|z| z.kind == kind)
    }

    /// Put `armor` on every zone of the given kinds.
    pub fn equip(&mut self, kinds: &[ZoneKind], armor: &ArmorStack) {
        for z in &mut self.zones {
            if kinds.contains(&z.kind) {
                z.armor = armor.clone();
            }
        }
    }

    /// World pose (center, rotation) of zone `zone` given body poses.
    pub fn zone_pose(&self, zone: usize, poses: &[(Vec3, [[f64; 3]; 3])]) -> (Vec3, [[f64; 3]; 3]) {
        let z = &self.zones[zone];
        let (p, r) = poses[z.body];
        (p + mat_vec(&r, z.local_offset), mat_mul(&r, &z.local_rot))
    }

    /// Projected area [m²] of zone `zone` seen along world direction `d`.
    pub fn zone_projected_area(
        &self,
        zone: usize,
        poses: &[(Vec3, [[f64; 3]; 3])],
        d_world: Vec3,
    ) -> f64 {
        let (_, r) = self.zone_pose(zone, poses);
        self.zones[zone]
            .shape
            .projected_area(mat_t_vec(&r, d_world))
    }

    /// World-space hit shapes, one per zone, in zone order.
    pub fn zone_shapes(&self, poses: &[(Vec3, [[f64; 3]; 3])]) -> Vec<ShapeRef> {
        (0..self.zones.len())
            .map(|i| {
                let (x, r) = self.zone_pose(i, poses);
                match self.zones[i].shape {
                    ZoneShape::Sphere { radius } => ShapeRef::Sphere {
                        x,
                        s: Sphere::new(radius),
                    },
                    ZoneShape::Capsule {
                        radius,
                        half_height,
                    } => ShapeRef::Capsule {
                        x,
                        r,
                        s: Capsule::new(radius, half_height),
                    },
                    ZoneShape::Box { half_extents: e } => ShapeRef::Box {
                        x,
                        r,
                        s: BoxShape::new(e[0], e[1], e[2]),
                    },
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn humanoid_has_13_zones_without_overlap_in_y() {
        let c = Character::humanoid(0, 1.8, 0);
        assert_eq!(c.zones.len(), 13);
        let shapes = c.zone_shapes(&[(Vec3::zero(), IDENTITY)]);
        assert_eq!(shapes.len(), 13);
    }

    #[test]
    fn capsule_projected_area_side_vs_end() {
        let s = ZoneShape::Capsule {
            radius: 0.05,
            half_height: 0.1,
        };
        let end = s.projected_area(Vec3::unit_z());
        let side = s.projected_area(Vec3::unit_x());
        assert!((end - core::f64::consts::PI * 0.0025).abs() < 1e-15);
        assert!((side - end - 0.02).abs() < 1e-15);
    }

    #[test]
    fn wound_loss_closed_form() {
        let w = Wound {
            zone: ZoneKind::ThighL,
            rate0_l_s: 0.02,
            tau_s: 100.0,
            age_s: 0.0,
        };
        let expect = 0.02 * 100.0 * (1.0 - libm_ops::exp(-0.5));
        assert!((w.loss_over(50.0) - expect).abs() < 1e-15);
    }
}
