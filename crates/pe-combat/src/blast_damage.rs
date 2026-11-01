//! Blast loading and injury per character zone.
//!
//! Each zone is treated as an exposed face toward the charge, with the zone's
//! projected area along the blast direction. Peak overpressure comes from
//! Kinney–Graham ([`pe_blast::peak_overpressure`]) and the positive impulse
//! from the Friedlander waveform ([`pe_blast::face_positive_impulse`]). Injury
//! uses the zone's [`crate::BlastTolerance`] pressure–impulse curve
//! ([`pe_blast::PressureImpulseCurve`]):
//!
//! ```text
//! excess = (P − P0)(I − I0) / C            (damaged when ≥ 1)
//! damage = damage_hp · (1 + ln excess)     [HP]
//! push   = I · A · d̂                       [N·s], d̂ = away from the charge
//! ```
//!
//! No shielding between charge and zone is modelled (caller's job).

use pe_blast::{
    face_positive_impulse, peak_overpressure, BlastSource, ExposedFace, PressureImpulseCurve,
};
use pe_math::libm_ops;
use pe_math::Vec3;

use crate::hitzones::{Character, HealthState, ZoneKind};

/// Blast result for one zone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneBlast {
    /// Zone index.
    pub zone: usize,
    /// Zone kind.
    pub kind: ZoneKind,
    /// Zone center [m].
    pub center: Vec3,
    /// Shock arrival time [s] (world clock).
    pub arrival_s: f64,
    /// Peak incident overpressure [Pa].
    pub p_peak_pa: f64,
    /// Positive-phase specific impulse [Pa·s].
    pub impulse_pa_s: f64,
    /// Projected area facing the charge [m²].
    pub area_m2: f64,
    /// Push impulse on the zone [N·s] (away from the charge).
    pub push_impulse: Vec3,
    /// Damage [HP] (0 below the tolerance curve).
    pub damage_hp: f64,
}

/// Evaluate a blast on every zone of `ch` posed by `poses`.
pub fn blast_on_character(
    src: &BlastSource,
    ch: &Character,
    poses: &[(Vec3, [[f64; 3]; 3])],
) -> Vec<ZoneBlast> {
    (0..ch.zones.len())
        .map(|zi| {
            let z = &ch.zones[zi];
            let (c, _) = ch.zone_pose(zi, poses);
            let r = c - src.origin;
            let dist = r.norm();
            let d = if dist > 1e-12 {
                r / dist
            } else {
                Vec3::unit_z()
            };
            let area = ch.zone_projected_area(zi, poses, d);
            let face = ExposedFace {
                body: z.body,
                center: c,
                normal: -d,
                area,
            };
            let p_peak = peak_overpressure(src.z_at(c), src.p0);
            let impulse = face_positive_impulse(src, &face);
            let tol = z.vulnerability.blast;
            let curve = PressureImpulseCurve {
                p0: tol.p0_pa,
                i0: tol.i0_pa_s,
                c: tol.c,
            };
            let damage = if curve.is_damaged(p_peak, impulse) {
                let excess = (p_peak - tol.p0_pa) * (impulse - tol.i0_pa_s) / tol.c;
                tol.damage_hp * (1.0 + libm_ops::log(excess.max(1.0)))
            } else {
                0.0
            };
            ZoneBlast {
                zone: zi,
                kind: z.kind,
                center: c,
                arrival_s: src.t0 + src.arrival_time_approx(c),
                p_peak_pa: p_peak,
                impulse_pa_s: impulse,
                area_m2: area,
                push_impulse: d * (impulse * area),
                damage_hp: damage,
            }
        })
        .collect()
}

impl HealthState {
    /// Apply zone blast results: damage, pain (`damage/40`) and limb loss
    /// (`function *= e^{−damage/50}` on limb zones).
    pub fn apply_blast(&mut self, results: &[ZoneBlast]) {
        let mut total = 0.0;
        for r in results {
            if r.damage_hp <= 0.0 {
                continue;
            }
            total += r.damage_hp;
            if let Some(l) = r.kind.limb() {
                self.limb_function[l as usize] *= libm_ops::exp(-r.damage_hp / 50.0);
            }
        }
        if total > 0.0 {
            self.add_damage(total, total / 40.0);
        }
    }
}

/// Total push impulse [N·s] on the character (sum over zones).
pub fn total_push(results: &[ZoneBlast]) -> Vec3 {
    results
        .iter()
        .fold(Vec3::zero(), |acc, r| acc + r.push_impulse)
}
