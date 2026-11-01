//! Spall and blast fragments as real projectiles.
//!
//! Fragments fly through the same [`ProjectileSystem`] as bullets, so they hit
//! walls, armor and character zones with the same physics.
//!
//! - [`fragment_profile`] — a sphere-equivalent fragment class (analytic drag,
//!   irregular-shape form factor). Fragments of other masses use the same
//!   profile with `drag_multiplier = (m / m_nom)^{2/3}`; since drag
//!   acceleration divides by the projectile's own mass, this is the correct
//!   area scaling for geometrically similar fragments.
//! - [`fragment_shot_id`] — deterministic ids (top bit set) derived from the
//!   parent shot and the impact, independent of how many other shots exist.

use core::f64::consts::PI;

use pe_ballistics::ammo::{
    AmmoProfile, BallisticCoefficient, FieldProvenance, AMMO_PROFILE_VERSION,
};
use pe_ballistics::drag::DragModelRef;
use pe_ballistics::impact::kinetic_energy;
use pe_ballistics::projectile::{ProjectileSystem, ShotId};
use pe_blast::mott_sample_many;
use pe_math::libm_ops;
use pe_math::{Vec3, Xoshiro256StarStar};

use crate::penetration::ImpactRecord;

/// Bit set on every fragment [`ShotId`].
pub const FRAGMENT_ID_BIT: u64 = 1 << 63;

/// Form factor applied to the analytic sphere-ish drag curve for irregular fragments.
pub const FRAGMENT_FORM_FACTOR: f64 = 3.0;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Deterministic fragment id from a parent id, a salt and the fragment index.
pub fn fragment_shot_id(parent: u64, salt: u64, k: u64) -> ShotId {
    let h = splitmix64(parent ^ splitmix64(salt ^ splitmix64(k)));
    ShotId(h | FRAGMENT_ID_BIT)
}

/// Sphere-equivalent fragment profile of nominal mass `nominal_mass_kg` [kg]
/// and material density `density_kg_m3` [kg/m³]. Nominal (`verified: false`).
pub fn fragment_profile(name: &str, nominal_mass_kg: f64, density_kg_m3: f64) -> AmmoProfile {
    let d = libm_ops::cbrt(6.0 * nominal_mass_kg / (PI * density_kg_m3));
    let area = 0.25 * PI * d * d;
    let src = "sphere-equivalent fragment; analytic Cd × irregular form factor (game tuning)";
    AmmoProfile {
        version: AMMO_PROFILE_VERSION,
        name: name.to_string(),
        caliber_m: d,
        mass_kg: nominal_mass_kg,
        muzzle_velocity_m_s: 0.0,
        bc: BallisticCoefficient {
            value_si: nominal_mass_kg / area,
            drag_model: DragModelRef::Analytic,
            table: None,
            provenance: FieldProvenance::nominal(src),
        },
        length_m: d,
        twist_m_per_rev: 0.0,
        form_factor: FRAGMENT_FORM_FACTOR,
        barrel_length_ref_m: None,
        velocity_std_m_s: 0.0,
        dispersion_std_rad: 0.0,
        provenance: FieldProvenance::nominal(src),
    }
}

/// Drag multiplier for a fragment of mass `m` using a profile of nominal mass `m_nom`.
pub fn fragment_drag_multiplier(m: f64, m_nom: f64) -> f64 {
    libm_ops::pow(m / m_nom, 2.0 / 3.0)
}

/// Spawn the spall of one impact record as projectiles using fragment
/// profile `profile_index`. Returns the number spawned.
///
/// Ids are derived from `(record.shot_id, record.shape_index, time, entry)`.
pub fn spawn_spall(
    sys: &mut ProjectileSystem,
    profile_index: usize,
    record: &ImpactRecord,
) -> usize {
    let m_nom = sys.profiles[profile_index].mass_kg;
    let salt = (record.shape_index as u64)
        ^ splitmix64(record.time_s.to_bits())
        ^ splitmix64(record.entry.x.to_bits() ^ record.entry.y.to_bits().rotate_left(21))
        ^ record.entry.z.to_bits().rotate_left(42);
    for (k, f) in record.spall.iter().enumerate() {
        sys.spawn_with_velocity(
            fragment_shot_id(record.shot_id.0, salt, k as u64),
            profile_index,
            f.position,
            f.velocity,
            f.mass_kg,
            fragment_drag_multiplier(f.mass_kg, m_nom),
        );
    }
    record.spall.len()
}

/// Spawn spall for every record (in order). Returns the number spawned.
pub fn spawn_all_spall(
    sys: &mut ProjectileSystem,
    profile_index: usize,
    records: &[ImpactRecord],
) -> usize {
    records
        .iter()
        .map(|r| spawn_spall(sys, profile_index, r))
        .sum()
}

/// Spawn an isotropic burst of `n` Mott-distributed fragments (characteristic
/// mass `mu_kg`) at `origin` with speed `speed_m_s` (e.g. from
/// [`pe_blast::gurney_velocity`]). Deterministic in `seed`. Returns total KE [J].
pub fn spawn_fragment_burst(
    sys: &mut ProjectileSystem,
    profile_index: usize,
    origin: Vec3,
    seed: u64,
    n: usize,
    mu_kg: f64,
    speed_m_s: f64,
) -> f64 {
    let m_nom = sys.profiles[profile_index].mass_kg;
    let masses = mott_sample_many(seed, mu_kg, n);
    let mut rng = Xoshiro256StarStar::seed(splitmix64(seed));
    let mut ke = 0.0;
    for (k, m) in masses.into_iter().enumerate() {
        if m <= 0.0 {
            continue;
        }
        let z = 2.0 * rng.next_f64() - 1.0;
        let phi = 2.0 * PI * rng.next_f64();
        let s = libm_ops::sqrt((1.0 - z * z).max(0.0));
        let dir = Vec3::new(s * libm_ops::cos(phi), s * libm_ops::sin(phi), z);
        let v = dir * speed_m_s;
        ke += kinetic_energy(m, v);
        sys.spawn_with_velocity(
            fragment_shot_id(seed, 0xB1A5_7000, k as u64),
            profile_index,
            origin,
            v,
            m,
            fragment_drag_multiplier(m, m_nom),
        );
    }
    ke
}
