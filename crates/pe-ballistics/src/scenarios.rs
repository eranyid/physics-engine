//! Canonical ballistics determinism scenario (cross-OS hash).
//!
//! Mixed volley of 100 bullets with wind; FNV-1a over packed projectile state.

use pe_aero::turbulence::{DrydenParams, DrydenTurbulence, SteadyWind, WindField};
use pe_math::libm_ops;
use pe_math::Vec3;

use crate::ammo::{bundled_by_name, bundled_profiles};
use crate::projectile::{MuzzlePose, ProjectileSystem, ShotId};

/// Steps in the canonical hash scenario.
pub const BALLISTICS_HASH_STEPS: usize = 120;
/// Timestep [s].
pub const BALLISTICS_HASH_DT: f64 = 1.0 / 120.0;
/// Number of bullets in the mixed volley.
pub const BALLISTICS_HASH_N: usize = 100;

fn fnv1a_f64(hash: &mut u64, x: f64) {
    const FNV_PRIME: u64 = 0x100000001b3;
    for b in x.to_bits().to_le_bytes() {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

/// Build the canonical mixed-volley system (100 bullets, several ammo classes, wind).
pub fn ballistics_volley_system() -> ProjectileSystem {
    let mut sys = ProjectileSystem::new(0.7); // ~40° lat
    let steady = SteadyWind::new(3.0, 0.5, 0.0);
    let dryden = DrydenTurbulence::new(DrydenParams::light_low_altitude(), 0xB0A7);
    sys.wind = WindField::with_dryden(steady, dryden);
    sys.enable_spin_effects = true;

    // Register a few bundled profiles.
    let names = [
        "generic_pistol",
        "generic_intermediate_rifle",
        "generic_battle_rifle",
        "generic_sniper_long_range",
        "generic_subsonic",
    ];
    let mut idxs = Vec::new();
    for n in names {
        let p = bundled_by_name(n).unwrap_or_else(|| bundled_profiles()[0].clone());
        // Dispersion/velocity noise remain; shot_id seeds each RNG → bit-stable.
        idxs.push(sys.add_profile(p));
    }

    for i in 0..BALLISTICS_HASH_N {
        let pidx = idxs[i % idxs.len()];
        let yaw = 0.002 * ((i % 10) as f64 - 4.5);
        let pitch = 0.01 + 0.001 * (i % 7) as f64;
        let dir = Vec3::new(
            libm_ops::cos(pitch) * libm_ops::cos(yaw),
            libm_ops::cos(pitch) * libm_ops::sin(yaw),
            libm_ops::sin(pitch),
        );
        let _ = sys.spawn(
            ShotId(1000 + i as u64),
            pidx,
            MuzzlePose {
                position: Vec3::new(0.0, (i as f64) * 0.01, 1.5),
                direction: dir,
                barrel_length_m: None,
            },
        );
    }
    sys
}

/// FNV-1a hash of all projectile positions/velocities (id order).
pub fn ballistics_volley_hash() -> u64 {
    let mut sys = ballistics_volley_system();
    for _ in 0..BALLISTICS_HASH_STEPS {
        sys.step(BALLISTICS_HASH_DT, &[], &[]);
    }
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    let mut hash = FNV_OFFSET;
    for p in &sys.projectiles {
        fnv1a_f64(&mut hash, p.id.0 as f64);
        for c in p.x.to_array() {
            fnv1a_f64(&mut hash, c);
        }
        for c in p.v.to_array() {
            fnv1a_f64(&mut hash, c);
        }
        fnv1a_f64(&mut hash, if p.alive { 1.0 } else { 0.0 });
    }
    hash
}
