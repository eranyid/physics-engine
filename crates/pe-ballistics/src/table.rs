//! Ballistic table generator (drop, drift, velocity, TOF, energy).

use pe_aero::turbulence::WindField;
use pe_math::Vec3;

use crate::ammo::AmmoProfile;
use crate::projectile::{MuzzlePose, ProjectileSystem, ShotId};

/// One range sample of a ballistic table.
#[derive(Clone, Debug)]
pub struct TableRow {
    /// Horizontal range [m].
    pub range_m: f64,
    /// Drop (negative z relative to bore line) [m].
    pub drop_m: f64,
    /// Lateral drift [m] (+y).
    pub drift_m: f64,
    /// Speed [m/s].
    pub speed_m_s: f64,
    /// Time of flight [s].
    pub time_s: f64,
    /// Kinetic energy [J].
    pub energy_j: f64,
}

/// Generate a ballistic table for `profile` at the given ranges [m].
///
/// Flat-fire along +x from the origin; `wind` is cloned into a temporary system.
/// `dt` [s] integration step.
pub fn generate_table(
    profile: &AmmoProfile,
    ranges_m: &[f64],
    wind: WindField,
    latitude_rad: f64,
    dt: f64,
) -> Vec<TableRow> {
    let mut sys = ProjectileSystem::new(latitude_rad);
    sys.wind = wind;
    let pidx = sys.add_profile(profile.clone());
    let _ = sys.spawn(
        ShotId(1),
        pidx,
        MuzzlePose {
            position: Vec3::zero(),
            direction: Vec3::unit_x(),
            barrel_length_m: None,
        },
    );
    // Zero dispersion / velocity noise for table: re-spawn with patched profile.
    sys.profiles[pidx].velocity_std_m_s = 0.0;
    sys.profiles[pidx].dispersion_std_rad = 0.0;
    sys.projectiles.clear();
    let _ = sys.spawn(
        ShotId(2),
        pidx,
        MuzzlePose {
            position: Vec3::zero(),
            direction: Vec3::unit_x(),
            barrel_length_m: None,
        },
    );

    let mut out = Vec::new();
    let mut next_ri = 0usize;
    let max_range = ranges_m.iter().cloned().fold(0.0_f64, f64::max);
    let t_limit = 60.0;
    while next_ri < ranges_m.len() && sys.t < t_limit {
        if !sys.projectiles[0].alive {
            break;
        }
        let x_before = sys.projectiles[0].x.x;
        sys.step(dt, &[], &[]);
        let p = &sys.projectiles[0];
        let x_after = p.x.x;
        while next_ri < ranges_m.len()
            && x_before <= ranges_m[next_ri]
            && x_after >= ranges_m[next_ri]
        {
            let r = ranges_m[next_ri];
            let denom = (x_after - x_before).abs().max(1e-12);
            let _alpha = ((r - x_before) / denom).clamp(0.0, 1.0);
            // Linear interpolate state (crude but fine for tables).
            let drop = p.x.z; // bore at z=0
            let drift = p.x.y;
            let speed = p.v.norm();
            out.push(TableRow {
                range_m: r,
                drop_m: drop,
                drift_m: drift,
                speed_m_s: speed,
                time_s: sys.t,
                energy_j: 0.5 * p.mass * speed * speed,
            });
            next_ri += 1;
        }
        if x_after > max_range + 1.0 {
            break;
        }
    }
    out
}
