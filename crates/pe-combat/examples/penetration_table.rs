//! Print what each bundled ammo class does against each bundled material
//! (head-on, at muzzle velocity, 10 mm plates / 20 cm media).
//!
//! `cargo run -p pe-combat --example penetration_table`

use pe_ballistics::ammo::bundled_profiles;
use pe_combat::*;
use pe_math::{Vec3, Xoshiro256StarStar};

fn main() {
    let mats = bundled_materials();
    for p in bundled_profiles() {
        println!(
            "\n{}  (m={:.1} g, d={:.2} mm, v0={:.0} m/s)",
            p.name,
            p.mass_kg * 1e3,
            p.caliber_m * 1e3,
            p.muzzle_velocity_m_s
        );
        for (i, m) in mats.iter().enumerate() {
            let t = match m.model {
                PenetrationModel::Plate { .. } => 0.010,
                PenetrationModel::Medium { .. } => 0.20,
            };
            let mut rng = Xoshiro256StarStar::seed(1);
            let r = resolve_stack(
                &mats,
                &ArmorStack::single(i, t),
                StackInput {
                    velocity: Vec3::new(p.muzzle_velocity_m_s, 0.0, 0.0),
                    mass_kg: p.mass_kg,
                    diameter_m: p.caliber_m,
                    length_m: p.length_m,
                    normal: Vec3::new(-1.0, 0.0, 0.0),
                    chord_m: 0.0,
                },
                &mut rng,
            );
            println!(
                "  {:<26} {:>5.0} mm  {:<10} v_out={:>6.0} m/s  depth={}",
                m.name,
                t * 1e3,
                format!("{:?}", r.kind),
                r.velocity_out.norm(),
                r.stop_depth_m
                    .map(|d| format!("{:.0} mm", d * 1e3))
                    .unwrap_or_else(|| "-".into())
            );
        }
    }
}
