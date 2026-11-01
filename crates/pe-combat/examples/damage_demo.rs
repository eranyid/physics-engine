//! Shots and a blast on a standing humanoid, with and without armor.
//!
//! `cargo run -p pe-combat --example damage_demo`

use pe_ballistics::ammo::bundled_by_name;
use pe_ballistics::projectile::{MuzzlePose, ProjectileSystem, ShotId};
use pe_blast::BlastSource;
use pe_combat::*;
use pe_math::Vec3;

const I3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn mat(name: &str) -> usize {
    bundled_materials()
        .iter()
        .position(|m| m.name == name)
        .unwrap()
}

fn shoot(ammo: &str, aim_z: f64, aim_y: f64, plate_mm: Option<f64>) {
    let mut ch = Character::humanoid(0, 1.8, mat("generic_tissue_simulant"));
    if let Some(t) = plate_mm {
        ch.equip(
            &[ZoneKind::UpperTorso, ZoneKind::LowerTorso],
            &ArmorStack::single(mat("generic_hard_steel_plate"), t * 1e-3),
        );
    }
    let poses = [(Vec3::new(20.0, 0.0, 0.0), I3)];
    let shapes = ch.zone_shapes(&poses);
    let mut targets = TargetTable::new();
    targets.push_character(0, &ch);
    let mut res = CombatResolver::with_targets(bundled_materials(), targets);
    let mut sys = ProjectileSystem::new(0.0);
    sys.enable_spin_effects = false;
    let p = sys.add_profile(bundled_by_name(ammo).unwrap());
    let mut profile = sys.profiles[p].clone();
    profile.dispersion_std_rad = 0.0;
    sys.profiles[p] = profile;
    let _ = sys.spawn(
        ShotId(1),
        p,
        MuzzlePose {
            position: Vec3::new(0.0, aim_y, aim_z),
            direction: Vec3::unit_x(),
            barrel_length_m: None,
        },
    );
    let ids = vec![None; shapes.len()];
    for _ in 0..30 {
        sys.step_with_resolver(1.0 / 60.0, &shapes, &ids, &mut res);
    }
    let mut chars = vec![ch];
    let outs = apply_damage_events(&mut chars, &res.damage_events);
    println!(
        "\n{ammo} at z={aim_z:.2} y={aim_y:.2} plate={:?} mm",
        plate_mm
    );
    for (ev, o) in res.damage_events.iter().zip(&outs) {
        println!(
            "  {:<11?} KE_in={:>6.0} J tissue={:>6.0} J armor={:>6.0} J through={} pen_armor={} -> {:>5.1} HP bleed={:.4} L/s blunt={:.0} J incap={}",
            o.zone, ev.ke_in_j, ev.e_tissue_j, ev.e_armor_j, ev.through_and_through,
            ev.penetrated_armor, o.damage_hp, o.bleed_l_s, o.blunt_j, o.instant_incap
        );
    }
    let h = &mut chars[0].health;
    println!("  now: {:?} dmg={:.1} HP", h.condition, h.damage_hp);
    h.step(60.0);
    println!(
        "  +60 s: {:?} blood loss={:.1}% mods={:?}",
        h.condition,
        100.0 * h.blood_loss_fraction(),
        h.modifiers()
    );
}

fn main() {
    shoot("generic_battle_rifle", 1.35, 0.0, None);
    shoot("generic_battle_rifle", 1.35, 0.0, Some(6.0));
    shoot("generic_battle_rifle", 1.35, 0.0, Some(10.0));
    shoot("generic_pistol", 1.35, 0.0, None);
    shoot("generic_pistol", 1.68, 0.0, None);
    shoot("generic_pistol", 0.70, 0.09, None);
    shoot("generic_intermediate_rifle", 1.30, 0.234, None);

    let ch = Character::humanoid(0, 1.8, mat("generic_tissue_simulant"));
    for d in [2.0, 3.0, 4.0, 5.0, 8.0] {
        let src = BlastSource::new(Vec3::new(-d, 0.0, 1.0), 1.0);
        let r = blast_on_character(&src, &ch, &[(Vec3::zero(), I3)]);
        let dmg: f64 = r.iter().map(|z| z.damage_hp).sum();
        let torso = &r[ch.zone_index(ZoneKind::UpperTorso).unwrap()];
        println!(
            "\nblast 1 kg TNT-eq at {d} m: torso P={:.0} kPa I={:.0} Pa·s, total damage={:.1} HP, push={:.1} N·s",
            torso.p_peak_pa * 1e-3,
            torso.impulse_pa_s,
            dmg,
            total_push(&r).norm()
        );
    }
}
