//! Hit zones, damage, blast and fragment tests for pe-combat.

use approx::assert_relative_eq;
use pe_ballistics::ammo::bundled_by_name;
use pe_ballistics::hitscan::ShapeRef;
use pe_ballistics::projectile::{MuzzlePose, ProjectileSystem, ShotId};
use pe_blast::{face_positive_impulse, BlastSource, ExposedFace};
use pe_collision::BoxShape;
use pe_combat::*;
use pe_math::{libm_ops, Vec3, Xoshiro256StarStar};
use proptest::prelude::*;

const I3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const CHAR_X: f64 = 20.0;

fn mat(name: &str) -> usize {
    bundled_materials()
        .iter()
        .position(|m| m.name == name)
        .unwrap()
}

fn humanoid() -> Character {
    Character::humanoid(0, 1.8, mat("generic_tissue_simulant"))
}

fn poses() -> [(Vec3, [[f64; 3]; 3]); 1] {
    [(Vec3::new(CHAR_X, 0.0, 0.0), I3)]
}

struct Scene {
    sys: ProjectileSystem,
    res: CombatResolver,
    shapes: Vec<ShapeRef>,
    profile: usize,
}

/// Character at x = 20 m, optional extra shapes (with targets) in front.
fn scene(ch: &Character, ammo: &str, extra: Vec<(ShapeRef, Target)>) -> Scene {
    let mut targets = TargetTable::new();
    let mut shapes = Vec::new();
    for (s, t) in extra {
        shapes.push(s);
        targets.push(t);
    }
    targets.push_character(0, ch);
    shapes.extend(ch.zone_shapes(&poses()));
    let mut sys = ProjectileSystem::new(0.0);
    sys.enable_spin_effects = false;
    let mut p = bundled_by_name(ammo).unwrap();
    p.dispersion_std_rad = 0.0;
    p.velocity_std_m_s = 0.0;
    let profile = sys.add_profile(p);
    Scene {
        sys,
        res: CombatResolver::with_targets(bundled_materials(), targets),
        shapes,
        profile,
    }
}

impl Scene {
    fn fire(&mut self, id: u64, y: f64, z: f64) {
        let _ = self.sys.spawn(
            ShotId(id),
            self.profile,
            MuzzlePose {
                position: Vec3::new(0.0, y, z),
                direction: Vec3::unit_x(),
                barrel_length_m: None,
            },
        );
    }

    fn run(&mut self, steps: usize) {
        let ids = vec![None; self.shapes.len()];
        for _ in 0..steps {
            self.sys
                .step_with_resolver(1.0 / 60.0, &self.shapes, &ids, &mut self.res);
        }
    }
}

// ------------------------------------------------------------ ledger

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Σ layer_dissipated + spall + KE_out = KE_in, and one entry per reached layer.
    #[test]
    fn per_layer_ledger_closes(
        seed in 0u64..10_000,
        speed in 50.0f64..1500.0,
        obliq in 0.0f64..1.5,
        n_layers in 1usize..4,
        layer_pick in proptest::collection::vec((0usize..8, 0.0005f64..0.3), 3),
    ) {
        let mats = bundled_materials();
        let stack = ArmorStack {
            layers: layer_pick[..n_layers]
                .iter()
                .map(|&(m, t)| ArmorLayer { material: m, thickness_m: t })
                .collect(),
        };
        let v = Vec3::new(libm_ops::cos(obliq), 0.0, -libm_ops::sin(obliq)) * speed;
        let mut rng = Xoshiro256StarStar::seed(seed);
        let r = resolve_stack(&mats, &stack, StackInput {
            velocity: v, mass_kg: 0.0095, diameter_m: 0.0078, length_m: 0.03,
            normal: Vec3::unit_z(), chord_m: 0.0,
        }, &mut rng);
        let sum: f64 = r.layer_dissipated_j.iter().sum::<f64>() + r.e_spall_j + r.ke_out_j;
        prop_assert!((sum - r.ke_in_j).abs() <= 1e-9 * r.ke_in_j);
        prop_assert!(r.layer_dissipated_j.iter().all(|&e| e >= 0.0));
        match r.kind {
            ImpactKind::Perforated => {
                prop_assert_eq!(r.layer_dissipated_j.len(), n_layers);
                prop_assert!(r.stopped_in_layer.is_none());
            }
            ImpactKind::Stopped => {
                let li = r.stopped_in_layer.unwrap();
                prop_assert_eq!(r.layer_dissipated_j.len(), li + 1);
                prop_assert_eq!(r.layers_perforated, li);
            }
            ImpactKind::Ricochet => prop_assert_eq!(r.layer_dissipated_j.len(), 1),
        }
    }
}

// ------------------------------------------------------------ bleeding

#[test]
fn bleeding_matches_closed_form_for_any_dt() {
    let mut h = HealthState::new(HealthParams::default());
    h.add_wound(ZoneKind::ThighL, 0.01, 90.0);
    h.add_wound(ZoneKind::UpperTorso, 0.004, 200.0);
    let mut t = 0.0;
    let mut k = 0usize;
    while t < 120.0 {
        let dt = [0.016, 0.5, 3.0, 0.1][k % 4];
        h.step(dt);
        t += dt;
        k += 1;
    }
    let lost = 0.01 * 90.0 * (1.0 - libm_ops::exp(-t / 90.0))
        + 0.004 * 200.0 * (1.0 - libm_ops::exp(-t / 200.0));
    assert_relative_eq!(5.0 - h.blood_l, lost, max_relative = 1e-12);
}

#[test]
fn heavy_bleeding_progresses_to_death() {
    let mut h = HealthState::new(HealthParams::default());
    h.add_wound(ZoneKind::Neck, 0.05, 300.0);
    let mut seen = Vec::new();
    for _ in 0..600 {
        h.step(1.0);
        if seen.last() != Some(&h.condition) {
            seen.push(h.condition);
        }
    }
    assert_eq!(
        seen,
        vec![
            Condition::Wounded,
            Condition::Incapacitated,
            Condition::Dead
        ]
    );
}

// ------------------------------------------------------------ bullets on characters

#[test]
fn rifle_through_unarmored_torso_keeps_flying() {
    let ch = humanoid();
    let mut sc = scene(&ch, "generic_battle_rifle", vec![]);
    sc.fire(1, 0.0, 1.35);
    sc.run(30);
    let torso = ch.zone_index(ZoneKind::UpperTorso).unwrap();
    let ev = sc
        .res
        .damage_events
        .iter()
        .find(|e| e.zone == torso)
        .expect("torso hit");
    assert!(ev.through_and_through && !ev.penetrated_armor);
    assert!(ev.e_tissue_j > 0.0 && ev.e_armor_j == 0.0);
    assert_relative_eq!(ev.wound_track_m, 2.0 * 0.055 * 1.8, max_relative = 1e-6);
    let p = &sc.sys.projectiles[0];
    assert!(
        p.alive && p.x.x > CHAR_X + 1.0,
        "bullet should exit: {:?}",
        p.x
    );
}

#[test]
fn plate_stops_rifle_and_gives_blunt_only() {
    let mut ch = humanoid();
    ch.equip(
        &[ZoneKind::UpperTorso],
        &ArmorStack::single(mat("generic_hard_steel_plate"), 0.010),
    );
    let mut sc = scene(&ch, "generic_battle_rifle", vec![]);
    sc.fire(1, 0.0, 1.35);
    sc.run(30);
    assert_eq!(sc.res.damage_events.len(), 1);
    let ev = sc.res.damage_events[0];
    assert!(!ev.penetrated_armor && !ev.through_and_through);
    assert_eq!(ev.e_tissue_j, 0.0);
    assert_relative_eq!(ev.e_armor_j, ev.ke_in_j, max_relative = 1e-12);
    let mut chars = vec![ch];
    let out = apply_damage_events(&mut chars, &sc.res.damage_events);
    let v = default_vulnerability(ZoneKind::UpperTorso);
    assert_relative_eq!(
        out[0].blunt_j,
        v.blunt_factor * ev.ke_in_j,
        max_relative = 1e-12
    );
    assert_eq!(out[0].bleed_l_s, 0.0);
    assert!(chars[0].health.wounds.is_empty());
    assert_eq!(chars[0].health.condition, Condition::Wounded);
}

#[test]
fn energy_split_between_armor_and_tissue_is_exact() {
    let mut ch = humanoid();
    ch.equip(
        &[ZoneKind::UpperTorso],
        &ArmorStack::single(mat("generic_mild_steel_plate"), 0.004),
    );
    let mut sc = scene(&ch, "generic_battle_rifle", vec![]);
    sc.fire(1, 0.0, 1.35);
    sc.run(30);
    let ev = sc.res.damage_events[0];
    assert!(ev.penetrated_armor);
    let rec = &sc.res.records[0];
    assert_relative_eq!(
        ev.e_armor_j + ev.e_tissue_j,
        rec.ke_in_j - rec.ke_out_j,
        max_relative = 1e-12
    );
    assert!(
        rec.spall.is_empty(),
        "worn-armor spall goes into the wearer"
    );
}

#[test]
fn head_shot_incapacitates_instantly() {
    let ch = humanoid();
    let mut sc = scene(&ch, "generic_pistol", vec![]);
    sc.fire(1, 0.0, 0.935 * 1.8);
    sc.run(30);
    let mut chars = vec![ch];
    let out = apply_damage_events(&mut chars, &sc.res.damage_events);
    assert_eq!(out[0].zone, ZoneKind::Head);
    assert!(out[0].instant_incap);
    assert!(chars[0].health.condition >= Condition::Incapacitated);
}

#[test]
fn limb_hits_degrade_modifiers_monotonically() {
    let mut h = HealthState::new(HealthParams::default());
    let v = default_vulnerability(ZoneKind::ThighL);
    let mut last = h.modifiers();
    for k in 0..4 {
        let ev = DamageEvent {
            character: 0,
            zone: 0,
            shot_id: ShotId(k),
            time_s: 0.0,
            position: Vec3::zero(),
            direction: Vec3::unit_x(),
            ke_in_j: 400.0,
            e_tissue_j: 150.0,
            e_armor_j: 0.0,
            penetrated_armor: false,
            through_and_through: true,
            ricochet: false,
            wound_track_m: 0.1,
            momentum_in: Vec3::zero(),
            momentum_out: Vec3::zero(),
        };
        h.apply_hit(&ev, ZoneKind::ThighL, &v);
        let m = h.modifiers();
        assert!(m.move_speed_mult < last.move_speed_mult);
        assert!(m.aim_sway_mult > last.aim_sway_mult);
        last = m;
    }
}

// ------------------------------------------------------------ blast

#[test]
fn blast_damage_decreases_with_distance() {
    let ch = humanoid();
    let pose = [(Vec3::zero(), I3)];
    let mut last = f64::INFINITY;
    for d in [1.5, 2.0, 2.5, 3.0, 4.0, 6.0] {
        let src = BlastSource::new(Vec3::new(-d, 0.0, 1.0), 1.0);
        let dmg: f64 = blast_on_character(&src, &ch, &pose)
            .iter()
            .map(|z| z.damage_hp)
            .sum();
        assert!(dmg <= last, "d={d}: {dmg} > {last}");
        last = dmg;
    }
    assert_eq!(last, 0.0);
}

#[test]
fn blast_push_matches_face_impulse_times_area() {
    let ch = humanoid();
    let pose = [(Vec3::zero(), I3)];
    let src = BlastSource::new(Vec3::new(-3.0, 0.0, 1.35), 2.0);
    let res = blast_on_character(&src, &ch, &pose);
    let zi = ch.zone_index(ZoneKind::UpperTorso).unwrap();
    let z = &res[zi];
    let d = (z.center - src.origin).normalize();
    let face = ExposedFace {
        body: 0,
        center: z.center,
        normal: -d,
        area: z.area_m2,
    };
    let i = face_positive_impulse(&src, &face);
    assert_relative_eq!(z.push_impulse.norm(), i * z.area_m2, max_relative = 1e-12);
    assert!(
        z.push_impulse.dot(d) > 0.0,
        "push points away from the charge"
    );
    // Torso box seen head-on along x: area = 4 hy hz.
    assert_relative_eq!(
        z.area_m2,
        4.0 * (0.10 * 1.8) * (0.075 * 1.8),
        max_relative = 1e-12
    );
}

// ------------------------------------------------------------ fragments

#[test]
fn spawned_spall_carries_e_spall() {
    let steel = mat("generic_hard_steel_plate");
    let plate = ShapeRef::Box {
        x: Vec3::new(10.0, 0.0, 1.35),
        r: I3,
        s: BoxShape::new(0.002, 1.0, 1.0),
    };
    let ch = humanoid();
    let mut sc = scene(
        &ch,
        "generic_heavy_machine_gun",
        vec![(plate, Target::Armor(ArmorStack::solid(steel)))],
    );
    sc.fire(1, 0.0, 1.35);
    sc.run(1); // reaches the plate at 10 m within one 1/60 s step
    let rec = sc.res.records[0].clone();
    assert_eq!(rec.kind, ImpactKind::Perforated);
    assert!(!rec.spall.is_empty());
    let frag = sc
        .sys
        .add_profile(fragment_profile("steel_spall", 2e-5, 7850.0));
    let n_before = sc.sys.projectiles.len();
    let n = spawn_spall(&mut sc.sys, frag, &rec);
    assert_eq!(sc.sys.projectiles.len(), n_before + n);
    let ke: f64 = sc
        .sys
        .projectiles
        .iter()
        .filter(|p| p.id.0 & FRAGMENT_ID_BIT != 0)
        .map(|p| 0.5 * p.mass * p.v.norm_squared())
        .sum();
    assert_relative_eq!(ke, rec.e_spall_j, max_relative = 1e-12);
}

#[test]
fn spall_from_plate_can_hit_character_behind_it() {
    let steel = mat("generic_mild_steel_plate");
    // 5 mm plate 1 m in front of the character's chest.
    let plate = ShapeRef::Box {
        x: Vec3::new(CHAR_X - 1.0, 0.0, 1.35),
        r: I3,
        s: BoxShape::new(0.0025, 1.0, 1.0),
    };
    let run = |crowd: usize| {
        let ch = humanoid();
        let mut sc = scene(
            &ch,
            "generic_heavy_machine_gun",
            vec![(plate.clone(), Target::Armor(ArmorStack::solid(steel)))],
        );
        let frag = sc
            .sys
            .add_profile(fragment_profile("steel_spall", 2e-5, 7850.0));
        sc.fire(1, 0.0, 1.35);
        for k in 0..crowd {
            sc.fire(100 + k as u64, 3.0 + 0.01 * k as f64, 1.35);
        }
        for _ in 0..20 {
            sc.run(1);
            let recs = std::mem::take(&mut sc.res.records);
            spawn_all_spall(&mut sc.sys, frag, &recs);
        }
        let mut frag_hits: Vec<_> = sc
            .res
            .damage_events
            .iter()
            .filter(|e| e.shot_id.0 & FRAGMENT_ID_BIT != 0)
            .map(|e| (e.shot_id.0, e.zone, e.e_tissue_j.to_bits()))
            .collect();
        frag_hits.sort();
        frag_hits
    };
    let alone = run(0);
    assert!(!alone.is_empty(), "spall should reach the character");
    assert_eq!(alone, run(200));
}

#[test]
fn fragment_burst_is_deterministic_and_sums_ke() {
    let mut a = ProjectileSystem::new(0.0);
    let pa = a.add_profile(fragment_profile("frag", 1e-4, 7850.0));
    let ke = spawn_fragment_burst(&mut a, pa, Vec3::new(0.0, 0.0, 1.0), 77, 64, 1e-4, 1500.0);
    let sum: f64 = a
        .projectiles
        .iter()
        .map(|p| 0.5 * p.mass * p.v.norm_squared())
        .sum();
    assert_relative_eq!(ke, sum, max_relative = 1e-12);
    let mut b = ProjectileSystem::new(0.0);
    let pb = b.add_profile(fragment_profile("frag", 1e-4, 7850.0));
    let _ = spawn_fragment_burst(&mut b, pb, Vec3::new(0.0, 0.0, 1.0), 77, 64, 1e-4, 1500.0);
    for (p, q) in a.projectiles.iter().zip(&b.projectiles) {
        assert_eq!(p.id, q.id);
        assert_eq!(p.v.to_array(), q.v.to_array());
    }
}

#[test]
fn fragment_drag_scales_with_area() {
    // Same-shape fragment 8× heavier → 4× area → drag multiplier 4.
    assert_relative_eq!(
        fragment_drag_multiplier(8e-4, 1e-4),
        4.0,
        max_relative = 1e-14
    );
    let p = fragment_profile("frag", 1e-4, 7850.0);
    assert_relative_eq!(
        p.reference_area_m2(),
        0.25 * core::f64::consts::PI * p.caliber_m * p.caliber_m,
        max_relative = 1e-12
    );
}
