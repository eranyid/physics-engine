//! Analytic oracles and system tests for pe-combat armor / penetration.

use approx::assert_relative_eq;
use pe_ballistics::ammo::bundled_by_name;
use pe_ballistics::hitscan::ShapeRef;
use pe_ballistics::projectile::{MuzzlePose, ProjectileSystem, ShotId};
use pe_ballistics::{recht_ipson_residual, FieldProvenance};
use pe_collision::BoxShape;
use pe_combat::*;
use pe_math::{libm_ops, Vec3, Xoshiro256StarStar};
use proptest::prelude::*;

const I3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn clean_plate(k: f64, a: f64, p: f64) -> ArmorMaterial {
    ArmorMaterial {
        version: ARMOR_MATERIAL_VERSION,
        name: "test_plate".into(),
        density_kg_m3: 7850.0,
        model: PenetrationModel::Plate {
            lambert_k: k,
            obliquity_exponent: 0.75,
            residual_a: a,
            residual_p: p,
            erosion_per_z: 0.0,
        },
        ricochet: RicochetParams::none(),
        spall: SpallParams::none(),
        tumble_drag_gain: 0.0,
        exit_deflection_std_rad: 0.0,
        provenance: FieldProvenance::nominal("test"),
    }
}

fn clean_medium(a: f64, b: f64) -> ArmorMaterial {
    ArmorMaterial {
        model: PenetrationModel::Medium {
            poncelet_a_pa: a,
            poncelet_b_kg_m3: b,
        },
        name: "test_medium".into(),
        ..clean_plate(1.0, 1.0, 2.0)
    }
}

fn head_on(v: f64, mass: f64, d: f64, len: f64) -> StackInput {
    StackInput {
        velocity: Vec3::new(v, 0.0, 0.0),
        mass_kg: mass,
        diameter_m: d,
        length_m: len,
        normal: Vec3::new(-1.0, 0.0, 0.0),
        chord_m: 0.0,
    }
}

#[test]
fn lambert_v50_closed_form() {
    let (k, m, d, l, t) = (126_491.0, 0.004, 0.0057, 0.023, 0.006);
    let z: f64 = t / d;
    let f = z + libm_ops::exp(-z) - 1.0;
    let expect = k * libm_ops::pow(l / d, 0.15) * libm_ops::sqrt(f * d * d * d / m);
    assert_relative_eq!(
        lambert_jonas_v50(k, m, d, l, t, 0.0, 0.75),
        expect,
        max_relative = 1e-14
    );
}

#[test]
fn lambert_z_obliquity_scaling() {
    let th = 0.7;
    let z0 = lambert_z(0.01, 0.005, 0.0, 0.75);
    let z1 = lambert_z(0.01, 0.005, th, 0.75);
    assert_relative_eq!(
        z1 / z0,
        libm_ops::pow(1.0 / libm_ops::cos(th), 0.75),
        max_relative = 1e-14
    );
    // n = 1 → pure line-of-sight thickness 1/cos θ.
    assert_relative_eq!(
        lambert_z(0.01, 0.005, th, 1.0) / z0,
        1.0 / libm_ops::cos(th),
        max_relative = 1e-14
    );
}

#[test]
fn v50_monotone_in_thickness_and_obliquity() {
    let mut last = 0.0;
    for i in 1..20 {
        let v = lambert_jonas_v50(1e5, 0.01, 0.0078, 0.03, i as f64 * 1e-3, 0.0, 0.75);
        assert!(v > last);
        last = v;
    }
    let a = lambert_jonas_v50(1e5, 0.01, 0.0078, 0.03, 0.005, 0.0, 0.75);
    let b = lambert_jonas_v50(1e5, 0.01, 0.0078, 0.03, 0.005, 1.0, 0.75);
    assert!(b > a);
}

#[test]
fn plate_residual_matches_recht_ipson_p2() {
    let mats = vec![clean_plate(1e5, 0.8, 2.0)];
    let stack = ArmorStack::single(0, 0.004);
    let (m, d, l) = (0.0095, 0.0078, 0.03);
    let v50 = lambert_jonas_v50(1e5, m, d, l, 0.004, 0.0, 0.75);
    let v0 = v50 * 1.5;
    let mut rng = Xoshiro256StarStar::seed(1);
    let r = resolve_stack(&mats, &stack, head_on(v0, m, d, l), &mut rng);
    assert_eq!(r.kind, ImpactKind::Perforated);
    let expect = 0.8 * libm_ops::sqrt(v0 * v0 - v50 * v50);
    assert_relative_eq!(r.velocity_out.norm(), expect, max_relative = 1e-12);
    assert_relative_eq!(
        r.velocity_out.norm(),
        recht_ipson_residual(v0, v50, 0.8, 2.0),
        max_relative = 1e-12
    );
}

#[test]
fn plate_stops_below_v50() {
    let mats = vec![clean_plate(1e5, 0.8, 2.0)];
    let stack = ArmorStack::single(0, 0.004);
    let v50 = lambert_jonas_v50(1e5, 0.0095, 0.0078, 0.03, 0.004, 0.0, 0.75);
    let mut rng = Xoshiro256StarStar::seed(1);
    let r = resolve_stack(
        &mats,
        &stack,
        head_on(v50 * 0.999, 0.0095, 0.0078, 0.03),
        &mut rng,
    );
    assert_eq!(r.kind, ImpactKind::Stopped);
    assert_relative_eq!(r.e_deposited_j, r.ke_in_j, max_relative = 1e-15);
}

#[test]
fn poncelet_exit_closed_form_matches_rk4() {
    let (m, v0, d, a, b) = (0.008, 360.0, 0.009, 5e6, 424.0);
    let area = 0.25 * core::f64::consts::PI * d * d;
    let path = 0.15;
    // Integrate m v dv/dx = −A (a + b v²) in x with RK4.
    let f = |v: f64| -area * (a + b * v * v) / (m * v);
    let n = 20_000;
    let h = path / n as f64;
    let mut v = v0;
    for _ in 0..n {
        let k1 = f(v);
        let k2 = f(v + 0.5 * h * k1);
        let k3 = f(v + 0.5 * h * k2);
        let k4 = f(v + h * k3);
        v += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    let closed = poncelet_exit_speed(m, v0, area, a, b, path).unwrap();
    assert_relative_eq!(closed, v, max_relative = 1e-9);
}

#[test]
fn poncelet_exit_vanishes_at_depth() {
    let (m, v0, area, a, b) = (0.008, 360.0, 6.36e-5, 5e6, 424.0);
    let depth = pe_ballistics::poncelet_depth(m, v0, area, a, b).unwrap();
    assert!(poncelet_exit_speed(m, v0, area, a, b, depth * 1.0001).is_none());
    let just_short = poncelet_exit_speed(m, v0, area, a, b, depth * 0.999).unwrap();
    assert!(just_short < 0.05 * v0);
}

#[test]
fn medium_stop_reports_depth() {
    let mats = vec![clean_medium(5e6, 424.0)];
    let stack = ArmorStack::single(0, 2.0); // 2 m block: stops
    let mut rng = Xoshiro256StarStar::seed(3);
    let r = resolve_stack(&mats, &stack, head_on(360.0, 0.008, 0.009, 0.015), &mut rng);
    assert_eq!(r.kind, ImpactKind::Stopped);
    let area = 0.25 * core::f64::consts::PI * 0.009 * 0.009;
    let d = pe_ballistics::poncelet_depth(0.008, 360.0, area, 5e6, 424.0).unwrap();
    assert_relative_eq!(r.stop_depth_m.unwrap(), d, max_relative = 1e-14);
}

#[test]
fn sharp_ricochet_at_graze() {
    let mut mat = clean_plate(1e6, 0.9, 2.0); // very strong plate
    mat.ricochet = RicochetParams {
        critical_graze_rad: 0.3,
        transition_width_rad: 0.0,
        normal_restitution: 0.2,
        tangential_retention: 0.8,
        deflection_std_rad: 0.0,
    };
    let mats = vec![mat];
    let stack = ArmorStack::single(0, 0.01);
    let graze: f64 = 0.1;
    let v = Vec3::new(libm_ops::cos(graze), 0.0, -libm_ops::sin(graze)) * 800.0;
    let mut rng = Xoshiro256StarStar::seed(9);
    let r = resolve_stack(
        &mats,
        &stack,
        StackInput {
            velocity: v,
            mass_kg: 0.004,
            diameter_m: 0.0057,
            length_m: 0.023,
            normal: Vec3::unit_z(),
            chord_m: 0.0,
        },
        &mut rng,
    );
    assert_eq!(r.kind, ImpactKind::Ricochet);
    let expect = Vec3::new(v.x * 0.8, 0.0, -v.z * 0.2);
    assert_relative_eq!(r.velocity_out.x, expect.x, max_relative = 1e-12);
    assert_relative_eq!(r.velocity_out.z, expect.z, max_relative = 1e-12);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// KE_in = KE_out + E_dep + E_spall for random stacks and hits; spall KE sums to E_spall;
    /// ricochet never gains energy and leaves the surface.
    #[test]
    fn energy_ledger_closes(
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
        let sum = r.ke_out_j + r.e_deposited_j + r.e_spall_j;
        prop_assert!((sum - r.ke_in_j).abs() <= 1e-9 * r.ke_in_j);
        prop_assert!(r.e_deposited_j >= 0.0 && r.e_spall_j >= 0.0);
        prop_assert!(r.ke_out_j <= r.ke_in_j * (1.0 + 1e-12));
        let frag_ke: f64 = r.spall.iter().map(|f| 0.5 * f.mass_kg * f.velocity.norm_squared()).sum();
        prop_assert!((frag_ke - r.e_spall_j).abs() <= 1e-9 * r.ke_in_j.max(1.0));
        if r.kind == ImpactKind::Ricochet {
            prop_assert!(r.velocity_out.norm() <= speed * (1.0 + 1e-12));
            prop_assert!(r.velocity_out.dot(Vec3::unit_z()) >= 0.0);
        }
        prop_assert!(r.velocity_out.x.is_finite() && r.drag_factor >= 1.0);
    }
}

// ---------------------------------------------------------------- system tests

fn wall(x: f64, half_thickness: f64) -> ShapeRef {
    ShapeRef::Box {
        x: Vec3::new(x, 0.0, 1.5),
        r: I3,
        s: BoxShape::new(half_thickness, 2.0, 2.0),
    }
}

fn rifle_system() -> (ProjectileSystem, usize) {
    let mut sys = ProjectileSystem::new(0.0);
    sys.enable_spin_effects = false;
    let p = bundled_by_name("generic_battle_rifle").unwrap();
    let idx = sys.add_profile(p);
    (sys, idx)
}

fn fire(sys: &mut ProjectileSystem, idx: usize, id: u64, y: f64) {
    let _ = sys.spawn(
        ShotId(id),
        idx,
        MuzzlePose {
            position: Vec3::new(0.0, y, 1.5),
            direction: Vec3::unit_x(),
            barrel_length_m: None,
        },
    );
}

fn material_index(name: &str) -> usize {
    bundled_materials()
        .iter()
        .position(|m| m.name == name)
        .unwrap()
}

#[test]
fn rifle_perforates_wood_then_stops_in_steel() {
    let (mut sys, idx) = rifle_system();
    let wood = material_index("generic_wood");
    let steel = material_index("generic_hard_steel_plate");
    let shapes = vec![wall(20.0, 0.02), wall(40.0, 0.025)];
    let mut res = CombatResolver::new(
        bundled_materials(),
        vec![
            Some(ArmorStack::solid(wood)),
            Some(ArmorStack::single(steel, 0.05)),
        ],
    );
    fire(&mut sys, idx, 1, 0.0);
    for _ in 0..60 {
        sys.step_with_resolver(1.0 / 60.0, &shapes, &[None, None], &mut res);
    }
    assert_eq!(res.records.len(), 2, "{:?}", res.records);
    let r0 = &res.records[0];
    assert_eq!(r0.kind, ImpactKind::Perforated);
    let exit = r0.exit.unwrap();
    assert_relative_eq!(exit.x, 20.02, epsilon = 1e-6);
    assert!(r0.ke_out_j < r0.ke_in_j);
    let r1 = &res.records[1];
    assert_eq!(r1.kind, ImpactKind::Stopped);
    assert!(!sys.projectiles[0].alive);
    // Bullet left the first wall slower than it arrived at the second? No: drag only.
    assert!(r1.velocity_in.norm() <= r0.velocity_out.norm());
}

#[test]
fn no_tunneling_through_thin_plate_at_low_rate() {
    let (mut sys, idx) = rifle_system();
    let steel = material_index("generic_hard_steel_plate");
    let shapes = vec![wall(30.0, 0.0005)]; // 1 mm plate
    let mut res = CombatResolver::new(bundled_materials(), vec![Some(ArmorStack::solid(steel))]);
    fire(&mut sys, idx, 7, 0.0);
    for _ in 0..30 {
        sys.step_with_resolver(1.0 / 60.0, &shapes, &[None], &mut res);
    }
    assert_eq!(res.records.len(), 1);
    assert_eq!(res.records[0].kind, ImpactKind::Perforated);
    assert!(sys.projectiles[0].x.x > 30.0);
    assert!(sys.projectiles[0].drag_multiplier >= 1.0);
}

#[test]
fn same_shot_same_outcome_regardless_of_crowd() {
    let concrete = material_index("generic_concrete");
    let run = |crowd: usize| {
        let (mut sys, idx) = rifle_system();
        let shapes = vec![wall(25.0, 0.05)];
        let mut res =
            CombatResolver::new(bundled_materials(), vec![Some(ArmorStack::solid(concrete))]);
        fire(&mut sys, idx, 42, 0.0);
        for k in 0..crowd {
            fire(&mut sys, idx, 1000 + k as u64, 0.3 + 0.001 * k as f64);
        }
        for _ in 0..30 {
            sys.step_with_resolver(1.0 / 60.0, &shapes, &[None], &mut res);
        }
        let p = sys
            .projectiles
            .iter()
            .find(|p| p.id == ShotId(42))
            .unwrap()
            .clone();
        let rec: Vec<_> = res
            .records
            .iter()
            .filter(|r| r.shot_id == ShotId(42))
            .map(|r| (r.kind, r.ke_out_j.to_bits(), r.spall.len()))
            .collect();
        (p.x.to_array().map(f64::to_bits), p.alive, rec)
    };
    assert_eq!(run(0), run(200));
}

#[test]
fn shapes_without_armor_stop_like_classic() {
    let (mut sys, idx) = rifle_system();
    let shapes = vec![wall(10.0, 0.01)];
    let mut res = CombatResolver::new(bundled_materials(), vec![None]);
    fire(&mut sys, idx, 3, 0.0);
    for _ in 0..30 {
        sys.step_with_resolver(1.0 / 60.0, &shapes, &[Some(5)], &mut res);
    }
    assert_eq!(res.records.len(), 1);
    assert_eq!(res.records[0].kind, ImpactKind::Stopped);
    assert_eq!(res.records[0].body_id, Some(5));
    assert_relative_eq!(sys.projectiles[0].x.x, 9.99, epsilon = 1e-9);
}
