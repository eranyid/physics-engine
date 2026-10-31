//! Analytic and system tests for pe-ballistics.

use approx::assert_relative_eq;
use pe_ballistics::ammo::{bundled_profiles, AmmoProfile};
use pe_ballistics::drag::analytic_cd;
use pe_ballistics::hitscan::{segment_vs_box, ShapeRef};
use pe_ballistics::impact::{poncelet_depth, recht_ipson_residual};
use pe_ballistics::oracles::{
    coriolis_freefall_east, g0, quadratic_terminal_speed, vacuum_max_height, vacuum_range,
};
use pe_ballistics::projectile::{MuzzlePose, ProjectileSystem, ShotId};
use pe_ballistics::{ballistics_volley_hash, EARTH_OMEGA};
use pe_collision::BoxShape;
use pe_math::{libm_ops, Vec3};

#[test]
fn vacuum_oracles() {
    let v = 100.0;
    let th = core::f64::consts::FRAC_PI_4;
    let r = vacuum_range(v, th, g0());
    assert_relative_eq!(r, v * v / g0(), epsilon = 1e-10);
    let h = vacuum_max_height(v, th, g0());
    assert_relative_eq!(h, (v * v) / (4.0 * g0()), epsilon = 1e-10);
}

#[test]
fn quadratic_terminal() {
    let vt = quadratic_terminal_speed(1.0, 0.01, g0());
    assert_relative_eq!(vt, libm_ops::sqrt(g0() / 0.01), epsilon = 1e-12);
}

#[test]
fn coriolis_freefall_small_time() {
    let t = 2.0;
    let lat = 0.0;
    let d = coriolis_freefall_east(t, lat);
    let expect = (1.0 / 3.0) * EARTH_OMEGA * g0() * t * t * t;
    assert_relative_eq!(d, expect, epsilon = 1e-14);
}

#[test]
fn muzzle_energy_consistency() {
    for p in bundled_profiles() {
        let e = p.muzzle_energy_j();
        let expect = 0.5 * p.mass_kg * p.muzzle_velocity_m_s * p.muzzle_velocity_m_s;
        assert_relative_eq!(e, expect, epsilon = 1e-12);
    }
}

#[test]
fn ammo_json_roundtrip() {
    let p = &bundled_profiles()[0];
    let j = p.to_json().unwrap();
    let q = AmmoProfile::from_json(&j).unwrap();
    assert_eq!(p.name, q.name);
    assert_relative_eq!(p.mass_kg, q.mass_kg, epsilon = 0.0);
}

#[test]
fn deterministic_replay_independent_of_count() {
    let mut a = ProjectileSystem::new(0.5);
    let mut b = ProjectileSystem::new(0.5);
    let p = bundled_profiles()
        .into_iter()
        .find(|p| p.name == "generic_intermediate_rifle")
        .unwrap();
    let mut p = p;
    p.velocity_std_m_s = 0.0;
    p.dispersion_std_rad = 0.0;
    let ia = a.add_profile(p.clone());
    let ib = b.add_profile(p);
    // System A: only shot 42
    a.spawn(
        ShotId(42),
        ia,
        MuzzlePose {
            position: Vec3::new(0.0, 0.0, 1.0),
            direction: Vec3::unit_x(),
            barrel_length_m: None,
        },
    );
    // System B: 1000 other shots then 42
    for i in 0..1000 {
        b.spawn(
            ShotId(10_000 + i),
            ib,
            MuzzlePose {
                position: Vec3::new(0.0, i as f64, 1.0),
                direction: Vec3::unit_x(),
                barrel_length_m: None,
            },
        );
    }
    b.spawn(
        ShotId(42),
        ib,
        MuzzlePose {
            position: Vec3::new(0.0, 0.0, 1.0),
            direction: Vec3::unit_x(),
            barrel_length_m: None,
        },
    );
    for _ in 0..50 {
        a.step(1.0 / 120.0, &[], &[]);
        b.step(1.0 / 120.0, &[], &[]);
    }
    let pa = a.projectiles.iter().find(|p| p.id == ShotId(42)).unwrap();
    let pb = b.projectiles.iter().find(|p| p.id == ShotId(42)).unwrap();
    assert_relative_eq!(pa.x.x, pb.x.x, epsilon = 1e-9);
    assert_relative_eq!(pa.x.y, pb.x.y, epsilon = 1e-9);
    assert_relative_eq!(pa.x.z, pb.x.z, epsilon = 1e-9);
}

#[test]
fn analytic_cd_transonic_rise() {
    let sub = analytic_cd(0.5);
    let peak = analytic_cd(1.1);
    let super_ = analytic_cd(2.0);
    assert!(peak > sub);
    assert!(super_ < peak);
}

#[test]
fn no_tunneling_fast_bullet_thin_wall() {
    let mut sys = ProjectileSystem::new(0.0);
    let mut p = bundled_profiles()[2].clone();
    p.muzzle_velocity_m_s = 1000.0;
    p.velocity_std_m_s = 0.0;
    p.dispersion_std_rad = 0.0;
    let idx = sys.add_profile(p);
    sys.spawn(
        ShotId(1),
        idx,
        MuzzlePose {
            position: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::unit_x(),
            barrel_length_m: None,
        },
    );
    // 1 cm thick wall (box) centered at x=10 m
    let wall = ShapeRef::Box {
        x: Vec3::new(10.0, 0.0, 0.0),
        r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        s: BoxShape::new(0.005, 1.0, 1.0),
    };
    let shapes = [wall];
    let bodies = [Some(0usize)];
    // Large dt = 1/60 would jump ~16.7 m — must still hit.
    sys.step(1.0 / 60.0, &shapes, &bodies);
    assert!(!sys.projectiles[0].alive);
    assert_eq!(sys.hits.len(), 1);
    assert!(sys.hits[0].position.x > 9.9 && sys.hits[0].position.x < 10.1);
}

#[test]
fn dispersion_stats_mean_near_aim() {
    let mut sys = ProjectileSystem::new(0.0);
    let mut p = bundled_profiles()[0].clone();
    p.dispersion_std_rad = 0.002;
    p.velocity_std_m_s = 0.0;
    let idx = sys.add_profile(p);
    let n = 400usize;
    for i in 0..n {
        sys.spawn(
            ShotId(i as u64 + 1),
            idx,
            MuzzlePose {
                position: Vec3::zero(),
                direction: Vec3::unit_x(),
                barrel_length_m: None,
            },
        );
    }
    // Sample direction via velocity at t=0
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut mz = 0.0;
    for p in &sys.projectiles {
        let d = p.v.normalize();
        mx += d.x;
        my += d.y;
        mz += d.z;
    }
    mx /= n as f64;
    my /= n as f64;
    mz /= n as f64;
    assert!(my.abs() < 0.002);
    assert!(mz.abs() < 0.002);
    assert!(mx > 0.99);
}

#[test]
fn recht_ipson_and_poncelet_closed_forms() {
    let vr = recht_ipson_residual(400.0, 200.0, 0.9, 2.0);
    assert_relative_eq!(
        vr,
        0.9 * libm_ops::sqrt(400.0_f64.powi(2) - 200.0_f64.powi(2)),
        epsilon = 1e-12
    );
    let x = poncelet_depth(0.01, 500.0, 1e-4, 1e8, 2e3).unwrap();
    let expect = (0.01 / (2.0 * 1e-4 * 2e3)) * libm_ops::log(1.0 + (2e3 / 1e8) * 500.0 * 500.0);
    assert_relative_eq!(x, expect, epsilon = 1e-12);
}

#[test]
fn segment_box_hit() {
    let h = segment_vs_box(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        BoxShape::new(0.1, 0.5, 0.5),
    );
    assert!(h.is_some());
}

#[test]
fn volley_hash_bit_stable() {
    assert_eq!(ballistics_volley_hash(), ballistics_volley_hash());
}
