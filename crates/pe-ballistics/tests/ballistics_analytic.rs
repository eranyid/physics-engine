//! Analytic and system tests for pe-ballistics.

use approx::assert_relative_eq;
use pe_ballistics::ammo::{bundled_by_name, bundled_profiles, AmmoProfile};
use pe_ballistics::drag::{analytic_cd, bc_imperial_to_si, g1_table, g7_table, DragModelRef};
use pe_ballistics::hitscan::{segment_vs_box, ShapeRef};
use pe_ballistics::impact::{poncelet_depth, recht_ipson_residual};
use pe_ballistics::oracles::{
    coriolis_freefall_east, g0, quadratic_terminal_speed, vacuum_max_height, vacuum_range,
};
use pe_ballistics::projectile::{MuzzlePose, ProjectileSystem, ShotId};
use pe_ballistics::{ballistics_volley_hash, EARTH_OMEGA};
use pe_aero::turbulence::WindField;
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

#[test]
fn g1_g7_tables_match_jbm_endpoints() {
    let g1 = g1_table();
    let g7 = g7_table();
    assert_eq!(g1.mach.len(), 79);
    assert_eq!(g7.mach.len(), 84);
    assert_relative_eq!(g1.cd_linear(1.0), 0.4805, epsilon = 0.0);
    assert_relative_eq!(g7.cd_linear(1.05), 0.4043, epsilon = 0.0);
}

#[test]
fn ammo_serde_roundtrip_and_legacy_drag_alias() {
    let p = bundled_by_name("5.56_55gr_fmjbt_federal_ae193").unwrap();
    let j = p.to_json().unwrap();
    let q = AmmoProfile::from_json(&j).unwrap();
    assert_eq!(p.name, q.name);
    assert_relative_eq!(p.mass_kg, q.mass_kg, epsilon = 0.0);
    assert_relative_eq!(p.muzzle_velocity_m_s, q.muzzle_velocity_m_s, epsilon = 0.0);
    assert!(matches!(q.bc.drag_model, DragModelRef::G1));
    // Legacy tag deserialises to G1.
    let legacy = r#"{
      "version": 1,
      "name": "legacy",
      "caliber_m": 0.009,
      "mass_kg": 0.008,
      "muzzle_velocity_m_s": 350.0,
      "bc": {
        "value_si": 100.0,
        "drag_model": "G1StandIn",
        "table": null,
        "provenance": { "note": "test", "verified": false }
      },
      "length_m": 0.015,
      "twist_m_per_rev": 0.25,
      "form_factor": 1.0,
      "barrel_length_ref_m": null,
      "velocity_std_m_s": 0.0,
      "dispersion_std_rad": 0.0,
      "provenance": { "note": "test", "verified": false }
    }"#;
    let leg = AmmoProfile::from_json(legacy).unwrap();
    assert!(matches!(leg.bc.drag_model, DragModelRef::G1));
}

#[test]
fn every_verified_ammo_row_has_citation() {
    for p in bundled_profiles() {
        if p.provenance.verified {
            assert!(!p.provenance.note.trim().is_empty(), "{}", p.name);
        }
        if p.bc.provenance.verified {
            assert!(!p.bc.provenance.note.trim().is_empty(), "bc {}", p.name);
        }
    }
}

/// G1 BC 0.5, 2800 fps, ICAO sea-level, sight height 0, flat fire.
///
/// Independent open calculator (py_ballisticcalc 2.2.0, TableG1, Atmo.icao(),
/// Weapon sight_height=0) measured on 2026-11-01:
///
/// | yd | drop (in) | TOF (s) | vel (fps) |
/// | 100 | -2.3199 | 0.110872 | 2615.05 |
/// | 200 | -9.7308 | 0.229701 | 2437.52 |
/// | 300 | -23.0023 | 0.357326 | 2266.97 |
/// | 400 | -43.0505 | 0.494724 | 2103.16 |
/// | 500 | -70.9732 | 0.643012 | 1946.36 |
///
/// Tolerance: drop within **5% relative or 1.0 inch abs** (whichever larger);
/// TOF within **3%**; velocity within **2%**. Linear Cd(Mach) interpolation.
#[test]
fn g1_bc05_2800fps_drop_tof_vs_py_ballisticcalc() {
    let mut profile = bundled_by_name("oracle_g1_bc0.5_2800fps").unwrap();
    assert_relative_eq!(profile.bc.value_si, bc_imperial_to_si(0.5), epsilon = 1e-9);
    assert!(matches!(profile.bc.drag_model, DragModelRef::G1));
    profile.velocity_std_m_s = 0.0;
    profile.dispersion_std_rad = 0.0;

    let ranges_yd = [100.0, 200.0, 300.0, 400.0, 500.0];
    let ranges_m: Vec<f64> = ranges_yd.iter().map(|y| y * 0.9144).collect();
    // Fine step; calm ICAO sea-level (pe-aero ISA); spin/Magnus off for point-mass oracle.
    let mut sys = ProjectileSystem::new(0.0);
    sys.enable_spin_effects = false;
    sys.wind = WindField::steady(Vec3::zero());
    let pidx = sys.add_profile(profile);
    let _ = sys.spawn(
        ShotId(1),
        pidx,
        MuzzlePose {
            position: Vec3::zero(),
            direction: Vec3::unit_x(),
            barrel_length_m: None,
        },
    );
    let mut rows = Vec::new();
    let dt = 1.0 / 4000.0;
    let mut next_ri = 0usize;
    let max_range = ranges_m[ranges_m.len() - 1];
    while next_ri < ranges_m.len() && sys.t < 5.0 {
        if !sys.projectiles[0].alive {
            break;
        }
        let x_before = sys.projectiles[0].x.x;
        let z_before = sys.projectiles[0].x.z;
        let t_before = sys.t;
        let v_before = sys.projectiles[0].v.norm();
        sys.step(dt, &[], &[]);
        let p = &sys.projectiles[0];
        let x_after = p.x.x;
        while next_ri < ranges_m.len()
            && x_before <= ranges_m[next_ri]
            && x_after >= ranges_m[next_ri]
        {
            let r = ranges_m[next_ri];
            let alpha = ((r - x_before) / (x_after - x_before).max(1e-12)).clamp(0.0, 1.0);
            let drop = z_before + alpha * (p.x.z - z_before);
            let speed = v_before + alpha * (p.v.norm() - v_before);
            let time = t_before + alpha * (sys.t - t_before);
            rows.push((drop, time, speed));
            next_ri += 1;
        }
        if x_after > max_range + 1.0 {
            break;
        }
    }

    assert_eq!(rows.len(), 5, "expected 5 range samples, got {}", rows.len());

    // py_ballisticcalc oracle (inches, seconds, fps)
    let ref_drop_in: [f64; 5] = [-2.3199, -9.7308, -23.0023, -43.0505, -70.9732];
    let ref_tof: [f64; 5] = [0.110872, 0.229701, 0.357326, 0.494724, 0.643012];
    let ref_fps: [f64; 5] = [2615.05, 2437.52, 2266.97, 2103.16, 1946.36];

    for i in 0..5 {
        let drop_in = rows[i].0 / 0.0254;
        let tof = rows[i].1;
        let fps = rows[i].2 / 0.3048;
        let tol_drop = (ref_drop_in[i].abs() * 0.05).max(1.0);
        let err_drop = (drop_in - ref_drop_in[i]).abs();
        assert!(
            err_drop <= tol_drop,
            "yd={} drop_in={:.3} ref={:.3} err={:.3} tol={:.3}",
            ranges_yd[i],
            drop_in,
            ref_drop_in[i],
            err_drop,
            tol_drop
        );
        let err_tof = ((tof - ref_tof[i]) / ref_tof[i]).abs();
        assert!(
            err_tof <= 0.03,
            "yd={} tof={:.5} ref={:.5} rel={:.4}",
            ranges_yd[i],
            tof,
            ref_tof[i],
            err_tof
        );
        let err_v = ((fps - ref_fps[i]) / ref_fps[i]).abs();
        assert!(
            err_v <= 0.02,
            "yd={} fps={:.1} ref={:.1} rel={:.4}",
            ranges_yd[i],
            fps,
            ref_fps[i],
            err_v
        );
    }
}
