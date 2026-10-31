//! Stage-3 hydro analytic / published-value tests (parity with Python suite).

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_core::forces::UniformGravity;
use pe_core::integrators::rk4;
use pe_core::{inertia_solid_box, RigidBody, World};
use pe_hydro::hull::{BoxHull, ConvexMeshHull, CylinderHull, Hull, SphereHull, RHO_FRESHWATER};
use pe_hydro::{
    angle_of_vanishing_stability, box_added_mass_heave, box_added_mass_surge,
    box_righting_arm_curve, check_imo_is_code, heave_period_earth, metacentric_height,
    righting_arm, righting_moment, roll_period_earth, Buoyancy, HullDrag, WaveDrag,
};
use pe_math::libm_ops;
use pe_math::{Quat, Vec3, STANDARD_GRAVITY};

#[test]
fn box_floating_draft_archimedes() {
    let hull = BoxHull::new(20.0, 5.0, 3.0);
    let rho = 1000.0;
    let mass = 50_000.0;
    let t = hull.upright_draft_for_mass(mass, rho);
    assert_abs_diff_eq!(t, 0.5, epsilon = 1e-15);
    let z_c = -t + 0.5 * hull.depth;
    let sub = hull.submerged(Vec3::new(0.0, 0.0, z_c), Quat::identity(), 0.0);
    assert_relative_eq!(sub.volume, mass / rho, max_relative = 1e-12);
    assert_abs_diff_eq!(sub.a_wp, hull.length * hull.beam, epsilon = 1e-12);
    assert_relative_eq!(sub.cob_world.z, -t + 0.5 * t, max_relative = 1e-12);
}

#[test]
fn cylinder_floating_draft() {
    let hull = CylinderHull::new(2.0, 4.0);
    let rho = 1000.0;
    let mass = 8000.0;
    let t = hull.upright_draft_for_mass(mass, rho);
    assert_relative_eq!(
        t,
        mass / (rho * core::f64::consts::PI * 4.0),
        max_relative = 1e-15
    );
    let z_c = -t + 0.5 * hull.height;
    let sub = hull.submerged(Vec3::new(0.0, 0.0, z_c), Quat::identity(), 0.0);
    assert_relative_eq!(sub.volume, mass / rho, max_relative = 1e-12);
}

#[test]
fn buoyancy_equilibrium_force() {
    let hull = BoxHull::new(10.0, 4.0, 2.0);
    let mass = 20_000.0;
    let t = hull.upright_draft_for_mass(mass, RHO_FRESHWATER);
    let z_c = -t + 0.5 * hull.depth;
    let body = RigidBody::new(mass)
        .with_inertia(inertia_solid_box(mass, 10.0, 4.0, 2.0))
        .with_position(Vec3::new(0.0, 0.0, z_c));
    let w = World::new(
        vec![body],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(Buoyancy::freshwater(0, Hull::Box(hull), 0.0)),
        ],
    );
    let (f, _) = w.forces(&w.state, 0.0);
    assert_abs_diff_eq!(f[0].norm(), 0.0, epsilon = 1e-8);
}

#[test]
fn barge_gm_published_formulas() {
    let hull = BoxHull::new(30.0, 8.0, 4.0);
    let draft = 2.0;
    let kg = 1.5;
    let kb = hull.upright_kb(draft);
    let bm = hull.upright_bm_roll(draft);
    let gm = hull.upright_gm(draft, kg, true);
    assert_abs_diff_eq!(kb, 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(bm, 64.0 / 24.0, epsilon = 1e-15);
    assert_abs_diff_eq!(gm, metacentric_height(kb, bm, kg), epsilon = 1e-15);
    assert!(gm > 0.0);
}

#[test]
fn gz_small_angle_matches_gm_sin() {
    let hull = BoxHull::new(30.0, 8.0, 4.0);
    let draft = 2.0;
    let kg = 1.5;
    let gm = hull.upright_gm(draft, kg, true);
    for phi in [0.01, 0.05, 0.1, 5.0_f64.to_radians()] {
        let gz_f = righting_arm(gm, phi);
        assert_relative_eq!(hull.righting_arm_small(gm, phi), gm * libm_ops::sin(phi));
        let gz_n = box_righting_arm_curve(&hull, draft, kg, &[phi])[0];
        assert_relative_eq!(gz_n, gz_f, max_relative = 2e-2);
    }
}

#[test]
fn avs_sign_change() {
    let hull = BoxHull::new(20.0, 8.0, 2.5);
    let draft = 2.0;
    let kg = 1.8;
    assert!(hull.upright_gm(draft, kg, true) > 0.0);
    let avs = angle_of_vanishing_stability(&hull, draft, kg, 89.0_f64.to_radians(), 180);
    assert!(avs > 50.0_f64.to_radians() && avs < 85.0_f64.to_radians());
    let before = box_righting_arm_curve(&hull, draft, kg, &[0.5 * avs])[0];
    let after = box_righting_arm_curve(&hull, draft, kg, &[avs + 5.0_f64.to_radians()])[0];
    assert!(before > 0.0);
    assert!(after < 0.0);
}

#[test]
fn heave_period_simulation() {
    let hull = BoxHull::new(12.0, 4.0, 2.0);
    let mass = 24_000.0;
    let t_draft = hull.upright_draft_for_mass(mass, 1000.0);
    let a_wp = hull.length * hull.beam;
    let t_heave = heave_period_earth(mass, 1000.0, a_wp);
    let z_eq = -t_draft + 0.5 * hull.depth;
    let body = RigidBody::new(mass)
        .with_inertia(inertia_solid_box(mass, 12.0, 4.0, 2.0))
        .with_position(Vec3::new(0.0, 0.0, z_eq + 0.05));
    let mut w = World::new(
        vec![body],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(Buoyancy::freshwater(0, Hull::Box(hull), 0.0)),
        ],
    );
    let dt = 1e-3;
    let t_end = t_heave * 3.5;
    let mut zs = Vec::new();
    let mut ts = Vec::new();
    while w.t < t_end {
        w.step(dt, rk4);
        zs.push(w.state.x[0].z - z_eq);
        ts.push(w.t);
    }
    let mut crosses = Vec::new();
    for i in 1..zs.len() {
        if zs[i - 1] <= 0.0 && zs[i] > 0.0 {
            crosses.push(ts[i]);
        }
    }
    assert!(crosses.len() >= 2);
    let t_meas = crosses[1] - crosses[0];
    assert_relative_eq!(t_meas, t_heave, max_relative = 2e-2);
}

#[test]
fn convex_mesh_matches_box_upright() {
    let hull = BoxHull::new(4.0, 2.0, 2.0);
    let (verts, faces) = hull.mesh_topology(0.0);
    let mesh = ConvexMeshHull::new(verts, faces);
    let x = Vec3::zero();
    let q = Quat::from_axis_angle(Vec3::unit_x(), 1e-4);
    let sub_m = mesh.submerged(x, q, 0.0);
    let sub_b = hull.submerged(x, Quat::identity(), 0.0);
    assert_relative_eq!(sub_m.volume, sub_b.volume, max_relative = 1e-3);
    assert_abs_diff_eq!(sub_m.cob_world.z, sub_b.cob_world.z, epsilon = 2e-3);
}

#[test]
fn hull_drag_opposes_surge() {
    let body = RigidBody::new(1000.0)
        .with_inertia_diag(100.0, 100.0, 100.0)
        .with_velocity(Vec3::new(5.0, 0.0, 0.0));
    let drag = HullDrag::new(
        0,
        20.0,
        10.0,
        0.004,
        WaveDrag::Constant(0.002),
        1000.0,
        STANDARD_GRAVITY,
        Vec3::zero(),
        false,
    );
    let w = World::new(vec![body], vec![Box::new(drag)]);
    let (f, _) = w.forces(&w.state, 0.0);
    let v = 5.0;
    let r = 0.5 * 1000.0 * (0.004 + 0.002) * 20.0 * v * v;
    assert_relative_eq!(f[0].x, -r, max_relative = 1e-12);
}

#[test]
fn added_mass_and_righting_moment() {
    assert!(box_added_mass_heave(1000.0, 40.0, 1.0) > 0.0);
    assert_abs_diff_eq!(
        box_added_mass_surge(1000.0, 40.0, 0.5),
        0.5 * 1000.0 * 40.0,
        epsilon = 1e-12
    );
    let m = righting_moment(1025.0, 100.0, 0.2, STANDARD_GRAVITY);
    assert_abs_diff_eq!(m, 1025.0 * STANDARD_GRAVITY * 100.0 * 0.2, epsilon = 1e-9);
    let _ = roll_period_earth(1.0, 0.5);
    let _ = SphereHull::new(1.0);
}

#[test]
fn imo_is_code_pass_and_fail() {
    let n = 81;
    let mut phis = Vec::new();
    let mut gz_good = Vec::new();
    for i in 0..n {
        let phi = 50.0_f64.to_radians() * (i as f64) / ((n - 1) as f64);
        phis.push(phi);
        gz_good.push(0.45 * libm_ops::sin(phi));
    }
    assert!(check_imo_is_code(&phis, &gz_good, 0.5).passed);

    let gz_bad: Vec<f64> = phis.iter().map(|&p| 0.05 * libm_ops::sin(p)).collect();
    assert!(!check_imo_is_code(&phis, &gz_bad, 0.05).passed);
}
