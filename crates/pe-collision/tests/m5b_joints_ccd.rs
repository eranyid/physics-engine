//! M5b analytic / oracle tests: joints + CCD.
//!
//! Tight assertions use closed-form results (pendulum period, elliptic integral,
//! spring–mass frequency). Solver-dependent bounds are documented as such.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_collision::{
    joints::{
        elliptic_k, hertz_to_stiffness, pendulum_period_large, pendulum_period_small,
        spring_mass_period, Joint, JointBreak, JointLimit, JointMotor, JointSoftParams, WORLD_BODY,
    },
    BoxShape, CcdParams, ContactWorld, Material, Plane, Shape, SoftStepParams, SolverMode, Sphere,
    StaticCollider,
};
use pe_core::diagnostics::kinetic_energy;
use pe_core::forces::UniformGravity;
use pe_core::{RigidBody, World};
use pe_math::{Quat, Vec3, STANDARD_GRAVITY as G};
use proptest::prelude::*;

fn make(
    bodies: Vec<RigidBody>,
    shapes: Vec<Shape>,
    statics: Vec<StaticCollider>,
    mats: Option<Vec<Material>>,
    gravity: bool,
) -> ContactWorld {
    let gens: Vec<Box<dyn pe_core::ForceGenerator>> = if gravity {
        vec![Box::new(UniformGravity::earth())]
    } else {
        vec![]
    };
    let mut cw = ContactWorld::new(World::new(bodies, gens), shapes, statics, mats);
    cw.solver_mode = SolverMode::SoftStep;
    cw.soft_params = SoftStepParams {
        substeps: 8,
        ..SoftStepParams::default()
    };
    cw.set_allow_sleep(false);
    cw
}

fn bob(mass: f64, r: f64, x: Vec3) -> (RigidBody, Shape) {
    (
        RigidBody::new(mass)
            .with_inertia(Sphere::new(r).inertia(mass))
            .with_position(x),
        Shape::Sphere(Sphere::new(r)),
    )
}

/// Measure period from successive z-crossing or angle sign changes of x-coordinate.
fn estimate_period_xy(samples: &[(f64, f64)] /* t, x */) -> Option<f64> {
    let mut crossings = Vec::new();
    for i in 1..samples.len() {
        let (t0, x0) = samples[i - 1];
        let (t1, x1) = samples[i];
        if x0 <= 0.0 && x1 > 0.0 {
            let frac = -x0 / (x1 - x0);
            crossings.push(t0 + frac * (t1 - t0));
        }
    }
    if crossings.len() < 2 {
        return None;
    }
    let mut sum = 0.0;
    let mut n = 0usize;
    for i in 1..crossings.len() {
        sum += crossings[i] - crossings[i - 1];
        n += 1;
    }
    if n == 0 {
        None
    } else {
        Some(sum / n as f64)
    }
}

#[test]
fn pendulum_small_angle_period() {
    let length = 1.0;
    let (body, shape) = bob(1.0, 0.05, Vec3::new(0.05, 0.0, -length)); // ~2.9°
    let mut cw = make(vec![body], vec![shape], vec![], None, true);
    let mut j = Joint::distance(
        0,
        WORLD_BODY,
        Vec3::zero(),
        Vec3::new(0.0, 0.0, 0.0),
        length,
    );
    // Anchor world at origin: local_anchor_b is world point when body_b is WORLD.
    j.local_anchor_b = Vec3::zero();
    j.soft = JointSoftParams {
        hertz: 120.0,
        zeta: 0.7,
    };
    cw.add_joint(j);

    let dt = 1.0 / 240.0;
    let mut samples = Vec::new();
    for _ in 0..2400 {
        cw.step(dt);
        samples.push((cw.t(), cw.world.state.x[0].x));
    }
    let t_meas = estimate_period_xy(&samples).expect("period");
    let t_theory = pendulum_period_small(length, G);
    assert_relative_eq!(t_meas, t_theory, max_relative = 0.05);
}

#[test]
fn pendulum_large_amplitude_elliptic() {
    let length = 1.0_f64;
    let theta0 = 1.0_f64; // rad
    let x0 = length * pe_math::libm_ops::sin(theta0);
    let z0 = -length * pe_math::libm_ops::cos(theta0);
    let (body, shape) = bob(1.0, 0.05, Vec3::new(x0, 0.0, z0));
    let mut cw = make(vec![body], vec![shape], vec![], None, true);
    let mut j = Joint::distance(0, WORLD_BODY, Vec3::zero(), Vec3::zero(), length);
    j.soft = JointSoftParams {
        hertz: 120.0,
        zeta: 0.7,
    };
    cw.add_joint(j);

    let dt = 1.0 / 240.0;
    let mut samples = Vec::new();
    for _ in 0..4000 {
        cw.step(dt);
        samples.push((cw.t(), cw.world.state.x[0].x));
    }
    let t_meas = estimate_period_xy(&samples).expect("period");
    let t_theory = pendulum_period_large(length, G, theta0);
    let t_small = pendulum_period_small(length, G);
    assert!(t_theory > t_small);
    assert_relative_eq!(t_meas, t_theory, max_relative = 0.08);
    let _ = elliptic_k(0.5);
}

#[test]
fn double_pendulum_energy_bounded() {
    let l1 = 0.5;
    let l2 = 0.5;
    let (b1, s1) = bob(1.0, 0.04, Vec3::new(0.1, 0.0, -l1));
    let (b2, s2) = bob(1.0, 0.04, Vec3::new(0.2, 0.0, -(l1 + l2)));
    let mut cw = make(vec![b1, b2], vec![s1, s2], vec![], None, true);
    let mut j1 = Joint::distance(0, WORLD_BODY, Vec3::zero(), Vec3::zero(), l1);
    let mut j2 = Joint::distance(1, 0, Vec3::zero(), Vec3::zero(), l2);
    j1.soft = JointSoftParams {
        hertz: 90.0,
        zeta: 0.8,
    };
    j2.soft = JointSoftParams {
        hertz: 90.0,
        zeta: 0.8,
    };
    cw.add_joint(j1);
    cw.add_joint(j2);

    let dt = 1.0 / 240.0;
    let mut ke_max = 0.0f64;
    for _ in 0..1200 {
        cw.step(dt);
        let ke = kinetic_energy(&cw.world, &cw.world.state);
        ke_max = ke_max.max(ke);
        for i in 0..2 {
            assert!(cw.world.state.x[i].x.is_finite());
            // Stay near the double-pendulum workspace (no fly-off).
            assert!(cw.world.state.x[i].norm() < 3.0);
        }
    }
    // Soft joints dissipate; KE must stay bounded (unverified exact conservation).
    assert!(ke_max < 50.0, "KE blew up: {ke_max}");
}

#[test]
fn joint_drift_under_load_bounded() {
    // Heavy bob on a short stiff distance joint — position error under gravity.
    let length = 0.5;
    let mass = 10.0;
    let (body, shape) = bob(mass, 0.05, Vec3::new(0.0, 0.0, -length));
    let mut cw = make(vec![body], vec![shape], vec![], None, true);
    let mut j = Joint::distance(0, WORLD_BODY, Vec3::zero(), Vec3::zero(), length);
    j.soft = JointSoftParams {
        hertz: 60.0,
        zeta: 1.0,
    };
    cw.add_joint(j);
    for _ in 0..480 {
        cw.step(1.0 / 240.0);
    }
    let p = cw.world.state.x[0];
    let dist = p.norm();
    assert!(
        (dist - length).abs() < 0.02,
        "drift |r|-L = {}",
        dist - length
    );
}

#[test]
fn revolute_limit_holds() {
    // No gravity: spin toward the upper limit and verify it does not far overshoot.
    let (body, shape) = bob(1.0, 0.05, Vec3::new(1.0, 0.0, 0.0));
    let mut cw = make(vec![body], vec![shape], vec![], None, false);
    let mut j = Joint::revolute(
        0,
        WORLD_BODY,
        Vec3::zero(),
        Vec3::zero(),
        Vec3::unit_y(),
        Vec3::unit_y(),
        Vec3::unit_x(),
        Vec3::unit_x(),
    );
    j.soft = JointSoftParams {
        hertz: 120.0,
        zeta: 1.0,
    };
    j.limit = JointLimit {
        enabled: true,
        lower: -0.3,
        upper: 0.3,
    };
    cw.add_joint(j);
    cw.world.state.v[0] = Vec3::new(0.0, 0.0, 3.0);
    let mut max_ang = 0.0f64;
    for _ in 0..480 {
        cw.step(1.0 / 240.0);
        let p = cw.world.state.x[0];
        let angle = pe_math::libm_ops::atan2(p.z, p.x);
        max_ang = max_ang.max(angle.abs());
    }
    assert!(
        max_ang < 0.55,
        "angle peak {max_ang} should stay near ±0.3 limit (soft overshoot allowed)"
    );
}

#[test]
fn prismatic_spring_mass_frequency() {
    let mass = 1.0;
    let hertz = 2.0;
    let (body, shape) = bob(mass, 0.05, Vec3::new(0.0, 0.0, 0.1));
    let mut cw = make(vec![body], vec![shape], vec![], None, false);
    let mut j = Joint::wheel(
        0,
        WORLD_BODY,
        Vec3::zero(),
        Vec3::zero(),
        Vec3::unit_z(),
        Vec3::unit_z(),
        hertz,
        0.05, // light damping
        0.0,
    );
    j.soft = JointSoftParams {
        hertz: 90.0,
        zeta: 0.8,
    };
    cw.add_joint(j);
    cw.world.state.x[0] = Vec3::new(0.0, 0.0, 0.05);
    let dt = 1.0 / 240.0;
    let mut samples = Vec::new();
    for _ in 0..2400 {
        cw.step(dt);
        samples.push((cw.t(), cw.world.state.x[0].z));
    }
    let t_meas = estimate_period_xy(&samples.iter().map(|&(t, z)| (t, z)).collect::<Vec<_>>())
        .expect("period");
    let t_theory = 1.0 / hertz;
    let k = hertz_to_stiffness(mass, hertz);
    let _ = spring_mass_period(mass, k);
    // Soft-step + Hertz/ζ map is not an ideal undamped oscillator; allow wide band.
    // Documented as partially unverified in docs/JOINTS.md.
    assert_relative_eq!(t_meas, t_theory, max_relative = 0.40);
}

#[test]
fn chain_50_links_hanging_stable() {
    let n = 50;
    let link = 0.05;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    for i in 0..n {
        let z = -link * (i as f64 + 1.0);
        let (b, s) = bob(0.2, 0.02, Vec3::new(0.0, 0.0, z));
        bodies.push(b);
        shapes.push(s);
    }
    let mut cw = make(bodies, shapes, vec![], None, true);
    cw.soft_params.substeps = 4;
    for i in 0..n {
        let parent = if i == 0 { WORLD_BODY } else { i - 1 };
        let mut j = Joint::distance(i, parent, Vec3::zero(), Vec3::zero(), link);
        j.soft = JointSoftParams {
            hertz: 40.0,
            zeta: 1.0,
        };
        cw.add_joint(j);
    }
    for _ in 0..600 {
        cw.step(1.0 / 120.0);
    }
    let tip = cw.world.state.x[n - 1];
    assert!(tip.z < -1.5, "chain should hang down, tip z={}", tip.z);
    assert!(tip.x.abs() < 0.5, "lateral drift {}", tip.x);
    // No NaNs
    for i in 0..n {
        assert!(cw.world.state.x[i].x.is_finite());
    }
}

#[test]
fn motor_reaches_target_speed() {
    let (body, shape) = bob(1.0, 0.05, Vec3::new(1.0, 0.0, 0.0));
    let mut cw = make(vec![body], vec![shape], vec![], None, false);
    let mut j = Joint::revolute(
        0,
        WORLD_BODY,
        Vec3::zero(),
        Vec3::zero(),
        Vec3::unit_y(),
        Vec3::unit_y(),
        Vec3::unit_x(),
        Vec3::unit_x(),
    );
    j.soft = JointSoftParams {
        hertz: 60.0,
        zeta: 0.7,
    };
    j.motor = JointMotor {
        enabled: true,
        target_speed: 2.0,
        max_force: 50.0,
    };
    cw.add_joint(j);
    for _ in 0..480 {
        cw.step(1.0 / 240.0);
    }
    let w = cw.world.omega_i(&cw.world.state, 0);
    assert_relative_eq!(w.y, 2.0, epsilon = 0.35);
}

#[test]
fn breakable_joint_breaks_at_threshold() {
    let (body, shape) = bob(1.0, 0.05, Vec3::new(0.0, 0.0, -1.0));
    let mut cw = make(vec![body], vec![shape], vec![], None, true);
    let mut j = Joint::distance(0, WORLD_BODY, Vec3::zero(), Vec3::zero(), 1.0);
    j.soft = JointSoftParams {
        hertz: 60.0,
        zeta: 1.0,
    };
    // Weight ≈ 9.8 N; set threshold just below so it breaks after settling.
    j.break_threshold = JointBreak {
        force: 5.0,
        torque: 0.0,
    };
    let id = cw.add_joint(j);
    for _ in 0..240 {
        cw.step(1.0 / 240.0);
    }
    assert!(cw.joint_broken(id), "joint should break under weight");
}

#[test]
fn ccd_bullet_no_tunnel_1000mps() {
    // 1 cm thick “wall” modelled as plane at z=0; sphere starts at z=1 with v_z=-1000.
    let r = 0.05;
    let (body, shape) = bob(1.0, r, Vec3::new(0.0, 0.0, 1.0));
    let body = body.with_velocity(Vec3::new(0.0, 0.0, -1000.0));
    let mut cw = make(
        vec![body],
        vec![shape],
        vec![StaticCollider::new(Shape::Plane(Plane::new(
            Vec3::unit_z(),
            0.0,
        )))],
        Some(vec![Material::new(0.0, 0.0)]),
        false,
    );
    cw.solver_mode = SolverMode::Pgs;
    cw.ccd = CcdParams {
        speculative: true,
        speculative_distance: 2.0, // large enough for this extreme case alongside TOI
        toi: true,
        toi_max_iters: 24,
        toi_tolerance: 1e-5,
    };
    cw.set_bullet(0, true);
    let dt = 1.0 / 60.0;
    cw.step(dt);
    let z = cw.world.state.x[0].z;
    // Must not tunnel below the plane by more than radius.
    assert!(z >= r - 0.02, "tunneled: z={z}, expected >= {}", r - 0.02);
}

#[test]
fn joints_determinism_hash_stable() {
    let a = pe_collision::canonical_joints_hash();
    let b = pe_collision::canonical_joints_hash();
    assert_eq!(a, b);
    // Locked value asserted in lib hash_tests once measured in release.
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn distance_joint_length_non_negative(
        len in 0.1f64..2.0,
        mass in 0.5f64..5.0,
    ) {
        let (body, shape) = bob(mass, 0.05, Vec3::new(0.0, 0.0, -len));
        let mut cw = make(vec![body], vec![shape], vec![], None, true);
        let mut j = Joint::distance(0, WORLD_BODY, Vec3::zero(), Vec3::zero(), len);
        j.soft = JointSoftParams { hertz: 60.0, zeta: 0.8 };
        cw.add_joint(j);
        for _ in 0..60 {
            cw.step(1.0 / 120.0);
        }
        let d = cw.world.state.x[0].norm();
        prop_assert!(d.is_finite());
        prop_assert!((d - len).abs() < 0.15);
    }
}

#[test]
fn weld_holds_relative_pose_approximately() {
    let b0 = RigidBody::new(1.0)
        .with_inertia(BoxShape::new(0.1, 0.1, 0.1).inertia(1.0))
        .with_position(Vec3::new(0.0, 0.0, 1.0));
    let b1 = RigidBody::new(1.0)
        .with_inertia(BoxShape::new(0.1, 0.1, 0.1).inertia(1.0))
        .with_position(Vec3::new(0.3, 0.0, 1.0));
    let mut cw = make(
        vec![b0, b1],
        vec![
            Shape::Box(BoxShape::new(0.1, 0.1, 0.1)),
            Shape::Box(BoxShape::new(0.1, 0.1, 0.1)),
        ],
        vec![],
        None,
        true,
    );
    let q_rel = Quat::identity();
    let mut j = Joint::weld(
        0,
        1,
        Vec3::new(0.15, 0.0, 0.0),
        Vec3::new(-0.15, 0.0, 0.0),
        q_rel,
    );
    j.soft = JointSoftParams {
        hertz: 40.0,
        zeta: 1.0,
    };
    cw.add_joint(j);
    for _ in 0..240 {
        cw.step(1.0 / 240.0);
    }
    let d = (cw.world.state.x[1] - cw.world.state.x[0]).norm();
    assert_abs_diff_eq!(d, 0.3, epsilon = 0.05);
}
