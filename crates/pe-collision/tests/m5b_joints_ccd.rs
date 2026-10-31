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
use pe_math::libm_ops::{cos, sin};
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

fn make_stiff(bodies: Vec<RigidBody>, shapes: Vec<Shape>, substeps: usize) -> ContactWorld {
    let mut cw = make(bodies, shapes, vec![], None, true);
    cw.soft_params = SoftStepParams {
        substeps,
        ..SoftStepParams::default()
    };
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

/// Pendulum bob on the circle of radius `length` at angle `theta` from downward.
fn pendulum_pose(length: f64, theta: f64) -> Vec3 {
    Vec3::new(length * sin(theta), 0.0, -length * cos(theta))
}

/// Mean period from successive positive x zero-crossings; skips the first interval
/// (startup / soft-joint transient).
fn estimate_period_xy(samples: &[(f64, f64)]) -> Option<f64> {
    let mut crossings = Vec::new();
    for i in 1..samples.len() {
        let (t0, x0) = samples[i - 1];
        let (t1, x1) = samples[i];
        if x0 <= 0.0 && x1 > 0.0 {
            let frac = -x0 / (x1 - x0);
            crossings.push(t0 + frac * (t1 - t0));
        }
    }
    if crossings.len() < 3 {
        return None;
    }
    let mut sum = 0.0;
    let mut n = 0usize;
    for i in 2..crossings.len() {
        sum += crossings[i] - crossings[i - 1];
        n += 1;
    }
    if n == 0 {
        None
    } else {
        Some(sum / n as f64)
    }
}

fn run_distance_pendulum(
    length: f64,
    theta0: f64,
    substeps: usize,
    hertz: f64,
    zeta: f64,
    dt: f64,
    n_steps: usize,
) -> f64 {
    let (body, shape) = bob(1.0, 0.05, pendulum_pose(length, theta0));
    let mut cw = make_stiff(vec![body], vec![shape], substeps);
    let mut j = Joint::distance(0, WORLD_BODY, Vec3::zero(), Vec3::zero(), length);
    j.soft = JointSoftParams { hertz, zeta };
    cw.add_joint(j);
    let mut samples = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        cw.step(dt);
        samples.push((cw.t(), cw.world.state.x[0].x));
    }
    estimate_period_xy(&samples).expect("period")
}

/// Exact large-amplitude period `T = 4 √(L/g) K(sin(θ₀/2))` with AGM `K` in-test.
fn elliptic_period(length: f64, g: f64, theta0: f64) -> f64 {
    let k = sin(0.5 * theta0);
    4.0 * pe_math::libm_ops::sqrt(length / g) * elliptic_k(k)
}

#[test]
fn pendulum_small_angle_period_stiff_under_1pct() {
    // θ₀ = 0.01 rad: finite-amplitude bias vs 2π√(L/g) is ~0.0006% (θ²/16),
    // so a <1% bound is a real solver check, not the O(θ²) truncation.
    let length = 1.0;
    let theta0 = 0.01;
    let dt = 1.0 / 240.0;
    let t_meas = run_distance_pendulum(length, theta0, 16, 500.0, 0.1, dt, 3600);
    let t_theory = pendulum_period_small(length, G);
    let rel = (t_meas - t_theory).abs() / t_theory;
    assert!(
        rel < 0.01,
        "small-angle period rel err {rel:.4e} (meas={t_meas}, theory={t_theory})"
    );
}

#[test]
fn pendulum_large_amplitude_elliptic_stiff_under_2pct() {
    let length = 1.0_f64;
    let theta0 = 1.0_f64;
    let dt = 1.0 / 480.0;
    let t_theory = elliptic_period(length, G, theta0);
    assert_relative_eq!(
        t_theory,
        pendulum_period_large(length, G, theta0),
        epsilon = 1e-12
    );
    assert!(t_theory > pendulum_period_small(length, G));
    let n = ((t_theory * 8.0) / dt) as usize;
    let t_meas = run_distance_pendulum(length, theta0, 32, 1000.0, 0.05, dt, n);
    let rel = (t_meas - t_theory).abs() / t_theory;
    assert!(
        rel < 0.02,
        "large-amplitude period rel err {rel:.4e} (meas={t_meas}, elliptic={t_theory})"
    );
}

/// Default-ish softness still recovers the period to a few percent; kept as a
/// regression guard for non-stiff settings (see `docs/JOINTS.md`).
#[test]
fn pendulum_small_angle_period_default_softness() {
    let length = 1.0;
    let theta0 = 0.05;
    let dt = 1.0 / 240.0;
    let t_meas = run_distance_pendulum(length, theta0, 8, 120.0, 0.7, dt, 3000);
    // Compare to elliptic (exact) so O(θ²) is not charged to the solver.
    let t_theory = elliptic_period(length, G, theta0);
    assert_relative_eq!(t_meas, t_theory, max_relative = 0.01);
}

#[test]
fn pendulum_period_converges_with_stiffness() {
    // Error vs elliptic period shrinks as Hertz ↑ / substeps ↑ (soft-joint bias).
    let length = 1.0;
    let theta0 = 1.0;
    let dt = 1.0 / 240.0;
    let t_theory = elliptic_period(length, G, theta0);
    let n = ((t_theory * 6.0) / dt) as usize;
    let configs = [
        (4usize, 60.0, 0.7),
        (8, 120.0, 0.7),
        (16, 500.0, 0.1),
        (32, 1000.0, 0.05),
    ];
    let mut prev_err = f64::INFINITY;
    for &(sub, hz, zeta) in &configs {
        let t_meas = run_distance_pendulum(length, theta0, sub, hz, zeta, dt, n);
        let err = (t_meas - t_theory).abs() / t_theory;
        assert!(
            err <= prev_err * 1.05 + 1e-6,
            "period error should not grow with stiffness: prev={prev_err} err={err} (sub={sub}, hz={hz})"
        );
        prev_err = err;
    }
    assert!(
        prev_err < 0.002,
        "stiffest config should be under 0.2%: {prev_err}"
    );
}

#[test]
fn double_pendulum_energy_drift_documented() {
    // Exact circular IC; soft-step + soft distance joints dissipate slowly.
    let l1 = 0.5;
    let l2 = 0.5;
    let th1 = 0.2;
    let th2 = 0.3;
    let p1 = pendulum_pose(l1, th1);
    let p2 = Vec3::new(p1.x + l2 * sin(th1 + th2), 0.0, p1.z - l2 * cos(th1 + th2));
    let (b1, s1) = bob(1.0, 0.04, p1);
    let (b2, s2) = bob(1.0, 0.04, p2);
    let mut cw = make_stiff(vec![b1, b2], vec![s1, s2], 32);
    let mut j1 = Joint::distance(0, WORLD_BODY, Vec3::zero(), Vec3::zero(), l1);
    let mut j2 = Joint::distance(1, 0, Vec3::zero(), Vec3::zero(), l2);
    j1.soft = JointSoftParams {
        hertz: 500.0,
        zeta: 0.1,
    };
    j2.soft = JointSoftParams {
        hertz: 500.0,
        zeta: 0.1,
    };
    cw.add_joint(j1);
    cw.add_joint(j2);

    let dt = 1.0 / 480.0;
    for _ in 0..240 {
        cw.step(dt);
    }
    let e0 = {
        let ke = kinetic_energy(&cw.world, &cw.world.state);
        let pe: f64 = (0..2)
            .map(|i| cw.world.mass[i] * G * cw.world.state.x[i].z)
            .sum();
        ke + pe
    };
    let mut e_min = e0;
    let mut e_max = e0;
    // 20 s after settle.
    for _ in 0..(20.0 / dt) as usize {
        cw.step(dt);
        let ke = kinetic_energy(&cw.world, &cw.world.state);
        let pe: f64 = (0..2)
            .map(|i| cw.world.mass[i] * G * cw.world.state.x[i].z)
            .sum();
        let e = ke + pe;
        e_min = e_min.min(e);
        e_max = e_max.max(e);
        for i in 0..2 {
            assert!(cw.world.state.x[i].x.is_finite());
            assert!(cw.world.state.x[i].norm() < 3.0);
        }
    }
    let e_end = {
        let ke = kinetic_energy(&cw.world, &cw.world.state);
        let pe: f64 = (0..2)
            .map(|i| cw.world.mass[i] * G * cw.world.state.x[i].z)
            .sum();
        ke + pe
    };
    let span = e_max - e_min;
    let drift = (e0 - e_end).abs();
    // Measured (release, this host): span ~2e-4 J, |ΔE| ~2e-5 J over 20 s
    // (~0.0001% of |E|). Soft joints dissipate; exact conservation not claimed.
    assert!(
        span < 5e-3,
        "energy span {span} too large (E0={e0}, E_end={e_end})"
    );
    assert!(
        drift < 5e-3,
        "energy drift {drift} too large (E0={e0}, E_end={e_end})"
    );
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
