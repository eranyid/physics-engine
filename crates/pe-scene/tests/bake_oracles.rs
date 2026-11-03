//! M24 bake oracles: closed-form free fall / projectile / slerp / pendulum,
//! keyframe reduction bound, bit-identical re-bake, coincident-time resampling.

use pe_math::{libm_ops, Quat, Vec3, STANDARD_GRAVITY};
use pe_scene::{
    bake_json_hash64, bake_scene, canonical_bake_document, canonical_bake_json_hash,
    canonical_bake_scene, BakeOptions, BakeReduceParams, BodyDesc, ForceDesc, JointDesc,
    JointKindDesc, MaterialDesc, SceneDocument, ShapeDesc, SolverSettings, BAKE_SCHEMA_VERSION,
    CANONICAL_BAKE_JSON_HASH,
};

const G: f64 = STANDARD_GRAVITY;

/// Symplectic-Euler global position error bound for constant acceleration:
/// `|x_n − x_exact| = ½ |a| t Δt` (see docs/BAKE.md).
fn symplectic_euler_pos_bound(accel_mag: f64, t: f64, dt: f64) -> f64 {
    0.5 * accel_mag * t * dt
}

fn free_fall_scene(z0: f64, v0: [f64; 3]) -> SceneDocument {
    SceneDocument {
        schema_version: pe_scene::SCENE_SCHEMA_VERSION,
        bodies: vec![BodyDesc {
            mass: 1.0,
            inertia_body: [[0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]],
            position: [0.0, 0.0, z0],
            velocity: v0,
            orientation: [1.0, 0.0, 0.0, 0.0],
            angular_velocity: [0.0, 0.0, 0.0],
            name: "body".into(),
            bullet: false,
        }],
        shapes: vec![ShapeDesc::Sphere { radius: 0.05 }],
        materials: vec![MaterialDesc::default()],
        statics: Vec::new(),
        joints: Vec::new(),
        forces: vec![ForceDesc::UniformGravity { g: [0.0, 0.0, -G] }],
        solver: SolverSettings {
            allow_sleep: false,
            ..SolverSettings::default()
        },
        state: None,
        runtime: None,
    }
}

#[test]
fn free_fall_and_projectile_vs_analytic_at_24fps() {
    let z0 = 10.0;
    let v0 = [5.0, 0.0, 2.0];
    let scene = free_fall_scene(z0, v0);
    let dt = 1.0 / 240.0;
    let duration = 1.0;
    let opt = BakeOptions {
        duration,
        physics_dt: dt,
        sample_fps: 24.0,
        include_velocities: true,
        reduce: None,
    };
    let doc = bake_scene(&scene, &opt).unwrap();
    assert_eq!(doc.schema_version, BAKE_SCHEMA_VERSION);
    let body = &doc.bodies[0];
    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    for (i, &t) in body.times.iter().enumerate() {
        let p = body.position[i];
        let v = body.linear_velocity.as_ref().unwrap()[i];
        let x_a = v0[0] * t;
        let z_a = z0 + v0[2] * t - 0.5 * G * t * t;
        let vx_a = v0[0];
        let vz_a = v0[2] - G * t;
        let err = libm_ops::sqrt((p[0] - x_a) * (p[0] - x_a) + (p[2] - z_a) * (p[2] - z_a));
        let verr = libm_ops::sqrt((v[0] - vx_a) * (v[0] - vx_a) + (v[2] - vz_a) * (v[2] - vz_a));
        if err > max_pos_err {
            max_pos_err = err;
        }
        if verr > max_vel_err {
            max_vel_err = verr;
        }
        // Position: symplectic Euler truncation ½ g T Δt; +1 ULP margin.
        let tol = symplectic_euler_pos_bound(G, duration, dt) + 1e-12;
        assert!(
            err <= tol,
            "t={t}: pos err {err:.6e} > tol {tol:.6e} (½ g T Δt)"
        );
        // Velocity is exact for constant accel under symplectic Euler (v_n = v0 + n a Δt).
        // Mid-interval lerp of v introduces ≤ ½ |a| Δt vs continuous; samples lie on
        // output grid so worst-case ≤ ½ g Δt.
        let vtol = 0.5 * G * dt + 1e-12;
        assert!(verr <= vtol, "t={t}: vel err {verr:.6e} > tol {vtol:.6e}");
    }
    // Recorded for the PR measured-vs-tolerance table.
    eprintln!(
        "free_fall/projectile @24fps: max_pos_err={max_pos_err:.6e} tol={:.6e}; max_vel_err={max_vel_err:.6e}",
        symplectic_euler_pos_bound(G, duration, dt)
    );
}

#[test]
fn uniform_rotation_slerp_vs_analytic() {
    // Torque-free spin about principal +z; no gravity.
    let omega = 2.0; // rad/s
    let scene = SceneDocument {
        schema_version: pe_scene::SCENE_SCHEMA_VERSION,
        bodies: vec![BodyDesc {
            mass: 1.0,
            inertia_body: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            orientation: [1.0, 0.0, 0.0, 0.0],
            angular_velocity: [0.0, 0.0, omega],
            name: "spinner".into(),
            bullet: false,
        }],
        shapes: vec![ShapeDesc::Sphere { radius: 0.1 }],
        materials: vec![MaterialDesc::default()],
        statics: Vec::new(),
        joints: Vec::new(),
        forces: Vec::new(),
        solver: SolverSettings {
            allow_sleep: false,
            ..SolverSettings::default()
        },
        state: None,
        runtime: None,
    };
    let dt = 1.0 / 240.0;
    let opt = BakeOptions {
        duration: 1.0,
        physics_dt: dt,
        sample_fps: 24.0,
        include_velocities: true,
        reduce: None,
    };
    let doc = bake_scene(&scene, &opt).unwrap();
    let body = &doc.bodies[0];
    let mut max_ang = 0.0_f64;
    for (i, &t) in body.times.iter().enumerate() {
        let q = Quat::from_slice(&body.rotation[i]);
        let analytic = Quat::from_axis_angle(Vec3::unit_z(), omega * t);
        let ang = q.angle_between(analytic);
        if ang > max_ang {
            max_ang = ang;
        }
        // Implicit-midpoint exp map on principal-axis spin is near machine accurate;
        // allow a small integrator + slerp budget (measured ≪ 1e-10).
        assert!(ang < 1e-10, "t={t}: rotation err {ang:.3e} rad (tol 1e-10)");
    }
    eprintln!("uniform rotation @24fps: max angle err={max_ang:.6e} rad (tol 1e-10)");
}

#[test]
fn pendulum_small_angle_period_from_bake() {
    let length = 1.0;
    let theta0 = 0.01;
    let x0 = length * libm_ops::sin(theta0);
    let z0 = -length * libm_ops::cos(theta0);
    let scene = SceneDocument {
        schema_version: pe_scene::SCENE_SCHEMA_VERSION,
        bodies: vec![BodyDesc {
            mass: 1.0,
            inertia_body: [[0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]],
            position: [x0, 0.0, z0],
            velocity: [0.0, 0.0, 0.0],
            orientation: [1.0, 0.0, 0.0, 0.0],
            angular_velocity: [0.0, 0.0, 0.0],
            name: "bob".into(),
            bullet: false,
        }],
        shapes: vec![ShapeDesc::Sphere { radius: 0.05 }],
        materials: vec![MaterialDesc::default()],
        statics: Vec::new(),
        joints: vec![JointDesc {
            body_a: 0,
            body_b: u64::MAX, // WORLD
            local_anchor_a: [0.0, 0.0, 0.0],
            local_anchor_b: [0.0, 0.0, 0.0],
            kind: JointKindDesc::Distance { length },
            soft_hertz: 500.0,
            soft_zeta: 0.1,
            motor_enabled: false,
            motor_target_speed: 0.0,
            motor_max_force: 0.0,
            limit_enabled: false,
            limit_lower: 0.0,
            limit_upper: 0.0,
            break_force: 0.0,
            break_torque: 0.0,
            broken: false,
            warm_bits: Vec::new(),
        }],
        forces: vec![ForceDesc::UniformGravity { g: [0.0, 0.0, -G] }],
        solver: SolverSettings {
            mode: "soft_step".into(),
            soft_substeps: 16,
            allow_sleep: false,
            ..SolverSettings::default()
        },
        state: None,
        runtime: None,
    };
    let dt = 1.0 / 240.0;
    let t_theory = 2.0 * core::f64::consts::PI * libm_ops::sqrt(length / G);
    let duration = t_theory * 4.0;
    let opt = BakeOptions {
        duration,
        physics_dt: dt,
        sample_fps: 60.0,
        include_velocities: false,
        reduce: None,
    };
    let doc = bake_scene(&scene, &opt).unwrap();
    let body = &doc.bodies[0];
    let mut crossings = Vec::new();
    for i in 1..body.times.len() {
        let (t0, x0) = (body.times[i - 1], body.position[i - 1][0]);
        let (t1, x1) = (body.times[i], body.position[i][0]);
        if x0 <= 0.0 && x1 > 0.0 {
            let frac = -x0 / (x1 - x0);
            crossings.push(t0 + frac * (t1 - t0));
        }
    }
    assert!(
        crossings.len() >= 3,
        "need ≥3 zero-crossings, got {}",
        crossings.len()
    );
    let mut sum = 0.0;
    let mut n = 0usize;
    for i in 2..crossings.len() {
        sum += crossings[i] - crossings[i - 1];
        n += 1;
    }
    let t_meas = sum / n as f64;
    let rel = (t_meas - t_theory).abs() / t_theory;
    // Matches pe-collision stiff distance-pendulum oracle (<1%).
    assert!(
        rel < 0.01,
        "pendulum period rel err {rel:.4e} (meas={t_meas}, theory={t_theory})"
    );
    eprintln!("pendulum small-angle: rel err={rel:.6e} (tol 1e-2)");
}

#[test]
fn keyframe_reduction_error_bound_holds() {
    let scene = free_fall_scene(10.0, [2.0, 0.0, 1.0]);
    let max_pos = 1e-3;
    let max_rot = 1e-4;
    let opt_dense = BakeOptions {
        duration: 1.0,
        physics_dt: 1.0 / 240.0,
        sample_fps: 60.0,
        include_velocities: true,
        reduce: None,
    };
    let dense = bake_scene(&scene, &opt_dense).unwrap();
    let opt_red = BakeOptions {
        reduce: Some(BakeReduceParams {
            max_position_error: max_pos,
            max_rotation_error: max_rot,
        }),
        ..opt_dense
    };
    let reduced = bake_scene(&scene, &opt_red).unwrap();
    assert!(
        reduced.bodies[0].times.len() < dense.bodies[0].times.len(),
        "reduction should drop keys ({} vs {})",
        reduced.bodies[0].times.len(),
        dense.bodies[0].times.len()
    );
    let mut worst_p = 0.0_f64;
    let mut worst_r = 0.0_f64;
    for (i, &t) in dense.bodies[0].times.iter().enumerate() {
        let (p, q) = reduced.sample_body_at(0, t).unwrap();
        let pd = Vec3::from_slice(&dense.bodies[0].position[i]);
        let qd = Quat::from_slice(&dense.bodies[0].rotation[i]);
        let dp = (p - pd).norm();
        let dr = q.angle_between(qd);
        if dp > worst_p {
            worst_p = dp;
        }
        if dr > worst_r {
            worst_r = dr;
        }
        assert!(
            dp <= max_pos + 1e-15,
            "pos err {dp} at t={t} exceeds bound {max_pos}"
        );
        assert!(
            dr <= max_rot + 1e-15,
            "rot err {dr} at t={t} exceeds bound {max_rot}"
        );
    }
    eprintln!(
        "keyframe reduction: worst_pos={worst_p:.6e} (bound {max_pos}); worst_rot={worst_r:.6e} (bound {max_rot})"
    );
}

#[test]
fn bake_bit_identical_and_json_hash_stable() {
    let a = canonical_bake_document().unwrap();
    let b = canonical_bake_document().unwrap();
    assert_eq!(a, b);
    let ja = a.to_json().unwrap();
    let jb = b.to_json().unwrap();
    assert_eq!(ja, jb);
    let ha = bake_json_hash64(&a).unwrap();
    let hb = bake_json_hash64(&b).unwrap();
    assert_eq!(ha, hb);
    assert_eq!(ha, canonical_bake_json_hash());
    assert_eq!(
        ha, CANONICAL_BAKE_JSON_HASH,
        "bake JSON hash drifted: got {ha:016x}"
    );
}

#[test]
fn coincident_times_across_fps_agree_exactly() {
    let scene = canonical_bake_scene();
    let base = BakeOptions {
        duration: 1.0,
        physics_dt: 1.0 / 240.0,
        sample_fps: 24.0,
        include_velocities: true,
        reduce: None,
    };
    let d24 = bake_scene(&scene, &base).unwrap();
    let d48 = bake_scene(
        &scene,
        &BakeOptions {
            sample_fps: 48.0,
            ..base.clone()
        },
    )
    .unwrap();
    let d30 = bake_scene(
        &scene,
        &BakeOptions {
            sample_fps: 30.0,
            ..base
        },
    )
    .unwrap();

    // Every 24 fps stamp must appear in 48 fps and match bit-exactly.
    for (i, &t) in d24.bodies[0].times.iter().enumerate() {
        let j = d48.bodies[0]
            .times
            .iter()
            .position(|&u| u == t)
            .unwrap_or_else(|| panic!("t={t} from 24fps missing in 48fps"));
        assert_eq!(
            d24.bodies[0].position[i], d48.bodies[0].position[j],
            "position mismatch at coincident t={t}"
        );
        assert_eq!(
            d24.bodies[0].rotation[i], d48.bodies[0].rotation[j],
            "rotation mismatch at coincident t={t}"
        );
        assert_eq!(
            d24.bodies[0].linear_velocity.as_ref().unwrap()[i],
            d48.bodies[0].linear_velocity.as_ref().unwrap()[j],
        );
    }

    // t=0 and t=1.0 must match across 24/30/48.
    for doc in [&d24, &d30, &d48] {
        assert_eq!(doc.bodies[0].times[0], 0.0);
        assert_eq!(*doc.bodies[0].times.last().unwrap(), 1.0);
    }
    assert_eq!(d24.bodies[0].position[0], d30.bodies[0].position[0]);
    assert_eq!(d24.bodies[0].position[0], d48.bodies[0].position[0]);
    let i24 = d24.bodies[0].times.len() - 1;
    let i30 = d30.bodies[0].times.len() - 1;
    let i48 = d48.bodies[0].times.len() - 1;
    assert_eq!(d24.bodies[0].position[i24], d30.bodies[0].position[i30]);
    assert_eq!(d24.bodies[0].position[i24], d48.bodies[0].position[i48]);
}

#[test]
fn bake_json_roundtrip_preserves_schema() {
    // Decimal JSON may lose ULPs on re-parse (same caveat as scene mass/inertia
    // fields in SCENE.md). Schema, lengths, and times must round-trip; positions
    // stay within a few ULPs of the producer values.
    let doc = canonical_bake_document().unwrap();
    let json = doc.to_json().unwrap();
    let back = pe_scene::BakeDocument::from_json(&json).unwrap();
    assert_eq!(back.schema_version, doc.schema_version);
    assert_eq!(back.bodies.len(), doc.bodies.len());
    assert_eq!(back.bodies[0].times.len(), doc.bodies[0].times.len());
    assert_eq!(back.bodies[0].position.len(), doc.bodies[0].position.len());
    let mut max_ulp_dist = 0.0_f64;
    for (ta, tb) in doc.bodies[0].times.iter().zip(back.bodies[0].times.iter()) {
        assert!((ta - tb).abs() <= 1e-15, "JSON roundtrip time drift");
    }
    for (a, b) in doc.bodies[0]
        .position
        .iter()
        .zip(back.bodies[0].position.iter())
    {
        for k in 0..3 {
            let d = (a[k] - b[k]).abs();
            if d > max_ulp_dist {
                max_ulp_dist = d;
            }
            // Decimal JSON can lose ~1 ULP on values ~O(10).
            assert!(d <= 1e-14, "JSON roundtrip position drift {d}");
        }
    }
    eprintln!("bake JSON roundtrip max |Δpos|={max_ulp_dist:.3e}");
}
