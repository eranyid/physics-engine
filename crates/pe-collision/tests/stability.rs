//! Long-run stability suite (M22). Marked `#[ignore]` for default CI.
//!
//! Run with: `cargo test -p pe-collision --test stability -- --ignored --nocapture`
//! Nightly / `workflow_dispatch`: see `.github/workflows/ci.yml` job `rust-stability`.
//! Measured drifts are recorded in `docs/STABILITY.md`.

use pe_collision::{
    BoxShape, ContactWorld, Joint, Material, Plane, Shape, Sphere, StaticCollider, WORLD_BODY,
};
use pe_core::diagnostics::kinetic_energy;
use pe_core::forces::UniformGravity;
use pe_core::{RigidBody, World};
use pe_math::Vec3;

const DT: f64 = 1.0 / 240.0;

fn plane_ground() -> StaticCollider {
    StaticCollider::new(Shape::Plane(Plane::new(Vec3::unit_z(), 0.0)))
        .with_material(Material::new(0.0, 1.0))
}

#[test]
#[ignore = "slow: 1e5-step box stack"]
fn stability_stack_1e5_steps() {
    let h = 0.25;
    let n = 5;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    let mut mats = Vec::new();
    for i in 0..n {
        bodies.push(
            RigidBody::new(1.0)
                .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
                .with_position(Vec3::new(0.0, 0.0, h + 2.0 * h * i as f64)),
        );
        shapes.push(Shape::Box(BoxShape::new(h, h, h)));
        mats.push(Material::new(0.0, 0.5));
    }
    let mut cw = ContactWorld::new(
        World::new(bodies, vec![Box::new(UniformGravity::earth())]),
        shapes,
        vec![plane_ground()],
        Some(mats),
    );
    cw.set_allow_sleep(true);
    for _ in 0..100_000 {
        cw.step(DT);
        assert!(cw.world.state.all_finite());
    }
    let top_z = cw.world.state.x[n - 1].z;
    let drift_xy = (0..n)
        .map(|i| (cw.world.state.x[i].x.powi(2) + cw.world.state.x[i].y.powi(2)).sqrt())
        .fold(0.0_f64, f64::max);
    let ke = kinetic_energy(&cw.world, &cw.world.state);
    eprintln!("stability_stack_1e5: top_z={top_z:.6} max_xy_drift={drift_xy:.6e} ke={ke:.6e}");
    // Bound: COM xy drift < 5 mm; top still above ground; KE < 1 J.
    assert!(drift_xy < 5e-3, "xy drift {drift_xy}");
    assert!(top_z > 0.0);
    assert!(ke < 1.0, "KE {ke}");
}

#[test]
#[ignore = "slow: 1e5-step pyramid"]
fn stability_pyramid_1e5_steps() {
    let h = 0.25;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    let mut mats = Vec::new();
    let mut placed = 0usize;
    let mut row = 0usize;
    while placed < 10 {
        let n = row + 1;
        let x0 = -(n as f64 - 1.0) * h;
        for i in 0..n {
            if placed >= 10 {
                break;
            }
            let x = x0 + 2.0 * h * (i as f64);
            let z = h + 2.0 * h * (row as f64);
            bodies.push(
                RigidBody::new(1.0)
                    .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
                    .with_position(Vec3::new(x, 0.0, z)),
            );
            shapes.push(Shape::Box(BoxShape::new(h, h, h)));
            mats.push(Material::new(0.0, 0.5));
            placed += 1;
        }
        row += 1;
    }
    let mut cw = ContactWorld::new(
        World::new(bodies, vec![Box::new(UniformGravity::earth())]),
        shapes,
        vec![plane_ground()],
        Some(mats),
    );
    cw.set_allow_sleep(true);
    for _ in 0..100_000 {
        cw.step(DT);
        assert!(cw.world.state.all_finite());
    }
    let ke = kinetic_energy(&cw.world, &cw.world.state);
    let max_z = cw
        .world
        .state
        .x
        .iter()
        .map(|p| p.z)
        .fold(f64::NEG_INFINITY, f64::max);
    eprintln!("stability_pyramid_1e5: max_z={max_z:.6} ke={ke:.6e}");
    assert!(ke < 2.0, "pyramid KE {ke}");
    assert!(max_z > 0.0 && max_z < 6.0);
}

#[test]
#[ignore = "slow: rolling sphere 1e5 steps"]
fn stability_rolling_sphere_1e5() {
    let r = 0.2;
    let body = RigidBody::new(1.0)
        .with_inertia(Sphere::new(r).inertia(1.0))
        .with_position(Vec3::new(0.0, 0.0, r + 0.001))
        .with_velocity(Vec3::new(2.0, 0.0, 0.0));
    let mut cw = ContactWorld::new(
        World::new(vec![body], vec![Box::new(UniformGravity::earth())]),
        vec![Shape::Sphere(Sphere::new(r))],
        vec![plane_ground().with_material(Material::new(0.0, 0.6))],
        Some(vec![Material::new(0.0, 0.6)]),
    );
    cw.set_allow_sleep(false);
    for _ in 0..100_000 {
        cw.step(DT);
        assert!(cw.world.state.all_finite());
        assert!(cw.world.state.x[0].z > 0.0);
    }
    let z = cw.world.state.x[0].z;
    eprintln!(
        "stability_rolling: final z={z:.6} x={:.6}",
        cw.world.state.x[0].x
    );
    // Bound: still on plane within 5 cm of radius rest height.
    assert!((z - r).abs() < 0.05, "z={z}");
}

#[test]
#[ignore = "slow: joint chain 1e5 steps"]
fn stability_joint_chain_1e5() {
    let n = 6;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    let mut mats = Vec::new();
    for i in 0..n {
        bodies.push(
            RigidBody::new(1.0)
                .with_inertia(Sphere::new(0.1).inertia(1.0))
                .with_position(Vec3::new(0.0, 0.0, 2.0 - 0.35 * i as f64)),
        );
        shapes.push(Shape::Sphere(Sphere::new(0.1)));
        mats.push(Material::new(0.0, 0.2));
    }
    let mut cw = ContactWorld::new(
        World::new(bodies, vec![Box::new(UniformGravity::earth())]),
        shapes,
        vec![],
        Some(mats),
    );
    cw.set_allow_sleep(false);
    cw.add_joint(Joint::distance(
        0,
        WORLD_BODY,
        Vec3::zero(),
        Vec3::new(0.0, 0.0, 2.0),
        0.0,
    ));
    for i in 0..n - 1 {
        cw.add_joint(Joint::distance(i, i + 1, Vec3::zero(), Vec3::zero(), 0.35));
    }
    for _ in 0..100_000 {
        cw.step(DT);
        assert!(cw.world.state.all_finite());
    }
    let tip = cw.world.state.x[n - 1];
    eprintln!(
        "stability_joint_chain: tip=({:.4},{:.4},{:.4})",
        tip.x, tip.y, tip.z
    );
    // Bound: tip stays within a few metres of the hang region (no blow-up).
    assert!(tip.norm() < 10.0, "tip blew up: {tip:?}");
}
