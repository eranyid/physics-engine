//! Manual timing for rigid-body step phases (not Criterion).
use pe_collision::{
    BoxShape, BroadPhaseKind, ContactWorld, Material, Plane, Shape, StaticCollider,
};
use pe_core::forces::UniformGravity;
use pe_core::{RigidBody, World};
use pe_math::Vec3;
use std::time::Instant;

fn pyramid(rows: usize) -> ContactWorld {
    let h = 0.5;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    let mut mats = Vec::new();
    for row in 0..rows {
        let n = row + 1;
        let y0 = -(n as f64 - 1.0) * h;
        for i in 0..n {
            let x = y0 + 2.0 * h * (i as f64);
            let z = h + 2.0 * h * (row as f64);
            bodies.push(
                RigidBody::new(1.0)
                    .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
                    .with_position(Vec3::new(x, 0.0, z)),
            );
            shapes.push(Shape::Box(BoxShape::new(h, h, h)));
            mats.push(Material::new(0.0, 0.5));
        }
    }
    let mut cw = ContactWorld::new(
        World::new(bodies, vec![Box::new(UniformGravity::earth())]),
        shapes,
        vec![
            StaticCollider::new(Shape::Plane(Plane::new(Vec3::unit_z(), 0.0)))
                .with_material(Material::new(0.0, 1.0)),
        ],
        Some(mats),
    );
    cw.iterations = 20;
    cw.profile_phases = true;
    cw
}

fn falling(n: usize) -> ContactWorld {
    let h = 0.25;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    let mut mats = Vec::new();
    let side = (n as f64).cbrt().ceil() as usize;
    let mut placed = 0usize;
    for ix in 0..side {
        for iy in 0..side {
            for iz in 0..side {
                if placed >= n {
                    break;
                }
                let x = (ix as f64 - side as f64 * 0.5) * 3.0 * h;
                let y = (iy as f64 - side as f64 * 0.5) * 3.0 * h;
                let z = 2.0 + iz as f64 * 3.0 * h;
                bodies.push(
                    RigidBody::new(1.0)
                        .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
                        .with_position(Vec3::new(x, y, z)),
                );
                shapes.push(Shape::Box(BoxShape::new(h, h, h)));
                mats.push(Material::new(0.0, 0.3));
                placed += 1;
            }
        }
    }
    let mut cw = ContactWorld::new(
        World::new(bodies, vec![Box::new(UniformGravity::earth())]),
        shapes,
        vec![
            StaticCollider::new(Shape::Plane(Plane::new(Vec3::unit_z(), 0.0)))
                .with_material(Material::new(0.0, 1.0)),
        ],
        Some(mats),
    );
    cw.profile_phases = true;
    cw
}

fn main() {
    let h = pe_collision::canonical_collision_hash();
    println!(
        "collision_hash={:016x} expect={:016x} ok={}",
        h,
        pe_collision::CANONICAL_COLLISION_HASH,
        h == pe_collision::CANONICAL_COLLISION_HASH
    );

    for rows in [10usize, 20, 40] {
        let mut cw = pyramid(rows);
        for _ in 0..5 {
            cw.step(1.0 / 240.0);
        }
        let steps = 60usize;
        let t0 = Instant::now();
        for _ in 0..steps {
            cw.step(1.0 / 240.0);
        }
        let dt = t0.elapsed().as_secs_f64();
        let n = rows * (rows + 1) / 2;
        let p = cw.last_phase_times;
        println!(
            "AFTER pyramid_{rows}_rows n={n} {steps} steps: {:.3} ms total, {:.1} steps/s, {:.3} ms/step | phase last-step broad={:.3}ms narrow={:.3}ms solver={:.3}ms",
            dt * 1e3,
            steps as f64 / dt,
            dt / steps as f64 * 1e3,
            p.broad * 1e3,
            p.narrow * 1e3,
            p.solver * 1e3
        );
    }

    for (n, kind, label) in [
        (1000usize, BroadPhaseKind::DynamicTree, "tree"),
        (1000, BroadPhaseKind::Sap, "sap"),
    ] {
        let mut cw = falling(n);
        cw.broad_phase = kind;
        for _ in 0..2 {
            cw.step(1.0 / 240.0);
        }
        let steps = 20usize;
        let t0 = Instant::now();
        for _ in 0..steps {
            cw.step(1.0 / 240.0);
        }
        let dt = t0.elapsed().as_secs_f64();
        let p = cw.last_phase_times;
        println!(
            "AFTER falling_{n} {label} {steps} steps: {:.3} ms total, {:.1} steps/s, {:.3} ms/step | broad={:.3}ms narrow={:.3}ms solver={:.3}ms",
            dt * 1e3,
            steps as f64 / dt,
            dt / steps as f64 * 1e3,
            p.broad * 1e3,
            p.narrow * 1e3,
            p.solver * 1e3
        );
    }
}
