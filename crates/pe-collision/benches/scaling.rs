//! Scaling Criterion benches: pyramids, falling boxes, sparse broad phase,
//! ragdoll chains, joint stacks. Reports steps/s via Criterion (throughput).
//!
//! Not run in default CI. Record medians + phase times in `docs/BENCHMARKS.md`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pe_collision::{
    BoxShape, BroadPhaseKind, ContactWorld, Joint, JointSoftParams, Material, Plane, Shape, Sphere,
    StaticCollider, WORLD_BODY,
};
use pe_core::forces::UniformGravity;
use pe_core::{RigidBody, World};
use pe_math::Vec3;

fn pyramid_world(rows: usize) -> ContactWorld {
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
    cw
}

fn falling_boxes(n: usize) -> ContactWorld {
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
    cw.broad_phase = BroadPhaseKind::DynamicTree;
    cw
}

fn sparse_scene(n: usize) -> ContactWorld {
    let h = 0.2;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    let mut mats = Vec::new();
    for i in 0..n {
        let x = (i as f64) * 5.0;
        bodies.push(
            RigidBody::new(1.0)
                .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
                .with_position(Vec3::new(x, 0.0, h + 0.01)),
        );
        shapes.push(Shape::Box(BoxShape::new(h, h, h)));
        mats.push(Material::new(0.0, 0.4));
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
    cw.broad_phase = BroadPhaseKind::DynamicTree;
    cw
}

fn ragdoll_chain(links: usize) -> ContactWorld {
    let r = 0.08;
    let spacing = 0.25;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    for i in 0..links {
        bodies.push(
            RigidBody::new(1.0)
                .with_inertia(Sphere::new(r).inertia(1.0))
                .with_position(Vec3::new(0.0, 0.0, -spacing * (i as f64 + 1.0))),
        );
        shapes.push(Shape::Sphere(Sphere::new(r)));
    }
    let mut cw = ContactWorld::new(
        World::new(bodies, vec![Box::new(UniformGravity::earth())]),
        shapes,
        vec![],
        None,
    );
    cw.set_allow_sleep(false);
    for i in 0..links {
        let parent = if i == 0 { WORLD_BODY } else { i - 1 };
        let anchor_parent = if i == 0 {
            Vec3::zero()
        } else {
            Vec3::new(0.0, 0.0, -spacing * 0.5)
        };
        let anchor_child = Vec3::new(0.0, 0.0, spacing * 0.5);
        cw.joints.push(
            Joint::distance(parent, i, anchor_parent, anchor_child, spacing)
                .with_soft(JointSoftParams::default()),
        );
    }
    cw
}

fn joints_stack(n: usize) -> ContactWorld {
    let h = 0.2;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    for i in 0..n {
        bodies.push(
            RigidBody::new(1.0)
                .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
                .with_position(Vec3::new(0.0, 0.0, h + 2.0 * h * (i as f64))),
        );
        shapes.push(Shape::Box(BoxShape::new(h, h, h)));
    }
    let mut cw = ContactWorld::new(
        World::new(bodies, vec![Box::new(UniformGravity::earth())]),
        shapes,
        vec![
            StaticCollider::new(Shape::Plane(Plane::new(Vec3::unit_z(), 0.0)))
                .with_material(Material::new(0.0, 1.0)),
        ],
        None,
    );
    cw.set_allow_sleep(false);
    for i in 0..n {
        let parent = if i == 0 { WORLD_BODY } else { i - 1 };
        let z = if i == 0 { 0.0 } else { h };
        cw.joints.push(
            Joint::distance(
                parent,
                i,
                Vec3::new(0.0, 0.0, z),
                Vec3::new(0.0, 0.0, -h),
                2.0 * h,
            )
            .with_soft(JointSoftParams::default()),
        );
    }
    cw
}

fn bench_pyramids(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_pyramid");
    group.sample_size(15);
    for rows in [10usize, 20, 40] {
        let steps = 60usize;
        group.throughput(Throughput::Elements(steps as u64));
        group.bench_with_input(BenchmarkId::from_parameter(rows), &rows, |b, &rows| {
            b.iter(|| {
                let mut cw = pyramid_world(rows);
                cw.profile_phases = true;
                for _ in 0..steps {
                    cw.step(1.0 / 240.0);
                }
                std::hint::black_box((cw.world.state.pack(), cw.last_phase_times))
            });
        });
    }
    group.finish();
}

fn bench_falling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_falling");
    group.sample_size(12);
    for n in [1_000usize, 5_000, 10_000] {
        let steps = if n >= 10_000 { 10 } else { 20 };
        group.throughput(Throughput::Elements(steps as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let mut cw = falling_boxes(n);
                cw.profile_phases = true;
                for _ in 0..steps {
                    cw.step(1.0 / 240.0);
                }
                std::hint::black_box((cw.world.state.pack(), cw.last_phase_times))
            });
        });
    }
    group.finish();
}

fn bench_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_sparse_broadphase");
    group.sample_size(15);
    let n = 5_000usize;
    let steps = 30usize;
    group.throughput(Throughput::Elements(steps as u64));
    group.bench_function(BenchmarkId::new("tree", n), |b| {
        b.iter(|| {
            let mut cw = sparse_scene(n);
            cw.profile_phases = true;
            for _ in 0..steps {
                cw.step(1.0 / 240.0);
            }
            std::hint::black_box(cw.last_phase_times)
        });
    });
    group.finish();
}

fn bench_ragdoll(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_ragdoll");
    group.sample_size(20);
    for links in [8usize, 16, 32] {
        let steps = 120usize;
        group.throughput(Throughput::Elements(steps as u64));
        group.bench_with_input(BenchmarkId::from_parameter(links), &links, |b, &links| {
            b.iter(|| {
                let mut cw = ragdoll_chain(links);
                for _ in 0..steps {
                    cw.step(1.0 / 240.0);
                }
                std::hint::black_box(cw.world.state.pack())
            });
        });
    }
    group.finish();
}

fn bench_joints_stack(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_joints_stack");
    group.sample_size(15);
    for n in [5usize, 10, 20] {
        let steps = 120usize;
        group.throughput(Throughput::Elements(steps as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let mut cw = joints_stack(n);
                for _ in 0..steps {
                    cw.step(1.0 / 240.0);
                }
                std::hint::black_box(cw.world.state.pack())
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pyramids,
    bench_falling,
    bench_sparse,
    bench_ragdoll,
    bench_joints_stack
);
criterion_main!(benches);
