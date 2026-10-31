//! Criterion benches: box pyramids (10 and 20 rows).
//!
//! Not executed in default CI (`cargo test` omits Criterion harnesses).
//! Record medians in `docs/BENCHMARKS.md` after numerical changes.

use criterion::{criterion_group, criterion_main, Criterion};
use pe_collision::{BoxShape, ContactWorld, Material, Shape, StaticCollider};
use pe_core::forces::UniformGravity;
use pe_core::{RigidBody, World};
use pe_math::Vec3;

fn pyramid_bodies(rows: usize) -> (Vec<RigidBody>, Vec<Shape>, Vec<Material>) {
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
    (bodies, shapes, mats)
}

fn make_pyramid(rows: usize) -> ContactWorld {
    let (bodies, shapes, mats) = pyramid_bodies(rows);
    let mut cw = ContactWorld::new(
        World::new(bodies, vec![Box::new(UniformGravity::earth())]),
        shapes,
        vec![
            StaticCollider::new(Shape::Plane(pe_collision::Plane::new(Vec3::unit_z(), 0.0)))
                .with_material(Material::new(0.0, 1.0)),
        ],
        Some(mats),
    );
    cw.iterations = 20;
    cw
}

fn bench_pyramid(c: &mut Criterion) {
    let mut group = c.benchmark_group("contact_pyramids");
    group.sample_size(20);

    group.bench_function("pyramid_10_rows_60_steps", |b| {
        b.iter(|| {
            let mut cw = make_pyramid(10);
            // 10 rows → 55 boxes; short run for hot-loop timing
            cw.run_for(60.0 / 240.0, 1.0 / 240.0);
            std::hint::black_box(cw.world.state.pack())
        });
    });

    group.bench_function("pyramid_20_rows_60_steps", |b| {
        b.iter(|| {
            let mut cw = make_pyramid(20);
            cw.run_for(60.0 / 240.0, 1.0 / 240.0);
            std::hint::black_box(cw.world.state.pack())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_pyramid);
criterion_main!(benches);
