//! Criterion baselines for pe-soft XPBD workloads.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pe_math::Vec3;
use pe_soft::builders::{make_cloth, make_rope, make_soft_cube};
use pe_soft::material::material_by_name;
use pe_soft::scenarios::{soft_canonical_hash, soft_canonical_world};

fn rope_hang_1s(c: &mut Criterion) {
    c.bench_function("rope_hang_1s", |b| {
        b.iter(|| {
            let mut w = make_rope(
                Vec3::new(-0.5, 0.0, 1.0),
                Vec3::new(0.5, 0.0, 1.0),
                16,
                0.5,
                1e-6,
                true,
            );
            w.params.substeps = 20;
            for _ in 0..60 {
                w.step(1.0 / 60.0);
            }
            black_box(w.particles[8].x.z);
        });
    });
}

fn cloth_4x4_2s(c: &mut Criterion) {
    c.bench_function("cloth_4x4_2s", |b| {
        b.iter(|| {
            let mut w = make_cloth(
                Vec3::new(0.0, 0.0, 1.0),
                4,
                4,
                0.1,
                0.1,
                0.2,
                1e-6,
                1e-4,
                true,
            );
            w.params.substeps = 20;
            for _ in 0..120 {
                w.step(1.0 / 60.0);
            }
            black_box(w.pack());
        });
    });
}

fn jelly_cube_drop(c: &mut Criterion) {
    c.bench_function("jelly_cube_drop_1s", |b| {
        b.iter(|| {
            let mat = material_by_name("jelly").unwrap();
            let mut w = make_soft_cube(Vec3::new(0.0, 0.0, 0.5), 0.25, 2, &mat, true);
            w.params.substeps = 20;
            for _ in 0..60 {
                w.step(1.0 / 60.0);
            }
            black_box(w.pack());
        });
    });
}

fn canonical_hash_bench(c: &mut Criterion) {
    c.bench_function("soft_canonical_hash", |b| {
        b.iter(|| black_box(soft_canonical_hash()));
    });
}

fn canonical_world_build(c: &mut Criterion) {
    c.bench_function("soft_canonical_world_build", |b| {
        b.iter(|| black_box(soft_canonical_world().len()));
    });
}

criterion_group!(
    soft_benches,
    rope_hang_1s,
    cloth_4x4_2s,
    jelly_cube_drop,
    canonical_hash_bench,
    canonical_world_build
);
criterion_main!(soft_benches);
