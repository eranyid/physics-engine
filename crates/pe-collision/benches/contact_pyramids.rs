//! Criterion benches: box pyramids (PGS vs soft-step) and broad-phase (SAP vs tree).
//!
//! Not executed in default CI (`cargo test` omits Criterion harnesses).
//! Record medians in `docs/BENCHMARKS.md` after numerical changes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pe_collision::ccd::sphere_plane_toi;
use pe_collision::shapes::Plane;
use pe_collision::{
    brute_force_pairs, dynamic_tree_pairs, sweep_and_prune, Aabb, BoxShape, BroadPhaseKind,
    ContactWorld, Joint, JointSoftParams, Material, Shape, SoftStepParams, SolverMode, Sphere,
    StaticCollider, WORLD_BODY,
};
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

fn make_pyramid(rows: usize, soft: bool) -> ContactWorld {
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
    if soft {
        cw.solver_mode = SolverMode::SoftStep;
        cw.soft_params = SoftStepParams::default();
    }
    cw
}

fn random_aabbs(n: usize, seed: u64) -> Vec<Aabb> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        state
    };
    let uni = |next: &mut dyn FnMut() -> u64, a: f64, b: f64| -> f64 {
        let u = (next() >> 11) as f64 / ((1u64 << 53) as f64);
        a + (b - a) * u
    };
    let mut aabbs = Vec::with_capacity(n);
    for _ in 0..n {
        let cx = uni(&mut next, 0.0, 50.0);
        let cy = uni(&mut next, 0.0, 50.0);
        let cz = uni(&mut next, 0.0, 50.0);
        let ex = uni(&mut next, 0.05, 0.8);
        let ey = uni(&mut next, 0.05, 0.8);
        let ez = uni(&mut next, 0.05, 0.8);
        aabbs.push(Aabb {
            lo: Vec3::new(cx - ex, cy - ey, cz - ez),
            hi: Vec3::new(cx + ex, cy + ey, cz + ez),
        });
    }
    aabbs
}

fn bench_pyramid(c: &mut Criterion) {
    let mut group = c.benchmark_group("contact_pyramids");
    group.sample_size(20);

    for (rows, soft, name) in [
        (10usize, false, "pyramid_10_rows_60_steps_pgs"),
        (10, true, "pyramid_10_rows_60_steps_soft"),
        (20, false, "pyramid_20_rows_60_steps_pgs"),
        (20, true, "pyramid_20_rows_60_steps_soft"),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut cw = make_pyramid(rows, soft);
                cw.run_for(60.0 / 240.0, 1.0 / 240.0);
                std::hint::black_box(cw.world.state.pack())
            });
        });
    }

    group.finish();
}

fn bench_broadphase(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadphase");
    group.sample_size(30);

    for n in [1_000usize, 10_000] {
        let aabbs = random_aabbs(n, 42);
        group.bench_with_input(BenchmarkId::new("sap", n), &aabbs, |b, aabbs| {
            b.iter(|| std::hint::black_box(sweep_and_prune(aabbs, None)));
        });
        group.bench_with_input(BenchmarkId::new("tree", n), &aabbs, |b, aabbs| {
            b.iter(|| std::hint::black_box(dynamic_tree_pairs(aabbs, None)));
        });
        if n <= 1_000 {
            group.bench_with_input(BenchmarkId::new("brute", n), &aabbs, |b, aabbs| {
                b.iter(|| std::hint::black_box(brute_force_pairs(aabbs, None)));
            });
        }
    }

    group.finish();
}

fn bench_pyramid_broad_kind(c: &mut Criterion) {
    let mut group = c.benchmark_group("pyramid_broad_kind");
    group.sample_size(15);
    for kind in [BroadPhaseKind::Sap, BroadPhaseKind::DynamicTree] {
        let name = match kind {
            BroadPhaseKind::Sap => "pyramid_10_sap",
            BroadPhaseKind::DynamicTree => "pyramid_10_tree",
            BroadPhaseKind::Brute => "pyramid_10_brute",
        };
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut cw = make_pyramid(10, false);
                cw.broad_phase = kind;
                cw.run_for(60.0 / 240.0, 1.0 / 240.0);
                std::hint::black_box(cw.world.state.pack())
            });
        });
    }
    group.finish();
}

fn bench_joints_and_ccd(c: &mut Criterion) {
    let mut group = c.benchmark_group("m5b_joints_ccd");
    group.sample_size(20);
    group.bench_function("joints_double_pendulum_600", |b| {
        b.iter(|| {
            std::hint::black_box(pe_collision::canonical_joints_hash());
        });
    });
    group.bench_function("ccd_bullet_toi_sphere_plane", |b| {
        b.iter(|| {
            let t = sphere_plane_toi(
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, -1000.0),
                0.05,
                Vec3::unit_z(),
                0.0,
                1.0 / 60.0,
            );
            std::hint::black_box(t)
        });
    });
    // Keep unused imports honest if criterion elides — touch Joint builder once.
    group.bench_function("joint_distance_build", |b| {
        b.iter(|| {
            let j = Joint::distance(0, WORLD_BODY, Vec3::zero(), Vec3::zero(), 1.0)
                .with_soft(JointSoftParams::default());
            std::hint::black_box(j)
        });
    });
    let _ = (Sphere::new(0.1), Plane::new(Vec3::unit_z(), 0.0));
    group.finish();
}

criterion_group!(
    benches,
    bench_pyramid,
    bench_broadphase,
    bench_pyramid_broad_kind,
    bench_joints_and_ccd
);
criterion_main!(benches);
