//! Criterion benches for integrator hot loops (stage 1).
//!
//! Baseline numbers are recorded in `docs/BENCHMARKS.md`.

use criterion::{criterion_group, criterion_main, Criterion};
use pe_core::forces::{NewtonianGravity, Spring, UniformGravity};
use pe_core::integrators::{rk4, symplectic_euler, velocity_verlet};
use pe_core::{RigidBody, World};
use pe_math::Vec3;

fn oscillator_world() -> World {
    World::new(
        vec![RigidBody::new(2.0).with_position(Vec3::new(1.0, 0.0, 0.0))],
        vec![Box::new(Spring::new(0, Vec3::zero(), 8.0))],
    )
}

fn nbody_world(n: usize) -> World {
    let mut bodies = Vec::with_capacity(n);
    for i in 0..n {
        let a = i as f64;
        bodies.push(
            RigidBody::new(1e6 + a * 1e5)
                .with_position(Vec3::new(a, 0.1 * a, -0.05 * a))
                .with_velocity(Vec3::new(0.0, 1e-3, 0.0)),
        );
    }
    World::new(bodies, vec![Box::new(NewtonianGravity::new())])
}

fn projectile_world() -> World {
    World::new(
        vec![RigidBody::new(1.0).with_velocity(Vec3::new(10.0, 0.0, 10.0))],
        vec![Box::new(UniformGravity::earth())],
    )
}

fn bench_integrators(c: &mut Criterion) {
    let mut group = c.benchmark_group("integrator_hot_loops");

    group.bench_function("ho_rk4_1k_steps", |b| {
        b.iter(|| {
            let mut w = oscillator_world();
            for _ in 0..1000 {
                w.step(1e-3, rk4);
            }
            w.state.x[0].x
        })
    });

    group.bench_function("ho_verlet_1k_steps", |b| {
        b.iter(|| {
            let mut w = oscillator_world();
            for _ in 0..1000 {
                w.step(1e-3, velocity_verlet);
            }
            w.state.x[0].x
        })
    });

    group.bench_function("ho_symplectic_1k_steps", |b| {
        b.iter(|| {
            let mut w = oscillator_world();
            for _ in 0..1000 {
                w.step(1e-3, symplectic_euler);
            }
            w.state.x[0].x
        })
    });

    group.bench_function("projectile_verlet_1k", |b| {
        b.iter(|| {
            let mut w = projectile_world();
            for _ in 0..1000 {
                w.step(1e-3, velocity_verlet);
            }
            w.state.x[0].z
        })
    });

    group.bench_function("nbody8_rk4_200_steps", |b| {
        b.iter(|| {
            let mut w = nbody_world(8);
            for _ in 0..200 {
                w.step(0.1, rk4);
            }
            w.state.x[0].x
        })
    });

    group.finish();
}

criterion_group!(benches, bench_integrators);
criterion_main!(benches);
