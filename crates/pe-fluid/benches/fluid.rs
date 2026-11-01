//! Criterion benches for pe-fluid (particles/sec workloads).

use criterion::{criterion_group, criterion_main, Criterion};
use pe_fluid::particles::{fill_box, fill_box_boundary, mass_from_spacing, Particles};
use pe_fluid::pbf::{PbfParams, PbfWorld};
use pe_fluid::scenarios::{dfsph_tank_hash, fluid_tank_hash};
use pe_fluid::wcsph::{WcsphParams, WcsphWorld};
use pe_fluid::Dim;
use pe_math::{Vec3, STANDARD_GRAVITY};

fn bench_fluid(c: &mut Criterion) {
    let mut group = c.benchmark_group("fluid");

    group.bench_function("fluid_tank_hash", |b| {
        b.iter(|| std::hint::black_box(fluid_tank_hash()))
    });

    group.bench_function("dfsph_tank_hash", |b| {
        b.iter(|| std::hint::black_box(dfsph_tank_hash()))
    });

    group.bench_function("pbf_1k_particles_10_steps", |b| {
        b.iter(|| {
            let dim = Dim::D3;
            let dx = 0.05;
            let h = 2.2 * dx;
            let rho0 = 1000.0;
            let mass = mass_from_spacing(rho0, dx, dim);
            let mut parts = Particles::new(dim, rho0, mass, h);
            // ~10×10×10 = 1000 fluid
            fill_box(
                &mut parts,
                Vec3::new(0.1, 0.1, 0.1),
                10,
                10,
                10,
                dx,
                Vec3::zero(),
            );
            fill_box_boundary(
                &mut parts,
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.7, 0.7, 0.7),
                dx,
                1,
            );
            let params = PbfParams {
                gravity: Vec3::new(0.0, 0.0, -STANDARD_GRAVITY),
                iterations: 2,
                ..PbfParams::default()
            };
            let mut world = PbfWorld::new(parts, params);
            for _ in 0..10 {
                world.step(1.0 / 120.0);
            }
            std::hint::black_box(world.particles.len())
        })
    });

    group.bench_function("wcsph_256_particles_20_steps", |b| {
        b.iter(|| {
            let dim = Dim::D3;
            let dx = 0.05;
            let h = 2.2 * dx;
            let rho0 = 1000.0;
            let mass = mass_from_spacing(rho0, dx, dim);
            let mut parts = Particles::new(dim, rho0, mass, h);
            fill_box(
                &mut parts,
                Vec3::new(0.1, 0.1, 0.1),
                8,
                8,
                4,
                dx,
                Vec3::zero(),
            );
            fill_box_boundary(
                &mut parts,
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.6, 0.6, 0.4),
                dx,
                1,
            );
            let params = WcsphParams::water(rho0, 25.0, Vec3::new(0.0, 0.0, -STANDARD_GRAVITY));
            let mut world = WcsphWorld::new(parts, params);
            for _ in 0..20 {
                world.step(0.0005);
            }
            std::hint::black_box(world.t)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_fluid);
criterion_main!(benches);
