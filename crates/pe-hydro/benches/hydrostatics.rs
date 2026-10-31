//! Criterion benches for hydrostatic buoyancy + waves (stage 3 / M9).
//!
//! Baseline numbers are recorded in `docs/BENCHMARKS.md`.

use criterion::{criterion_group, criterion_main, Criterion};
use pe_core::forces::UniformGravity;
use pe_core::integrators::rk4;
use pe_core::{inertia_solid_box, RigidBody, World};
use pe_hydro::hull::{BoxHull, Hull, RHO_FRESHWATER};
use pe_hydro::waves::{AiryWave, SpectralOcean, Wave, WaveField};
use pe_hydro::Buoyancy;
use pe_math::{Quat, Vec3};

fn floating_world() -> World {
    let hull = BoxHull::new(20.0, 6.0, 3.0);
    let mass = RHO_FRESHWATER * hull.length * hull.beam * 1.2;
    let draft = hull.upright_draft_for_mass(mass, RHO_FRESHWATER);
    let z = -draft + 0.5 * hull.depth;
    World::new(
        vec![RigidBody::new(mass)
            .with_inertia(inertia_solid_box(mass, 20.0, 6.0, 3.0))
            .with_position(Vec3::new(0.0, 0.0, z))
            .with_orientation(Quat::from_axis_angle(Vec3::unit_x(), 0.05))],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(Buoyancy::freshwater(0, Hull::Box(hull), 0.0)),
        ],
    )
}

fn bench_hydro(c: &mut Criterion) {
    let mut group = c.benchmark_group("hydrostatics");
    group.bench_function("buoyancy_forces_1k", |b| {
        let w = floating_world();
        b.iter(|| {
            let _ = w.forces(&w.state, 0.0);
        });
    });
    group.bench_function("heave_rk4_500_steps", |b| {
        b.iter(|| {
            let mut w = floating_world();
            for _ in 0..500 {
                w.step(1e-3, rk4);
            }
        });
    });
    group.bench_function("gz_curve_60_angles", |b| {
        let hull = BoxHull::new(20.0, 8.0, 2.5);
        let phis: Vec<f64> = (0..60).map(|i| (i as f64) * 1.0_f64.to_radians()).collect();
        b.iter(|| {
            let _ = pe_hydro::box_righting_arm_curve(&hull, 2.0, 1.5, &phis);
        });
    });
    group.bench_function("airy_height_1k", |b| {
        let w = AiryWave::earth(0.5, 8.0, 50.0, 0.3);
        b.iter(|| {
            let mut s = 0.0;
            for i in 0..1000 {
                s += w.height(i as f64 * 0.1, 0.0, 0.01 * i as f64);
            }
            s
        });
    });
    group.bench_function("spectral_height_eval", |b| {
        let ocean = SpectralOcean::pierson_moskowitz(10.0, 100.0, 24, 8, 1);
        b.iter(|| ocean.height(0.0, 0.0, 1.0));
    });
    group.bench_function("wave_buoyancy_rk4_200", |b| {
        b.iter(|| {
            let hull = BoxHull::new(12.0, 4.0, 2.0);
            let mass = 24_000.0;
            let draft = hull.upright_draft_for_mass(mass, RHO_FRESHWATER);
            let z = -draft + 0.5 * hull.depth;
            let wave = AiryWave::earth(0.25, 8.0, 50.0, 0.0);
            let mut w = World::new(
                vec![RigidBody::new(mass)
                    .with_inertia(inertia_solid_box(mass, 12.0, 4.0, 2.0))
                    .with_position(Vec3::new(0.0, 0.0, z))],
                vec![
                    Box::new(UniformGravity::earth()),
                    Box::new(
                        Buoyancy::freshwater(0, Hull::Box(hull), 0.0)
                            .with_wave(WaveField::Airy(wave)),
                    ),
                ],
            );
            for _ in 0..200 {
                w.step(1e-2, rk4);
            }
        });
    });
    group.finish();
}

criterion_group!(benches, bench_hydro);
criterion_main!(benches);
