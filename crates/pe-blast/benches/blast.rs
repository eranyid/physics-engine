//! Criterion benches for pe-blast.

use criterion::{criterion_group, criterion_main, Criterion};
use pe_blast::euler::Euler1d;
use pe_blast::kinney::peak_overpressure;
use pe_blast::sedov::{shock_radius, solve_xi0, SedovGeometry};
use pe_blast::{blast_pyramid_hash, friedlander_pressure};

fn bench_blast(c: &mut Criterion) {
    let mut group = c.benchmark_group("blast");
    group.bench_function("kinney_1k", |b| {
        b.iter(|| {
            let mut s = 0.0;
            for i in 0..1000 {
                s += peak_overpressure(0.5 + 0.01 * i as f64, 101_325.0);
            }
            std::hint::black_box(s)
        })
    });
    group.bench_function("friedlander_10k", |b| {
        b.iter(|| {
            let mut s = 0.0;
            for i in 0..10_000 {
                s += friedlander_pressure(1e-5 * i as f64, 1e5, 0.01, 1.0);
            }
            std::hint::black_box(s)
        })
    });
    group.bench_function("sod_n200_to_0.2", |b| {
        b.iter(|| {
            let mut s = Euler1d::sod(200, 1.4);
            s.run_cfl(0.2, 0.4);
            std::hint::black_box(s.t)
        })
    });
    group.bench_function("sedov_radius_1k", |b| {
        let sol = solve_xi0(1.4, SedovGeometry::Spherical);
        b.iter(|| {
            let mut s = 0.0;
            for i in 1..1001 {
                s += shock_radius(&sol, 1e12, 1.225, 0.01 * i as f64);
            }
            std::hint::black_box(s)
        })
    });
    group.bench_function("blast_pyramid_hash", |b| {
        b.iter(|| std::hint::black_box(blast_pyramid_hash()))
    });
    group.finish();
}

criterion_group!(benches, bench_blast);
criterion_main!(benches);
