//! Criterion benches for pe-ballistics.

use criterion::{criterion_group, criterion_main, Criterion};
use pe_ballistics::ammo::bundled_profiles;
use pe_ballistics::projectile::{MuzzlePose, ProjectileSystem, ShotId};
use pe_ballistics::{analytic_cd, ballistics_volley_hash};
use pe_math::Vec3;

fn bench_ballistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("ballistics");
    group.bench_function("analytic_cd_100k", |b| {
        b.iter(|| {
            let mut s = 0.0;
            for i in 0..100_000 {
                s += analytic_cd(0.001 * i as f64);
            }
            std::hint::black_box(s)
        })
    });
    group.bench_function("step_10k_bullets", |b| {
        let mut sys = ProjectileSystem::new(0.5);
        let mut p = bundled_profiles()[2].clone();
        p.velocity_std_m_s = 0.0;
        p.dispersion_std_rad = 0.0;
        let idx = sys.add_profile(p);
        for i in 0..10_000u64 {
            sys.spawn(
                ShotId(i + 1),
                idx,
                MuzzlePose {
                    position: Vec3::new(0.0, (i as f64) * 0.001, 1.0),
                    direction: Vec3::unit_x(),
                    barrel_length_m: None,
                },
            );
        }
        b.iter(|| {
            sys.step(1.0 / 120.0, &[], &[]);
            std::hint::black_box(sys.alive_count())
        })
    });
    group.bench_function("volley_hash", |b| {
        b.iter(|| std::hint::black_box(ballistics_volley_hash()))
    });
    group.finish();
}

criterion_group!(benches, bench_ballistics);
criterion_main!(benches);
