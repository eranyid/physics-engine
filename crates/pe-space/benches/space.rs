//! Criterion baselines for pe-space (see docs/BENCHMARKS.md).

use criterion::{criterion_group, criterion_main, Criterion};
use pe_math::Vec3;
use pe_space::constants::EARTH_MU;
use pe_space::field::{CentralMu, NBodyPointMass};
use pe_space::integrator::Integrator;
use pe_space::integrators::{Leapfrog, Rk4, Yoshida4};
use pe_space::scenarios::{figure_eight_state, FIGURE_EIGHT_PERIOD};
use pe_space::state::NBodyState;

fn leo_state() -> NBodyState {
    let r = 7_000_000.0;
    let v = (EARTH_MU / r).sqrt();
    NBodyState::new(
        vec![1.0],
        vec![Vec3::new(r, 0.0, 0.0)],
        vec![Vec3::new(0.0, v, 0.0)],
    )
}

fn bench_leapfrog_leo(c: &mut Criterion) {
    c.bench_function("leo_leapfrog_2k_steps", |b| {
        b.iter(|| {
            let mut state = leo_state();
            let field = CentralMu::new(EARTH_MU);
            let mut integ = Leapfrog::default();
            let dt = 5828.5 / 2000.0;
            let mut t = 0.0;
            for _ in 0..2000 {
                let res = integ.step(&field, &mut state, t, dt);
                t = res.t;
            }
            t
        })
    });
}

fn bench_yoshida_leo(c: &mut Criterion) {
    c.bench_function("leo_yoshida4_2k_steps", |b| {
        b.iter(|| {
            let mut state = leo_state();
            let field = CentralMu::new(EARTH_MU);
            let mut integ = Yoshida4::default();
            let dt = 5828.5 / 2000.0;
            let mut t = 0.0;
            for _ in 0..2000 {
                let res = integ.step(&field, &mut state, t, dt);
                t = res.t;
            }
            t
        })
    });
}

fn bench_rk4_leo(c: &mut Criterion) {
    c.bench_function("leo_rk4_2k_steps", |b| {
        b.iter(|| {
            let mut state = leo_state();
            let field = CentralMu::new(EARTH_MU);
            let mut integ = Rk4;
            let dt = 5828.5 / 2000.0;
            let mut t = 0.0;
            for _ in 0..2000 {
                let res = integ.step(&field, &mut state, t, dt);
                t = res.t;
            }
            t
        })
    });
}

fn bench_figure_eight(c: &mut Criterion) {
    c.bench_function("figure8_leapfrog_2k_steps", |b| {
        b.iter(|| {
            let mut state = figure_eight_state();
            let field = NBodyPointMass::geometric();
            let mut integ = Leapfrog::default();
            let dt = FIGURE_EIGHT_PERIOD / 2000.0;
            let mut t = 0.0;
            for _ in 0..2000 {
                let res = integ.step(&field, &mut state, t, dt);
                t = res.t;
            }
            t
        })
    });
}

criterion_group!(
    benches,
    bench_leapfrog_leo,
    bench_yoshida_leo,
    bench_rk4_leo,
    bench_figure_eight
);
criterion_main!(benches);
