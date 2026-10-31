//! Criterion baselines for pe-space (see docs/BENCHMARKS.md).

use criterion::{criterion_group, criterion_main, Criterion};
use pe_math::Vec3;
use pe_space::constants::EARTH_MU;
use pe_space::field::{CentralMu, NBodyPointMass};
use pe_space::integrator::{AdaptiveStep, Integrator};
use pe_space::integrators::{Ias15, Leapfrog, Rk4, Whfast, WhfastCoords, Yoshida4};
use pe_space::kepler::{elements_to_state, orbital_period, KeplerElements};
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

fn eccentric_state() -> NBodyState {
    let el = KeplerElements {
        a: 1.0e7,
        e: 0.9,
        i: 0.1,
        raan: 0.0,
        arg_p: 0.0,
        true_anomaly: 0.0,
    };
    let (r, v) = elements_to_state(EARTH_MU, el).unwrap();
    NBodyState::new(vec![1.0], vec![r], vec![v])
}

fn synthetic_5body() -> NBodyState {
    let m0 = 1.0;
    let masses = [1.0e-3, 5.0e-4, 2.0e-4, 1.0e-4];
    let a = [1.0, 2.0, 4.0, 8.0];
    let mut m = vec![m0];
    let mut x = vec![Vec3::zero()];
    let mut v = vec![Vec3::zero()];
    for (&mi, &ai) in masses.iter().zip(a.iter()) {
        let vc = f64::sqrt((m0 + mi) / ai);
        m.push(mi);
        x.push(Vec3::new(ai, 0.0, 0.0));
        v.push(Vec3::new(0.0, vc, 0.0));
    }
    let mut state = NBodyState::new(m, x, v);
    state.rebase_to_com();
    state
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

fn bench_ias15_eccentric_orbit(c: &mut Criterion) {
    c.bench_function("ias15_e09_one_orbit", |b| {
        b.iter(|| {
            let mut state = eccentric_state();
            state.enable_compensated_positions();
            let field = CentralMu::new(EARTH_MU);
            let period = orbital_period(EARTH_MU, 1.0e7);
            let mut integ = Ias15::new(AdaptiveStep {
                dt: period / 100.0,
                dt_min: 1e-6,
                dt_max: period / 10.0,
                ..AdaptiveStep::default()
            });
            integ.integrate_to(&field, &mut state, 0.0, period, period / 100.0)
        })
    });
}

fn bench_whfast_5body(c: &mut Criterion) {
    c.bench_function("whfast_5body_10k_steps", |b| {
        b.iter(|| {
            let mut state = synthetic_5body();
            let field = NBodyPointMass::geometric();
            let mut integ = Whfast::new(1.0, WhfastCoords::Jacobi);
            let dt = 0.05;
            let mut t = 0.0;
            for _ in 0..10_000 {
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
    bench_figure_eight,
    bench_ias15_eccentric_orbit,
    bench_whfast_5body
);
criterion_main!(benches);
