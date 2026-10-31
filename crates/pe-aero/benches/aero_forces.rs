//! Criterion benches for aero force evaluation + compressibility / Dryden (M9).
//!
//! Baseline numbers are recorded in `docs/BENCHMARKS.md`.

use criterion::{criterion_group, criterion_main, Criterion};
use pe_aero::atmosphere::{DensityModel, RHO0};
use pe_aero::bem::{nominal_rotor_polar, BemRotor, RotorGeometry};
use pe_aero::compressibility::{normal_shock, prandtl_glauert};
use pe_aero::turbulence::{DrydenParams, DrydenTurbulence};
use pe_aero::{AircraftAero, FixedThrust, WingPolar};
use pe_core::forces::UniformGravity;
use pe_core::integrators::rk4;
use pe_core::{inertia_solid_box, RigidBody, World};
use pe_math::{Quat, Vec3};

fn trimmed_world() -> World {
    let polar = WingPolar::light_aircraft();
    let mass = 1000.0;
    let q = Quat::from_axis_angle(Vec3::unit_y(), -0.08);
    World::new(
        vec![RigidBody::new(mass)
            .with_inertia(inertia_solid_box(mass, 8.0, 10.0, 2.0))
            .with_position(Vec3::new(0.0, 0.0, 1000.0))
            .with_velocity(Vec3::new(55.0, 0.0, 0.0))
            .with_orientation(q)],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(AircraftAero::new(
                0,
                polar,
                DensityModel::Constant(RHO0),
                Vec3::zero(),
            )),
            Box::new(FixedThrust::new(0, 1200.0, Vec3::unit_x())),
        ],
    )
}

fn bench_aero(c: &mut Criterion) {
    let mut group = c.benchmark_group("aero_forces");
    group.bench_function("forces_eval_1k", |b| {
        let w = trimmed_world();
        b.iter(|| {
            let _ = w.forces(&w.state, 0.0);
        });
    });
    group.bench_function("rk4_1k_steps", |b| {
        b.iter(|| {
            let mut w = trimmed_world();
            for _ in 0..1000 {
                w.step(1e-3, rk4);
            }
        });
    });
    group.bench_function("normal_shock_1k", |b| {
        b.iter(|| {
            let mut s = 0.0;
            for i in 0..1000 {
                let m = 1.1 + 0.002 * i as f64;
                if let Some(n) = normal_shock(m, 1.4) {
                    s += n.p_ratio;
                }
            }
            s
        });
    });
    group.bench_function("prandtl_glauert_1k", |b| {
        b.iter(|| {
            let mut s = 0.0;
            for i in 0..1000 {
                s += prandtl_glauert(1.0, 0.1 + 0.0005 * i as f64);
            }
            s
        });
    });
    group.bench_function("dryden_1k_steps", |b| {
        b.iter(|| {
            let mut d = DrydenTurbulence::new(DrydenParams::light_low_altitude(), 7);
            let mut s = 0.0;
            for _ in 0..1000 {
                s += d.step(50.0, 0.01).x;
            }
            s
        });
    });
    group.bench_function("bem_hover_eval", |b| {
        let geom = RotorGeometry::uniform_chord(
            2,
            4.0,
            0.3,
            0.2,
            10.0_f64.to_radians(),
            4.0_f64.to_radians(),
            16,
        );
        let bem = BemRotor::new(geom, nominal_rotor_polar(0.2, 0.012));
        b.iter(|| bem.evaluate(35.0, 0.0, RHO0));
    });
    group.finish();
}

criterion_group!(benches, bench_aero);
criterion_main!(benches);
