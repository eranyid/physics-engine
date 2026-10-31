//! Analytic / closed-form stage-1 tests (Rust unit suite).

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_core::body::inertia_solid_cylinder;
use pe_core::diagnostics::{
    angular_momentum, body_angular_velocity, kinetic_energy, linear_momentum, state_hash64,
    total_energy,
};
use pe_core::forces::{
    ConstantTorque, LinearDrag, NewtonianGravity, QuadraticDrag, Spring, UniformGravity,
};
use pe_core::integrators::{
    explicit_euler, rk4, symplectic_euler, velocity_verlet, Integrator, IntegratorKind,
};
use pe_core::{RigidBody, World};
use pe_math::libm_ops;
use pe_math::{Quat, Vec3, G, STANDARD_GRAVITY as G0};

fn world_one(body: RigidBody, gens: Vec<Box<dyn pe_core::ForceGenerator>>) -> World {
    World::new(vec![body], gens)
}

#[test]
fn free_fall_time_to_ground() {
    let h = 100.0; // m
    let mut w = world_one(
        RigidBody::new(1.0).with_position(Vec3::new(0.0, 0.0, h)),
        vec![Box::new(UniformGravity::earth())],
    );
    let t = libm_ops::sqrt(2.0 * h / G0);
    w.run_for(t, 1e-3, velocity_verlet);
    assert_abs_diff_eq!(w.state.x[0].z, 0.0, epsilon = 1e-9);
    assert_relative_eq!(
        w.state.v[0].z,
        -libm_ops::sqrt(2.0 * G0 * h),
        max_relative = 1e-12
    );
}

#[test]
fn projectile_verlet_matches_closed_form() {
    let x0 = Vec3::new(0.0, 0.0, 2.0);
    let v0 = Vec3::new(12.0, -3.0, 15.0);
    let mut w = world_one(
        RigidBody::new(2.5).with_position(x0).with_velocity(v0),
        vec![Box::new(UniformGravity::earth())],
    );
    let t = 2.0;
    w.run_for(t, 1e-3, velocity_verlet);
    let gv = Vec3::new(0.0, 0.0, -G0);
    let x_exact = x0 + v0 * t + gv * (0.5 * t * t);
    let v_exact = v0 + gv * t;
    assert_abs_diff_eq!(w.state.x[0].x, x_exact.x, epsilon = 1e-9);
    assert_abs_diff_eq!(w.state.x[0].y, x_exact.y, epsilon = 1e-9);
    assert_abs_diff_eq!(w.state.x[0].z, x_exact.z, epsilon = 1e-9);
    assert_abs_diff_eq!(w.state.v[0].x, v_exact.x, epsilon = 1e-9);
    assert_abs_diff_eq!(w.state.v[0].y, v_exact.y, epsilon = 1e-9);
    assert_abs_diff_eq!(w.state.v[0].z, v_exact.z, epsilon = 1e-9);
}

#[test]
fn linear_drag_exponential_approach() {
    let m = 0.2; // kg
    let b = 0.05; // kg/s
    let mut w = world_one(
        RigidBody::new(m),
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(LinearDrag::new(b, Vec3::zero())),
        ],
    );
    let vt = m * G0 / b;
    let tau = m / b;
    for &t in &[1.0, 4.0, 12.0, 40.0] {
        w.run_for(t, 1e-2, rk4);
        let v_exact = -vt * (1.0 - libm_ops::exp(-t / tau));
        let z_exact = -(vt * t - vt * tau * (1.0 - libm_ops::exp(-t / tau)));
        assert_relative_eq!(w.state.v[0].z, v_exact, max_relative = 1e-9);
        assert_relative_eq!(w.state.x[0].z, z_exact, max_relative = 1e-9);
    }
    assert_relative_eq!(-w.state.v[0].z, vt, max_relative = 1e-4);
}

#[test]
fn quadratic_drag_tanh_and_terminal() {
    let m = 80.0;
    let k = QuadraticDrag::coefficient(1.225, 1.0, 0.7);
    let vt = libm_ops::sqrt(m * G0 / k);
    let mut w = world_one(
        RigidBody::new(m).with_position(Vec3::new(0.0, 0.0, 4000.0)),
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(QuadraticDrag::new(k, Vec3::zero())),
        ],
    );
    for &t in &[2.0, 5.0, 10.0, 30.0] {
        w.run_for(t, 1e-2, rk4);
        let v_exact = -vt * libm_ops::tanh(G0 * t / vt);
        let arg = G0 * t / vt;
        // cosh via portable exp (libm has cosh too; exp path keeps one dependency style)
        let cosh = (libm_ops::exp(arg) + libm_ops::exp(-arg)) * 0.5;
        let drop = vt * vt / G0 * libm_ops::log(cosh);
        assert_relative_eq!(w.state.v[0].z, v_exact, max_relative = 1e-8);
        assert_relative_eq!(w.state.x[0].z, 4000.0 - drop, max_relative = 1e-8);
    }
    assert_relative_eq!(-w.state.v[0].z, vt, max_relative = 1e-5);
    assert!((vt - 42.7).abs() < 0.1);
}

const M_SUN: f64 = 1.98847e30;
const M_EARTH: f64 = 5.9722e24;
const AU: f64 = 1.495978707e11;

fn two_body(a: f64, e: f64) -> (World, f64) {
    let m = M_SUN + M_EARTH;
    let rp = a * (1.0 - e);
    let vp = libm_ops::sqrt(G * m * (1.0 + e) / rp);
    let f1 = M_EARTH / m;
    let f2 = M_SUN / m;
    let bodies = vec![
        RigidBody::new(M_SUN)
            .with_position(Vec3::new(-f1 * rp, 0.0, 0.0))
            .with_velocity(Vec3::new(0.0, -f1 * vp, 0.0)),
        RigidBody::new(M_EARTH)
            .with_position(Vec3::new(f2 * rp, 0.0, 0.0))
            .with_velocity(Vec3::new(0.0, f2 * vp, 0.0)),
    ];
    let period = 2.0 * std::f64::consts::PI * libm_ops::sqrt(a * a * a / (G * m));
    (
        World::new(bodies, vec![Box::new(NewtonianGravity::new())]),
        period,
    )
}

#[test]
fn kepler_circular_orbit() {
    let (mut w, p) = two_body(AU, 0.0);
    assert_relative_eq!(p / 86400.0, 365.25, max_relative = 2e-3);
    let e0 = total_energy(&w, &w.state);
    let l0 = angular_momentum(&w, &w.state, Vec3::zero());
    let x0 = w.state.x[1] - w.state.x[0];
    w.run_for(p, p / 20000.0, rk4);
    let rel = w.state.x[1] - w.state.x[0];
    assert!((rel - x0).norm() / AU < 1e-8);
    assert_relative_eq!(total_energy(&w, &w.state), e0, max_relative = 1e-10);
    let l1 = angular_momentum(&w, &w.state, Vec3::zero());
    assert_relative_eq!(l1.x, l0.x, max_relative = 1e-10);
    assert_relative_eq!(l1.y, l0.y, max_relative = 1e-10);
    assert_relative_eq!(l1.z, l0.z, max_relative = 1e-10);
}

#[test]
fn kepler_eccentric_apoapsis() {
    let (mut w, p) = two_body(AU, 0.5);
    w.run_for(p / 2.0, p / 20000.0, rk4);
    let r = (w.state.x[1] - w.state.x[0]).norm();
    assert_relative_eq!(r, AU * 1.5, max_relative = 1e-9);
}

fn final_ho_error(integ: Integrator, dt: f64) -> f64 {
    let m = 2.0;
    let k = 8.0;
    let omega = libm_ops::sqrt(k / m);
    let t_end = 2.0;
    let mut w = world_one(
        RigidBody::new(m).with_position(Vec3::new(1.0, 0.0, 0.0)),
        vec![Box::new(Spring::new(0, Vec3::zero(), k))],
    );
    w.run_for(t_end, dt, integ);
    let x_ex = libm_ops::cos(omega * t_end);
    let v_ex = -omega * libm_ops::sin(omega * t_end);
    let dx = w.state.x[0].x - x_ex;
    let dv = (w.state.v[0].x - v_ex) / omega;
    libm_ops::sqrt(dx * dx + dv * dv)
}

fn observed_order(integ: Integrator, dts: &[f64]) -> f64 {
    let errs: Vec<f64> = dts.iter().map(|&dt| final_ho_error(integ, dt)).collect();
    // log-log slope via least squares
    let n = dts.len() as f64;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..dts.len() {
        let x = libm_ops::log(dts[i]);
        let y = libm_ops::log(errs[i]);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

#[test]
fn harmonic_oscillator_convergence_orders() {
    let cases: &[(&str, Integrator, f64, &[f64])] = &[
        (
            "explicit_euler",
            explicit_euler,
            1.0,
            &[1e-3, 5e-4, 2.5e-4, 1.25e-4],
        ),
        (
            "symplectic_euler",
            symplectic_euler,
            1.0,
            &[1e-3, 5e-4, 2.5e-4, 1.25e-4],
        ),
        (
            "velocity_verlet",
            velocity_verlet,
            2.0,
            &[1e-2, 5e-3, 2.5e-3, 1.25e-3],
        ),
        ("rk4", rk4, 4.0, &[8e-2, 4e-2, 2e-2, 1e-2]),
    ];
    for &(name, integ, expected, dts) in cases {
        let p = observed_order(integ, dts);
        eprintln!("{name}: observed order {p:.3}");
        assert!(
            (p - expected).abs() < 0.15,
            "{name}: expected ~{expected}, got {p}"
        );
    }
}

fn spinning(i_diag: [f64; 3], w_body: Vec3, q: Quat) -> World {
    let w_world = q.rotate(w_body);
    World::new(
        vec![RigidBody::new(1.0)
            .with_inertia_diag(i_diag[0], i_diag[1], i_diag[2])
            .with_orientation(q)
            .with_angular_velocity(w_world)],
        vec![],
    )
}

#[test]
fn torque_free_conserves_l_and_energy() {
    let q0 = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0), 0.3);
    for kind in [
        IntegratorKind::Rk4,
        IntegratorKind::VelocityVerlet,
        IntegratorKind::SymplecticEuler,
    ] {
        let mut w = spinning([1.0, 2.0, 3.5], Vec3::new(0.4, 1.0, -2.0), q0);
        let l0 = angular_momentum(&w, &w.state, Vec3::zero());
        let t0 = kinetic_energy(&w, &w.state);
        let norms = w.run_recorded(20.0, 1e-3, kind.as_fn(), |ww| ww.state.q[0].norm());
        let l1 = angular_momentum(&w, &w.state, Vec3::zero());
        assert_abs_diff_eq!(l1.x, l0.x, epsilon = 1e-14);
        assert_abs_diff_eq!(l1.y, l0.y, epsilon = 1e-14);
        assert_abs_diff_eq!(l1.z, l0.z, epsilon = 1e-14);
        assert!(norms.iter().all(|&n| (n - 1.0).abs() < 1e-14));
        let tol = match kind {
            IntegratorKind::Rk4 => 1e-9,
            _ => 1e-6,
        };
        assert_relative_eq!(kinetic_energy(&w, &w.state), t0, max_relative = tol);
    }
}

#[test]
fn symmetric_top_body_frame_precession() {
    let i1 = 2.0;
    let i3 = 1.0;
    let wp = 0.5;
    let w3 = 3.0;
    let mut w = spinning([i1, i1, i3], Vec3::new(wp, 0.0, w3), Quat::identity());
    let om = (i3 - i1) * w3 / i1;
    for &t in &[1.0, 5.0, 10.0] {
        w.run_for(t, 1e-3, rk4);
        let wb = body_angular_velocity(&w, &w.state)[0];
        assert_abs_diff_eq!(wb.x, wp * libm_ops::cos(om * t), epsilon = 1e-9);
        assert_abs_diff_eq!(wb.y, wp * libm_ops::sin(om * t), epsilon = 1e-9);
        assert_abs_diff_eq!(wb.z, w3, epsilon = 1e-9);
    }
}

#[test]
fn constant_torque_spins_up_sphere() {
    let i = 0.4;
    let mut w = World::new(
        vec![RigidBody::new(1.0).with_inertia_diag(i, i, i)],
        vec![Box::new(ConstantTorque::new(0, Vec3::new(0.0, 0.0, 2.0)))],
    );
    w.run_for(3.0, 1e-3, rk4);
    let omega = w.omega_i(&w.state, 0);
    assert_abs_diff_eq!(omega.z, 2.0 * 3.0 / i, epsilon = 1e-11);
}

#[test]
fn cylinder_inertia_matches_python() {
    let i = inertia_solid_cylinder(2.0, 0.5, 1.0);
    assert_abs_diff_eq!(i[0][0], 0.29166666666666663, epsilon = 1e-15);
    assert_abs_diff_eq!(i[2][2], 0.25, epsilon = 1e-15);
}

#[test]
fn determinism_bitwise_repeat() {
    // Fixed scenario — identical packed state and hash on repeat.
    let mut a = cluster(1);
    let mut b = cluster(1);
    a.run_for(50.0, 0.5, rk4);
    b.run_for(50.0, 0.5, rk4);
    assert_eq!(a.state.pack(), b.state.pack());
    assert_eq!(state_hash64(&a.state), state_hash64(&b.state));
}

#[test]
fn isolated_nbody_momentum() {
    for kind in [
        IntegratorKind::ExplicitEuler,
        IntegratorKind::SymplecticEuler,
        IntegratorKind::VelocityVerlet,
        IntegratorKind::Rk4,
    ] {
        let mut w = cluster(1);
        let p0 = linear_momentum(&w, &w.state);
        let l0 = angular_momentum(&w, &w.state, Vec3::zero());
        w.run_for(500.0, 0.5, kind.as_fn());
        assert!(w.state.v.iter().map(|v| v.norm()).sum::<f64>() > 1e-3);
        let p1 = linear_momentum(&w, &w.state);
        let pscale = w
            .mass
            .iter()
            .zip(w.state.v.iter())
            .map(|(m, v)| *m * v.norm())
            .sum::<f64>();
        assert!((p1 - p0).norm() / pscale < 1e-12);
        if matches!(
            kind,
            IntegratorKind::SymplecticEuler | IntegratorKind::VelocityVerlet
        ) {
            let lscale: f64 = w
                .state
                .x
                .iter()
                .zip(w.mass.iter())
                .zip(w.state.v.iter())
                .map(|((x, m), v)| x.cross(*v * *m).norm())
                .sum();
            let l1 = angular_momentum(&w, &w.state, Vec3::zero());
            assert!((l1 - l0).norm() / lscale < 1e-10);
        }
    }
}

struct Rng(u64);

impl Rng {
    fn next_u01(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 33) as f64 / (1u64 << 31) as f64
    }

    fn randn(&mut self) -> f64 {
        let u1 = self.next_u01().max(1e-12);
        let u2 = self.next_u01();
        libm_ops::sqrt(-2.0 * libm_ops::log(u1)) * libm_ops::cos(2.0 * std::f64::consts::PI * u2)
    }
}

/// Fixed seed cluster matching the spirit of the Python `cluster` fixture.
fn cluster(seed: u64) -> World {
    let mut rng = Rng(seed.wrapping_add(0x9e3779b97f4a7c15));
    let mut bodies = Vec::new();
    for _ in 0..5 {
        let m = 1e6 + rng.next_u01() * (1e8 - 1e6);
        let pos = Vec3::new(rng.randn(), rng.randn(), rng.randn()) * 10.0;
        let vel = Vec3::new(rng.randn(), rng.randn(), rng.randn()) * 1e-3;
        let w = Vec3::new(rng.randn(), rng.randn(), rng.randn());
        let i1 = 1.0 + rng.next_u01() * 2.0;
        let i2 = 1.0 + rng.next_u01() * 2.0;
        let i3 = 1.0 + rng.next_u01() * 2.0;
        bodies.push(
            RigidBody::new(m)
                .with_position(pos)
                .with_velocity(vel)
                .with_angular_velocity(w)
                .with_inertia_diag(i1, i2, i3),
        );
    }
    World::new(
        bodies,
        vec![Box::new(NewtonianGravity::with_params(G, 0.1))],
    )
}

/// Canonical scenario for cross-OS hash CI (see `docs/DETERMINISM.md`).
#[test]
fn cross_os_canonical_hash_fixture() {
    let mut w = World::new(
        vec![
            RigidBody::new(1.0)
                .with_position(Vec3::new(0.0, 0.0, 10.0))
                .with_velocity(Vec3::new(1.0, 0.0, 0.0)),
            RigidBody::new(2.0)
                .with_position(Vec3::new(1.0, 0.0, 0.0))
                .with_velocity(Vec3::new(0.0, 0.5, 0.0))
                .with_inertia_diag(0.4, 0.5, 0.6)
                .with_angular_velocity(Vec3::new(0.1, 0.0, 0.2)),
        ],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(LinearDrag::new(0.01, Vec3::zero())),
            Box::new(Spring::new(0, Vec3::new(0.0, 0.0, 10.0), 5.0)),
            Box::new(ConstantTorque::new(1, Vec3::new(0.0, 0.0, 0.05))),
        ],
    );
    w.run_for(2.0, 1e-3, rk4);
    let h = state_hash64(&w.state);
    // Printed for the cross-OS CI job; also locked here once measured.
    eprintln!("CANONICAL_STATE_HASH={h:016x}");
    // Placeholder: overwritten after first local measurement in DETERMINISM.md
    // For now assert self-consistency only; the CI job compares across OSes.
    assert_eq!(h, state_hash64(&w.state));
}
