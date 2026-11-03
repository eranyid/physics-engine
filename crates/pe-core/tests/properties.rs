//! Property-based tests for stage-1 rigid worlds (M22).
//!
//! Default CI keeps case counts modest; set `PROPTEST_CASES` for a larger opt-in run
//! (see the `rust-stability` CI job / `docs/STABILITY.md`).

use pe_core::diagnostics::{linear_momentum, state_hash64, total_energy};
use pe_core::forces::{LinearDrag, NewtonianGravity, Spring, UniformGravity};
use pe_core::integrators::{rk4, symplectic_euler, velocity_verlet};
use pe_core::{inertia_solid_sphere, RigidBody, World};
use pe_math::{Quat, Vec3, Xoshiro256StarStar};
use proptest::prelude::*;

fn finite_state(w: &World) -> bool {
    w.state.all_finite() && w.t.is_finite()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn random_nbody_step_stays_finite(
        seed in any::<u64>(),
        n in 1usize..6,
        steps in 1usize..40,
        kind in 0u8..3,
    ) {
        let mut rng = Xoshiro256StarStar::seed(seed);
        let mut bodies = Vec::with_capacity(n);
        for i in 0..n {
            let mass = rng.uniform(0.5, 3.0);
            let r = rng.uniform(0.05, 0.4);
            bodies.push(
                RigidBody::try_new(mass)
                    .unwrap()
                    .try_with_inertia(inertia_solid_sphere(mass, r))
                    .unwrap()
                    .with_position(Vec3::new(
                        rng.uniform(-2.0, 2.0),
                        rng.uniform(-2.0, 2.0),
                        rng.uniform(0.5, 4.0) + i as f64,
                    ))
                    .with_velocity(Vec3::new(
                        rng.uniform(-1.0, 1.0),
                        rng.uniform(-1.0, 1.0),
                        rng.uniform(-1.0, 1.0),
                    ))
                    .with_orientation(Quat::from_axis_angle(
                        Vec3::new(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
                            .normalize(),
                        rng.uniform(-0.5, 0.5),
                    )),
            );
        }
        let mut gens: Vec<Box<dyn pe_core::ForceGenerator>> =
            vec![Box::new(UniformGravity::earth())];
        gens.push(Box::new(Spring::new(0, Vec3::new(0.0, 0.0, 10.0), 5.0)));
        gens.push(Box::new(LinearDrag::new(0.02, Vec3::zero())));
        let mut w = World::new(bodies, gens);
        let integ = match kind {
            0 => velocity_verlet as pe_core::Integrator,
            1 => symplectic_euler,
            _ => rk4,
        };
        for _ in 0..steps {
            w.step(1e-3, integ);
            prop_assert!(finite_state(&w), "non-finite state escaped step");
        }
    }

    #[test]
    fn isolated_nbody_momentum_conserved_no_external(
        seed in any::<u64>(),
        steps in 10usize..80,
    ) {
        // Softened Newtonian pair: internal forces only → total P conserved.
        let mut rng = Xoshiro256StarStar::seed(seed);
        let v = rng.uniform(0.2, 2.0);
        let bodies = vec![
            RigidBody::new(1.0)
                .with_position(Vec3::new(-1.0, 0.0, 0.0))
                .with_velocity(Vec3::new(0.0, v, 0.0))
                .with_inertia_diag(0.4, 0.4, 0.4),
            RigidBody::new(1.0)
                .with_position(Vec3::new(1.0, 0.0, 0.0))
                .with_velocity(Vec3::new(0.0, -v, 0.0))
                .with_inertia_diag(0.4, 0.4, 0.4),
        ];
        let mut w = World::new(
            bodies,
            vec![Box::new(NewtonianGravity::with_params(1.0, 0.05))],
        );
        let p0 = linear_momentum(&w, &w.state);
        for _ in 0..steps {
            w.step(1e-3, rk4);
        }
        prop_assert!(w.state.all_finite());
        let p1 = linear_momentum(&w, &w.state);
        prop_assert!((p1 - p0).norm() < 1e-9, "ΔP={}", (p1 - p0).norm());
    }

    #[test]
    fn determinism_repeat_hash_identical(seed in any::<u64>()) {
        let mut rng = Xoshiro256StarStar::seed(seed);
        let body = RigidBody::new(1.5)
            .with_position(Vec3::new(rng.uniform(-1.0, 1.0), 0.0, rng.uniform(1.0, 3.0)))
            .with_velocity(Vec3::new(rng.uniform(-0.5, 0.5), 0.0, 0.0))
            .with_inertia_diag(0.5, 0.6, 0.7);
        let build = || {
            let mut w = World::new(
                vec![body.clone()],
                vec![
                    Box::new(UniformGravity::earth()),
                    Box::new(LinearDrag::new(0.01, Vec3::zero())),
                ],
            );
            for _ in 0..50 {
                w.step(1e-3, rk4);
            }
            state_hash64(&w.state)
        };
        prop_assert_eq!(build(), build());
    }
}

#[test]
fn try_new_typed_errors_do_not_panic() {
    assert!(pe_core::RigidBody::try_new(f64::NAN).is_err());
    assert!(pe_core::RigidBody::try_new(0.0).is_err());
}

#[test]
fn harmonic_energy_bounded_symplectic() {
    let body = RigidBody::new(1.0)
        .with_position(Vec3::new(1.0, 0.0, 0.0))
        .with_inertia_diag(1.0, 1.0, 1.0);
    let mut w = World::new(
        vec![body],
        vec![Box::new(Spring::new(0, Vec3::zero(), 40.0))],
    );
    let e0 = total_energy(&w, &w.state);
    for _ in 0..2000 {
        w.step(1e-3, symplectic_euler);
    }
    assert!(w.state.all_finite());
    let e1 = total_energy(&w, &w.state);
    assert!(
        (e1 - e0).abs() / e0.abs().max(1.0) < 0.05,
        "energy drift {} → {}",
        e0,
        e1
    );
}
