//! Invariant checks: NaN/Inf never escape a step; energy/momentum bounds; determinism.

use pe_core::diagnostics::{
    angular_momentum, kinetic_energy, linear_momentum, state_hash64, total_energy,
};
use pe_core::forces::{NewtonianGravity, Spring, UniformGravity};
use pe_core::integrators::{rk4, symplectic_euler};
use pe_core::{RigidBody, World};
use pe_math::Vec3;

#[test]
fn step_never_emits_nan_inf_projectile() {
    let mut w = World::new(
        vec![RigidBody::new(2.0)
            .with_position(Vec3::new(0.0, 0.0, 10.0))
            .with_velocity(Vec3::new(5.0, 0.0, 0.0))
            .with_inertia_diag(0.4, 0.4, 0.4)],
        vec![Box::new(UniformGravity::earth())],
    );
    for _ in 0..5_000 {
        w.step(1e-3, rk4);
        assert!(w.state.all_finite(), "t={}", w.t);
    }
}

#[test]
fn inelastic_free_particles_kinetic_energy_non_increasing_without_forces() {
    // No forces: KE constant (within ULP). Documented bound: relative |ΔKE|/KE0 < 1e-12.
    let mut w = World::new(
        vec![
            RigidBody::new(1.0)
                .with_velocity(Vec3::new(1.0, 0.0, 0.0))
                .with_inertia_diag(0.4, 0.4, 0.4),
            RigidBody::new(2.0)
                .with_velocity(Vec3::new(-0.5, 0.0, 0.0))
                .with_inertia_diag(0.8, 0.8, 0.8),
        ],
        vec![],
    );
    let ke0 = kinetic_energy(&w, &w.state);
    for _ in 0..1_000 {
        w.step(1e-3, symplectic_euler);
    }
    let ke1 = kinetic_energy(&w, &w.state);
    assert!(w.state.all_finite());
    assert!(
        (ke1 - ke0).abs() / ke0.max(1e-15) < 1e-12,
        "KE drifted {ke0} → {ke1}"
    );
}

#[test]
fn contact_free_newtonian_momentum_conserved() {
    let mut w = World::new(
        vec![
            RigidBody::new(1.0)
                .with_position(Vec3::new(-1.0, 0.0, 0.0))
                .with_velocity(Vec3::new(0.0, 0.5, 0.0))
                .with_inertia_diag(0.4, 0.4, 0.4),
            RigidBody::new(1.0)
                .with_position(Vec3::new(1.0, 0.0, 0.0))
                .with_velocity(Vec3::new(0.0, -0.5, 0.0))
                .with_inertia_diag(0.4, 0.4, 0.4),
        ],
        vec![Box::new(NewtonianGravity::with_params(1.0, 0.1))],
    );
    let p0 = linear_momentum(&w, &w.state);
    let l0 = angular_momentum(&w, &w.state, Vec3::zero());
    for _ in 0..500 {
        w.step(1e-3, rk4);
    }
    let p1 = linear_momentum(&w, &w.state);
    let l1 = angular_momentum(&w, &w.state, Vec3::zero());
    assert!((p1 - p0).norm() < 1e-10);
    assert!((l1 - l0).norm() < 1e-9);
}

#[test]
fn determinism_identical_hash_on_repeat() {
    let run = || {
        let mut w = World::new(
            vec![RigidBody::new(1.0)
                .with_position(Vec3::new(0.0, 0.0, 2.0))
                .with_velocity(Vec3::new(0.1, 0.0, 0.0))
                .with_inertia_diag(0.4, 0.5, 0.6)],
            vec![
                Box::new(UniformGravity::earth()),
                Box::new(Spring::new(0, Vec3::new(0.0, 0.0, 2.0), 3.0)),
            ],
        );
        for _ in 0..200 {
            w.step(1e-3, rk4);
        }
        state_hash64(&w.state)
    };
    assert_eq!(run(), run());
}

#[test]
fn spring_total_energy_bounded_symplectic() {
    let mut w = World::new(
        vec![RigidBody::new(1.0)
            .with_position(Vec3::new(0.5, 0.0, 0.0))
            .with_inertia_diag(1.0, 1.0, 1.0)],
        vec![Box::new(Spring::new(0, Vec3::zero(), 25.0))],
    );
    let e0 = total_energy(&w, &w.state);
    let mut e_max = e0;
    for _ in 0..5_000 {
        w.step(5e-4, symplectic_euler);
        let e = total_energy(&w, &w.state);
        e_max = e_max.max(e);
    }
    // Bound: symplectic energy oscillation ≤ 2% of |E0| peak excursion for this HO.
    assert!(
        (e_max - e0).abs() / e0.abs().max(1.0) < 0.02,
        "peak energy {e_max} vs e0 {e0}"
    );
}
