//! Canonical scenarios for cross-OS determinism hashes.

use pe_aero::atmosphere::{DensityModel, RHO0};
use pe_aero::{nominal_rotor_polar, BemRotor, BemRotorForce, RotorGeometry};
use pe_core::diagnostics::state_hash64;
use pe_core::forces::UniformGravity;
use pe_core::integrators::rk4;
use pe_core::{inertia_solid_box, RigidBody, World};
use pe_math::Vec3;

use crate::force::Buoyancy;
use crate::hull::{BoxHull, Hull, RHO_FRESHWATER, RHO_SEAWATER};
use crate::morison::{MorisonForce, NOMINAL_CD_CIRCULAR, NOMINAL_CM_CIRCULAR};
use crate::waves::{AiryWave, WaveField};

/// Wave-driven floating box: Airy η=0.25 m, T=8 s, h=50 m; RK4 2 s @ 1e-2.
///
/// Must stay in sync with `docs/DETERMINISM.md`, the `wave_body_hash` binary, and
/// `physics_engine_rs.canonical_wave_body_hash()`.
pub fn canonical_wave_body_hash() -> u64 {
    let hull = BoxHull::new(12.0, 4.0, 2.0);
    let mass = 24_000.0;
    let draft = hull.upright_draft_for_mass(mass, RHO_FRESHWATER);
    let z_c = -draft + 0.5 * hull.depth;
    let wave = AiryWave::earth(0.25, 8.0, 50.0, 0.0);
    let body = RigidBody::new(mass)
        .with_inertia(inertia_solid_box(mass, 12.0, 4.0, 2.0))
        .with_position(Vec3::new(0.0, 0.0, z_c));
    let mut w = World::new(
        vec![body],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(
                Buoyancy::freshwater(0, Hull::Box(hull), 0.0)
                    .with_wave(WaveField::Airy(wave))
                    .with_froude_krylov(true),
            ),
        ],
    );
    w.run_for(2.0, 1e-2, rk4);
    state_hash64(&w.state)
}

/// M11 locked scenario: Morison pile + BEM rotor hover body; RK4 1 s @ 1e-2.
///
/// Must stay in sync with `docs/DETERMINISM.md`, the `m11_hash` binary, and
/// `physics_engine_rs.canonical_m11_hash()`.
///
/// - Body 0: vertical slender cylinder (D=0.5 m, L=10 m), mass 2000 kg, free surge
///   under Morison loads (Cm=2, Cd=1 nominal) in Airy A=0.5 m, T=8 s, h=30 m.
/// - Body 1: 500 kg rotorcraft at z=50 m with BEM rotor (R=4 m, 2 blades, Ω=40 rad/s)
///   thrusting +z against gravity (hover demo).
pub fn canonical_m11_hash() -> u64 {
    let wave = AiryWave::earth(0.5, 8.0, 30.0, 0.0);
    let d = 0.5;
    let z_bot = -10.0;
    let z_top = 0.0;
    let mass_pile = 2000.0;
    let pile = RigidBody::new(mass_pile)
        .with_inertia(inertia_solid_box(mass_pile, d, d, 10.0))
        .with_position(Vec3::new(0.0, 0.0, 0.5 * (z_bot + z_top)));

    let mass_rot = 500.0;
    let rotor_body = RigidBody::new(mass_rot)
        .with_inertia(inertia_solid_box(mass_rot, 2.0, 2.0, 1.0))
        .with_position(Vec3::new(20.0, 0.0, 50.0));

    let geom = RotorGeometry::uniform_chord(
        2,
        4.0,
        0.4,
        0.25,
        12.0_f64.to_radians(),
        6.0_f64.to_radians(),
        12,
    );
    let polar = nominal_rotor_polar(0.25, 0.01);
    let bem = BemRotor::new(geom, polar);

    let mut w = World::new(
        vec![pile, rotor_body],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(MorisonForce::new(
                0,
                Vec3::new(0.0, 0.0, z_bot - 0.5 * (z_bot + z_top)),
                Vec3::new(0.0, 0.0, z_top - 0.5 * (z_bot + z_top)),
                d,
                NOMINAL_CM_CIRCULAR,
                NOMINAL_CD_CIRCULAR,
                32,
                RHO_SEAWATER,
                WaveField::Airy(wave),
            )),
            Box::new(BemRotorForce::new(
                1,
                bem,
                40.0,
                DensityModel::Constant(RHO0),
                Vec3::unit_z(),
                Vec3::zero(),
            )),
        ],
    );
    w.run_for(1.0, 1e-2, rk4);
    state_hash64(&w.state)
}
