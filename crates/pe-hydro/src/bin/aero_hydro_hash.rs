//! Print the canonical stage-3 aero+hydro state hash for cross-OS CI.
//!
//! Usage: `cargo run -p pe-hydro --bin aero_hydro_hash --release`
//! Scenario must stay in lockstep with `docs/DETERMINISM.md` and
//! `physics_engine_rs.canonical_aero_hydro_hash()`.

use pe_aero::atmosphere::{DensityModel, RHO0};
use pe_aero::{AircraftAero, FixedThrust, WingPolar};
use pe_core::diagnostics::state_hash64;
use pe_core::forces::UniformGravity;
use pe_core::integrators::rk4;
use pe_core::{inertia_solid_box, RigidBody, World};
use pe_hydro::hull::{BoxHull, Hull, RHO_FRESHWATER};
use pe_hydro::Buoyancy;
use pe_math::{Quat, Vec3};

fn main() {
    let polar = WingPolar::new(
        16.0,
        1.6,
        10.0,
        0.025,
        0.8,
        0.2,
        5.7,
        14.0_f64.to_radians(),
        1.2,
        None,
        0.0,
        0.0,
        0.0,
    )
    .expect("polar");
    let mass_ac = 1000.0;
    let q_att = Quat::from_axis_angle(Vec3::unit_y(), -0.05);
    let aircraft = RigidBody::new(mass_ac)
        .with_inertia(inertia_solid_box(mass_ac, 8.0, 10.0, 2.0))
        .with_position(Vec3::new(0.0, 0.0, 500.0))
        .with_velocity(Vec3::new(40.0, 0.0, 0.0))
        .with_orientation(q_att);

    let hull = BoxHull::new(12.0, 4.0, 2.0);
    let mass_ship = 24_000.0;
    let draft = hull.upright_draft_for_mass(mass_ship, RHO_FRESHWATER);
    let z_c = -draft + 0.5 * hull.depth;
    let ship = RigidBody::new(mass_ship)
        .with_inertia(inertia_solid_box(mass_ship, 12.0, 4.0, 2.0))
        .with_position(Vec3::new(30.0, 0.0, z_c))
        .with_orientation(Quat::from_axis_angle(Vec3::unit_x(), 0.02));

    let mut w = World::new(
        vec![aircraft, ship],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(AircraftAero::new(
                0,
                polar,
                DensityModel::Constant(RHO0),
                Vec3::zero(),
            )),
            Box::new(FixedThrust::new(0, 800.0, Vec3::unit_x())),
            Box::new(Buoyancy::freshwater(1, Hull::Box(hull), 0.0)),
        ],
    );
    w.run_for(1.0, 1e-2, rk4);
    let h = state_hash64(&w.state);
    println!("{h:016x}");
}
