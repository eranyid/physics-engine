//! GA-like aircraft: compare 10 s at trim thrust vs zero thrust (altitude loss).

use pe_aero::{level_flight_trim_earth, AircraftAero, DensityModel, FixedThrust, WingPolar, RHO0};
use pe_core::forces::UniformGravity;
use pe_core::integrators::rk4;
use pe_core::{inertia_solid_box, RigidBody, World};
use pe_math::libm_ops;
use pe_math::{Quat, Vec3};

fn make_world(thrust: f64) -> World {
    let mass = 1000.0;
    let polar = WingPolar::light_aircraft();
    let trim = level_flight_trim_earth(mass, &polar, RHO0);
    let q_att = Quat::from_axis_angle(Vec3::unit_y(), -trim.alpha);
    let e_v_body = Vec3::new(libm_ops::cos(trim.alpha), 0.0, -libm_ops::sin(trim.alpha));
    let body = RigidBody::new(mass)
        .with_inertia(inertia_solid_box(mass, 8.0, 10.0, 2.0))
        .with_position(Vec3::new(0.0, 0.0, 1000.0))
        .with_velocity(Vec3::new(trim.v, 0.0, 0.0))
        .with_orientation(q_att);
    World::new(
        vec![body],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(AircraftAero::new(
                0,
                polar,
                DensityModel::Constant(RHO0),
                Vec3::zero(),
            )),
            Box::new(FixedThrust::new(0, thrust, e_v_body)),
        ],
    )
}

fn main() {
    let mass = 1000.0;
    let polar = WingPolar::light_aircraft();
    let trim = level_flight_trim_earth(mass, &polar, RHO0);

    let mut held = make_world(trim.thrust);
    let mut cut = make_world(0.0);
    for _ in 0..1_000 {
        held.step(1e-2, rk4);
        cut.step(1e-2, rk4);
    }
    println!(
        "falling_aircraft: after 10 s, trim z={:.2} m; zero-thrust z={:.2} m (started at 1000 m)",
        held.state.x[0].z, cut.state.x[0].z
    );
}
