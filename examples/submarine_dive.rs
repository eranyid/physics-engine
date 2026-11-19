//! M34 demo: Fossen 6-DOF submarine under thrust + ballast + current.
//!
//! ```text
//! cargo run -p pe-examples --bin submarine_dive --release
//! ```

use pe_hydro::hull::RHO_SEAWATER;
use pe_hydro::underwater::{
    canonical_underwater_hash, spheroid_added_mass6, CurrentField, UnderwaterVehicle, VolumeModel,
};
use pe_math::{Quat, Vec3, STANDARD_GRAVITY};

fn main() {
    let rho = RHO_SEAWATER;
    let added = spheroid_added_mass6(0.9, 0.11, rho);
    let d_quad0 = 0.5 * rho * 0.35 * 0.074;
    let mut veh = UnderwaterVehicle::try_new(
        51.0,
        [0.31, 14.0, 14.0],
        added,
        [2.0, 10.0, 10.0, 0.5, 5.0, 5.0],
        [d_quad0, 30.0, 30.0, 0.1, 10.0, 10.0],
        VolumeModel::Submerged {
            volume: 0.0499,
            cob_body: Vec3::new(0.0, 0.0, 0.02),
        },
        rho,
        STANDARD_GRAVITY,
        CurrentField::Uniform(Vec3::new(0.2, 0.1, 0.0)),
        None,
        false,
        Vec3::new(0.0, 0.0, -5.0),
        Quat::identity(),
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .expect("vehicle");

    let tau = [13.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    veh.run(10.0, 0.01, &tau);
    println!(
        "t={:.2} s  depth={:.3} m  u={:.3} m/s  pressure(p0=0)={:.1} Pa",
        veh.t,
        veh.depth(),
        veh.nu[0],
        veh.pressure(0.0)
    );
    println!(
        "canonical_underwater_hash = {:016x}",
        canonical_underwater_hash()
    );
}
