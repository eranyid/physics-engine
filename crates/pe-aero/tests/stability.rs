//! Aircraft trim-holding long run (M22). `#[ignore]` for default CI.
//!
//! Matches the Python slow oracle (`tests/test_aero.py::test_trimmed_level_flight_holds_altitude`):
//! zero pitch derivatives (force-balance trim, no pitching moment), 5 s RK4.
//! Bound in `docs/STABILITY.md`: `|Δz| < 0.05 m`, `|v|` within 0.1% of trim.

use approx::assert_relative_eq;
use pe_aero::{level_flight_trim_earth, AircraftAero, DensityModel, FixedThrust, WingPolar, RHO0};
use pe_core::forces::UniformGravity;
use pe_core::integrators::rk4;
use pe_core::{inertia_solid_box, RigidBody, World};
use pe_math::libm_ops;
use pe_math::{Quat, Vec3};

#[test]
#[ignore = "slow: aircraft trim hold 5 s (force-balance polar)"]
fn stability_aircraft_trim_hold_5s() {
    let mass = 1000.0;
    let base = WingPolar::light_aircraft();
    // Freeze attitude moments: force-balance only (same as Python slow oracle).
    let polar = WingPolar::new(
        base.s,
        base.c,
        base.b,
        base.cd0,
        base.e,
        base.cl0,
        base.cl_alpha,
        base.alpha_stall,
        base.cl_fall,
        None,
        0.0,
        0.0,
        0.0,
    )
    .unwrap();
    let trim = level_flight_trim_earth(mass, &polar, RHO0);
    let q_att = Quat::from_axis_angle(Vec3::unit_y(), -trim.alpha);
    let e_v_body = Vec3::new(libm_ops::cos(trim.alpha), 0.0, -libm_ops::sin(trim.alpha));
    let body = RigidBody::new(mass)
        .with_inertia(inertia_solid_box(mass, 8.0, 10.0, 2.0))
        .with_position(Vec3::new(0.0, 0.0, 1000.0))
        .with_velocity(Vec3::new(trim.v, 0.0, 0.0))
        .with_orientation(q_att);
    let mut w = World::new(
        vec![body],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(AircraftAero::new(
                0,
                polar,
                DensityModel::Constant(RHO0),
                Vec3::zero(),
            )),
            Box::new(FixedThrust::new(0, trim.thrust, e_v_body)),
        ],
    );

    let (f, _) = w.forces(&w.state, 0.0);
    assert_relative_eq!(f[0].x, 0.0, epsilon = 1e-4);
    assert_relative_eq!(f[0].z, 0.0, epsilon = 1e-4);

    let z0 = w.state.x[0].z;
    let v0 = w.state.v[0].norm();
    let dt = 1e-3;
    let steps = (5.0 / dt) as usize;
    for _ in 0..steps {
        w.step(dt, rk4);
        assert!(w.state.all_finite());
    }
    let z1 = w.state.x[0].z;
    let v1 = w.state.v[0].norm();
    let dz = (z1 - z0).abs();
    let dv_rel = (v1 - v0).abs() / v0;
    eprintln!("stability_trim_hold: Δz={dz:.6} m |v|={v1:.6} (v0={v0:.6}) rel_dv={dv_rel:.3e}");
    // Bound (matches Python slow oracle): |Δz| < 0.05 m, |v| within 0.1%.
    assert!(dz < 0.05, "altitude drift {dz} m exceeds 0.05 m");
    assert!(dv_rel < 1e-3, "speed relative drift {dv_rel} exceeds 1e-3");
}
