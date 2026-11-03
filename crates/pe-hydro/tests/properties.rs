//! Property / fuzz tests for pe-hydro finite-input hardening (M27).

use pe_hydro::{
    froude_number, ittc57_cf, try_calm_water_resistance, try_froude_number, try_holtrop_mennen,
    try_ittc57_cf, BoxHull, HoltropInput, HydroError, MorisonMember, NomotoKT, SphereHull,
    NU_SEAWATER_15C_NOMINAL,
};
use pe_math::Vec3;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn ittc57_cf_finite_and_non_increasing_in_rn(
        rn in 1.0e3f64..1.0e10,
    ) {
        let cf = try_ittc57_cf(rn).unwrap();
        prop_assert!(cf.is_finite());
        prop_assert!(cf >= 0.0);
        let cf2 = ittc57_cf(rn * 1.5);
        prop_assert!(cf2.is_finite());
        // ITTC-57 friction coefficient decreases with Reynolds number.
        prop_assert!(cf2 <= cf + 1e-15);
    }

    #[test]
    fn froude_and_resistance_no_panic(
        speed in 0.0f64..30.0,
        length in 1.0f64..400.0,
    ) {
        let fr = try_froude_number(speed, length, 9.80665).unwrap();
        prop_assert!(fr.is_finite());
        prop_assert!(fr >= 0.0);
        let _ = froude_number(speed, length, 9.80665);
        let r = try_calm_water_resistance(
            1025.0,
            speed,
            length * 10.0,
            length,
            NU_SEAWATER_15C_NOMINAL,
            0.0,
            1.0,
        )
        .unwrap();
        prop_assert!(r.is_finite());
        prop_assert!(r >= 0.0);
    }

    #[test]
    fn holtrop_speed_ramp_resistance_non_decreasing(
        knots in 5.0f64..28.0,
    ) {
        let mut a = HoltropInput::holtrop_1982_example();
        let mut b = a;
        a.speed = knots * pe_hydro::KNOTS_TO_MS;
        b.speed = (knots + 1.0) * pe_hydro::KNOTS_TO_MS;
        let ra = try_holtrop_mennen(&a).unwrap();
        let rb = try_holtrop_mennen(&b).unwrap();
        prop_assert!(ra.r_total.is_finite() && rb.r_total.is_finite());
        // Calm-water total resistance grows with speed for this displacement hull.
        prop_assert!(rb.r_total + 1.0 >= ra.r_total);
    }
}

#[test]
fn try_constructors_typed_errors_do_not_panic() {
    assert!(matches!(
        BoxHull::try_new(f64::NAN, 1.0, 1.0),
        Err(HydroError::InvalidSize("length"))
    ));
    assert!(matches!(
        BoxHull::try_new(0.0, 1.0, 1.0),
        Err(HydroError::InvalidSize("length"))
    ));
    assert!(matches!(
        SphereHull::try_new(-1.0),
        Err(HydroError::InvalidSize("radius"))
    ));
    assert!(matches!(
        NomotoKT::try_new(1.0, 0.0),
        Err(HydroError::InvalidSize("nomoto.t"))
    ));
    assert!(matches!(
        MorisonMember::try_new(Vec3::zero(), Vec3::unit_z(), 0.0, 2.0, 1.0, 4),
        Err(HydroError::InvalidSize("diameter"))
    ));
    assert!(try_ittc57_cf(f64::NAN).is_err());
    assert!(try_froude_number(1.0, f64::NEG_INFINITY, 9.81).is_err());
}
