//! Property / fuzz tests for pe-blast finite-input hardening (M27).
//!
//! Default CI keeps case counts modest; set `PROPTEST_CASES` for a larger run
//! (see `docs/STABILITY.md`).

use pe_blast::{
    friedlander_pressure, peak_overpressure, positive_impulse, try_friedlander_pressure,
    try_gurney_velocity, try_peak_overpressure, try_positive_impulse, try_scaled_distance,
    BlastError, BlastSource, ExposedFace, GurneyGeometry,
};
use pe_math::Vec3;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn kinney_finite_nonneg_z_stays_finite(
        z in 0.0f64..200.0,
        p0 in 1.0e3f64..2.0e5,
    ) {
        let ps = try_peak_overpressure(z, p0).unwrap();
        prop_assert!(ps.is_finite());
        prop_assert!(ps >= 0.0);
        // Monotonicity in Z: larger standoff → smaller overpressure (Kinney).
        let ps2 = peak_overpressure(z + 1.0, p0);
        prop_assert!(ps2.is_finite());
        prop_assert!(ps2 <= ps + 1e-9);
    }

    #[test]
    fn friedlander_no_panic_finite_outputs(
        t in -0.1f64..0.2,
        p_s in -1.0e6f64..1.0e6,
        t_d in 1.0e-4f64..0.5,
        b in 0.0f64..5.0,
    ) {
        let p = try_friedlander_pressure(t, p_s, t_d, b).unwrap();
        prop_assert!(p.is_finite());
        let i = try_positive_impulse(p_s, t_d, b).unwrap();
        prop_assert!(i.is_finite());
        // Legacy infallible path must not panic.
        let _ = friedlander_pressure(t, p_s, t_d, b);
        let _ = positive_impulse(p_s, t_d, b);
    }

    #[test]
    fn scaled_distance_rejects_non_physical(
        range in -10.0f64..100.0,
        mass in -10.0f64..100.0,
        teq in -2.0f64..5.0,
    ) {
        let r = try_scaled_distance(range, mass, teq);
        if range >= 0.0 && range.is_finite() && mass > 0.0 && teq > 0.0 {
            prop_assert!(r.unwrap().is_finite());
        } else {
            prop_assert!(r.is_err());
        }
    }

    #[test]
    fn gurney_velocity_monotonic_in_cm(
        sqrt2e in 100.0f64..4000.0,
        cm in 0.05f64..2.0,
    ) {
        let v1 = try_gurney_velocity(sqrt2e, cm, GurneyGeometry::Cylindrical).unwrap();
        let v2 = try_gurney_velocity(sqrt2e, cm + 0.05, GurneyGeometry::Cylindrical).unwrap();
        prop_assert!(v1.is_finite() && v2.is_finite());
        prop_assert!(v2 >= v1);
    }
}

#[test]
fn try_constructors_typed_errors_do_not_panic() {
    assert!(matches!(
        BlastSource::try_new(Vec3::zero(), f64::NAN),
        Err(BlastError::InvalidSize("w_eq_kg"))
    ));
    assert!(matches!(
        BlastSource::try_new(Vec3::new(f64::INFINITY, 0.0, 0.0), 1.0),
        Err(BlastError::NonFinite("origin"))
    ));
    assert!(matches!(
        ExposedFace::try_new(0, Vec3::zero(), Vec3::unit_x(), 0.0),
        Err(BlastError::InvalidSize("face.area"))
    ));
    assert!(try_peak_overpressure(f64::NAN, 101_325.0).is_err());
    assert!(try_friedlander_pressure(0.0, 1.0, -1.0, 1.0).is_err());
    assert!(try_gurney_velocity(-1.0, 1.0, GurneyGeometry::Flat).is_err());
}
