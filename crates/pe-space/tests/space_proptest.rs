//! Property tests: energy and angular-momentum conservation on random circular
//! two-body / mild N-body problems (avoids close-encounter blow-ups).

use pe_math::libm_ops;
use pe_math::Vec3;
use pe_space::diagnostics::{angular_momentum, total_energy};
use pe_space::field::{CentralMu, NBodyPointMass};
use pe_space::integrator::Integrator;
use pe_space::integrators::{Leapfrog, Yoshida4};
use pe_space::state::NBodyState;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]

    #[test]
    fn leapfrog_circular_kepler_conserves_e_and_l(
        radius in 1.0f64..5.0,
        mu in 0.5f64..2.0,
    ) {
        let v = libm_ops::sqrt(mu / radius);
        let mut state = NBodyState::new(
            vec![1.0],
            vec![Vec3::new(radius, 0.0, 0.0)],
            vec![Vec3::new(0.0, v, 0.0)],
        );
        let field = CentralMu::new(mu);
        let e0 = total_energy(&field, &state);
        let l0 = angular_momentum(&state, Vec3::zero());
        let period = pe_space::kepler::orbital_period(mu, radius);
        let dt = period / 512.0;
        let mut integ = Leapfrog::default();
        let mut t = 0.0;
        for _ in 0..512 {
            let res = integ.step(&field, &mut state, t, dt);
            t = res.t;
        }
        let e1 = total_energy(&field, &state);
        let l1 = angular_momentum(&state, Vec3::zero());
        let de = ((e1 - e0) / e0).abs();
        prop_assert!(de < 1e-8, "energy relative drift {de}");
        prop_assert!((l1 - l0).norm() < 1e-10, "L drift {}", (l1 - l0).norm());
    }

    #[test]
    fn yoshida4_circular_kepler_energy_bound(
        radius in 1.0f64..5.0,
        mu in 0.5f64..2.0,
    ) {
        let v = libm_ops::sqrt(mu / radius);
        let mut state = NBodyState::new(
            vec![1.0],
            vec![Vec3::new(radius, 0.0, 0.0)],
            vec![Vec3::new(0.0, v, 0.0)],
        );
        let field = CentralMu::new(mu);
        let e0 = total_energy(&field, &state);
        let period = pe_space::kepler::orbital_period(mu, radius);
        let dt = period / 256.0;
        let mut integ = Yoshida4::default();
        let mut t = 0.0;
        for _ in 0..(256 * 5) {
            let res = integ.step(&field, &mut state, t, dt);
            t = res.t;
        }
        let e1 = total_energy(&field, &state);
        let de = ((e1 - e0) / e0).abs();
        prop_assert!(de < 1e-10, "Yoshida4 energy drift {de}");
    }

    #[test]
    fn nbody_softened_conserves_l(
        scale in 0.8f64..1.5,
    ) {
        // Equilateral three-body with modest velocities; softening avoids collisions.
        let mut state = NBodyState::new(
            vec![1.0, 1.0, 1.0],
            vec![
                Vec3::new(scale, 0.0, 0.0),
                Vec3::new(-0.5 * scale, 0.8660254037844386 * scale, 0.0),
                Vec3::new(-0.5 * scale, -0.8660254037844386 * scale, 0.0),
            ],
            vec![
                Vec3::new(0.0, 0.3 / scale, 0.0),
                Vec3::new(-0.2598076211353316 / scale, -0.15 / scale, 0.0),
                Vec3::new(0.2598076211353316 / scale, -0.15 / scale, 0.0),
            ],
        );
        state.rebase_to_com();
        let field = NBodyPointMass::with_params(1.0, 0.1);
        let l0 = angular_momentum(&state, Vec3::zero());
        let mut integ = Leapfrog::default();
        let mut t = 0.0;
        for _ in 0..200 {
            let res = integ.step(&field, &mut state, t, 1e-3);
            t = res.t;
        }
        let l1 = angular_momentum(&state, Vec3::zero());
        prop_assert!((l1 - l0).norm() < 1e-9, "L drift {}", (l1 - l0).norm());
    }
}
