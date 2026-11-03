//! Orbital long-run stability (M22). `#[ignore]` for default CI.
//!
//! Bound documented in `docs/STABILITY.md`: Yoshida4 e=0.5 over **10⁴ orbits**,
//! relative energy drift `< 1e-9` (stricter than the 1000-orbit `< 1e-10` oracle
//! scaled conservatively for 10× longer).

use pe_math::libm_ops;
use pe_space::constants::EARTH_MU;
use pe_space::diagnostics::total_energy;
use pe_space::field::CentralMu;
use pe_space::integrator::Integrator;
use pe_space::integrators::Yoshida4;
use pe_space::kepler::{elements_to_state, orbital_period, KeplerElements};
use pe_space::state::NBodyState;

#[test]
#[ignore = "slow: 1e4 orbital periods Yoshida4"]
fn stability_yoshida4_1e4_orbits_energy_drift() {
    let mu = EARTH_MU;
    let a = 10_000_000.0;
    let e = 0.5;
    let el = KeplerElements {
        a,
        e,
        i: 0.2,
        raan: 0.1,
        arg_p: 0.3,
        true_anomaly: 0.0,
    };
    let (r, v) = elements_to_state(mu, el).unwrap();
    let field = CentralMu::new(mu);
    let period = orbital_period(mu, a);
    let steps_per_orbit = 4_096usize;
    let dt = period / steps_per_orbit as f64;
    let n_orbits = 10_000usize;

    let mut state = NBodyState::new(vec![1.0], vec![r], vec![v]);
    let e0 = total_energy(&field, &state);
    let mut integ = Yoshida4::default();
    let mut t = 0.0;
    let mut max_rel = 0.0_f64;
    for orbit in 0..n_orbits {
        for _ in 0..steps_per_orbit {
            let res = integ.step(&field, &mut state, t, dt);
            t = res.t;
        }
        let e_now = total_energy(&field, &state);
        let rel = libm_ops::fabs((e_now - e0) / e0);
        if rel > max_rel {
            max_rel = rel;
        }
        if orbit % 1000 == 0 {
            eprintln!("orbit {orbit}: rel_energy={rel:.3e} (max {max_rel:.3e})");
        }
    }
    eprintln!("stability_yoshida4_1e4: max_rel_energy={max_rel:.6e} (bound 1e-9)");
    assert!(
        max_rel < 1e-9,
        "Yoshida4 relative energy over 1e4 orbits: {max_rel}"
    );
}
