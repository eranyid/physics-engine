//! Circular LEO for one period with leapfrog; prints radius error.

use pe_math::libm_ops;
use pe_math::Vec3;
use pe_space::constants::EARTH_MU;
use pe_space::field::CentralMu;
use pe_space::integrator::Integrator;
use pe_space::integrators::Leapfrog;
use pe_space::kepler::orbital_period;
use pe_space::state::NBodyState;

fn main() {
    let r0 = 7_000_000.0;
    let mu = EARTH_MU;
    let v = libm_ops::sqrt(mu / r0);
    let period = orbital_period(mu, r0);
    let mut state = NBodyState::new(
        vec![1.0],
        vec![Vec3::new(r0, 0.0, 0.0)],
        vec![Vec3::new(0.0, v, 0.0)],
    );
    let field = CentralMu::new(mu);
    let steps = 20_000;
    let dt = period / steps as f64;
    let mut integ = Leapfrog::default();
    let mut t = 0.0;
    for _ in 0..steps {
        let res = integ.step(&field, &mut state, t, dt);
        t = res.t;
    }
    let err = (state.x[0].norm() - r0).abs();
    println!("orbit: period={period:.3} s, radius error after 1 orbit = {err:.6} m");
}
