//! Translational N-body integrators.
//!
//! | Method | Order | Symplectic | Adaptive | Notes |
//! | --- | --- | --- | --- | --- |
//! | Leapfrog (velocity Verlet) | 2 | yes (pos.-dep.) | no | kick–drift–kick |
//! | Yoshida 4 | 4 | yes | no | composition of leapfrogs (Yoshida 1990) |
//! | RK4 | 4 | no | no | classical |
//! | RKF78 | 7(8) | no | yes | Fehlberg 1968 |
//! | DOP853 | 8(5,3) | no | yes | Hairer–Nørsett–Wanner; SciPy BSD tables |
//! | IAS15 | 15 | no | yes | Rein & Spiegel 2015 Gauss–Radau |
//! | WHFast-style | 2 (+ correctors) | yes (Kepler split) | no | Wisdom–Holman; Jacobi or democratic-heliocentric |
//!
//! References (papers only; clean-room except DOP853 coefficient tables —
//! see [`THIRD_PARTY.md`](../../../../THIRD_PARTY.md)):
//! - Yoshida, H. (1990). Construction of higher order symplectic integrators.
//!   *Phys. Lett. A* 150.
//! - Fehlberg, E. (1968). Classical fifth-, sixth-, seventh-, and eighth-order
//!   Runge–Kutta formulas with stepsize control. NASA TR R-287.
//! - Hairer, E., Nørsett, S. P. & Wanner, G. *Solving ODEs I* — DOP853.
//! - Rein & Spiegel (2015). IAS15. *MNRAS* 446, 1424 (arXiv:1409.4779).
//! - Rein & Tamayo (2015). WHFast. *MNRAS* 452, 376 (arXiv:1506.01084).
//! - Wisdom (2006). Symplectic correctors for canonical heliocentric n-body maps.
//! - Wisdom & Holman (1991). Symplectic maps for the N-body problem. *AJ* 102.

mod dop853;
mod ias15;
mod leapfrog;
mod radau;
mod rk4;
mod rkf78;
mod whfast;
mod yoshida;

pub use dop853::{
    butcher_consistency_residual, dop853, dormand_prince_8_7, ode_step, ode_step_fixed, Dop853,
    B as DOP853_B, C as DOP853_C, E3 as DOP853_E3, E5 as DOP853_E5, INTERPOLATOR_POWER,
    N_STAGES as DOP853_N_STAGES, N_STAGES_EXTENDED as DOP853_N_STAGES_EXTENDED,
};
pub use ias15::{ias15, Ias15, DEFAULT_EPSILON_B};
pub use leapfrog::{leapfrog, Leapfrog};
pub use radau::{compute_radau_nodes, radau_polynomial, N_RADAU, N_RADAU_FREE, RADAU_H};
pub use rk4::{rk4, Rk4};
pub use rkf78::{rkf78, Rkf78};
pub use whfast::{whfast, whfast_with, Whfast, WhfastCoords, WhfastCorrector};
pub use yoshida::{yoshida4, Yoshida4};

use pe_math::Vec3;

use crate::field::GravityField;
use crate::state::NBodyState;

pub(crate) fn accel_buf(field: &dyn GravityField, state: &NBodyState, t: f64) -> Vec<Vec3> {
    let mut a = vec![Vec3::zero(); state.n()];
    field.accelerations(state, t, &mut a);
    a
}
