//! Translational N-body integrators.
//!
//! | Method | Order | Symplectic | Adaptive | Notes |
//! | --- | --- | --- | --- | --- |
//! | Leapfrog (velocity Verlet) | 2 | yes (pos.-dep.) | no | kick–drift–kick |
//! | Yoshida 4 | 4 | yes | no | composition of leapfrogs (Yoshida 1990) |
//! | RK4 | 4 | no | no | classical |
//! | RKF78 | 7(8) | no | yes | Fehlberg; **DOP853 stand-in** (tableau not reproduced) |
//! | IAS15 | 15 | no | yes | Rein & Spiegel 2015 Gauss–Radau |
//! | WHFast-style | 2 | yes (Kepler split) | no | Wisdom–Holman; Jacobi or democratic-heliocentric |
//!
//! References (papers only; clean-room):
//! - Yoshida, H. (1990). Construction of higher order symplectic integrators.
//!   *Phys. Lett. A* 150.
//! - Fehlberg, E. (1968). Classical fifth-, sixth-, seventh-, and eighth-order
//!   Runge–Kutta formulas with stepsize control. NASA TR R-287.
//! - Rein & Spiegel (2015). IAS15. *MNRAS* 446, 1424 (arXiv:1409.4779).
//! - Rein & Tamayo (2015). WHFast. *MNRAS* 452, 376 (arXiv:1506.01084).
//! - Wisdom & Holman (1991). Symplectic maps for the N-body problem. *AJ* 102.

mod ias15;
mod leapfrog;
mod radau;
mod rk4;
mod rkf78;
mod whfast;
mod yoshida;

pub use ias15::{ias15, Ias15, DEFAULT_EPSILON_B};
pub use leapfrog::{leapfrog, Leapfrog};
pub use radau::{compute_radau_nodes, radau_polynomial, N_RADAU, N_RADAU_FREE, RADAU_H};
pub use rk4::{rk4, Rk4};
pub use rkf78::{dormand_prince_8_7, rkf78, Dop853, Rkf78};
pub use whfast::{whfast, whfast_with, Whfast, WhfastCoords};
pub use yoshida::{yoshida4, Yoshida4};

use pe_math::Vec3;

use crate::field::GravityField;
use crate::state::NBodyState;

pub(crate) fn accel_buf(field: &dyn GravityField, state: &NBodyState, t: f64) -> Vec<Vec3> {
    let mut a = vec![Vec3::zero(); state.n()];
    field.accelerations(state, t, &mut a);
    a
}
