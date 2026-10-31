//! Translational N-body integrators.
//!
//! | Method | Order | Symplectic | Adaptive | Notes |
//! | --- | --- | --- | --- | --- |
//! | Leapfrog (velocity Verlet) | 2 | yes (pos.-dep.) | no | kick–drift–kick |
//! | Yoshida 4 | 4 | yes | no | composition of leapfrogs (Yoshida 1990) |
//! | RK4 | 4 | no | no | classical |
//! | RKF78 | 7(8) | no | yes | Fehlberg; **chosen over full DOP853** for coefficient size |
//! | IAS15 | 15 | no | yes | **stub** — Rein & Spiegel 2015 tracked TODO |
//! | WHFast-style | ~2 | yes (Kepler split) | no | **stub** — Wisdom–Holman + universal drift TODO |
//!
//! References (papers only; clean-room):
//! - Yoshida, H. (1990). Construction of higher order symplectic integrators.
//!   *Phys. Lett. A* 150.
//! - Fehlberg, E. (1968). Classical fifth-, sixth-, seventh-, and eighth-order
//!   Runge–Kutta formulas with stepsize control. NASA TR R-287.
//! - Rein & Spiegel (2015). IAS15. *MNRAS* 446, 1424 (stub only).
//! - Wisdom & Holman (1991). Symplectic maps for the N-body problem. *AJ* 102.

mod leapfrog;
mod rk4;
mod rkf78;
mod stubs;
mod yoshida;

pub use leapfrog::{leapfrog, Leapfrog};
pub use rk4::{rk4, Rk4};
pub use rkf78::{dormand_prince_8_7, rkf78, Dop853, Rkf78};
pub use stubs::{ias15, whfast, Ias15, Whfast};
pub use yoshida::{yoshida4, Yoshida4};

use pe_math::Vec3;

use crate::field::GravityField;
use crate::state::NBodyState;

pub(crate) fn accel_buf(field: &dyn GravityField, state: &NBodyState, t: f64) -> Vec<Vec3> {
    let mut a = vec![Vec3::zero(); state.n()];
    field.accelerations(state, t, &mut a);
    a
}
