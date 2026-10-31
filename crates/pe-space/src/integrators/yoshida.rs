//! Fourth-order symplectic integrator by composition of leapfrogs.
//!
//! Coefficients from Yoshida (1990), *Phys. Lett. A* **150**: solution of
//! `x₀ + 2 x₁ = 1`, `x₀³ + 2 x₁³ = 0` giving
//! `x₁ = −2^{1/3}/(2−2^{1/3})`, `x₀ = 1/(2−2^{1/3})`.
//!
//! Clean-room from the published formulas (no third-party source code).

use pe_math::libm_ops;

use crate::field::GravityField;
use crate::integrator::{Integrator, StepResult};
use crate::state::NBodyState;

use super::leapfrog::Leapfrog;

/// Yoshida 4th-order composition.
#[derive(Clone, Debug, Default)]
pub struct Yoshida4 {
    inner: Leapfrog,
}

/// Free function.
pub fn yoshida4(field: &dyn GravityField, state: &mut NBodyState, t: f64, dt: f64) -> StepResult {
    Yoshida4::default().step(field, state, t, dt)
}

fn yoshida_coeffs() -> (f64, f64) {
    // w1 = -2^(1/3) / (2 - 2^(1/3)) ; w0 = 1 / (2 - 2^(1/3))
    let two_cbrt = libm_ops::cbrt(2.0);
    let denom = 2.0 - two_cbrt;
    let w0 = 1.0 / denom;
    let w1 = -two_cbrt / denom;
    (w0, w1)
}

impl Integrator for Yoshida4 {
    fn name(&self) -> &'static str {
        "yoshida4"
    }

    fn step(
        &mut self,
        field: &dyn GravityField,
        state: &mut NBodyState,
        t: f64,
        dt: f64,
    ) -> StepResult {
        // Literature: w₁ = 1/(2−2^{1/3}), w₀ = −2^{1/3}/(2−2^{1/3}); sequence w₁,w₀,w₁.
        // Local names: (pos, neg) = (w₁, w₀).
        let (pos, neg) = yoshida_coeffs();
        let mut tt = t;
        for &w in &[pos, neg, pos] {
            let res = self.inner.step(field, state, tt, w * dt);
            tt = res.t;
        }
        StepResult {
            t: tt,
            dt_taken: dt,
            err_est: 0.0,
        }
    }
}
