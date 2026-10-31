//! Velocity-Verlet / leapfrog (kick–drift–kick), order 2, symplectic for
//! position-dependent forces.

use pe_math::Vec3;

use crate::field::GravityField;
use crate::integrator::{Integrator, StepResult};
use crate::state::NBodyState;

/// Leapfrog integrator with reusable acceleration scratch (avoids per-step alloc).
#[derive(Clone, Debug, Default)]
pub struct Leapfrog {
    scratch: Vec<Vec3>,
}

/// Free function matching the trait method.
pub fn leapfrog(field: &dyn GravityField, state: &mut NBodyState, t: f64, dt: f64) -> StepResult {
    Leapfrog::default().step(field, state, t, dt)
}

impl Integrator for Leapfrog {
    fn name(&self) -> &'static str {
        "leapfrog"
    }

    fn step(
        &mut self,
        field: &dyn GravityField,
        state: &mut NBodyState,
        t: f64,
        dt: f64,
    ) -> StepResult {
        let n = state.n();
        let h = 0.5 * dt;
        if self.scratch.len() != n {
            self.scratch.resize(n, Vec3::zero());
        }
        field.accelerations(state, t, &mut self.scratch);
        // Half kick
        for i in 0..n {
            state.v[i] += self.scratch[i] * h;
        }
        // Full drift
        for i in 0..n {
            let dx = state.v[i] * dt;
            state.add_position(i, dx);
        }
        // Closing kick at new positions
        field.accelerations(state, t + dt, &mut self.scratch);
        for i in 0..n {
            state.v[i] += self.scratch[i] * h;
        }
        StepResult {
            t: t + dt,
            dt_taken: dt,
            err_est: 0.0,
        }
    }
}
