//! Integrator interface for translational N-body systems.

use crate::field::GravityField;
use crate::state::NBodyState;

/// Result of one integrator advance (fixed or adaptive).
#[derive(Clone, Debug)]
pub struct StepResult {
    /// Simulation time after the step [s].
    pub t: f64,
    /// Step size actually taken [s].
    pub dt_taken: f64,
    /// Estimated local error (adaptive methods); `0` for fixed-step.
    pub err_est: f64,
}

/// Adaptive controller settings (tolerances are absolute on position [m] /
/// velocity [m/s] mixed via a simple RMS norm — see each adaptive method).
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveStep {
    /// Absolute tolerance.
    pub atol: f64,
    /// Relative tolerance.
    pub rtol: f64,
    /// Minimum allowed step [s].
    pub dt_min: f64,
    /// Maximum allowed step [s].
    pub dt_max: f64,
    /// Initial / current step proposal [s].
    pub dt: f64,
}

impl Default for AdaptiveStep {
    fn default() -> Self {
        Self {
            atol: 1e-10,
            rtol: 1e-10,
            dt_min: 1e-14,
            dt_max: 1e6,
            dt: 1e-2,
        }
    }
}

/// N-body integrator.
///
/// Implementations advance only translational DOF `(x, v)`. Rotational rigid-body
/// state lives in `pe-core`; this module is for orbital / N-body problems.
pub trait Integrator: Send + Sync {
    /// Human-readable name (for bindings / logs).
    fn name(&self) -> &'static str;

    /// Advance `state` by one step. `dt` is the proposed step [s]; adaptive
    /// methods may take a different `dt_taken` (returned in [`StepResult`]).
    ///
    /// `t` is the time at the beginning of the step [s].
    fn step(
        &mut self,
        field: &dyn GravityField,
        state: &mut NBodyState,
        t: f64,
        dt: f64,
    ) -> StepResult;

    /// Integrate until `t_end` [s] with fixed `dt` [s] (or adaptive internals).
    fn integrate_to(
        &mut self,
        field: &dyn GravityField,
        state: &mut NBodyState,
        t0: f64,
        t_end: f64,
        dt: f64,
    ) -> f64 {
        let mut t = t0;
        while t < t_end {
            let h = if t + dt > t_end { t_end - t } else { dt };
            if h <= 0.0 {
                break;
            }
            let res = self.step(field, state, t, h);
            t = res.t;
            state.after_step();
        }
        t
    }
}
