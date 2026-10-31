//! Documented stubs for IAS15 and WHFast-style Wisdom–Holman.
//!
//! Full implementations are deferred (tracked TODOs below). Calling
//! [`Integrator::step`] on these types returns a clear error-path via a
//! fixed-step leapfrog fallback **only when** `allow_fallback` is set; by
//! default they panic in debug and return a zero-length step in release after
//! logging via the `last_status` field — actually we keep them as explicit
//! stubs that do **not** silently integrate: `step` leaves the state unchanged
//! and reports `dt_taken = 0` with `err_est = −1` as a sentinel.
//!
//! # TODO(M7-followup): IAS15
//!
//! Implement Rein & Spiegel (2015), *MNRAS* **446**, 1424 — Gauss–Radau 15th-order
//! adaptive integrator with compensated summation. Clean-room from the paper
//! only (do **not** translate REBOUND source).
//!
//! # TODO(M7-followup): WHFast
//!
//! Implement Wisdom & Holman (1991) symplectic map with universal-variable
//! Kepler drift (Stumpff / Danby–Newton start — helpers already in
//! [`crate::kepler`]). Clean-room; do **not** translate REBOUND WHFast.

use crate::field::GravityField;
use crate::integrator::{Integrator, StepResult};
use crate::state::NBodyState;

/// IAS15 stub (Rein & Spiegel 2015). Not yet integrating.
#[derive(Clone, Debug, Default)]
pub struct Ias15 {
    /// Human-readable status after the last `step` call.
    pub last_status: &'static str,
}

/// Construct the IAS15 stub.
pub fn ias15() -> Ias15 {
    Ias15::default()
}

impl Integrator for Ias15 {
    fn name(&self) -> &'static str {
        "ias15_stub"
    }

    fn step(
        &mut self,
        _field: &dyn GravityField,
        _state: &mut NBodyState,
        t: f64,
        _dt: f64,
    ) -> StepResult {
        self.last_status =
            "IAS15 stub: full Gauss-Radau 15th-order adaptive not yet implemented (TODO M7-followup; Rein & Spiegel 2015)";
        StepResult {
            t,
            dt_taken: 0.0,
            err_est: -1.0,
        }
    }
}

/// WHFast-style Wisdom–Holman stub. Universal Kepler helpers exist in
/// [`crate::kepler`]; the symplectic split map is not wired yet.
#[derive(Clone, Debug, Default)]
pub struct Whfast {
    /// Dominant-body gravitational parameter μ [m³/s²] (reserved for future drift).
    pub mu: f64,
    /// Status after last `step`.
    pub last_status: &'static str,
}

/// Construct the WHFast stub with primary μ [m³/s²].
pub fn whfast(mu: f64) -> Whfast {
    Whfast {
        mu,
        last_status: "",
    }
}

impl Integrator for Whfast {
    fn name(&self) -> &'static str {
        "whfast_stub"
    }

    fn step(
        &mut self,
        _field: &dyn GravityField,
        _state: &mut NBodyState,
        t: f64,
        _dt: f64,
    ) -> StepResult {
        self.last_status =
            "WHFast stub: Wisdom-Holman symplectic map not yet implemented (TODO M7-followup; uses pe_space::kepler Stumpff helpers when landed)";
        let _ = self.mu;
        StepResult {
            t,
            dt_taken: 0.0,
            err_est: -1.0,
        }
    }
}
