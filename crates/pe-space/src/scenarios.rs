//! Canonical scenarios for determinism hashing and demos.

use pe_math::Vec3;

use crate::diagnostics::state_hash64;
use crate::field::{CentralMu, NBodyPointMass};
use crate::integrator::{AdaptiveStep, Integrator};
use crate::integrators::{Dop853, Ias15, Leapfrog};
use crate::kepler::{elements_to_state, orbital_period, KeplerElements};
use crate::state::NBodyState;

/// Chenciner–Montgomery figure-eight initial conditions (G = m = 1).
///
/// Positions: `x₁ = (0.97000436, −0.24308753)`, `x₂ = −x₁`, `x₃ = 0`;
/// velocities: `v₃ = (−0.93240737, −0.86473146)`, `v₁ = v₂ = −v₃/2`
/// (Simó / Chenciner–Montgomery; 8 published digits).
pub fn figure_eight_state() -> NBodyState {
    let x1 = Vec3::new(0.970_004_36, -0.243_087_53, 0.0);
    let x2 = Vec3::new(-0.970_004_36, 0.243_087_53, 0.0);
    let x3 = Vec3::zero();
    let v3 = Vec3::new(-0.932_407_37, -0.864_731_46, 0.0);
    let v12 = v3 * (-0.5);
    NBodyState::new(vec![1.0, 1.0, 1.0], vec![x1, x2, x3], vec![v12, v12, v3])
}

/// Figure-eight period used by the **hash** scenario [dimensionless time].
///
/// Locked with the leapfrog cross-OS hash — do not change without re-locking
/// `docs/DETERMINISM.md`.
pub const FIGURE_EIGHT_PERIOD: f64 = 6.325_913_98;

/// Period that minimises IAS15 return error for the 8-digit IC above.
///
/// The truncated published digits are not exactly periodic at
/// [`FIGURE_EIGHT_PERIOD`]; this refined value (found by minimising the
/// phase-space return under IAS15) is used only by accuracy tests.
pub const FIGURE_EIGHT_PERIOD_RETURN: f64 = 6.325_914_011_9;

/// Number of leapfrog steps in the canonical cross-OS hash.
pub const FIGURE_EIGHT_HASH_STEPS: usize = 2000;

/// Integrate the figure-eight for [`FIGURE_EIGHT_HASH_STEPS`] leapfrog steps and
/// return the FNV-1a state hash (IEEE bits of packed `x||v`).
pub fn figure_eight_hash() -> u64 {
    let mut state = figure_eight_state();
    let field = NBodyPointMass::geometric();
    let mut integ = Leapfrog::default();
    let dt = FIGURE_EIGHT_PERIOD / FIGURE_EIGHT_HASH_STEPS as f64;
    let mut t = 0.0;
    for _ in 0..FIGURE_EIGHT_HASH_STEPS {
        let res = integ.step(&field, &mut state, t, dt);
        t = res.t;
        state.after_step();
    }
    state_hash64(&state)
}

/// Number of **accepted** IAS15 steps in the hash scenario.
pub const IAS15_HASH_STEPS: usize = 64;

/// IAS15 hash scenario initial state (SI).
pub fn ias15_hash_state() -> NBodyState {
    let mu = crate::constants::EARTH_MU;
    let a = 1.0e7;
    let el = KeplerElements {
        a,
        e: 0.5,
        i: 0.2,
        raan: 0.1,
        arg_p: 0.3,
        true_anomaly: 0.0,
    };
    let (r, v) = elements_to_state(mu, el).expect("ias15 hash IC");
    let mut state = NBodyState::new(vec![1.0], vec![r], vec![v]);
    state.enable_compensated_positions();
    state
}

/// Run [`IAS15_HASH_STEPS`] accepted IAS15 steps and return the FNV-1a state hash.
pub fn ias15_hash() -> u64 {
    let mu = crate::constants::EARTH_MU;
    let mut state = ias15_hash_state();
    let field = CentralMu::new(mu);
    let period = orbital_period(mu, 1.0e7);
    let mut integ = Ias15::new(AdaptiveStep {
        dt: period / 100.0,
        dt_min: 1e-6,
        dt_max: period / 5.0,
        atol: 1e-9,
        rtol: 1e-9,
    });
    let mut t = 0.0;
    let mut accepted = 0usize;
    while accepted < IAS15_HASH_STEPS {
        let res = integ.step(&field, &mut state, t, integ.ctrl.dt);
        if res.dt_taken <= 0.0 {
            break;
        }
        t = res.t;
        state.after_step();
        accepted += 1;
    }
    state_hash64(&state)
}

/// Number of **accepted** DOP853 steps in the hash scenario.
pub const DOP853_HASH_STEPS: usize = 48;

/// DOP853 hash scenario: same IC as IAS15; 48 accepted adaptive steps.
pub fn dop853_hash() -> u64 {
    let mu = crate::constants::EARTH_MU;
    let mut state = ias15_hash_state();
    let field = CentralMu::new(mu);
    let period = orbital_period(mu, 1.0e7);
    let mut integ = Dop853::new(AdaptiveStep {
        dt: period / 80.0,
        dt_min: 1e-6,
        dt_max: period / 5.0,
        atol: 1e-10,
        rtol: 1e-10,
    });
    let mut t = 0.0;
    let mut accepted = 0usize;
    while accepted < DOP853_HASH_STEPS {
        let res = integ.step(&field, &mut state, t, integ.ctrl.dt);
        if res.dt_taken <= 0.0 {
            break;
        }
        t = res.t;
        state.after_step();
        accepted += 1;
    }
    state_hash64(&state)
}
