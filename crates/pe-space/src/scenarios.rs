//! Canonical scenarios for determinism hashing and demos.

use pe_math::Vec3;

use crate::diagnostics::state_hash64;
use crate::field::NBodyPointMass;
use crate::integrator::Integrator;
use crate::integrators::Leapfrog;
use crate::state::NBodyState;

/// Chenciner–Montgomery figure-eight initial conditions (G = m = 1).
///
/// Positions: `x₁ = (0.97000436, −0.24308753)`, `x₂ = −x₁`, `x₃ = 0`;
/// velocities: `v₃ = (−0.93240737, −0.86473146)`, `v₁ = v₂ = −v₃/2`;
/// period `T ≈ 6.32591398` (Simó / Chenciner–Montgomery).
pub fn figure_eight_state() -> NBodyState {
    let x1 = Vec3::new(0.970_004_36, -0.243_087_53, 0.0);
    let x2 = Vec3::new(-0.970_004_36, 0.243_087_53, 0.0);
    let x3 = Vec3::zero();
    let v3 = Vec3::new(-0.932_407_37, -0.864_731_46, 0.0);
    let v12 = v3 * (-0.5);
    NBodyState::new(vec![1.0, 1.0, 1.0], vec![x1, x2, x3], vec![v12, v12, v3])
}

/// Figure-eight period used by the hash scenario [dimensionless time].
pub const FIGURE_EIGHT_PERIOD: f64 = 6.325_913_98;

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
