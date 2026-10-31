//! Classical 4th-order Runge–Kutta on `(x, v)`. Not symplectic: secular energy
//! drift on long Kepler runs.

use pe_math::Vec3;

use crate::field::GravityField;
use crate::integrator::{Integrator, StepResult};
use crate::state::NBodyState;

use super::accel_buf;

/// Classical RK4.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rk4;

/// Free function.
pub fn rk4(field: &dyn GravityField, state: &mut NBodyState, t: f64, dt: f64) -> StepResult {
    Rk4.step(field, state, t, dt)
}

fn eval_deriv(
    field: &dyn GravityField,
    mass_n: usize,
    x: &[Vec3],
    v: &[Vec3],
    mass: &[f64],
    t: f64,
) -> (Vec<Vec3>, Vec<Vec3>) {
    // Temporary state view via a lightweight NBodyState borrow pattern:
    // build a scratch state for acceleration evaluation.
    let scratch = NBodyState::new(mass.to_vec(), x.to_vec(), v.to_vec());
    let a = accel_buf(field, &scratch, t);
    let mut dx = Vec::with_capacity(mass_n);
    let mut dv = Vec::with_capacity(mass_n);
    for i in 0..mass_n {
        dx.push(v[i]);
        dv.push(a[i]);
    }
    (dx, dv)
}

impl Integrator for Rk4 {
    fn name(&self) -> &'static str {
        "rk4"
    }

    fn step(
        &mut self,
        field: &dyn GravityField,
        state: &mut NBodyState,
        t: f64,
        dt: f64,
    ) -> StepResult {
        let n = state.n();
        let x0 = state.x.clone();
        let v0 = state.v.clone();
        let mass = state.mass.clone();

        let (k1x, k1v) = eval_deriv(field, n, &x0, &v0, &mass, t);

        let mut x = vec![Vec3::zero(); n];
        let mut v = vec![Vec3::zero(); n];
        for i in 0..n {
            x[i] = x0[i] + k1x[i] * (0.5 * dt);
            v[i] = v0[i] + k1v[i] * (0.5 * dt);
        }
        let (k2x, k2v) = eval_deriv(field, n, &x, &v, &mass, t + 0.5 * dt);

        for i in 0..n {
            x[i] = x0[i] + k2x[i] * (0.5 * dt);
            v[i] = v0[i] + k2v[i] * (0.5 * dt);
        }
        let (k3x, k3v) = eval_deriv(field, n, &x, &v, &mass, t + 0.5 * dt);

        for i in 0..n {
            x[i] = x0[i] + k3x[i] * dt;
            v[i] = v0[i] + k3v[i] * dt;
        }
        let (k4x, k4v) = eval_deriv(field, n, &x, &v, &mass, t + dt);

        for i in 0..n {
            let dx = (k1x[i] + k2x[i] * 2.0 + k3x[i] * 2.0 + k4x[i]) * (dt / 6.0);
            let dv = (k1v[i] + k2v[i] * 2.0 + k3v[i] * 2.0 + k4v[i]) * (dt / 6.0);
            state.add_position(i, dx);
            state.v[i] = v0[i] + dv;
        }

        StepResult {
            t: t + dt,
            dt_taken: dt,
            err_est: 0.0,
        }
    }
}
