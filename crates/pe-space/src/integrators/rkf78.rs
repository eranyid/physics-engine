//! Adaptive Runge–Kutta–Fehlberg 7(8) (RKF78).
//!
//! Classical **Fehlberg 7(8)** embedded pair (NASA TR R-287 / Fehlberg 1968).
//! For the Hairer–Nørsett–Wanner Dormand–Prince 8(5,3) method see [`super::dop853`].
//!
//! Step-size control uses a mixed absolute/relative RMS error on the `(x, v)`
//! state vector.

use pe_math::libm_ops;
use pe_math::Vec3;

use crate::field::GravityField;
use crate::integrator::{AdaptiveStep, Integrator, StepResult};
use crate::state::NBodyState;

use super::accel_buf;

/// Adaptive RKF78 controller.
#[derive(Clone, Debug, Default)]
pub struct Rkf78 {
    /// Step-size / tolerance settings.
    pub ctrl: AdaptiveStep,
}

/// Free-function one-shot with default tolerances.
pub fn rkf78(field: &dyn GravityField, state: &mut NBodyState, t: f64, dt: f64) -> StepResult {
    let mut integ = Rkf78 {
        ctrl: AdaptiveStep {
            dt,
            ..AdaptiveStep::default()
        },
    };
    integ.step(field, state, t, dt)
}

// Fehlberg 7(8): 13 stages. A_{i,j}, c_i, b8_j (order 8), b7_j (order 7).
// Fractions from Fehlberg (1968) / NASA TR R-287 as commonly tabulated.

fn feh_c() -> [f64; 13] {
    [
        0.0,
        2.0 / 27.0,
        1.0 / 9.0,
        1.0 / 6.0,
        5.0 / 12.0,
        0.5,
        5.0 / 6.0,
        1.0 / 6.0,
        2.0 / 3.0,
        1.0 / 3.0,
        1.0,
        0.0,
        1.0,
    ]
}

/// Lower-triangular A: `a[i][j]` for j < i. Stored densely per row.
#[allow(clippy::excessive_precision)]
fn feh_a(i: usize, j: usize) -> f64 {
    // Only defined for j < i; return 0 otherwise.
    const A: &[&[f64]] = &[
        &[],
        &[2.0 / 27.0],
        &[1.0 / 36.0, 1.0 / 12.0],
        &[1.0 / 24.0, 0.0, 1.0 / 8.0],
        &[5.0 / 12.0, 0.0, -25.0 / 16.0, 25.0 / 16.0],
        &[0.05, 0.0, 0.0, 0.25, 0.2],
        &[
            -25.0 / 108.0,
            0.0,
            0.0,
            125.0 / 108.0,
            -65.0 / 27.0,
            125.0 / 54.0,
        ],
        &[
            31.0 / 300.0,
            0.0,
            0.0,
            0.0,
            61.0 / 225.0,
            -2.0 / 9.0,
            13.0 / 900.0,
        ],
        &[
            2.0,
            0.0,
            0.0,
            -53.0 / 6.0,
            704.0 / 45.0,
            -107.0 / 9.0,
            67.0 / 90.0,
            3.0,
        ],
        &[
            -91.0 / 108.0,
            0.0,
            0.0,
            23.0 / 108.0,
            -976.0 / 135.0,
            311.0 / 54.0,
            -19.0 / 60.0,
            17.0 / 6.0,
            -1.0 / 12.0,
        ],
        &[
            2383.0 / 4100.0,
            0.0,
            0.0,
            -341.0 / 164.0,
            4496.0 / 1025.0,
            -301.0 / 82.0,
            2133.0 / 4100.0,
            45.0 / 82.0,
            45.0 / 164.0,
            18.0 / 41.0,
        ],
        &[
            3.0 / 205.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -6.0 / 41.0,
            -3.0 / 205.0,
            -3.0 / 41.0,
            3.0 / 41.0,
            6.0 / 41.0,
            0.0,
        ],
        &[
            -1777.0 / 4100.0,
            0.0,
            0.0,
            -341.0 / 164.0,
            4496.0 / 1025.0,
            -289.0 / 82.0,
            2193.0 / 4100.0,
            51.0 / 82.0,
            33.0 / 164.0,
            12.0 / 41.0,
            0.0,
            1.0,
        ],
    ];
    if j < A[i].len() {
        A[i][j]
    } else {
        0.0
    }
}

fn feh_b8() -> [f64; 13] {
    [
        41.0 / 840.0,
        0.0,
        0.0,
        0.0,
        0.0,
        34.0 / 105.0,
        9.0 / 35.0,
        9.0 / 35.0,
        9.0 / 280.0,
        9.0 / 280.0,
        41.0 / 840.0,
        0.0,
        0.0,
    ]
}

fn feh_b7() -> [f64; 13] {
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        34.0 / 105.0,
        9.0 / 35.0,
        9.0 / 35.0,
        9.0 / 280.0,
        9.0 / 280.0,
        0.0,
        41.0 / 840.0,
        41.0 / 840.0,
    ]
}

struct Phase {
    x: Vec<Vec3>,
    v: Vec<Vec3>,
}

fn add_scaled(base: &Phase, k: &[(Vec<Vec3>, Vec<Vec3>)], scales: &[f64], h: f64) -> Phase {
    let n = base.x.len();
    let mut x = base.x.clone();
    let mut v = base.v.clone();
    for (s, (kx, kv)) in scales.iter().zip(k.iter()) {
        let f = h * *s;
        for i in 0..n {
            x[i] += kx[i] * f;
            v[i] += kv[i] * f;
        }
    }
    Phase { x, v }
}

fn deriv(field: &dyn GravityField, mass: &[f64], p: &Phase, t: f64) -> (Vec<Vec3>, Vec<Vec3>) {
    let scratch = NBodyState::new(mass.to_vec(), p.x.clone(), p.v.clone());
    let a = accel_buf(field, &scratch, t);
    (p.v.clone(), a)
}

fn err_norm(e_x: &[Vec3], e_v: &[Vec3], x: &[Vec3], v: &[Vec3], atol: f64, rtol: f64) -> f64 {
    let n = e_x.len();
    let mut acc = 0.0;
    let mut count = 0.0;
    for i in 0..n {
        for (e, y) in [
            (e_x[i].x, x[i].x),
            (e_x[i].y, x[i].y),
            (e_x[i].z, x[i].z),
            (e_v[i].x, v[i].x),
            (e_v[i].y, v[i].y),
            (e_v[i].z, v[i].z),
        ] {
            let sc = atol + rtol * libm_ops::fabs(y);
            let w = e / sc;
            acc += w * w;
            count += 1.0;
        }
    }
    libm_ops::sqrt(acc / count)
}

impl Integrator for Rkf78 {
    fn name(&self) -> &'static str {
        "rkf78"
    }

    fn step(
        &mut self,
        field: &dyn GravityField,
        state: &mut NBodyState,
        t: f64,
        dt: f64,
    ) -> StepResult {
        let mut h = dt.clamp(self.ctrl.dt_min, self.ctrl.dt_max);
        let mass = state.mass.clone();
        let n = state.n();
        let c = feh_c();
        let b8 = feh_b8();
        let b7 = feh_b7();

        // At most a few reject/retry cycles per call.
        for _attempt in 0..12 {
            let base = Phase {
                x: state.x.clone(),
                v: state.v.clone(),
            };
            let mut ks: Vec<(Vec<Vec3>, Vec<Vec3>)> = Vec::with_capacity(13);
            for stage in 0..13 {
                let mut scales = [0.0f64; 13];
                for j in 0..stage {
                    scales[j] = feh_a(stage, j);
                }
                let p = add_scaled(&base, &ks, &scales[..stage], h);
                let d = deriv(field, &mass, &p, t + c[stage] * h);
                ks.push(d);
            }

            // Order-8 and order-7 updates
            let mut x8 = base.x.clone();
            let mut v8 = base.v.clone();
            let mut x7 = base.x.clone();
            let mut v7 = base.v.clone();
            let mut e_x = vec![Vec3::zero(); n];
            let mut e_v = vec![Vec3::zero(); n];
            for s in 0..13 {
                let f8 = h * b8[s];
                let f7 = h * b7[s];
                let (ref kx, ref kv) = ks[s];
                for i in 0..n {
                    x8[i] += kx[i] * f8;
                    v8[i] += kv[i] * f8;
                    x7[i] += kx[i] * f7;
                    v7[i] += kv[i] * f7;
                }
            }
            for i in 0..n {
                e_x[i] = x8[i] - x7[i];
                e_v[i] = v8[i] - v7[i];
            }
            let err = err_norm(&e_x, &e_v, &x8, &v8, self.ctrl.atol, self.ctrl.rtol);

            let order = 7.0;
            let mut accept = err <= 1.0;
            if err == 0.0 {
                accept = true;
            }
            // Step-size update
            let factor = if err == 0.0 {
                2.0
            } else {
                0.9 * libm_ops::pow(1.0 / err, 1.0 / (order + 1.0))
            };
            let factor = factor.clamp(0.2, 5.0);
            let h_new = (h * factor).clamp(self.ctrl.dt_min, self.ctrl.dt_max);

            if accept || h <= self.ctrl.dt_min * 1.000_000_1 {
                for i in 0..n {
                    let dx = x8[i] - state.x[i];
                    state.add_position(i, dx);
                    state.v[i] = v8[i];
                }
                self.ctrl.dt = h_new;
                return StepResult {
                    t: t + h,
                    dt_taken: h,
                    err_est: err,
                };
            }
            h = h_new;
        }

        // Force a tiny step if we never accepted (should be rare).
        let h = self.ctrl.dt_min;
        let mut rk = crate::integrators::Rk4;
        let res = rk.step(field, state, t, h);
        self.ctrl.dt = h;
        res
    }
}
