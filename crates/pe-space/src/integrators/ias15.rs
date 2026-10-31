//! IAS15: 15th-order adaptive Gauss–Radau integrator (Rein & Spiegel 2015).
//!
//! Clean-room from Rein & Spiegel, *MNRAS* **446**, 1424 (2015), arXiv:1409.4779
//! — formulas only. **Do not** translate REBOUND (GPL) source.
//!
//! # Algorithm summary
//!
//! Expand acceleration on a step of size `dt` as
//! `y''[h] ≈ y0'' + Σ_{k=0}^{6} b_k h^{k+1}` with dimensionless `h = t/dt`,
//! equivalently in Newton form with divided-difference coefficients `g_k` at the
//! Gauss–Radau nodes ([`crate::integrators::radau::RADAU_H`]). Predictor–corrector
//! iterates until the `b` series converges; step-size control uses
//! `b̃_6 = max|b_6| / max|y''|` and `dt_req = dt (ε_b / b̃_6)^{1/7}` (paper eqs.
//! 9–11). End-of-step position/velocity updates use compensated (Neumaier)
//! summation when the state has compensation enabled.
//!
//! Default `ε_b = 1e-9` follows the paper’s machine-precision regime.

use pe_math::libm_ops;
use pe_math::Vec3;

use crate::field::GravityField;
use crate::integrator::{AdaptiveStep, Integrator, StepResult};
use crate::integrators::radau::{N_RADAU, N_RADAU_FREE, RADAU_H};
use crate::state::NBodyState;

/// Default dimensionless accuracy parameter `ε_b` (Rein & Spiegel 2015).
pub const DEFAULT_EPSILON_B: f64 = 1e-9;

/// IAS15 adaptive Gauss–Radau integrator.
#[derive(Clone, Debug)]
pub struct Ias15 {
    /// Step-size / tolerance controller (`atol`/`rtol` unused; `ε_b` drives control).
    pub ctrl: AdaptiveStep,
    /// Dimensionless accuracy `ε_b` (paper eq. 11).
    pub epsilon_b: f64,
    /// Last accepted step size [s] (for `b`-coefficient rescaling).
    last_dt: f64,
    /// Predictor `b` coefficients `[body][k]` after last accepted step.
    b: Vec<[Vec3; N_RADAU_FREE]>,
    /// Extrapolation corrections `e_k` (paper §2.1).
    e: Vec<[Vec3; N_RADAU_FREE]>,
    /// Acceleration at the start of the current step.
    a0: Vec<Vec3>,
    /// Whether the next step is the first (cold start).
    first: bool,
    /// Scratch for force evaluations.
    a_scratch: Vec<Vec3>,
    /// Conversion matrix `b = M g` (precomputed from [`RADAU_H`]).
    m_bg: [[f64; N_RADAU_FREE]; N_RADAU_FREE],
}

impl Default for Ias15 {
    fn default() -> Self {
        Self::new(AdaptiveStep {
            atol: 1e-9,
            rtol: 1e-9,
            dt_min: 1e-16,
            dt_max: 1e6,
            dt: 1e-2,
        })
    }
}

/// Construct IAS15 with default `ε_b`.
pub fn ias15() -> Ias15 {
    Ias15::default()
}

impl Ias15 {
    /// Construct with an adaptive controller (initial `dt` used as first trial).
    pub fn new(ctrl: AdaptiveStep) -> Self {
        Self {
            ctrl,
            epsilon_b: DEFAULT_EPSILON_B,
            last_dt: ctrl.dt,
            b: Vec::new(),
            e: Vec::new(),
            a0: Vec::new(),
            first: true,
            a_scratch: Vec::new(),
            m_bg: build_b_from_g_matrix(&RADAU_H),
        }
    }

    fn ensure_buffers(&mut self, n: usize) {
        if self.b.len() != n {
            self.b = vec![[Vec3::zero(); N_RADAU_FREE]; n];
            self.e = vec![[Vec3::zero(); N_RADAU_FREE]; n];
            self.a0 = vec![Vec3::zero(); n];
            self.a_scratch = vec![Vec3::zero(); n];
            self.first = true;
        }
    }

    /// Predict position and velocity at dimensionless substep `h` for body `i`.
    fn predict_body(
        r0: Vec3,
        v0: Vec3,
        a0: Vec3,
        b: &[Vec3; N_RADAU_FREE],
        h: f64,
        dt: f64,
    ) -> (Vec3, Vec3) {
        // Velocity nesting (paper eq. 6): h/2, 2h/3, 3h/4, …, 7h/8
        let mut s = b[6];
        s = b[5] + s * (7.0 * h / 8.0);
        s = b[4] + s * (6.0 * h / 7.0);
        s = b[3] + s * (5.0 * h / 6.0);
        s = b[2] + s * (4.0 * h / 5.0);
        s = b[1] + s * (3.0 * h / 4.0);
        s = b[0] + s * (2.0 * h / 3.0);
        s = a0 + s * (h / 2.0);
        let vh = v0 + s * (h * dt);

        // Position nesting — coefficients  h/3, h/2, 3h/5, 2h/3, 5h/7, 3h/4, 7h/9
        // (double integral of the power series; the paper’s “…” hides the varying
        // factors after the first two).
        let mut s = b[6];
        s = b[5] + s * (7.0 * h / 9.0);
        s = b[4] + s * (3.0 * h / 4.0);
        s = b[3] + s * (5.0 * h / 7.0);
        s = b[2] + s * (2.0 * h / 3.0);
        s = b[1] + s * (3.0 * h / 5.0);
        s = b[0] + s * (h / 2.0);
        s = a0 + s * (h / 3.0);
        let rh = r0 + v0 * (h * dt) + s * (0.5 * h * h * dt * dt);
        (rh, vh)
    }

    fn g_to_b(&self, g: &[[Vec3; N_RADAU_FREE]], n_bodies: usize) -> Vec<[Vec3; N_RADAU_FREE]> {
        let mut b = vec![[Vec3::zero(); N_RADAU_FREE]; n_bodies];
        for i in 0..n_bodies {
            for row in 0..N_RADAU_FREE {
                let mut acc = Vec3::zero();
                for col in 0..N_RADAU_FREE {
                    acc += g[i][col] * self.m_bg[row][col];
                }
                b[i][row] = acc;
            }
        }
        b
    }

    fn max_norm_acc(a: &[Vec3]) -> f64 {
        let mut m = 0.0_f64;
        for ai in a {
            let n = ai.norm();
            if n > m {
                m = n;
            }
        }
        if m == 0.0 {
            1e-300
        } else {
            m
        }
    }

    fn max_norm_b6(b: &[[Vec3; N_RADAU_FREE]]) -> f64 {
        let mut m = 0.0_f64;
        for bi in b {
            let n = bi[6].norm();
            if n > m {
                m = n;
            }
        }
        m
    }

    fn max_delta_b(a: &[[Vec3; N_RADAU_FREE]], c: &[[Vec3; N_RADAU_FREE]]) -> f64 {
        let mut m = 0.0_f64;
        for i in 0..a.len() {
            for k in 0..N_RADAU_FREE {
                let d = (a[i][k] - c[i][k]).norm();
                if d > m {
                    m = d;
                }
            }
        }
        m
    }
}

impl Integrator for Ias15 {
    fn name(&self) -> &'static str {
        "ias15"
    }

    fn step(
        &mut self,
        field: &dyn GravityField,
        state: &mut NBodyState,
        t: f64,
        dt: f64,
    ) -> StepResult {
        let n = state.n();
        if n == 0 || dt <= 0.0 {
            return StepResult {
                t,
                dt_taken: 0.0,
                err_est: 0.0,
            };
        }
        self.ensure_buffers(n);

        // Acceleration at the beginning of the step.
        field.accelerations(state, t, &mut self.a0);

        let mut dt_try = dt.clamp(self.ctrl.dt_min, self.ctrl.dt_max);

        for _attempt in 0..40 {
            // Predictor: previous b + e, rescaled to dt_try.
            let mut b_pred = vec![[Vec3::zero(); N_RADAU_FREE]; n];
            if !self.first && self.last_dt > 0.0 {
                let fac = dt_try / self.last_dt;
                for i in 0..n {
                    for k in 0..N_RADAU_FREE {
                        let f = libm_ops::pow(fac, (k + 1) as f64);
                        b_pred[i][k] = (self.b[i][k] + self.e[i][k]) * f;
                    }
                }
            }
            let mut b_work = b_pred.clone();

            // Predictor–corrector
            for _it in 0..25 {
                let mut g = vec![[Vec3::zero(); N_RADAU_FREE]; n];
                // Temporary state buffer for substep force evals
                let mut sub = state.clone();
                for node in 1..N_RADAU {
                    let h = RADAU_H[node];
                    for i in 0..n {
                        let (rh, vh) = Self::predict_body(
                            state.x[i], state.v[i], self.a0[i], &b_work[i], h, dt_try,
                        );
                        sub.set_position(i, rh);
                        sub.v[i] = vh;
                    }
                    field.accelerations(&sub, t + h * dt_try, &mut self.a_scratch);
                    // Divided differences → g (paper eq. 5)
                    for i in 0..n {
                        let mut num = self.a_scratch[i] - self.a0[i];
                        let mut prod = RADAU_H[node];
                        for j in 0..node - 1 {
                            num -= g[i][j] * prod;
                            prod *= RADAU_H[node] - RADAU_H[j + 1];
                        }
                        g[i][node - 1] = num / prod;
                    }
                }
                let b_new = self.g_to_b(&g, n);
                let delta = Self::max_delta_b(&b_new, &b_work);
                b_work = b_new;
                let anorm = Self::max_norm_acc(&self.a0);
                if delta <= 1e-15 * anorm {
                    break;
                }
            }

            let anorm = Self::max_norm_acc(&self.a0);
            let b6t = Self::max_norm_b6(&b_work) / anorm;
            let err_est = b6t;
            let dt_req = if b6t < 1e-300 {
                dt_try * 2.0
            } else {
                dt_try * libm_ops::pow(self.epsilon_b / b6t, 1.0 / 7.0)
            };
            let dt_req = dt_req.clamp(self.ctrl.dt_min, self.ctrl.dt_max);

            if dt_try > dt_req * (1.0 + 1e-12) {
                dt_try = (dt_req * 0.9).max(self.ctrl.dt_min);
                continue;
            }

            // Accept: advance state with series at h = 1.
            for i in 0..n {
                let mut s = self.a0[i];
                for k in 0..N_RADAU_FREE {
                    s += b_work[i][k] * (1.0 / ((k + 2) as f64));
                }
                let dv = s * dt_try;
                // Position: r += v*dt + dt²/2 * (a0 + Σ b_k * 2/((k+2)(k+3)))
                let mut sp = self.a0[i];
                for k in 0..N_RADAU_FREE {
                    let kk = (k + 2) as f64;
                    sp += b_work[i][k] * (2.0 / (kk * (kk + 1.0)));
                }
                let dr = state.v[i] * dt_try + sp * (0.5 * dt_try * dt_try);
                state.add_position(i, dr);
                state.v[i] += dv;
            }

            // Store e = b_final − b_predicted; keep b for next rescale.
            for i in 0..n {
                for k in 0..N_RADAU_FREE {
                    self.e[i][k] = b_work[i][k] - b_pred[i][k];
                    self.b[i][k] = b_work[i][k];
                }
            }
            self.last_dt = dt_try;
            // Suggest next step (grow at most ×2).
            self.ctrl.dt = dt_req.min(dt_try * 2.0).min(self.ctrl.dt_max);
            self.first = false;

            return StepResult {
                t: t + dt_try,
                dt_taken: dt_try,
                err_est,
            };
        }

        // Failed to find an acceptable step — return zero advance.
        StepResult {
            t,
            dt_taken: 0.0,
            err_est: -1.0,
        }
    }

    fn integrate_to(
        &mut self,
        field: &dyn GravityField,
        state: &mut NBodyState,
        t0: f64,
        t_end: f64,
        dt: f64,
    ) -> f64 {
        let mut t = t0;
        if self.ctrl.dt <= 0.0 {
            self.ctrl.dt = dt;
        }
        while t < t_end {
            let remain = t_end - t;
            let h = self.ctrl.dt.min(remain).min(self.ctrl.dt_max);
            if h <= 0.0 {
                break;
            }
            let res = self.step(field, state, t, h);
            if res.dt_taken <= 0.0 {
                break;
            }
            t = res.t;
            state.after_step();
        }
        t
    }
}

/// Build the linear map `b = M g` by expanding the Newton basis
/// `φ_k(h) = Π_{j=0}^{k−1}(h − h_j)` into powers `h, h², …, h⁷`.
fn build_b_from_g_matrix(h: &[f64; N_RADAU]) -> [[f64; N_RADAU_FREE]; N_RADAU_FREE] {
    let mut m = [[0.0; N_RADAU_FREE]; N_RADAU_FREE];
    for k in 1..=N_RADAU_FREE {
        // coeffs of φ_k as power series: c[m] = coeff of h^m
        let mut c = [0.0; N_RADAU];
        c[0] = 1.0;
        for (deg, _) in (0..k).enumerate() {
            // multiply by (h - h[deg])
            let mut next = [0.0; N_RADAU];
            for m in 0..=deg {
                next[m] -= c[m] * h[deg];
                next[m + 1] += c[m];
            }
            c = next;
        }
        for power in 1..=N_RADAU_FREE {
            m[power - 1][k - 1] = c[power];
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::EARTH_MU;
    use crate::diagnostics::total_energy;
    use crate::field::CentralMu;
    use crate::kepler::{elements_to_state, orbital_period, KeplerElements};

    #[test]
    fn ias15_eccentric_kepler_energy_machine_precision() {
        let mu = EARTH_MU;
        let a = 1.0e10;
        let e = 0.9;
        let el = KeplerElements {
            a,
            e,
            i: 0.1,
            raan: 0.0,
            arg_p: 0.0,
            true_anomaly: 0.0,
        };
        let (r, v) = elements_to_state(mu, el).unwrap();
        let mut state = NBodyState::new(vec![1.0], vec![r], vec![v]);
        state.enable_compensated_positions();
        let field = CentralMu::new(mu);
        let e0 = total_energy(&field, &state);
        let period = orbital_period(mu, a);
        let mut integ = Ias15::new(AdaptiveStep {
            dt: period / 200.0,
            dt_min: 1e-6,
            dt_max: period / 10.0,
            ..AdaptiveStep::default()
        });
        let t_end = 1000.0 * period;
        integ.integrate_to(&field, &mut state, 0.0, t_end, period / 200.0);
        let e1 = total_energy(&field, &state);
        let rel = libm_ops::fabs((e1 - e0) / e0);
        assert!(
            rel < 1e-12,
            "IAS15 e=0.9 / 1000 orbits relative energy {rel}"
        );
    }
}
