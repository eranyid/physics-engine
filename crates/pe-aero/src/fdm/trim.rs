//! Steady level-flight trim solver (Newton on α, δe, thrust).

use pe_math::libm_ops;
use pe_math::{Vec3, STANDARD_GRAVITY};

use super::derivatives::ControlInputs;
use super::forces::{air_angles, eval_aero_body, level_attitude, SixDofModel};
use crate::atmosphere::RHO0;

/// Initial guess / bounds for the trim Newton solve.
#[derive(Clone, Copy, Debug)]
pub struct TrimGuess {
    /// Airspeed [m/s].
    pub v: f64,
    /// Initial angle of attack [rad].
    pub alpha: f64,
    /// Initial elevator [rad].
    pub delta_e: f64,
    /// Initial thrust [N].
    pub thrust: f64,
    /// Gravity magnitude [m/s²].
    pub g: f64,
    /// Density override [kg/m³]; `None` → model density at z=0 / constant.
    pub rho: Option<f64>,
}

impl Default for TrimGuess {
    fn default() -> Self {
        Self {
            v: 50.0,
            alpha: 0.05,
            delta_e: 0.0,
            thrust: 500.0,
            g: STANDARD_GRAVITY,
            rho: Some(RHO0),
        }
    }
}

/// Outcome of a trim attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimStatus {
    /// Residuals within tolerance.
    Converged,
    /// Hit iteration limit.
    MaxIters,
}

/// Steady wings-level trim: `Σ F_x = 0`, `Σ F_z = 0`, `M = 0` in body axes
/// with `β = p = q = r = 0`, flight-path angle ≈ 0.
#[derive(Clone, Copy, Debug)]
pub struct TrimResult {
    /// Trim airspeed [m/s].
    pub v: f64,
    /// Trim angle of attack [rad] (absolute).
    pub alpha: f64,
    /// Elevator [rad].
    pub delta_e: f64,
    /// Thrust [N].
    pub thrust: f64,
    /// Lift coefficient at trim.
    pub cl: f64,
    /// Drag coefficient at trim.
    pub cd: f64,
    /// Pitching-moment coefficient at trim.
    pub cm: f64,
    /// Lift-to-drag ratio.
    pub l_over_d: f64,
    /// Max |residual| force [N] / moment [N m] after solve.
    pub residual_norm: f64,
    /// Solver status.
    pub status: TrimStatus,
}

fn residual(
    model: &SixDofModel,
    v: f64,
    alpha: f64,
    de: f64,
    thrust: f64,
    rho: f64,
    g: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    // Body velocity for level flight at AoA α: u = V cosα, w = −V sinα.
    let v_body = Vec3::new(libm_ops::cos(alpha) * v, 0.0, -libm_ops::sin(alpha) * v);
    let mut m = model.clone();
    m.thrust = thrust;
    m.controls = ControlInputs {
        delta_e: de,
        delta_a: 0.0,
        delta_r: 0.0,
    };
    let loads = eval_aero_body(&m, v_body, Vec3::zero(), rho, None);
    // World forces with level attitude (pitch = α): rotate body→world.
    let q = level_attitude(alpha);
    let f_w = q.rotate(loads.force_body);
    // Gravity in world −z.
    let fx = f_w.x; // should be 0 (thrust vs drag)
    let fz = f_w.z - model.mass * g; // should be 0 (lift vs weight)
    let my = loads.moment_body.y; // should be 0
    (fx, fz, my, loads.cl, loads.cd, loads.cm)
}

/// Solve steady level flight for `(α, δe, T)` at fixed airspeed `guess.v`.
///
/// Uses a damped Newton iteration on the three residuals
/// `(F_x, F_z + mg wait — F_z − (−mg), M_y)`. Portable `libm` only.
pub fn trim_steady_level(model: &SixDofModel, guess: TrimGuess) -> TrimResult {
    let rho = guess.rho.unwrap_or_else(|| model.density.density_at(0.0));
    let v = guess.v;
    let mut alpha = guess.alpha;
    let mut de = guess.delta_e;
    let mut thrust = guess.thrust.max(0.0);
    let g = guess.g;
    let eps: f64 = 1e-6;
    let mut status = TrimStatus::MaxIters;
    let mut res_norm = f64::INFINITY;
    let mut cl = 0.0;
    let mut cd = 0.0;
    let mut cm = 0.0;

    for _ in 0..40 {
        let (fx, fz, my, cl_i, cd_i, cm_i) = residual(model, v, alpha, de, thrust, rho, g);
        cl = cl_i;
        cd = cd_i;
        cm = cm_i;
        res_norm = libm_ops::sqrt(fx * fx + fz * fz + my * my);
        if res_norm < 1e-4 {
            status = TrimStatus::Converged;
            break;
        }

        // Numerical Jacobian columns ∂R/∂(α, δe, T).
        let r0 = [fx, fz, my];
        let mut jac = [[0.0; 3]; 3];
        let eps_th: f64 = 1e-3;
        let cols = [(eps, 0.0, 0.0), (0.0, eps, 0.0), (0.0, 0.0, eps_th)];
        for (col, (dalpha, dde, dth)) in cols.iter().enumerate() {
            let (fx2, fz2, my2, _, _, _) =
                residual(model, v, alpha + dalpha, de + dde, thrust + dth, rho, g);
            let denom = if col == 2 {
                *dth
            } else if col == 1 {
                *dde
            } else {
                *dalpha
            };
            jac[0][col] = (fx2 - r0[0]) / denom;
            jac[1][col] = (fz2 - r0[1]) / denom;
            jac[2][col] = (my2 - r0[2]) / denom;
        }

        // Solve J Δ = −R (3×3 Gaussian elimination).
        let mut a = jac;
        let mut b = [-r0[0], -r0[1], -r0[2]];
        for i in 0..3 {
            // Pivot
            let mut piv = i;
            for r in i + 1..3 {
                if libm_ops::fabs(a[r][i]) > libm_ops::fabs(a[piv][i]) {
                    piv = r;
                }
            }
            if piv != i {
                a.swap(i, piv);
                b.swap(i, piv);
            }
            let diag = a[i][i];
            if libm_ops::fabs(diag) < 1e-14 {
                break;
            }
            for c in i..3 {
                a[i][c] /= diag;
            }
            b[i] /= diag;
            for r in 0..3 {
                if r == i {
                    continue;
                }
                let f = a[r][i];
                for c in i..3 {
                    a[r][c] -= f * a[i][c];
                }
                b[r] -= f * b[i];
            }
        }

        let damp = 0.5;
        alpha += damp * b[0];
        de += damp * b[1];
        thrust = (thrust + damp * b[2]).max(0.0);
        // Keep α in a sane band.
        alpha = alpha.clamp(-0.35, 0.35);
        de = de.clamp(-0.5, 0.5);
    }

    let ld = if libm_ops::fabs(cd) < 1e-15 {
        f64::INFINITY
    } else {
        cl / cd
    };

    // Sanity: air angles at the trimmed state.
    let v_body = Vec3::new(libm_ops::cos(alpha) * v, 0.0, -libm_ops::sin(alpha) * v);
    let (a_check, _, _) = air_angles(v_body);
    debug_assert!((a_check - alpha).abs() < 1e-12);

    TrimResult {
        v,
        alpha,
        delta_e: de,
        thrust,
        cl,
        cd,
        cm,
        l_over_d: ld,
        residual_norm: res_norm,
        status,
    }
}
