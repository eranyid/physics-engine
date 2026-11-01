//! Longitudinal state-space linearisation and textbook phugoid / turn helpers.

use pe_math::libm_ops;
use pe_math::{Vec3, STANDARD_GRAVITY};

use super::derivatives::ControlInputs;
use super::eigen::{classify_longitudinal_modes, LongitudinalModes};
use super::forces::{eval_aero_body, SixDofModel};
use super::trim::TrimResult;

/// Longitudinal state-space about a wings-level trim.
///
/// State `x = [u, w, q, θ]ᵀ` (body axial/normal speed [m/s], pitch rate [rad/s],
/// pitch attitude [rad]). `ẋ = A x` (controls fixed at trim).
#[derive(Clone, Debug)]
pub struct LongitudinalStateSpace {
    /// 4×4 system matrix [mixed SI units].
    pub a: [[f64; 4]; 4],
    /// Trim state `[u, w, q, θ]`.
    pub x0: [f64; 4],
    /// Extracted phugoid / short-period modes (if classification succeeds).
    pub modes: Option<LongitudinalModes>,
}

/// Analytic Lanchester / Etkin phugoid approximations:
/// `ω_ph ≈ g √2 / V`, `ζ_ph ≈ C_D / (√2 C_L)`.
///
/// Returns `(omega_n [rad/s], zeta [dimensionless])`.
pub fn phugoid_approximation(v: f64, cl: f64, cd: f64, g: f64) -> (f64, f64) {
    let omega = g * libm_ops::sqrt(2.0) / v.max(1e-9);
    let zeta = if cl.abs() < 1e-15 {
        f64::INFINITY
    } else {
        cd / (libm_ops::sqrt(2.0) * cl)
    };
    (omega, zeta)
}

/// Coordinated-turn load factor `n = 1 / cos φ` [dimensionless].
pub fn coordinated_turn_load_factor(bank_rad: f64) -> f64 {
    1.0 / libm_ops::cos(bank_rad)
}

/// Coordinated-turn rate `ψ̇ = g tan(φ) / V` [rad/s].
pub fn coordinated_turn_rate(bank_rad: f64, v: f64, g: f64) -> f64 {
    g * libm_ops::tan(bank_rad) / v.max(1e-9)
}

fn long_accel(
    model: &SixDofModel,
    u: f64,
    w: f64,
    q: f64,
    theta: f64,
    rho: f64,
    g: f64,
) -> [f64; 4] {
    // Body velocity (v=0); α from (u,w) with z-up: w_body = −V sinα ⇒ here w is
    // the +z body component.
    let v_body = Vec3::new(u, 0.0, w);
    let omega_b = Vec3::new(0.0, q, 0.0);
    let loads = eval_aero_body(model, v_body, omega_b, rho, None);
    let (ixx, iyy, izz) = model.inertia_diag;
    let _ = (ixx, izz);
    // Body-axis Newton–Euler (flat Earth, no Coriolis of navigation frame):
    //   m (ú + q w) = X − m g sinθ   (θ nose-up; our pitch quaternion uses −θ about y
    //                                 so world gravity body components:)
    // With q_att = exp(−θ ŷ), R maps body→world; g_world = (0,0,−g);
    // g_body = Rᵀ g_world.
    let s = libm_ops::sin(theta);
    let c = libm_ops::cos(theta);
    // Pitch-only attitude `Quat::from_axis_angle(ŷ, −θ)` (nose-up θ) gives
    // `g_body = Rᵀ (0,0,−g) = (−g sinθ, 0, −g cosθ)` in the z-up convention.
    let gx = -g * s;
    let gz = -g * c;
    // Body-axis kinematics (z-up): ú = X/m − g sinθ + q w,  ẇ = Z/m − g cosθ − q u.
    let ax = loads.force_body.x / model.mass + gx + q * w;
    let az = loads.force_body.z / model.mass + gz - q * u;
    let q_dot = loads.moment_body.y / iyy;
    // θ̇ = q for wings-level (no bank).
    let theta_dot = q;
    [ax, az, q_dot, theta_dot]
}

/// Numerically linearise longitudinal dynamics about a trim point.
///
/// Builds `A` by central differences (ε = 1e-5 relative / absolute mix).
pub fn linearise_longitudinal(
    model: &SixDofModel,
    trim: &TrimResult,
    rho: f64,
    g: f64,
) -> LongitudinalStateSpace {
    let mut m = model.clone();
    m.thrust = trim.thrust;
    m.controls = ControlInputs {
        delta_e: trim.delta_e,
        delta_a: 0.0,
        delta_r: 0.0,
    };
    let u0 = trim.v * libm_ops::cos(trim.alpha);
    let w0 = -trim.v * libm_ops::sin(trim.alpha);
    let q0 = 0.0;
    let theta0 = trim.alpha; // level flight path ⇒ θ = α
    let x0 = [u0, w0, q0, theta0];

    let f0 = long_accel(&m, u0, w0, q0, theta0, rho, g);
    let mut a = [[0.0; 4]; 4];
    let eps = [1e-4, 1e-4, 1e-5, 1e-5];
    for j in 0..4 {
        let mut xp = x0;
        let mut xm = x0;
        xp[j] += eps[j];
        xm[j] -= eps[j];
        let fp = long_accel(&m, xp[0], xp[1], xp[2], xp[3], rho, g);
        let fm = long_accel(&m, xm[0], xm[1], xm[2], xm[3], rho, g);
        for i in 0..4 {
            a[i][j] = (fp[i] - fm[i]) / (2.0 * eps[j]);
        }
    }
    let _ = f0; // trim residual should be near zero; kept for debugging
    let modes = classify_longitudinal_modes(&a);
    LongitudinalStateSpace { a, x0, modes }
}

/// Convenience: phugoid approx at Earth g.
pub fn phugoid_approximation_earth(v: f64, cl: f64, cd: f64) -> (f64, f64) {
    phugoid_approximation(v, cl, cd, STANDARD_GRAVITY)
}
