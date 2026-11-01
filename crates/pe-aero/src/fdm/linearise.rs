//! Longitudinal / lateral state-space linearisation and textbook mode helpers.

use pe_math::libm_ops;
use pe_math::{Quat, Vec3, STANDARD_GRAVITY};

use super::derivatives::ControlInputs;
use super::eigen::{
    classify_lateral_modes, classify_longitudinal_modes, LateralModes, LongitudinalModes,
};
use super::forces::{eval_aero_body, eval_aero_body_with_alphadot, SixDofModel};
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

/// Lateral-directional state-space about a wings-level trim.
///
/// State `x = [v, p, r, φ]ᵀ` (body lateral speed [m/s], roll/yaw rates [rad/s],
/// bank [rad]). `ẋ = A x`.
#[derive(Clone, Debug)]
pub struct LateralStateSpace {
    /// 4×4 system matrix.
    pub a: [[f64; 4]; 4],
    /// Trim state `[v, p, r, φ]` (typically zeros at wings-level).
    pub x0: [f64; 4],
    /// Roll / spiral / Dutch-roll classification.
    pub modes: Option<LateralModes>,
}

/// Analytic Lanchester / Etkin phugoid approximations:
/// `ω_ph ≈ g √2 / V`, `ζ_ph ≈ C_D / (√2 C_L)`.
///
/// Returns `(omega_n [rad/s], zeta [dimensionless])`.
///
/// This is a **coarse** sanity check only: the Lanchester gap vs a full
/// linearisation is typically O(10–20%) in `ω_n` for GA aircraft (see
/// `docs/AERO.md`). Prefer published Nelson/Teper roots for validation.
pub fn phugoid_approximation(v: f64, cl: f64, cd: f64, g: f64) -> (f64, f64) {
    let omega = g * libm_ops::sqrt(2.0) / v.max(1e-9);
    let zeta = if cl.abs() < 1e-15 {
        f64::INFINITY
    } else {
        cd / (libm_ops::sqrt(2.0) * cl)
    };
    (omega, zeta)
}

/// Reduced-order short-period approximation (Etkin / Nelson 2-DOF α–q),
/// **without** `C_mα̇` / `C_Lα̇`:
///
/// `ω_sp ≈ √(−Z_α M_q / V_T + M_α)` (dimensional), with the usual small-α
/// body-axis derivatives from `C_Lα`, `C_mq`, `C_mα`.
///
/// Returns `(omega_n, zeta)`. Including `C_mα̇` increases damping relative to
/// this approximation (see oracle tests).
pub fn short_period_approximation(
    model: &SixDofModel,
    v: f64,
    rho: f64,
    _alpha: f64,
) -> (f64, f64) {
    let qbar = 0.5 * rho * v * v;
    let s = model.s;
    let c = model.c;
    let m = model.mass;
    let iy = model.inertia_diag.1;
    let vt = v.max(1e-9);
    // Dimensional Z_α ≈ −q̄ S CL_α / m  (body z-up: lift → −z at α=0, so
    // ∂a_z/∂α ≈ −(q̄ S / m) CL_α  for the normal accel in wind axes; in our
    // z-up body frame at small α, w≈−Vα so the α-stiffness maps similarly).
    let z_alpha = -qbar * s * model.deriv.cl_alpha / m;
    let m_alpha = qbar * s * c * model.deriv.cm_alpha / iy;
    let m_q = qbar * s * c * c * model.deriv.cm_q / (2.0 * vt * iy);
    let z_q = -qbar * s * c * model.deriv.cl_q / (2.0 * m * vt);
    // Classic: ω² ≈ Z_α M_q / V − M_α  (signs: Z_α<0, M_α<0, M_q<0).
    // With our Z_α definition (negative): ω² = −M_α + Z_α M_q / V.
    let w2 = (-m_alpha + z_alpha * m_q / vt).max(0.0);
    let omega = libm_ops::sqrt(w2);
    // 2 ζ ω ≈ −(M_q + Z_α/V + M_α̇…) without α̇: −(M_q + Z_α/V)
    let two_zw = -(m_q + z_alpha / vt + z_q * 0.0);
    let zeta = if omega > 1e-12 {
        two_zw / (2.0 * omega)
    } else {
        0.0
    };
    let _ = z_q;
    (omega, zeta)
}

/// Same as [`short_period_approximation`] but folds `C_mα̇` into an effective
/// `M_q` / damping term via `M_α̇ = q̄ S c (c/2V) C_mα̇ / I_y` and the usual
/// `α̇ ≈ q` short-period kinematic substitution (`M_q_eff = M_q + M_α̇`).
pub fn short_period_approximation_with_alphadot(
    model: &SixDofModel,
    v: f64,
    rho: f64,
    alpha: f64,
) -> (f64, f64) {
    let qbar = 0.5 * rho * v * v;
    let s = model.s;
    let c = model.c;
    let m = model.mass;
    let iy = model.inertia_diag.1;
    let vt = v.max(1e-9);
    let z_alpha = -qbar * s * model.deriv.cl_alpha / m;
    let m_alpha = qbar * s * c * model.deriv.cm_alpha / iy;
    let m_q = qbar * s * c * c * model.deriv.cm_q / (2.0 * vt * iy);
    let m_ad = qbar * s * c * c * model.deriv.cm_alphadot / (2.0 * vt * iy);
    // α̇ ≈ q in the reduced short-period model ⇒ M_q_eff = M_q + M_α̇.
    let m_q_eff = m_q + m_ad;
    let w2 = (-m_alpha + z_alpha * m_q_eff / vt).max(0.0);
    let omega = libm_ops::sqrt(w2);
    let two_zw = -(m_q_eff + z_alpha / vt);
    let zeta = if omega > 1e-12 {
        two_zw / (2.0 * omega)
    } else {
        0.0
    };
    let _ = alpha;
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

#[allow(clippy::too_many_arguments)]
fn long_accel(
    model: &SixDofModel,
    u: f64,
    w: f64,
    q: f64,
    theta: f64,
    rho: f64,
    g: f64,
    use_alphadot: bool,
) -> [f64; 4] {
    use super::forces::alpha_dot_kinematic;
    let v_body = Vec3::new(u, 0.0, w);
    let omega_b = Vec3::new(0.0, q, 0.0);
    let (_ixx, iyy, _izz) = model.inertia_diag;
    let s = libm_ops::sin(theta);
    let c = libm_ops::cos(theta);
    let gx = -g * s;
    let gz = -g * c;

    // Fixed-point on α̇: forces depend on α̇, and α̇ = (w ú − u ẇ)/(u²+w²)
    // from the full body accelerations (including gravity + kinematics).
    let mut adot = 0.0;
    let mut ax = 0.0;
    let mut az = 0.0;
    let mut q_dot = 0.0;
    let passes = if use_alphadot { 3 } else { 1 };
    for _ in 0..passes {
        let loads = eval_aero_body_with_alphadot(model, v_body, omega_b, rho, None, adot);
        ax = loads.force_body.x / model.mass + gx + q * w;
        az = loads.force_body.z / model.mass + gz - q * u;
        q_dot = loads.moment_body.y / iyy;
        if use_alphadot {
            adot = alpha_dot_kinematic(v_body, Vec3::new(ax, 0.0, az));
        } else {
            adot = 0.0;
        }
    }
    let theta_dot = q;
    [ax, az, q_dot, theta_dot]
}

/// Numerically linearise longitudinal dynamics about a trim point.
///
/// Builds `A` by central differences. Includes `C_Lα̇` / `C_mα̇` via the
/// one-pass kinematic α̇ inside [`eval_aero_body`].
pub fn linearise_longitudinal(
    model: &SixDofModel,
    trim: &TrimResult,
    rho: f64,
    g: f64,
) -> LongitudinalStateSpace {
    linearise_longitudinal_ex(model, trim, rho, g, true)
}

/// Longitudinal linearisation with optional α̇ derivatives.
pub fn linearise_longitudinal_ex(
    model: &SixDofModel,
    trim: &TrimResult,
    rho: f64,
    g: f64,
    use_alphadot: bool,
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
    let theta0 = trim.alpha;
    let x0 = [u0, w0, q0, theta0];

    let f0 = long_accel(&m, u0, w0, q0, theta0, rho, g, use_alphadot);
    let mut a = [[0.0; 4]; 4];
    let eps = [1e-4, 1e-4, 1e-5, 1e-5];
    for j in 0..4 {
        let mut xp = x0;
        let mut xm = x0;
        xp[j] += eps[j];
        xm[j] -= eps[j];
        let fp = long_accel(&m, xp[0], xp[1], xp[2], xp[3], rho, g, use_alphadot);
        let fm = long_accel(&m, xm[0], xm[1], xm[2], xm[3], rho, g, use_alphadot);
        for i in 0..4 {
            a[i][j] = (fp[i] - fm[i]) / (2.0 * eps[j]);
        }
    }
    let _ = f0;
    let modes = classify_longitudinal_modes(&a);
    LongitudinalStateSpace { a, x0, modes }
}

fn attitude_pitch_bank_quat(theta: f64, phi: f64) -> Quat {
    let q_pitch = Quat::from_axis_angle(Vec3::unit_y(), -theta);
    let q_roll = Quat::from_axis_angle(Vec3::unit_x(), phi);
    q_pitch.multiply(q_roll)
}

#[allow(clippy::too_many_arguments)]
fn lat_accel(
    model: &SixDofModel,
    u0: f64,
    w0: f64,
    theta0: f64,
    v: f64,
    p: f64,
    r: f64,
    phi: f64,
    rho: f64,
    g: f64,
) -> [f64; 4] {
    let v_body = Vec3::new(u0, v, w0);
    let omega_b = Vec3::new(p, 0.0, r);
    let loads = eval_aero_body(model, v_body, omega_b, rho, None);
    let (ixx, iyy, izz) = model.inertia_diag;
    let _ = iyy;
    // Gravity in body axes for pitch θ, bank φ (yaw=0):
    // g_world = (0,0,−g); R = R_pitch(θ) R_roll(φ).
    let q_att = attitude_pitch_bank_quat(theta0, phi);
    let g_body = q_att.inverse_rotate(Vec3::new(0.0, 0.0, -g));
    // Body-axis kinematics:
    // ṿ = Y/m + g_y − r u + p w
    let a_v = loads.force_body.y / model.mass + g_body.y - r * u0 + p * w0;
    // Euler (Ixz=0): ṗ = L/Ixx − (Izz−Iyy)/Ixx * q r → q=0 ⇒ ṗ = L/Ixx
    //               ṙ = N/Izz − (Iyy−Ixx)/Izz * p q → q=0 ⇒ ṙ = N/Izz
    // (product-of-inertia terms omitted; Nelson Navion I_xz = 0).
    let p_dot = loads.moment_body.x / ixx;
    let r_dot = loads.moment_body.z / izz;
    // φ̇ = p + tanθ (q sinφ + r cosφ); q=0 ⇒ φ̇ = p + r tanθ cosφ
    // At small φ about trim, use full expression for the difference Jacobian.
    let phi_dot = p + libm_ops::tan(theta0) * (r * libm_ops::cos(phi));
    [a_v, p_dot, r_dot, phi_dot]
}

/// Numerically linearise lateral-directional dynamics about a wings-level trim.
pub fn linearise_lateral(
    model: &SixDofModel,
    trim: &TrimResult,
    rho: f64,
    g: f64,
) -> LateralStateSpace {
    let mut m = model.clone();
    m.thrust = trim.thrust;
    m.controls = ControlInputs {
        delta_e: trim.delta_e,
        delta_a: 0.0,
        delta_r: 0.0,
    };
    let u0 = trim.v * libm_ops::cos(trim.alpha);
    let w0 = -trim.v * libm_ops::sin(trim.alpha);
    let theta0 = trim.alpha;
    let x0 = [0.0, 0.0, 0.0, 0.0]; // v, p, r, φ

    let f0 = lat_accel(&m, u0, w0, theta0, 0.0, 0.0, 0.0, 0.0, rho, g);
    let mut a = [[0.0; 4]; 4];
    let eps = [1e-4, 1e-5, 1e-5, 1e-5];
    for j in 0..4 {
        let mut xp = x0;
        let mut xm = x0;
        xp[j] += eps[j];
        xm[j] -= eps[j];
        let fp = lat_accel(&m, u0, w0, theta0, xp[0], xp[1], xp[2], xp[3], rho, g);
        let fm = lat_accel(&m, u0, w0, theta0, xm[0], xm[1], xm[2], xm[3], rho, g);
        for i in 0..4 {
            a[i][j] = (fp[i] - fm[i]) / (2.0 * eps[j]);
        }
    }
    let _ = f0;
    let modes = classify_lateral_modes(&a);
    LateralStateSpace { a, x0, modes }
}

/// Convenience: phugoid approx at Earth g.
pub fn phugoid_approximation_earth(v: f64, cl: f64, cd: f64) -> (f64, f64) {
    phugoid_approximation(v, cl, cd, STANDARD_GRAVITY)
}
