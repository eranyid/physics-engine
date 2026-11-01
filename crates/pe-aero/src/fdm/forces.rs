//! Body-frame 6-DOF aero + thrust force/moment evaluation.

use pe_core::forces::ForceGenerator;
use pe_core::world::{State, World};
use pe_math::libm_ops;
use pe_math::{Quat, Vec3};

use super::derivatives::{ControlInputs, StabilityDerivatives};
use crate::atmosphere::DensityModel;
use crate::ground_effect::apply_ground_effect;
use crate::polar::dynamic_pressure;
use crate::turbulence::HeldDrydenGust;

/// SI aircraft geometry, mass properties, and derivative set for 6-DOF aero.
#[derive(Clone, Debug)]
pub struct SixDofModel {
    /// Mass [kg].
    pub mass: f64,
    /// Reference wing area [m²].
    pub s: f64,
    /// Wing span [m].
    pub b: f64,
    /// Mean aerodynamic chord [m].
    pub c: f64,
    /// Principal inertias `(I_xx, I_yy, I_zz)` [kg m²] about COM.
    pub inertia_diag: (f64, f64, f64),
    /// Stability / control derivatives.
    pub deriv: StabilityDerivatives,
    /// Thrust magnitude [N] along [`Self::thrust_axis`].
    pub thrust: f64,
    /// Unit thrust axis in the body frame.
    pub thrust_axis: Vec3,
    /// Thrust application point relative to COM [m] (body frame).
    pub thrust_offset: Vec3,
    /// Density model (ISA or constant).
    pub density: DensityModel,
    /// Wind velocity [m/s] (world frame; blowing *to*). Steady component only;
    /// turbulence is attached on [`SixDofAero`] via [`HeldDrydenGust`].
    pub wind: Vec3,
    /// Control-surface deflections [rad].
    pub controls: ControlInputs,
    /// Trim angle of attack [rad] used when interpreting incremental derivatives
    /// that were tabulated about a flight condition (0 if absolute).
    pub alpha_trim: f64,
    /// Whether geometry/derivatives are from cited public sources.
    pub verified: bool,
    /// Enable Wieselsberger/McCormick ground effect when `Some(h_agl)` [m]
    /// (wing height above ground). `None` disables.
    pub ground_height: Option<f64>,
}

impl SixDofModel {
    /// Diagonal inertia tensor [kg m²].
    pub fn inertia_matrix(&self) -> [[f64; 3]; 3] {
        let (ix, iy, iz) = self.inertia_diag;
        [[ix, 0.0, 0.0], [0.0, iy, 0.0], [0.0, 0.0, iz]]
    }
}

/// Instantaneous body-frame aero + thrust loads.
#[derive(Clone, Copy, Debug)]
pub struct SixDofForces {
    /// Force in body axes [N].
    pub force_body: Vec3,
    /// Moment about COM in body axes [N m].
    pub moment_body: Vec3,
    /// Airspeed [m/s].
    pub v: f64,
    /// Angle of attack [rad] (absolute, pe-aero convention).
    pub alpha: f64,
    /// Sideslip [rad].
    pub beta: f64,
    /// Lift coefficient.
    pub cl: f64,
    /// Drag coefficient.
    pub cd: f64,
    /// Side-force coefficient.
    pub cy: f64,
    /// Rolling-moment coefficient.
    pub c_l: f64,
    /// Pitching-moment coefficient.
    pub cm: f64,
    /// Yawing-moment coefficient.
    pub cn: f64,
    /// Dynamic pressure [Pa].
    pub qbar: f64,
    /// Density [kg/m³].
    pub rho: f64,
}

/// Angle of attack and sideslip from body-frame air-relative velocity.
///
/// Convention (matches `AircraftAero`): `α = atan2(−v_z, v_x)`,
/// `β = asin(clamp(v_y / V, −1, 1))`. At `V < 1e-9`, returns zeros.
pub fn air_angles(v_body: Vec3) -> (f64, f64, f64) {
    let v = v_body.norm();
    if v < 1e-9 {
        return (0.0, 0.0, 0.0);
    }
    let alpha = libm_ops::atan2(-v_body.z, v_body.x);
    let sbeta = (v_body.y / v).clamp(-1.0, 1.0);
    let beta = libm_ops::asin(sbeta);
    (alpha, beta, v)
}

/// Nondimensional body rates `(p̂, q̂, r̂)` given `ω_body` [rad/s], chord/span, `V`.
pub fn body_rates_hat(omega_b: Vec3, b: f64, c: f64, v: f64) -> (f64, f64, f64) {
    let v_safe = v.max(1e-6);
    let p_hat = omega_b.x * b / (2.0 * v_safe);
    let q_hat = omega_b.y * c / (2.0 * v_safe);
    let r_hat = omega_b.z * b / (2.0 * v_safe);
    (p_hat, q_hat, r_hat)
}

/// Nondimensional α̇: `α̂̇ = α̇ c̄ / (2 V)`.
pub fn alpha_dot_hat(alpha_dot: f64, c: f64, v: f64) -> f64 {
    alpha_dot * c / (2.0 * v.max(1e-6))
}

/// Kinematic α̇ [rad/s] from body air-relative velocity and its body-frame
/// derivative: `α̇ = (u ẇ − w ú) / (u² + w²)` with `α = atan2(−w, u)`.
pub fn alpha_dot_kinematic(v_body: Vec3, a_body: Vec3) -> f64 {
    let u = v_body.x;
    let w = v_body.z;
    let den = u * u + w * w;
    if den < 1e-12 {
        return 0.0;
    }
    // α = atan2(−w, u) ⇒ α̇ = (−u ẇ + w ú) / (u² + w²)?
    // d/dt atan2(−w,u) = [u (−ẇ) − (−w) ú] / (u²+w²) = (−u ẇ + w ú)/(u²+w²)
    (w * a_body.x - u * a_body.z) / den
}

/// Evaluate body-frame aero + thrust force/moment for a model state sample.
///
/// `v_body` [m/s] air-relative velocity in body axes; `omega_b` [rad/s] body rates;
/// `rho` [kg/m³]; controls from `model.controls` unless `controls` is `Some`.
///
/// Uses `α̇ = 0` (steady / quasi-steady). Pass a nonzero rate through
/// [`eval_aero_body_with_alphadot`] — linearisation does a gravity-consistent
/// fixed-point; [`SixDofAero`] can supply a lagged `α̇` from attitude history.
pub fn eval_aero_body(
    model: &SixDofModel,
    v_body: Vec3,
    omega_b: Vec3,
    rho: f64,
    controls: Option<ControlInputs>,
) -> SixDofForces {
    eval_aero_body_with_alphadot(model, v_body, omega_b, rho, controls, 0.0)
}

/// As [`eval_aero_body`] with an explicit `α̇` [rad/s] for `C_Lα̇` / `C_mα̇`.
pub fn eval_aero_body_with_alphadot(
    model: &SixDofModel,
    v_body: Vec3,
    omega_b: Vec3,
    rho: f64,
    controls: Option<ControlInputs>,
    alpha_dot: f64,
) -> SixDofForces {
    let ctrl = controls.unwrap_or(model.controls);
    let (alpha_abs, beta, v) = air_angles(v_body);
    // Derivatives tabulated about trim AoA: use incremental α.
    let alpha = alpha_abs - model.alpha_trim;
    let (p_hat, q_hat, r_hat) = body_rates_hat(omega_b, model.b, model.c, v);
    let adot_hat = alpha_dot_hat(alpha_dot, model.c, v);

    let mut cl = model.deriv.cl_full(alpha, q_hat, ctrl.delta_e, adot_hat);
    let mut cd = model.deriv.cd(alpha);
    if let Some(h) = model.ground_height {
        let (cl_ge, cd_ge, _, _) = apply_ground_effect(cl, cd, model.deriv.cd0, h, model.b);
        cl = cl_ge;
        cd = cd_ge;
    }
    let cy = model
        .deriv
        .cy(beta, p_hat, r_hat, ctrl.delta_a, ctrl.delta_r);
    let c_l = model
        .deriv
        .c_l_roll(beta, p_hat, r_hat, ctrl.delta_a, ctrl.delta_r);
    let cm = model.deriv.cm_full(alpha, q_hat, ctrl.delta_e, adot_hat);
    let cn = model
        .deriv
        .cn(beta, p_hat, r_hat, ctrl.delta_a, ctrl.delta_r);

    let qbar = dynamic_pressure(rho, v);
    let qs = qbar * model.s;

    let f_aero_b = aero_force_body(v_body, alpha_abs, cl, cd, cy, qs);
    let m_aero_b = Vec3::new(qs * model.b * c_l, qs * model.c * cm, qs * model.b * cn);
    let f_thrust = thrust_force(model);
    let m_thrust = model.thrust_offset.cross(f_thrust);

    SixDofForces {
        force_body: f_aero_b + f_thrust,
        moment_body: m_aero_b + m_thrust,
        v,
        alpha: alpha_abs,
        beta,
        cl,
        cd,
        cy,
        c_l,
        cm,
        cn,
        qbar,
        rho,
    }
}

fn thrust_force(model: &SixDofModel) -> Vec3 {
    let n = model.thrust_axis.norm();
    let t_axis = if n < 1e-15 {
        Vec3::unit_x()
    } else {
        model.thrust_axis / n
    };
    t_axis * model.thrust
}

fn aero_force_body(v_body: Vec3, alpha_abs: f64, cl: f64, cd: f64, cy: f64, qs: f64) -> Vec3 {
    let v = v_body.norm();
    if v < 1e-9 {
        return Vec3::zero();
    }
    let e_v = v_body / v;
    // Lift in the body x–z plane, ⟂ to the α-projection of V (z-up).
    let e_lift = Vec3::new(libm_ops::sin(alpha_abs), 0.0, libm_ops::cos(alpha_abs));
    // Side force: stability-derivative `C_Y` is the body-+y coefficient
    // (Nelson / Etkin). Do **not** use `e_v × e_lift` — that equals −ŷ at
    // α=β=0 and would flip the sign of Y_β (and Dutch-roll damping).
    e_lift * (qs * cl) + Vec3::unit_y() * (qs * cy) + (-e_v) * (qs * cd)
}

/// [`ForceGenerator`] wrapping a [`SixDofModel`] attached to one body.
///
/// Optional [`HeldDrydenGust`] supplies a **piecewise-constant** gust held over
/// each discrete interval `[n Δt, (n+1) Δt)`. RK4 stages that land in the same
/// interval therefore see identical wind, so integrator replay stays
/// bit-deterministic for a fixed seed/`Δt` (see `docs/AERO.md`).
#[derive(Clone, Debug)]
pub struct SixDofAero {
    /// Body index in the [`World`].
    pub body: usize,
    /// Aircraft model (controls / thrust / steady wind are part of the model).
    pub model: SixDofModel,
    /// Optional Dryden gust (held per discrete step).
    pub gust: Option<HeldDrydenGust>,
}

impl SixDofAero {
    /// Attach 6-DOF aero to `body` (no turbulence).
    pub fn new(body: usize, model: SixDofModel) -> Self {
        Self {
            body,
            model,
            gust: None,
        }
    }

    /// Attach 6-DOF aero with a held Dryden gust schedule.
    pub fn with_gust(body: usize, model: SixDofModel, gust: HeldDrydenGust) -> Self {
        Self {
            body,
            model,
            gust: Some(gust),
        }
    }

    fn wind_at(&self, t: f64, airspeed: f64) -> Vec3 {
        let gust = match &self.gust {
            Some(g) => g.gust_at(t, airspeed),
            None => Vec3::zero(),
        };
        self.model.wind + gust
    }

    /// Diagnostic loads at the current state (`t = 0` wind hold).
    pub fn loads(&self, world: &World, s: &State) -> SixDofForces {
        self.loads_at(world, s, 0.0)
    }

    /// Diagnostic loads at simulation time `t` [s] (gust schedule keyed by `t`).
    pub fn loads_at(&self, world: &World, s: &State, t: f64) -> SixDofForces {
        let i = self.body;
        let airspeed = (s.v[i] - self.model.wind).norm().max(1.0);
        let wind = self.wind_at(t, airspeed);
        let v_rel = s.v[i] - wind;
        let v_b = s.q[i].inverse_rotate(v_rel);
        let omega_b = s.q[i].inverse_rotate(world.omega_i(s, i));
        let model = &self.model;
        let rho = model.density.density_at(s.x[i].z);
        eval_aero_body(model, v_b, omega_b, rho, None)
    }
}

impl ForceGenerator for SixDofAero {
    fn apply(&self, world: &World, s: &State, t: f64, f: &mut [Vec3], torque: &mut [Vec3]) {
        let i = self.body;
        let airspeed = (s.v[i] - self.model.wind).norm().max(1.0);
        let wind = self.wind_at(t, airspeed);
        let v_rel = s.v[i] - wind;
        let v_b = s.q[i].inverse_rotate(v_rel);
        let omega_b = s.q[i].inverse_rotate(world.omega_i(s, i));
        let mut model = self.model.clone();
        // If ground effect is armed with height source = body altitude (z AGL
        // when ground is z=0), update from state when a non-negative sentinel
        // was stored; callers set `ground_height` explicitly for offline eval.
        if let Some(h0) = model.ground_height {
            if h0 < 0.0 {
                model.ground_height = Some(s.x[i].z.max(0.0));
            }
        }
        let rho = model.density.density_at(s.x[i].z);
        let loads = eval_aero_body(&model, v_b, omega_b, rho, None);
        f[i] += s.q[i].rotate(loads.force_body);
        torque[i] += s.q[i].rotate(loads.moment_body);
    }
}

/// Pitch attitude quaternion for level flight at angle of attack `alpha` [rad]
/// (nose up ⇒ negative rotation about +y in z-up world).
pub fn level_attitude(alpha: f64) -> Quat {
    Quat::from_axis_angle(Vec3::unit_y(), -alpha)
}

/// Banked attitude: bank `phi` [rad] then pitch `theta` [rad] (3-2-1 / yaw-pitch-roll
/// with yaw=0): `q = q_yaw(0) ⊗ q_pitch(θ) ⊗ q_roll(φ)`.
pub fn attitude_pitch_bank(theta: f64, phi: f64) -> Quat {
    let q_pitch = Quat::from_axis_angle(Vec3::unit_y(), -theta);
    let q_roll = Quat::from_axis_angle(Vec3::unit_x(), phi);
    // Apply roll in body, then pitch in world: q_pitch ⊗ q_roll.
    q_pitch.multiply(q_roll)
}
