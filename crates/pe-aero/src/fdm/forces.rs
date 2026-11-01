//! Body-frame 6-DOF aero + thrust force/moment evaluation.

use pe_core::forces::ForceGenerator;
use pe_core::world::{State, World};
use pe_math::libm_ops;
use pe_math::{Quat, Vec3};

use super::derivatives::{ControlInputs, StabilityDerivatives};
use crate::atmosphere::DensityModel;
use crate::polar::dynamic_pressure;

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
    /// Wind velocity [m/s] (world frame; blowing *to*).
    pub wind: Vec3,
    /// Control-surface deflections [rad].
    pub controls: ControlInputs,
    /// Trim angle of attack [rad] used when interpreting incremental derivatives
    /// that were tabulated about a flight condition (0 if absolute).
    pub alpha_trim: f64,
    /// Whether geometry/derivatives are from cited public sources.
    pub verified: bool,
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

/// Evaluate body-frame aero + thrust force/moment for a model state sample.
///
/// `v_body` [m/s] air-relative velocity in body axes; `omega_b` [rad/s] body rates;
/// `rho` [kg/m³]; controls from `model.controls` unless `controls` is `Some`.
pub fn eval_aero_body(
    model: &SixDofModel,
    v_body: Vec3,
    omega_b: Vec3,
    rho: f64,
    controls: Option<ControlInputs>,
) -> SixDofForces {
    let ctrl = controls.unwrap_or(model.controls);
    let (alpha_abs, beta, v) = air_angles(v_body);
    // Derivatives tabulated about trim AoA: use incremental α.
    let alpha = alpha_abs - model.alpha_trim;
    let (p_hat, q_hat, r_hat) = body_rates_hat(omega_b, model.b, model.c, v);

    let cl = model.deriv.cl(alpha, q_hat, ctrl.delta_e);
    let cd = model.deriv.cd(alpha);
    let cy = model
        .deriv
        .cy(beta, p_hat, r_hat, ctrl.delta_a, ctrl.delta_r);
    let c_l = model
        .deriv
        .c_l_roll(beta, p_hat, r_hat, ctrl.delta_a, ctrl.delta_r);
    let cm = model.deriv.cm(alpha, q_hat, ctrl.delta_e);
    let cn = model
        .deriv
        .cn(beta, p_hat, r_hat, ctrl.delta_a, ctrl.delta_r);

    let qbar = dynamic_pressure(rho, v);
    let qs = qbar * model.s;

    // Wind-axis → body (z-up): drag along −V̂, side along ŷ_wind, lift ⟂ V in α-plane.
    // For small β the body-axis approximation:
    //   X = qs (−CD cosα cosβ + CY sinβ − ? ) …
    // We use the same geometric construction as planar AircraftAero, extended for β:
    let (f_aero_b, _e_v) = if v < 1e-9 {
        (Vec3::zero(), Vec3::unit_x())
    } else {
        let e_v = v_body / v;
        // Lift direction in the plane of body x–z, perpendicular to the projection of V.
        let e_lift = Vec3::new(libm_ops::sin(alpha_abs), 0.0, libm_ops::cos(alpha_abs));
        // Side unit: complete right-handed wind basis ≈ body y for modest β.
        let e_side = e_v.cross(e_lift);
        let e_side = {
            let n = e_side.norm();
            if n < 1e-12 {
                Vec3::unit_y()
            } else {
                e_side / n
            }
        };
        let f = e_lift * (qs * cl) + e_side * (qs * cy) + (-e_v) * (qs * cd);
        (f, e_v)
    };

    let m_aero_b = Vec3::new(qs * model.b * c_l, qs * model.c * cm, qs * model.b * cn);

    let t_axis = {
        let n = model.thrust_axis.norm();
        if n < 1e-15 {
            Vec3::unit_x()
        } else {
            model.thrust_axis / n
        }
    };
    let f_thrust = t_axis * model.thrust;
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

/// [`ForceGenerator`] wrapping a [`SixDofModel`] attached to one body.
#[derive(Clone, Debug)]
pub struct SixDofAero {
    /// Body index in the [`World`].
    pub body: usize,
    /// Aircraft model (controls / thrust / wind are part of the model).
    pub model: SixDofModel,
}

impl SixDofAero {
    /// Attach 6-DOF aero to `body`.
    pub fn new(body: usize, model: SixDofModel) -> Self {
        Self { body, model }
    }

    /// Diagnostic loads at the current state.
    pub fn loads(&self, world: &World, s: &State) -> SixDofForces {
        let i = self.body;
        let v_rel = s.v[i] - self.model.wind;
        let v_b = s.q[i].inverse_rotate(v_rel);
        let omega_b = s.q[i].inverse_rotate(world.omega_i(s, i));
        let rho = self.model.density.density_at(s.x[i].z);
        eval_aero_body(&self.model, v_b, omega_b, rho, None)
    }
}

impl ForceGenerator for SixDofAero {
    fn apply(&self, world: &World, s: &State, _t: f64, f: &mut [Vec3], torque: &mut [Vec3]) {
        let i = self.body;
        let v_rel = s.v[i] - self.model.wind;
        let v_b = s.q[i].inverse_rotate(v_rel);
        let omega_b = s.q[i].inverse_rotate(world.omega_i(s, i));
        let rho = self.model.density.density_at(s.x[i].z);
        let loads = eval_aero_body(&self.model, v_b, omega_b, rho, None);
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
