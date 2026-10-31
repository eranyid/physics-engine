//! Aircraft aero force/moment generator attached to one [`World`] body.
//!
//! Lift ⟂ freestream in the body x–z plane; drag ∥ −freestream; sideforce = 0.
//! Pitching moment `M = q̄ S c C_m` about body +y [N m].
//! At `V < 1e-9` m/s, α/β are treated as 0 and no aero force is applied.

use pe_core::forces::ForceGenerator;
use pe_core::world::{State, World};
use pe_math::libm_ops;
use pe_math::Vec3;

use crate::atmosphere::DensityModel;
use crate::polar::{dynamic_pressure, WingPolar};

/// Snapshot of air-data / aero coefficients for diagnostics.
#[derive(Clone, Copy, Debug)]
pub struct AirData {
    /// Airspeed [m/s].
    pub v: f64,
    /// Angle of attack [rad].
    pub alpha: f64,
    /// Lift coefficient [dimensionless].
    pub cl: f64,
    /// Drag coefficient [dimensionless].
    pub cd: f64,
    /// Pitching-moment coefficient [dimensionless].
    pub cm: f64,
    /// Lift force magnitude [N].
    pub lift: f64,
    /// Drag force magnitude [N].
    pub drag: f64,
    /// Ambient density [kg/m³].
    pub rho: f64,
    /// Body pitch rate [rad/s].
    pub q: f64,
}

/// Body-frame aero forces/moments from a [`WingPolar`] + density model.
#[derive(Clone, Debug)]
pub struct AircraftAero {
    /// Body index.
    pub body: usize,
    /// Wing polar / reference geometry.
    pub polar: WingPolar,
    /// Density lookup.
    pub density: DensityModel,
    /// Wind velocity [m/s] (world frame).
    pub wind: Vec3,
}

impl AircraftAero {
    /// Attach aero to `body` with the given polar and density model.
    pub fn new(body: usize, polar: WingPolar, density: DensityModel, wind: Vec3) -> Self {
        Self {
            body,
            polar,
            density,
            wind,
        }
    }

    /// Diagnostic air data at the current state (SI).
    pub fn air_data(&self, world: &World, s: &State) -> AirData {
        let i = self.body;
        let v_rel = s.v[i] - self.wind;
        let v_b = s.q[i].inverse_rotate(v_rel);
        let v = v_b.norm();
        let rho = self.density.density_at(s.x[i].z);
        if v < 1e-9 {
            return AirData {
                v: 0.0,
                alpha: 0.0,
                cl: 0.0,
                cd: self.polar.cd0,
                cm: self.polar.cm0,
                lift: 0.0,
                drag: 0.0,
                rho,
                q: 0.0,
            };
        }
        let alpha = libm_ops::atan2(-v_b.z, v_b.x);
        let omega_b = s.q[i].inverse_rotate(world.omega_i(s, i));
        let q_rate = omega_b.y;
        let cl = self.polar.cl(alpha);
        let cd = self.polar.cd(cl);
        let cm = self.polar.cm(alpha, q_rate, v);
        let qbar = dynamic_pressure(rho, v);
        AirData {
            v,
            alpha,
            cl,
            cd,
            cm,
            lift: qbar * self.polar.s * cl,
            drag: qbar * self.polar.s * cd,
            rho,
            q: q_rate,
        }
    }
}

impl ForceGenerator for AircraftAero {
    fn apply(&self, world: &World, s: &State, _t: f64, f: &mut [Vec3], torque: &mut [Vec3]) {
        let i = self.body;
        let v_rel = s.v[i] - self.wind;
        let v_b = s.q[i].inverse_rotate(v_rel);
        let v = v_b.norm();
        if v < 1e-9 {
            return;
        }
        let e_v = v_b / v;
        let alpha = libm_ops::atan2(-v_b.z, v_b.x);
        let omega_b = s.q[i].inverse_rotate(world.omega_i(s, i));
        let q_rate = omega_b.y;
        let rho = self.density.density_at(s.x[i].z);
        let cl = self.polar.cl(alpha);
        let cd = self.polar.cd(cl);
        let cm = self.polar.cm(alpha, q_rate, v);
        let qbar = dynamic_pressure(rho, v);
        let e_lift = Vec3::new(libm_ops::sin(alpha), 0.0, libm_ops::cos(alpha));
        let e_drag = -e_v;
        let f_b = (e_lift * cl + e_drag * cd) * (qbar * self.polar.s);
        let m_b = Vec3::new(0.0, qbar * self.polar.s * self.polar.c * cm, 0.0);
        f[i] += s.q[i].rotate(f_b);
        torque[i] += s.q[i].rotate(m_b);
    }
}
