//! Buoyancy and hull-drag force generators for a [`World`] body.
//!
//! Buoyancy: `F = ρ g ∇` along +z [N], torque `(r_COB − r_COM) × F` [N m].
//! Hull drag: quadratic surge resistance opposing horizontal velocity.

use pe_core::forces::ForceGenerator;
use pe_core::world::{State, World};
use pe_math::libm_ops;
use pe_math::{Vec3, STANDARD_GRAVITY};

use crate::hull::{Hull, SubmergedState, RHO_FRESHWATER};

/// Archimedes buoyancy on one body from a [`Hull`].
#[derive(Clone, Debug)]
pub struct Buoyancy {
    /// Body index.
    pub body: usize,
    /// Hull geometry.
    pub hull: Hull,
    /// Fluid density [kg/m³].
    pub rho: f64,
    /// Gravity magnitude [m/s²] (force along +z).
    pub g: f64,
    /// Waterplane world z [m].
    pub water_level: f64,
    /// COM offset in the geometric body frame [m].
    pub com_body: Vec3,
}

impl Buoyancy {
    /// Attach buoyancy to `body`. `rho` [kg/m³], `water_level` [m], `com_body` [m].
    pub fn new(
        body: usize,
        hull: Hull,
        rho: f64,
        g: f64,
        water_level: f64,
        com_body: Vec3,
    ) -> Self {
        Self {
            body,
            hull,
            rho,
            g,
            water_level,
            com_body,
        }
    }

    /// Freshwater Earth-g convenience.
    pub fn freshwater(body: usize, hull: Hull, water_level: f64) -> Self {
        Self::new(
            body,
            hull,
            RHO_FRESHWATER,
            STANDARD_GRAVITY,
            water_level,
            Vec3::zero(),
        )
    }

    /// Submerged state at the current pose (geometric origin corrected by `com_body`).
    pub fn submerged(&self, s: &State) -> SubmergedState {
        let i = self.body;
        let x_geom = s.x[i] - s.q[i].rotate(self.com_body);
        self.hull.submerged(x_geom, s.q[i], self.water_level)
    }
}

impl ForceGenerator for Buoyancy {
    fn apply(&self, _world: &World, s: &State, _t: f64, f: &mut [Vec3], torque: &mut [Vec3]) {
        let i = self.body;
        let sub = self.submerged(s);
        if sub.volume <= 0.0 {
            return;
        }
        let fb = Vec3::new(0.0, 0.0, self.rho * self.g * sub.volume);
        f[i] += fb;
        torque[i] += (sub.cob_world - s.x[i]).cross(fb);
    }
}

/// Wave-making / residual drag coefficient: constant or `C_w(Fr)`.
#[derive(Clone, Debug)]
pub enum WaveDrag {
    /// Constant residual coefficient [dimensionless].
    Constant(f64),
    /// Linear in Froude number: `C_w = k · Fr` (`k` dimensionless).
    LinearFr(f64),
}

impl WaveDrag {
    /// Evaluate `C_w` [dimensionless] at Froude number `fr`.
    pub fn eval(&self, fr: f64) -> f64 {
        match self {
            WaveDrag::Constant(c) => *c,
            WaveDrag::LinearFr(k) => k * fr,
        }
    }
}

/// Quadratic hull resistance `R = ½ ρ V² S (C_f + C_w)` opposing velocity [N].
#[derive(Clone, Debug)]
pub struct HullDrag {
    /// Body index.
    pub body: usize,
    /// Wetted reference area [m²].
    pub s: f64,
    /// Reference length [m] for Froude number.
    pub length: f64,
    /// Frictional coefficient [dimensionless].
    pub c_f: f64,
    /// Residual / wave coefficient.
    pub c_w: WaveDrag,
    /// Fluid density [kg/m³].
    pub rho: f64,
    /// Gravity [m/s²] for Fr.
    pub g: f64,
    /// Wind velocity [m/s].
    pub wind: Vec3,
    /// If false, zero the heave component of relative velocity before drag.
    pub include_heave: bool,
}

impl HullDrag {
    /// Build hull drag. `S` [m²], `L` [m], coefficients dimensionless.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        body: usize,
        s: f64,
        length: f64,
        c_f: f64,
        c_w: WaveDrag,
        rho: f64,
        g: f64,
        wind: Vec3,
        include_heave: bool,
    ) -> Self {
        Self {
            body,
            s,
            length,
            c_f,
            c_w,
            rho,
            g,
            wind,
            include_heave,
        }
    }
}

impl ForceGenerator for HullDrag {
    fn apply(&self, _world: &World, s: &State, _t: f64, f: &mut [Vec3], _torque: &mut [Vec3]) {
        let i = self.body;
        let mut u = s.v[i] - self.wind;
        if !self.include_heave {
            u.z = 0.0;
        }
        let v = u.norm();
        if v < 1e-12 {
            return;
        }
        let fr = v / libm_ops::sqrt(self.g * self.length);
        let cw = self.c_w.eval(fr);
        f[i] -= u * (0.5 * self.rho * (self.c_f + cw) * self.s * v);
    }
}
