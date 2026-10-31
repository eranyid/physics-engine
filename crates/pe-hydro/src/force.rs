//! Buoyancy and hull-drag force generators for a [`World`] body.
//!
//! Buoyancy: `F = ρ g ∇` along +z [N], torque `(r_COB − r_COM) × F` [N m].
//! With an optional [`WaveField`], the local free-surface elevation sets the
//! instantaneous waterplane (pressure/immersion at the wave surface). An optional
//! long-wave Froude–Krylov correction uses the undisturbed Airy pressure head
//! at the geometric centre (Dean & Dalrymple).
//!
//! Hull drag: quadratic surge resistance opposing horizontal velocity.

use pe_core::forces::ForceGenerator;
use pe_core::world::{State, World};
use pe_math::libm_ops;
use pe_math::{Vec3, STANDARD_GRAVITY};

use crate::hull::{Hull, SubmergedState, RHO_FRESHWATER};
use crate::waves::{AiryWave, Wave, WaveField};

/// Archimedes buoyancy on one body from a [`Hull`], optionally on a wave surface.
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
    /// Still / mean waterplane world z [m] (used when `wave` is `None`).
    pub water_level: f64,
    /// COM offset in the geometric body frame [m].
    pub com_body: Vec3,
    /// Optional wave field: local η shifts the waterplane.
    pub wave: Option<WaveField>,
    /// If true and `wave` is [`WaveField::Airy`], add a long-wave Froude–Krylov
    /// vertical force from the undisturbed dynamic pressure head at the COM
    /// horizontal position / draft mid-depth (tractable FK approximation).
    pub froude_krylov: bool,
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
            wave: None,
            froude_krylov: false,
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

    /// Attach a wave field (local elevation as waterplane).
    pub fn with_wave(mut self, wave: WaveField) -> Self {
        self.wave = Some(wave);
        self
    }

    /// Enable / disable long-wave Froude–Krylov (Airy only).
    pub fn with_froude_krylov(mut self, enable: bool) -> Self {
        self.froude_krylov = enable;
        self
    }

    /// Instantaneous waterplane z [m] at the body horizontal position.
    pub fn waterplane_z(&self, s: &State, t: f64) -> f64 {
        let i = self.body;
        match &self.wave {
            Some(w) => w.mean_water_level() + w.height(s.x[i].x, s.x[i].y, t),
            None => self.water_level,
        }
    }

    /// Submerged state at the current pose (geometric origin corrected by `com_body`).
    pub fn submerged_at(&self, s: &State, t: f64) -> SubmergedState {
        let i = self.body;
        let x_geom = s.x[i] - s.q[i].rotate(self.com_body);
        let wl = self.waterplane_z(s, t);
        self.hull.submerged(x_geom, s.q[i], wl)
    }

    /// Submerged state at t = 0 (compat with still-water callers).
    pub fn submerged(&self, s: &State) -> SubmergedState {
        self.submerged_at(s, 0.0)
    }

    /// Long-wave FK vertical force [N] from Airy pressure head (optional).
    ///
    /// Approximates `F_z ≈ ρ g A_wp · (p_dyn/(ρ g))` evaluated at mid-draft under
    /// the COM — the hydrostatic part of the undisturbed wave pressure already
    /// enters via the elevated waterplane; this term adds the depth-attenuated
    /// dynamic head relative to the surface elevation (difference of
    /// `cosh(k(z+h))/cosh(kh)` vs 1 at the free surface). For very long waves
    /// (`kh → 0`) the difference vanishes.
    fn froude_krylov_force(&self, airy: &AiryWave, s: &State, t: f64, a_wp: f64) -> Vec3 {
        if a_wp <= 0.0 {
            return Vec3::zero();
        }
        let i = self.body;
        let x = s.x[i].x;
        let y = s.x[i].y;
        let eta = airy.height(x, y, t);
        // Evaluation depth: geometric COM z (under surface).
        let z = s.x[i].z;
        let head = airy.pressure_head(x, y, z, t);
        // Dynamic FK residual vs pure surface hydrostatic: head − η * attenuation≈1
        // at surface. Using (head − η) removes double-counting with waterplane shift.
        let residual = head - eta;
        Vec3::new(0.0, 0.0, self.rho * self.g * a_wp * residual)
    }
}

impl ForceGenerator for Buoyancy {
    fn apply(&self, _world: &World, s: &State, t: f64, f: &mut [Vec3], torque: &mut [Vec3]) {
        let i = self.body;
        let sub = self.submerged_at(s, t);
        if sub.volume <= 0.0 {
            return;
        }
        let fb = Vec3::new(0.0, 0.0, self.rho * self.g * sub.volume);
        f[i] += fb;
        torque[i] += (sub.cob_world - s.x[i]).cross(fb);

        if self.froude_krylov {
            if let Some(WaveField::Airy(ref airy)) = self.wave {
                f[i] += self.froude_krylov_force(airy, s, t, sub.a_wp);
            }
        }
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
