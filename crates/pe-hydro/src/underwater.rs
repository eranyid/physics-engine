//! Underwater vehicle hydrodynamics (M34): pressure, ballast, current, Fossen 6-DOF.
//!
//! # Body frame
//!
//! Standalone [`UnderwaterVehicle`] uses a **FLU** body frame: +x forward, +y left,
//! +z up, with the CG at the origin. Diagonal added-mass / damping coefficients make
//! the left/right (FLU vs starboard) convention immaterial for the linearised terms.
//! Existing [`crate::hull`] docs use +y starboard; [`UnderwaterDamping`] / [`BallastTrim`]
//! on the [`pe_core::World`] path inherit that hull convention and do **not** apply
//! added mass (`World` remains `v̇ = F/m`).
//!
//! # Equations ([`UnderwaterVehicle`])
//!
//! With rigid-body mass `M_RB`, diagonal added mass `M_A`, body velocity `ν = [ν₁; ν₂]`,
//! relative velocity `ν_r = ν − [Rᵀ u_f; 0]`, and fluid velocity / acceleration
//! `u_f`, `a_f` at the CG (uniform current + optional Airy / wave field):
//!
//! ```text
//! (M_RB + M_A) ν̇ = τ + τ_hs + τ_FK
//!   − [m ν₂×ν₁; ν₂×I ν₂]
//!   − [ν₂×A₁₁ ν₁r; ν₁r×A₁₁ ν₁r + ν₂×A₂₂ ν₂]
//!   − D(ν_r) ν_r
//!   + M_A ν̇_c
//! ```
//!
//! where `ν̇_c,lin = Rᵀ a_f − ν₂×Rᵀ u_f`, Froude–Krylov `τ_FK,lin = ρ ∇ Rᵀ a_f`
//! (optional), hydrostatics are the body wrench of weight `(0,0,−m g)` at the CG
//! and buoyancy `(0,0, ρ g ∇)` at the COB, and
//! `D(ν_r) = diag(d_lin + d_quad ∘ |ν_r|)`.
//!
//! Kinematics: `ẋ = R ν₁`, `q̇ = ½ q ⊗ (0, ν₂)`. Integration: classical RK4 with
//! quaternion renormalisation after each step.
//!
//! # Limits
//!
//! - No added mass on the `World` force-generator path ([`UnderwaterDamping`],
//!   [`BallastTrim`]).
//! - Current is uniform ([`CurrentField::Uniform`]) or none.
//! - Ballast is a buoyancy-volume delta (no free-surface tank dynamics).
//! - [`crate::hull::SphereHull::submerged`] jumps ~2.8 % in volume when tilt
//!   crosses ~1.15° (UV-mesh fallback); leave that path alone — prefer
//!   [`VolumeModel::Submerged`] for deeply submerged spheres.
//!
//! # Determinism
//!
//! Portable [`pe_math::libm_ops`] only; no FMA / `mul_add`; no `HashMap`; fixed
//! DOF loop order.
//!
//! # References (clean-room formulas; no MSS code)
//! - Fossen, T. I. *Handbook of Marine Craft Hydrodynamics and Motion Control*.
//! - Imlay, F. H. (1961). The complete expressions for added mass of a rigid
//!   body moving in an ideal fluid. DTMB Report 1528.
//! - Lamb, H. *Hydrodynamics*. Cambridge University Press.
//! - Newman, J. N. *Marine Hydrodynamics*. MIT Press.
//! - Dean & Dalrymple — Airy dynamic pressure head (see [`crate::waves::AiryWave`]).

use pe_core::forces::ForceGenerator;
use pe_core::world::{State, World};
use pe_math::libm_ops;
use pe_math::{Quat, Vec3, STANDARD_GRAVITY};

use crate::cummins6::solve6;
use crate::error::{require_finite, require_non_negative, require_positive, HydroError};
use crate::hull::{Hull, RHO_SEAWATER};
use crate::waves::{AiryWave, Wave, WaveField};

/// Hydrostatic (gauge or absolute) pressure [Pa]: `p₀ + ρ g max(d, 0)`.
///
/// `depth` [m] positive downward immersion, `rho` [kg/m³], `g` [m/s²], `p0` [Pa]
/// (use `0` for gauge, `101325` for standard absolute).
#[inline]
pub fn hydrostatic_pressure(depth: f64, rho: f64, g: f64, p0: f64) -> f64 {
    let d = if depth > 0.0 { depth } else { 0.0 };
    p0 + rho * g * d
}

/// Fallible hydrostatic pressure (all inputs finite; `rho`, `g` ≥ 0).
pub fn try_hydrostatic_pressure(depth: f64, rho: f64, g: f64, p0: f64) -> Result<f64, HydroError> {
    require_finite(depth, "depth")?;
    require_finite(p0, "p0")?;
    require_non_negative(rho, "rho")?;
    require_non_negative(g, "g")?;
    Ok(hydrostatic_pressure(depth, rho, g, p0))
}

/// Total Airy pressure [Pa]: `p₀ + ρ g (mwl − z) + ρ g · pressure_head`.
///
/// Combines still-water hydrostatic (z up, MWL at `airy.mwl`) with the
/// undisturbed dynamic head from [`AiryWave::pressure_head`] (Dean & Dalrymple).
pub fn airy_total_pressure(
    airy: &AiryWave,
    x: f64,
    y: f64,
    z: f64,
    t: f64,
    rho: f64,
    p0: f64,
) -> f64 {
    let head = airy.pressure_head(x, y, z, t);
    p0 + rho * airy.g * (airy.mwl - z) + rho * airy.g * head
}

/// Translational added mass of a sphere [kg]: `½ ρ · (4/3) π r³`.
pub fn sphere_added_mass(r: f64, rho: f64) -> f64 {
    0.5 * rho * (4.0 / 3.0) * core::f64::consts::PI * r * r * r
}

/// Lamb / Imlay (1961) spheroid inertia coefficients `(k₁, k₂, k')`.
///
/// Prolate spheroid semi-axes `a ≥ b > 0` (longitudinal / transverse). Eccentricity
/// `e = √(1 − (b/a)²)`; for `e < 1e-3` returns the sphere limit `(0.5, 0.5, 0)`.
///
/// ```text
/// α₀ = 2(1−e²)/e³ · (½ ln((1+e)/(1−e)) − e)
/// β₀ = 1/e² − (1−e²)/(2 e³) · ln((1+e)/(1−e))
/// k₁ = α₀/(2−α₀),  k₂ = β₀/(2−β₀)
/// k' = e⁴ (β₀−α₀) / ((2−e²) (2 e² − (2−e²)(β₀−α₀)))
/// ```
pub fn spheroid_lamb_k(a: f64, b: f64) -> (f64, f64, f64) {
    debug_assert!(a > 0.0 && b > 0.0);
    let ratio = b / a;
    let e2 = 1.0 - ratio * ratio;
    if e2 < 1e-6 {
        // e < 1e-3
        return (0.5, 0.5, 0.0);
    }
    let e = libm_ops::sqrt(e2);
    if e < 1e-3 {
        return (0.5, 0.5, 0.0);
    }
    let ln_term = libm_ops::log((1.0 + e) / (1.0 - e));
    let one_e2 = 1.0 - e2;
    let e3 = e * e * e;
    let alpha0 = 2.0 * one_e2 / e3 * (0.5 * ln_term - e);
    let beta0 = 1.0 / e2 - one_e2 / (2.0 * e3) * ln_term;
    let k1 = alpha0 / (2.0 - alpha0);
    let k2 = beta0 / (2.0 - beta0);
    let e4 = e2 * e2;
    let db = beta0 - alpha0;
    let denom = (2.0 - e2) * (2.0 * e2 - (2.0 - e2) * db);
    let kp = e4 * db / denom;
    (k1, k2, kp)
}

/// Diagonal 6-DOF added-mass coefficients for a prolate spheroid [SI mixed].
///
/// Returns `[A₁₁, A₂₂, A₂₂, 0, A₅₅, A₅₅]` with volume `V = 4/3 π a b²`,
/// `A₁₁ = k₁ ρ V`, `A₂₂ = k₂ ρ V`,
/// `A₅₅ = k' · (4/15) π ρ a b² (a² + b²)` (Imlay / Lamb).
pub fn spheroid_added_mass6(a: f64, b: f64, rho: f64) -> [f64; 6] {
    let (k1, k2, kp) = spheroid_lamb_k(a, b);
    let v = (4.0 / 3.0) * core::f64::consts::PI * a * b * b;
    let a11 = k1 * rho * v;
    let a22 = k2 * rho * v;
    let a55 = kp * (4.0 / 15.0) * core::f64::consts::PI * rho * a * b * b * (a * a + b * b);
    [a11, a22, a22, 0.0, a55, a55]
}

/// Ambient current field (world frame).
#[derive(Clone, Debug)]
pub enum CurrentField {
    /// Still water.
    None,
    /// Spatially uniform current velocity [m/s].
    Uniform(Vec3),
}

impl CurrentField {
    /// World-frame current velocity [m/s].
    #[inline]
    pub fn velocity(&self) -> Vec3 {
        match self {
            CurrentField::None => Vec3::zero(),
            CurrentField::Uniform(u) => *u,
        }
    }
}

/// Volume / buoyancy model for [`UnderwaterVehicle`].
#[derive(Clone, Debug)]
pub enum VolumeModel {
    /// Fully submerged: fixed displaced volume [m³] and COB in the body frame [m].
    Submerged {
        /// Displaced volume ∇ [m³].
        volume: f64,
        /// Centre of buoyancy in the body (CG) frame [m].
        cob_body: Vec3,
    },
    /// Geometric hull clipped at the free surface (no added mass on this path).
    Hull {
        /// Hull geometry.
        hull: Hull,
        /// COM offset in the geometric body frame [m] (same as [`crate::force::Buoyancy`]).
        com_body: Vec3,
    },
}

/// Body-frame linear + quadratic damping on a [`World`] body (no added mass).
///
/// Relative velocity `ν_r = [Rᵀ(v − u_f); Rᵀ ω]` with `u_f` from current + optional
/// wave velocity at the COM; body wrench
/// `τ_b = −(d_lin + d_quad ∘ |ν_r|) ∘ ν_r`, then rotated to the world frame.
#[derive(Clone, Debug)]
pub struct UnderwaterDamping {
    /// Body index.
    pub body: usize,
    /// Linear damping coefficients `[X_u, Y_v, Z_w, K_p, M_q, N_r]` [SI].
    pub d_lin: [f64; 6],
    /// Quadratic damping coefficients [SI].
    pub d_quad: [f64; 6],
    /// Ambient current.
    pub current: CurrentField,
    /// Optional wave orbital velocity at the COM.
    pub wave: Option<WaveField>,
}

impl UnderwaterDamping {
    /// Construct damping for `body`.
    pub fn new(
        body: usize,
        d_lin: [f64; 6],
        d_quad: [f64; 6],
        current: CurrentField,
        wave: Option<WaveField>,
    ) -> Self {
        Self {
            body,
            d_lin,
            d_quad,
            current,
            wave,
        }
    }
}

impl ForceGenerator for UnderwaterDamping {
    fn apply(&self, world: &World, s: &State, t: f64, f: &mut [Vec3], torque: &mut [Vec3]) {
        let i = self.body;
        let mut u_f = self.current.velocity();
        if let Some(ref w) = self.wave {
            u_f += w.velocity(s.x[i].x, s.x[i].y, s.x[i].z, t);
        }
        let q = s.q[i];
        let v_b = q.inverse_rotate(s.v[i] - u_f);
        let w_b = q.inverse_rotate(world.omega_i(s, i));
        let nu_r = [v_b.x, v_b.y, v_b.z, w_b.x, w_b.y, w_b.z];
        let mut tau_b = [0.0; 6];
        for j in 0..6 {
            let abs_n = libm_ops::fabs(nu_r[j]);
            tau_b[j] = -(self.d_lin[j] + self.d_quad[j] * abs_n) * nu_r[j];
        }
        let f_b = Vec3::new(tau_b[0], tau_b[1], tau_b[2]);
        let m_b = Vec3::new(tau_b[3], tau_b[4], tau_b[5]);
        f[i] += q.rotate(f_b);
        torque[i] += q.rotate(m_b);
    }
}

/// Ballast / trim as a buoyancy volume delta on a [`World`] body.
///
/// World force `F = (0, 0, ρ g ΔV)` applied at `x + R · point_body`.
#[derive(Clone, Debug)]
pub struct BallastTrim {
    /// Body index.
    pub body: usize,
    /// Displaced-volume change ΔV [m³] (positive → upward force).
    pub delta_volume: f64,
    /// Application point in the body frame [m].
    pub point_body: Vec3,
    /// Fluid density [kg/m³].
    pub rho: f64,
    /// Gravity magnitude [m/s²].
    pub g: f64,
}

impl BallastTrim {
    /// Construct ballast for `body`.
    pub fn new(body: usize, delta_volume: f64, point_body: Vec3, rho: f64, g: f64) -> Self {
        Self {
            body,
            delta_volume,
            point_body,
            rho,
            g,
        }
    }
}

impl ForceGenerator for BallastTrim {
    fn apply(&self, _world: &World, s: &State, _t: f64, f: &mut [Vec3], torque: &mut [Vec3]) {
        let i = self.body;
        let force = Vec3::new(0.0, 0.0, self.rho * self.g * self.delta_volume);
        let r = s.q[i].rotate(self.point_body);
        f[i] += force;
        torque[i] += r.cross(force);
    }
}

/// Standalone Fossen 6-DOF underwater vehicle (body FLU, CG at origin).
#[derive(Clone, Debug)]
pub struct UnderwaterVehicle {
    /// Dry mass [kg].
    pub mass: f64,
    /// Principal inertias `[I_xx, I_yy, I_zz]` [kg·m²].
    pub inertia: [f64; 3],
    /// Diagonal added mass `[A₁₁…A₆₆]` [SI mixed].
    pub added_mass: [f64; 6],
    /// Linear damping [SI].
    pub d_lin: [f64; 6],
    /// Quadratic damping [SI].
    pub d_quad: [f64; 6],
    /// Displaced volume / hull model.
    pub volume: VolumeModel,
    /// Fluid density [kg/m³].
    pub rho: f64,
    /// Gravity magnitude [m/s²].
    pub g: f64,
    /// Ambient current.
    pub current: CurrentField,
    /// Optional wave field at the CG.
    pub wave: Option<WaveField>,
    /// Enable Froude–Krylov `ρ ∇ Rᵀ a_f` when a wave field is present.
    pub froude_krylov: bool,
    /// World-frame COM position [m].
    pub x: Vec3,
    /// Orientation quaternion `[w,x,y,z]`.
    pub q: Quat,
    /// Body-frame velocity `ν = [u,v,w,p,q,r]` [m/s, rad/s].
    pub nu: [f64; 6],
    /// Simulation time [s].
    pub t: f64,
}

impl UnderwaterVehicle {
    /// Fallible constructor (finite inputs; `mass`, inertia > 0; added/damping ≥ 0).
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        mass: f64,
        inertia: [f64; 3],
        added_mass: [f64; 6],
        d_lin: [f64; 6],
        d_quad: [f64; 6],
        volume: VolumeModel,
        rho: f64,
        g: f64,
        current: CurrentField,
        wave: Option<WaveField>,
        froude_krylov: bool,
        x: Vec3,
        q: Quat,
        nu: [f64; 6],
    ) -> Result<Self, HydroError> {
        require_positive(mass, "mass")?;
        for (j, &ii) in inertia.iter().enumerate() {
            require_positive(
                ii,
                match j {
                    0 => "inertia[0]",
                    1 => "inertia[1]",
                    _ => "inertia[2]",
                },
            )?;
        }
        for (j, &a) in added_mass.iter().enumerate() {
            require_non_negative(
                a,
                match j {
                    0 => "added_mass[0]",
                    1 => "added_mass[1]",
                    2 => "added_mass[2]",
                    3 => "added_mass[3]",
                    4 => "added_mass[4]",
                    _ => "added_mass[5]",
                },
            )?;
        }
        for (j, &d) in d_lin.iter().enumerate() {
            require_non_negative(
                d,
                match j {
                    0 => "d_lin[0]",
                    1 => "d_lin[1]",
                    2 => "d_lin[2]",
                    3 => "d_lin[3]",
                    4 => "d_lin[4]",
                    _ => "d_lin[5]",
                },
            )?;
        }
        for (j, &d) in d_quad.iter().enumerate() {
            require_non_negative(
                d,
                match j {
                    0 => "d_quad[0]",
                    1 => "d_quad[1]",
                    2 => "d_quad[2]",
                    3 => "d_quad[3]",
                    4 => "d_quad[4]",
                    _ => "d_quad[5]",
                },
            )?;
        }
        require_positive(rho, "rho")?;
        require_positive(g, "g")?;
        require_finite(x.x, "x.x")?;
        require_finite(x.y, "x.y")?;
        require_finite(x.z, "x.z")?;
        if !q.is_finite() {
            return Err(HydroError::NonFinite("q"));
        }
        for (j, &n) in nu.iter().enumerate() {
            require_finite(
                n,
                match j {
                    0 => "nu[0]",
                    1 => "nu[1]",
                    2 => "nu[2]",
                    3 => "nu[3]",
                    4 => "nu[4]",
                    _ => "nu[5]",
                },
            )?;
        }
        match &volume {
            VolumeModel::Submerged {
                volume: vol,
                cob_body,
            } => {
                require_positive(*vol, "volume")?;
                require_finite(cob_body.x, "cob_body.x")?;
                require_finite(cob_body.y, "cob_body.y")?;
                require_finite(cob_body.z, "cob_body.z")?;
            }
            VolumeModel::Hull { com_body, .. } => {
                require_finite(com_body.x, "com_body.x")?;
                require_finite(com_body.y, "com_body.y")?;
                require_finite(com_body.z, "com_body.z")?;
            }
        }
        if let CurrentField::Uniform(u) = &current {
            require_finite(u.x, "current.x")?;
            require_finite(u.y, "current.y")?;
            require_finite(u.z, "current.z")?;
        }
        Ok(Self {
            mass,
            inertia,
            added_mass,
            d_lin,
            d_quad,
            volume,
            rho,
            g,
            current,
            wave,
            froude_krylov,
            x,
            q: q.normalize(),
            nu,
            t: 0.0,
        })
    }

    /// Instantaneous geometric depth below MWL [m] (`−z` with MWL at 0, or wave MWL).
    pub fn depth(&self) -> f64 {
        let mwl = match &self.wave {
            Some(w) => w.mean_water_level(),
            None => 0.0,
        };
        mwl - self.x.z
    }

    /// Hydrostatic pressure [Pa] at the COM depth with reference `p0`.
    pub fn pressure(&self, p0: f64) -> f64 {
        hydrostatic_pressure(self.depth(), self.rho, self.g, p0)
    }

    /// Mechanical energy proxy: `½ νᵀ (M_RB+M_A) ν + m g z`.
    pub fn energy(&self) -> f64 {
        let mut ke = 0.0;
        for i in 0..3 {
            let mii = self.mass + self.added_mass[i];
            ke += mii * self.nu[i] * self.nu[i];
        }
        for i in 0..3 {
            let iii = self.inertia[i] + self.added_mass[i + 3];
            ke += iii * self.nu[i + 3] * self.nu[i + 3];
        }
        0.5 * ke + self.mass * self.g * self.x.z
    }

    /// Pack state as `[x(3), q(4), nu(6)]`.
    pub fn pack(&self) -> [f64; 13] {
        let qa = self.q.to_array();
        [
            self.x.x, self.x.y, self.x.z, qa[0], qa[1], qa[2], qa[3], self.nu[0], self.nu[1],
            self.nu[2], self.nu[3], self.nu[4], self.nu[5],
        ]
    }

    /// Displaced volume ∇ [m³] and COB in the body frame [m] at the current pose.
    fn volume_cob(&self, x: Vec3, q: Quat, t: f64) -> (f64, Vec3) {
        match &self.volume {
            VolumeModel::Submerged { volume, cob_body } => (*volume, *cob_body),
            VolumeModel::Hull { hull, com_body } => {
                let x_geom = x - q.rotate(*com_body);
                let wl = match &self.wave {
                    Some(w) => w.mean_water_level() + w.height(x.x, x.y, t),
                    None => 0.0,
                };
                let sub = hull.submerged(x_geom, q, wl);
                let cob_body = q.inverse_rotate(sub.cob_world - x);
                (sub.volume, cob_body)
            }
        }
    }

    /// Fluid velocity / acceleration at the CG (world frame).
    fn fluid_at(&self, x: Vec3, t: f64) -> (Vec3, Vec3) {
        let mut u_f = self.current.velocity();
        let mut a_f = Vec3::zero();
        if let Some(ref w) = self.wave {
            u_f += w.velocity(x.x, x.y, x.z, t);
            a_f += w.acceleration(x.x, x.y, x.z, t);
        }
        (u_f, a_f)
    }

    /// Total mass matrix `M_RB + M_A` (diagonal).
    fn mass_total(&self) -> [[f64; 6]; 6] {
        let mut m = [[0.0; 6]; 6];
        for i in 0..3 {
            m[i][i] = self.mass + self.added_mass[i];
        }
        for i in 0..3 {
            m[i + 3][i + 3] = self.inertia[i] + self.added_mass[i + 3];
        }
        m
    }

    /// Body-frame acceleration `ν̇` at `(x, q, nu, t)` with control wrench `tau`.
    fn accel_at(&self, x: Vec3, q: Quat, nu: &[f64; 6], t: f64, tau: &[f64; 6]) -> [f64; 6] {
        let (vol, cob_body) = self.volume_cob(x, q, t);
        let (u_f, a_f) = self.fluid_at(x, t);
        let u_fb = q.inverse_rotate(u_f);
        let a_fb = q.inverse_rotate(a_f);

        let nu1 = Vec3::new(nu[0], nu[1], nu[2]);
        let nu2 = Vec3::new(nu[3], nu[4], nu[5]);
        let nu1r = nu1 - u_fb;

        // ν̇_c,lin = Rᵀ a_f − ν₂ × Rᵀ u_f
        let nudot_c_lin = a_fb - nu2.cross(u_fb);

        // Rigid-body Coriolis / centripetal: [m ν₂×ν₁; ν₂×I ν₂]
        let i_nu2 = Vec3::new(
            self.inertia[0] * nu2.x,
            self.inertia[1] * nu2.y,
            self.inertia[2] * nu2.z,
        );
        let crb_lin = nu2.cross(nu1) * self.mass;
        let crb_ang = nu2.cross(i_nu2);

        // Added-mass Coriolis (diagonal A): [ν₂×A₁₁ν₁r; ν₁r×A₁₁ν₁r + ν₂×A₂₂ν₂]
        let a11_nu1r = Vec3::new(
            self.added_mass[0] * nu1r.x,
            self.added_mass[1] * nu1r.y,
            self.added_mass[2] * nu1r.z,
        );
        let a22_nu2 = Vec3::new(
            self.added_mass[3] * nu2.x,
            self.added_mass[4] * nu2.y,
            self.added_mass[5] * nu2.z,
        );
        let ca_lin = nu2.cross(a11_nu1r);
        let ca_ang = nu1r.cross(a11_nu1r) + nu2.cross(a22_nu2);

        // Damping
        let nu_r = [nu1r.x, nu1r.y, nu1r.z, nu2.x, nu2.y, nu2.z];
        let mut d_tau = [0.0; 6];
        for j in 0..6 {
            let abs_n = libm_ops::fabs(nu_r[j]);
            d_tau[j] = (self.d_lin[j] + self.d_quad[j] * abs_n) * nu_r[j];
        }

        // Hydrostatics (body frame)
        let w_world = Vec3::new(0.0, 0.0, -self.mass * self.g);
        let b_world = Vec3::new(0.0, 0.0, self.rho * self.g * vol);
        let f_hs = q.inverse_rotate(w_world + b_world);
        let m_hs = cob_body.cross(q.inverse_rotate(b_world));

        // Froude–Krylov (optional)
        let mut f_fk = Vec3::zero();
        if self.froude_krylov && self.wave.is_some() {
            f_fk = a_fb * (self.rho * vol);
        }

        // M_A ν̇_c contribution to RHS
        let ma_nudot = Vec3::new(
            self.added_mass[0] * nudot_c_lin.x,
            self.added_mass[1] * nudot_c_lin.y,
            self.added_mass[2] * nudot_c_lin.z,
        );

        let mut rhs = [0.0; 6];
        rhs[0] = tau[0] + f_hs.x + f_fk.x - crb_lin.x - ca_lin.x - d_tau[0] + ma_nudot.x;
        rhs[1] = tau[1] + f_hs.y + f_fk.y - crb_lin.y - ca_lin.y - d_tau[1] + ma_nudot.y;
        rhs[2] = tau[2] + f_hs.z + f_fk.z - crb_lin.z - ca_lin.z - d_tau[2] + ma_nudot.z;
        rhs[3] = tau[3] + m_hs.x - crb_ang.x - ca_ang.x - d_tau[3];
        rhs[4] = tau[4] + m_hs.y - crb_ang.y - ca_ang.y - d_tau[4];
        rhs[5] = tau[5] + m_hs.z - crb_ang.z - ca_ang.z - d_tau[5];

        solve6(&self.mass_total(), &rhs)
    }

    /// Kinematic + dynamic derivatives at a stage state.
    fn deriv_at(
        &self,
        x: Vec3,
        q: Quat,
        nu: &[f64; 6],
        t: f64,
        tau: &[f64; 6],
    ) -> (Vec3, Quat, [f64; 6]) {
        let nu1 = Vec3::new(nu[0], nu[1], nu[2]);
        let dx = q.rotate(nu1);
        let wq = Quat::new(0.0, nu[3], nu[4], nu[5]);
        let dq = q.multiply(wq).scale(0.5);
        let dnu = self.accel_at(x, q, nu, t, tau);
        (dx, dq, dnu)
    }

    /// Classical RK4 step with body-frame control wrench `tau` [N, N·m].
    pub fn step_rk4(&mut self, dt: f64, tau: &[f64; 6]) {
        let x0 = self.x;
        let q0 = self.q;
        let nu0 = self.nu;
        let t0 = self.t;

        let (dx1, dq1, dnu1) = self.deriv_at(x0, q0, &nu0, t0, tau);

        let x2 = x0 + dx1 * (0.5 * dt);
        let q2 = (q0 + dq1.scale(0.5 * dt)).normalize();
        let mut nu2 = [0.0; 6];
        for i in 0..6 {
            nu2[i] = nu0[i] + 0.5 * dt * dnu1[i];
        }
        let (dx2, dq2, dnu2) = self.deriv_at(x2, q2, &nu2, t0 + 0.5 * dt, tau);

        let x3 = x0 + dx2 * (0.5 * dt);
        let q3 = (q0 + dq2.scale(0.5 * dt)).normalize();
        let mut nu3 = [0.0; 6];
        for i in 0..6 {
            nu3[i] = nu0[i] + 0.5 * dt * dnu2[i];
        }
        let (dx3, dq3, dnu3) = self.deriv_at(x3, q3, &nu3, t0 + 0.5 * dt, tau);

        let x4 = x0 + dx3 * dt;
        let q4 = (q0 + dq3.scale(dt)).normalize();
        let mut nu4 = [0.0; 6];
        for i in 0..6 {
            nu4[i] = nu0[i] + dt * dnu3[i];
        }
        let (dx4, dq4, dnu4) = self.deriv_at(x4, q4, &nu4, t0 + dt, tau);

        self.x = x0 + (dx1 + dx2 * 2.0 + dx3 * 2.0 + dx4) * (dt / 6.0);
        self.q = (q0
            + dq1.scale(dt / 6.0)
            + dq2.scale(dt / 3.0)
            + dq3.scale(dt / 3.0)
            + dq4.scale(dt / 6.0))
        .normalize();
        for i in 0..6 {
            self.nu[i] = nu0[i] + (dt / 6.0) * (dnu1[i] + 2.0 * dnu2[i] + 2.0 * dnu3[i] + dnu4[i]);
        }
        self.t = t0 + dt;
    }

    /// Integrate with fixed `dt` and constant `tau` until `t_end` (exclusive of overshoot).
    pub fn run(&mut self, t_end: f64, dt: f64, tau: &[f64; 6]) {
        while self.t + dt * 0.5 < t_end {
            self.step_rk4(dt, tau);
        }
    }
}

/// FNV-1a 64 over little-endian IEEE bits of an `f64` slice.
fn fnv1a64_f64_slice(xs: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in xs {
        for b in x.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// Canonical M34 underwater hash (locked scenario in `docs/DETERMINISM.md`).
///
/// Mass 51 kg, inertia `[0.31, 14, 14]`, submerged ∇=0.0499 m³ with COB
/// `(0,0,0.02)`, seawater, spheroid added mass `a=0.9,b=0.11`, damping as in
/// the roadmap brief, pose at `z=−5` with roll 10° / pitch 5°, `ν=(0.5,0,…)`,
/// uniform current `(0.2,0.1,0)`, Airy `A=0.25,T=6,h=20`, Froude–Krylov on,
/// control `τ=(13,0,…)`, RK4 `dt=0.01` × 1000 steps; FNV-1a 64 over `pack()`
/// after each step.
pub fn canonical_underwater_hash() -> u64 {
    let rho = RHO_SEAWATER;
    let added = spheroid_added_mass6(0.9, 0.11, rho);
    let d_quad0 = 0.5 * rho * 0.35 * 0.074;
    let d_lin = [2.0, 10.0, 10.0, 0.5, 5.0, 5.0];
    let d_quad = [d_quad0, 30.0, 30.0, 0.1, 10.0, 10.0];
    let volume = VolumeModel::Submerged {
        volume: 0.0499,
        cob_body: Vec3::new(0.0, 0.0, 0.02),
    };
    let roll = 10.0_f64.to_radians();
    let pitch = 5.0_f64.to_radians();
    let q = Quat::from_axis_angle(Vec3::unit_y(), pitch)
        .multiply(Quat::from_axis_angle(Vec3::unit_x(), roll));
    let wave = AiryWave::earth(0.25, 6.0, 20.0, 0.0);
    let mut veh = UnderwaterVehicle::try_new(
        51.0,
        [0.31, 14.0, 14.0],
        added,
        d_lin,
        d_quad,
        volume,
        rho,
        STANDARD_GRAVITY,
        CurrentField::Uniform(Vec3::new(0.2, 0.1, 0.0)),
        Some(WaveField::Airy(wave)),
        true,
        Vec3::new(0.0, 0.0, -5.0),
        q,
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .expect("canonical underwater vehicle");
    let tau = [13.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let dt = 0.01;
    let mut samples = Vec::with_capacity(13 * 1000);
    for _ in 0..1000 {
        veh.step_rk4(dt, &tau);
        samples.extend_from_slice(&veh.pack());
    }
    fnv1a64_f64_slice(&samples)
}

#[cfg(test)]
mod unit_smoke {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sphere_added_mass_oracle() {
        let a = sphere_added_mass(0.1, 1000.0);
        assert_relative_eq!(a, 2.0943951023931957, max_relative = 1e-15);
    }

    #[test]
    fn lamb_k_oracle() {
        let (k1, k2, kp) = spheroid_lamb_k(1.0, 0.5);
        assert_relative_eq!(k1, 0.21001504897664153, max_relative = 1e-14);
        assert_relative_eq!(k2, 0.704210425850353, max_relative = 1e-14);
        assert_relative_eq!(kp, 0.23942389319515428, max_relative = 1e-14);
    }
}
