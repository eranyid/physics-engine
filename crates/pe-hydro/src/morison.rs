//! Morison equation for slender members under wave kinematics.
//!
//! Force per unit length (normal to the member axis), relative-velocity form
//! (API RP 2A / DNV recommended practice style, clean-room):
//!
//! ```text
//! dF = ρ C_m A a_n + ½ ρ C_d D |u_rn| u_rn
//! ```
//!
//! where `A = π D²/4` [m²], `a_n` is the fluid acceleration normal to the
//! member [m/s²], and `u_rn = (u_fluid − u_structure)_n` [m/s]. The inertia
//! coefficient `C_m` includes the Froude–Krylov contribution (`C_m = 1 + C_a`
//! with added-mass coefficient `C_a`). Structure acceleration is **not**
//! folded into the inertia term here (caller may add −ρ C_a A a_struct when
//! modelling a moving member with relative-acceleration inertia).
//!
//! Integration uses a fixed number of equal segments along the member and
//! portable [`pe_math::libm_ops`] only (no FMA).
//!
//! # References
//! - Morison, J. R., O’Brien, M. P., Johnson, J. W. & Schaaf, S. A. (1950).
//!   The force exerted by surface waves on piles. *Petrol. Trans. AIME* **189**.
//! - Dean & Dalrymple — Airy kinematics used for closed-form pile oracles.
//! - API RP 2A / DNV-RP-C205 — recommended `C_m`, `C_d` ranges (values used
//!   here are **nominal / unverified** engineering defaults, labelled as such).
//!
//! # Deferred
//! - Member shadowing / wake interference; depth-varying `C_d`/`C_m`;
//!   MacCamy–Fuchs diffraction correction for large `D/λ`.

use pe_core::forces::ForceGenerator;
use pe_core::world::{State, World};
use pe_math::libm_ops;
use pe_math::Vec3;

use crate::waves::{AiryWave, Wave, WaveField};

/// Nominal (unverified) Morison coefficients for a smooth circular cylinder.
///
/// Source: API RP 2A / DNV-RP-C205 typical post-critical ranges (order of
/// magnitude). **Not** fitted to a specific experiment in this tree.
pub const NOMINAL_CM_CIRCULAR: f64 = 2.0;
/// Nominal drag coefficient for a smooth circular cylinder (see [`NOMINAL_CM_CIRCULAR`]).
pub const NOMINAL_CD_CIRCULAR: f64 = 1.0;

/// Straight slender circular member for Morison integration.
#[derive(Clone, Debug)]
pub struct MorisonMember {
    /// World-frame endpoint A [m].
    pub a: Vec3,
    /// World-frame endpoint B [m].
    pub b: Vec3,
    /// Diameter D [m] (>0).
    pub diameter: f64,
    /// Inertia coefficient C_m [dimensionless] (nominal unless caller documents).
    pub cm: f64,
    /// Drag coefficient C_d [dimensionless] (nominal unless caller documents).
    pub cd: f64,
    /// Number of equal integration segments (≥1).
    pub n_segments: usize,
}

impl MorisonMember {
    /// Build a member. `diameter` [m] > 0; `n_segments` ≥ 1.
    pub fn new(a: Vec3, b: Vec3, diameter: f64, cm: f64, cd: f64, n_segments: usize) -> Self {
        assert!(diameter > 0.0 && n_segments >= 1);
        Self {
            a,
            b,
            diameter,
            cm,
            cd,
            n_segments,
        }
    }

    /// Vertical circular cylinder on the z-axis from `z_bottom` to `z_top` [m].
    pub fn vertical_cylinder(
        z_bottom: f64,
        z_top: f64,
        diameter: f64,
        cm: f64,
        cd: f64,
        n_segments: usize,
    ) -> Self {
        Self::new(
            Vec3::new(0.0, 0.0, z_bottom),
            Vec3::new(0.0, 0.0, z_top),
            diameter,
            cm,
            cd,
            n_segments,
        )
    }

    /// Member length [m].
    pub fn length(&self) -> f64 {
        (self.b - self.a).norm()
    }

    /// Unit tangent from A→B.
    pub fn tangent(&self) -> Vec3 {
        let d = self.b - self.a;
        let n = d.norm();
        if n < 1e-15 {
            Vec3::unit_z()
        } else {
            d / n
        }
    }

    /// Cross-sectional area `A = π D²/4` [m²].
    pub fn area(&self) -> f64 {
        core::f64::consts::PI * self.diameter * self.diameter / 4.0
    }
}

/// Integrated Morison force [N] and moment about `moment_ref` [N·m].
#[derive(Clone, Copy, Debug, Default)]
pub struct MorisonLoad {
    /// Total force [N].
    pub force: Vec3,
    /// Total moment about `moment_ref` [N·m]: `Σ r × dF`.
    pub moment: Vec3,
}

/// Project `v` onto the plane normal to unit tangent `t`: `v − (v·t) t`.
#[inline]
pub fn normal_component(v: Vec3, t: Vec3) -> Vec3 {
    v - t * v.dot(t)
}

/// Integrate Morison loads on a fixed member (structure velocity = 0).
///
/// `rho` [kg/m³]; wave kinematics from `wave` at time `t` [s].
pub fn integrate_morison_fixed(
    member: &MorisonMember,
    wave: &impl Wave,
    rho: f64,
    t: f64,
    moment_ref: Vec3,
) -> MorisonLoad {
    integrate_morison(member, wave, rho, t, moment_ref, Vec3::zero(), Vec3::zero())
}

/// Integrate Morison loads with rigid-body structure kinematics at the COM.
///
/// Structure velocity at a station is `v_com + ω × (r − r_com)` with
/// `ω = 0` when `omega` is zero. Structure acceleration is **not** used in
/// the inertia term (see module docs).
pub fn integrate_morison(
    member: &MorisonMember,
    wave: &impl Wave,
    rho: f64,
    t: f64,
    moment_ref: Vec3,
    v_com: Vec3,
    r_com: Vec3,
) -> MorisonLoad {
    let n = member.n_segments;
    let t_hat = member.tangent();
    let ds = member.length() / n as f64;
    let a_sec = member.area();
    let mut force = Vec3::zero();
    let mut moment = Vec3::zero();
    for i in 0..n {
        let s = (i as f64 + 0.5) / n as f64;
        let r = member.a + (member.b - member.a) * s;
        let u_f = wave.velocity(r.x, r.y, r.z, t);
        let a_f = wave.acceleration(r.x, r.y, r.z, t);
        let v_s = v_com; // ω≡0 path; extend with ω×(r−r_com) when needed
        let _ = r_com;
        let u_n = normal_component(u_f - v_s, t_hat);
        let a_n = normal_component(a_f, t_hat);
        let u_n_mag = u_n.norm();
        let d_f_inertia = a_n * (rho * member.cm * a_sec);
        let d_f_drag = if u_n_mag > 0.0 {
            u_n * (0.5 * rho * member.cd * member.diameter * u_n_mag)
        } else {
            Vec3::zero()
        };
        let d_f = (d_f_inertia + d_f_drag) * ds;
        force += d_f;
        moment += (r - moment_ref).cross(d_f);
    }
    MorisonLoad { force, moment }
}

/// Closed-form Airy inertia force and mudline moment on a fixed vertical pile.
///
/// Pile axis at `(x, y)`, from `z_bottom` to `z_top` [m] (relative to MWL for
/// depth factors — pass absolute world z; MWL from the wave). Heading from the
/// wave. Drag is omitted (`C_d = 0` oracle).
///
/// ```text
/// F_∥ = ρ C_m A · A_w ω² sin χ · I₁
/// M_mud = ρ C_m A · A_w ω² sin χ · I₂
/// ```
/// with `I₁ = ∫ cosh(k(z+h))/sinh(kh) dz` and
/// `I₂ = ∫ (z − z_bottom) cosh(k(z+h))/sinh(kh) dz` along the pile.
///
/// Returns `(force, moment_about_bottom)` in world frame [N], [N·m].
#[allow(clippy::too_many_arguments)]
pub fn airy_inertia_closed_form(
    wave: &AiryWave,
    x: f64,
    y: f64,
    z_bottom: f64,
    z_top: f64,
    diameter: f64,
    cm: f64,
    rho: f64,
    t: f64,
) -> (Vec3, Vec3) {
    assert!(z_top > z_bottom);
    let area = core::f64::consts::PI * diameter * diameter / 4.0;
    let chi = {
        let kx = wave.k * libm_ops::cos(wave.theta);
        let ky = wave.k * libm_ops::sin(wave.theta);
        kx * x + ky * y - wave.omega * t + wave.phase
    };
    let sin_chi = libm_ops::sin(chi);
    let amp = wave.amplitude * wave.omega * wave.omega * sin_chi;
    let (i1, i2) = airy_cosh_integrals(wave, z_bottom - wave.mwl, z_top - wave.mwl, z_bottom);
    let f_mag = rho * cm * area * amp * i1;
    let m_mag = rho * cm * area * amp * i2;
    let ct = libm_ops::cos(wave.theta);
    let st = libm_ops::sin(wave.theta);
    let force = Vec3::new(f_mag * ct, f_mag * st, 0.0);
    // Horizontal force → moment about bottom: M = (z_arm cross F); for vertical
    // pile the lever is vertical so M = (−F_y * 0 + …) → M = (0,0,0)× wait:
    // r = (0,0,z−z_b), F = (Fx,Fy,0) ⇒ M = r × F = (−(z−zb) Fy, (z−zb) Fx, 0)
    // integrated already into I₂ as scalar along heading, so:
    let moment = Vec3::new(-m_mag * st, m_mag * ct, 0.0);
    (force, moment)
}

/// Integrals `I₁`, `I₂` of the Airy horizontal depth factor over `[z1,z2]`
/// (z relative to MWL for the hyperbolic functions; `z_bottom_abs` for moment arm).
fn airy_cosh_integrals(wave: &AiryWave, z1: f64, z2: f64, z_bottom_abs: f64) -> (f64, f64) {
    let k = wave.k;
    let h = wave.depth;
    let kh = k * h;
    let mwl = wave.mwl;
    // Moment arm uses absolute z: (z_abs − z_bottom) = (z_rel + mwl − z_bottom)
    let zb_rel_offset = mwl - z_bottom_abs;

    if kh > 30.0 {
        // Deep: factor ≈ e^{k z_rel}
        let e1 = libm_ops::exp(k * z1);
        let e2 = libm_ops::exp(k * z2);
        let i1 = (e2 - e1) / k;
        // ∫ (z_rel + zb_rel_offset) e^{k z} dz
        // ∫ z e^{kz} dz = e^{kz}(kz − 1)/k²
        let int_z_exp = |z: f64| {
            let ez = libm_ops::exp(k * z);
            ez * (k * z - 1.0) / (k * k)
        };
        let i2 = (int_z_exp(z2) - int_z_exp(z1)) + zb_rel_offset * i1;
        return (i1, i2);
    }

    let sinh_kh = libm_ops::sinh(kh);
    // ∫ cosh(k(z+h)) dz = sinh(k(z+h))/k
    let sinh_at = |z: f64| libm_ops::sinh(k * (z + h));
    let i1 = (sinh_at(z2) - sinh_at(z1)) / (k * sinh_kh);

    // ∫ z_rel cosh(k(z+h)) dz via parts:
    // u=z, dv=cosh(k(z+h))dz → du=dz, v=sinh(k(z+h))/k
    // = z sinh(k(z+h))/k − ∫ sinh(k(z+h))/k dz = z sinh/k − cosh/k²
    let int_z_cosh = |z: f64| {
        let kz = k * (z + h);
        z * libm_ops::sinh(kz) / k - libm_ops::cosh(kz) / (k * k)
    };
    let i2_rel = (int_z_cosh(z2) - int_z_cosh(z1)) / sinh_kh;
    let i2 = i2_rel + zb_rel_offset * i1;
    (i1, i2)
}

/// Closed-form Airy **drag-only** force on a fixed vertical pile at a phase
/// where `cos χ ≥ 0` everywhere so `u` does not reverse along the pile
/// (e.g. crest `χ = 0`). Uses `u ≥ 0 ⇒ |u|u = u²`.
///
/// ```text
/// F_∥ = ½ ρ C_d D (A_w ω)² cos²χ · ∫ [cosh(k(z+h))/sinh(kh)]² dz
/// ```
#[allow(clippy::too_many_arguments)]
pub fn airy_drag_closed_form_no_reversal(
    wave: &AiryWave,
    x: f64,
    y: f64,
    z_bottom: f64,
    z_top: f64,
    diameter: f64,
    cd: f64,
    rho: f64,
    t: f64,
) -> Vec3 {
    let chi = {
        let kx = wave.k * libm_ops::cos(wave.theta);
        let ky = wave.k * libm_ops::sin(wave.theta);
        kx * x + ky * y - wave.omega * t + wave.phase
    };
    let c = libm_ops::cos(chi);
    assert!(
        c >= -1e-12,
        "drag closed form requires cos χ ≥ 0 (no velocity reversal)"
    );
    let z1 = z_bottom - wave.mwl;
    let z2 = z_top - wave.mwl;
    let i = airy_cosh_sq_integral(wave, z1, z2);
    let f_mag = 0.5 * rho * cd * diameter * (wave.amplitude * wave.omega).powi(2) * c * c * i;
    let ct = libm_ops::cos(wave.theta);
    let st = libm_ops::sin(wave.theta);
    Vec3::new(f_mag * ct, f_mag * st, 0.0)
}

fn airy_cosh_sq_integral(wave: &AiryWave, z1: f64, z2: f64) -> f64 {
    let k = wave.k;
    let h = wave.depth;
    let kh = k * h;
    if kh > 30.0 {
        // ∫ e^{2 k z} dz = (e^{2kz2} − e^{2kz1})/(2k)
        return (libm_ops::exp(2.0 * k * z2) - libm_ops::exp(2.0 * k * z1)) / (2.0 * k);
    }
    let sinh_kh = libm_ops::sinh(kh);
    let denom = sinh_kh * sinh_kh;
    // cosh²(u) = (cosh(2u)+1)/2, u=k(z+h)
    // ∫ cosh²(k(z+h)) dz = ∫ (cosh(2k(z+h))+1)/2 dz
    //   = sinh(2k(z+h))/(4k) + z/2
    let int_cosh2 = |z: f64| {
        let u = k * (z + h);
        libm_ops::sinh(2.0 * u) / (4.0 * k) + 0.5 * z
    };
    (int_cosh2(z2) - int_cosh2(z1)) / denom
}

/// Force generator: Morison loads on a body from a slender member in a wave field.
///
/// The member endpoints are given in the **body frame** relative to the COM
/// (translated/rotated with the body each step). Relative velocity uses the
/// body COM velocity (rotation neglected in the relative-velocity term unless
/// `include_spin` is set).
#[derive(Clone, Debug)]
pub struct MorisonForce {
    /// Body index.
    pub body: usize,
    /// Endpoints in the body frame [m].
    pub a_body: Vec3,
    /// Endpoints in the body frame [m].
    pub b_body: Vec3,
    /// Diameter [m].
    pub diameter: f64,
    /// Inertia coefficient (nominal unless documented by caller).
    pub cm: f64,
    /// Drag coefficient (nominal unless documented by caller).
    pub cd: f64,
    /// Integration segments.
    pub n_segments: usize,
    /// Fluid density [kg/m³].
    pub rho: f64,
    /// Wave field.
    pub wave: WaveField,
    /// If true, include `ω × r` in structure velocity at each station.
    pub include_spin: bool,
}

impl MorisonForce {
    /// Build a Morison force generator. Coefficients are caller-specified.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        body: usize,
        a_body: Vec3,
        b_body: Vec3,
        diameter: f64,
        cm: f64,
        cd: f64,
        n_segments: usize,
        rho: f64,
        wave: WaveField,
    ) -> Self {
        assert!(diameter > 0.0 && n_segments >= 1 && rho > 0.0);
        Self {
            body,
            a_body,
            b_body,
            diameter,
            cm,
            cd,
            n_segments,
            rho,
            wave,
            include_spin: false,
        }
    }

    /// Enable / disable spin contribution to relative velocity.
    pub fn with_spin(mut self, enable: bool) -> Self {
        self.include_spin = enable;
        self
    }
}

impl ForceGenerator for MorisonForce {
    fn apply(&self, world: &World, s: &State, t: f64, f: &mut [Vec3], torque: &mut [Vec3]) {
        let i = self.body;
        let q = s.q[i];
        let x = s.x[i];
        let a_w = x + q.rotate(self.a_body);
        let b_w = x + q.rotate(self.b_body);
        let member = MorisonMember::new(a_w, b_w, self.diameter, self.cm, self.cd, self.n_segments);
        let t_hat = member.tangent();
        let ds = member.length() / self.n_segments as f64;
        let a_sec = member.area();
        let omega = if self.include_spin {
            world.omega_i(s, i)
        } else {
            Vec3::zero()
        };
        let mut force = Vec3::zero();
        let mut tau = Vec3::zero();
        for seg in 0..self.n_segments {
            let sf = (seg as f64 + 0.5) / self.n_segments as f64;
            let r = a_w + (b_w - a_w) * sf;
            let r_rel = r - x;
            let v_s = s.v[i] + omega.cross(r_rel);
            let u_f = self.wave.velocity(r.x, r.y, r.z, t);
            let a_f = self.wave.acceleration(r.x, r.y, r.z, t);
            let u_n = normal_component(u_f - v_s, t_hat);
            let a_n = normal_component(a_f, t_hat);
            let u_n_mag = u_n.norm();
            let d_f_inertia = a_n * (self.rho * self.cm * a_sec);
            let d_f_drag = if u_n_mag > 0.0 {
                u_n * (0.5 * self.rho * self.cd * self.diameter * u_n_mag)
            } else {
                Vec3::zero()
            };
            let d_f = (d_f_inertia + d_f_drag) * ds;
            force += d_f;
            tau += r_rel.cross(d_f);
        }
        f[i] += force;
        torque[i] += tau;
    }
}
