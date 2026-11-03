//! Planar ship maneuvering: Nomoto first-order K–T and a linear MMG-style 3-DOF.
//!
//! # Nomoto first-order
//!
//! Yaw rate `r` under rudder angle `δ`:
//!
//! ```text
//! T ṙ + r = K δ
//! ```
//!
//! Analytic step response (`δ = δ₀` for `t ≥ 0`, `r(0) = 0`):
//!
//! ```text
//! r(t) = K δ₀ (1 − e^{−t/T})
//! ψ(t) = K δ₀ (t − T (1 − e^{−t/T}))
//! ```
//!
//! Turning circle at steady state: `r_∞ = K δ₀`, radius `R = U / |r_∞|`
//! (constant forward speed `U`). Zig-zag 10°/10° switching uses the same ODE
//! with piecewise-constant `δ`.
//!
//! Validation targets:
//! - **analytic Nomoto K–T** (closed form);
//! - **KVLCC2** MMG coefficients from Yasukawa & Yoshimura (2015) Table 3
//!   (primary PDF) with turning-circle advance / tactical diameter vs the
//!   free-running experiment reported in that paper (MARIN / SIMMAN L7-model).
//!
//! The linear MMG block still offers [`nominal_linear_mmg_unverified`] for
//! demos; prefer [`Kvlcc2Mmg`] for cited derivatives.
//!
//! # Linear MMG-style surge–sway–yaw
//!
//! Body-frame planar model (small deviation from steady surge `U`):
//!
//! ```text
//! (m+m_x) u̇ − (m+m_y) v r = X_u u + (1−t_p) T_prop
//! (m+m_y) v̇ + (m+m_x) U r = Y_v v + Y_r r + Y_δ δ
//! (I_z+J_z) ṙ            = N_v v + N_r r + N_δ δ
//! ```
//!
//! Coefficients are dimensional SI. Propeller thrust `T_prop` and rudder `δ`
//! are controls. **Derivative values marked unverified** unless the caller
//! supplies a cited set.
//!
//! # References
//! - Nomoto, K. et al. (1957). On the steering qualities of ships.
//! - Yasukawa, H. & Yoshimura, Y. (2015). Introduction of MMG standard method
//!   for ship maneuvering predictions. *J. Mar. Sci. Technol.* **20** — KVLCC2
//!   Tables 1/3/4 transcribed as published numbers (clean-room code).

use pe_math::libm_ops;

/// Nomoto first-order steering indices.
#[derive(Clone, Copy, Debug)]
pub struct NomotoKT {
    /// Gain `K` [1/s] (yaw rate per rudder radian).
    pub k: f64,
    /// Time constant `T` [s].
    pub t: f64,
}

impl NomotoKT {
    /// Construct with `K` [1/s], `T` [s] (`T > 0`).
    pub fn new(k: f64, t: f64) -> Self {
        assert!(t > 0.0, "Nomoto T must be positive");
        Self { k, t }
    }

    /// Analytic yaw-rate step response [rad/s] for constant `delta` [rad].
    pub fn r_step(&self, delta: f64, time: f64) -> f64 {
        if time <= 0.0 {
            return 0.0;
        }
        self.k * delta * (1.0 - libm_ops::exp(-time / self.t))
    }

    /// Analytic heading step response [rad] (`ψ(0)=0`).
    pub fn psi_step(&self, delta: f64, time: f64) -> f64 {
        if time <= 0.0 {
            return 0.0;
        }
        let e = libm_ops::exp(-time / self.t);
        self.k * delta * (time - self.t * (1.0 - e))
    }

    /// Steady turning rate [rad/s] at constant rudder.
    pub fn r_steady(&self, delta: f64) -> f64 {
        self.k * delta
    }

    /// Steady turning radius [m] at forward speed `u` [m/s].
    pub fn turning_radius(&self, u: f64, delta: f64) -> f64 {
        let r = self.r_steady(delta);
        if r.abs() < 1e-15 {
            return f64::INFINITY;
        }
        libm_ops::fabs(u / r)
    }
}

/// Nomoto yaw ODE state (numerical integration for zig-zag / comparison).
#[derive(Clone, Debug)]
pub struct NomotoState {
    /// Model.
    pub model: NomotoKT,
    /// Yaw rate [rad/s].
    pub r: f64,
    /// Heading [rad].
    pub psi: f64,
    /// Time [s].
    pub t: f64,
    /// Forward speed [m/s] (held constant for path integration).
    pub u: f64,
    /// Earth-frame position [m].
    pub x: f64,
    /// Earth-frame position [m].
    pub y: f64,
}

impl NomotoState {
    /// Rest initial condition at origin, heading 0.
    pub fn new(model: NomotoKT, u: f64) -> Self {
        Self {
            model,
            r: 0.0,
            psi: 0.0,
            t: 0.0,
            u,
            x: 0.0,
            y: 0.0,
        }
    }

    /// Semi-implicit Euler step with rudder `delta` [rad].
    pub fn step(&mut self, dt: f64, delta: f64) {
        // T ṙ = −r + K δ
        let r_dot = (-self.r + self.model.k * delta) / self.model.t;
        self.r += dt * r_dot;
        self.psi += dt * self.r;
        self.x += dt * self.u * libm_ops::cos(self.psi);
        self.y += dt * self.u * libm_ops::sin(self.psi);
        self.t += dt;
    }

    /// Simulate a constant-rudder turning manoeuvre; returns final state.
    pub fn turning_circle(mut self, delta: f64, dt: f64, n_steps: usize) -> Self {
        for _ in 0..n_steps {
            self.step(dt, delta);
        }
        self
    }
}

/// 10°/10° zig-zag: switch rudder when `|ψ|` reaches `ψ_switch`, reverse sign.
///
/// Returns samples of `(t, ψ, r, δ)`. First overshoot heading and time to
/// second rudder execute are useful manoeuvring metrics (vs Nomoto analytic
/// switching is numerical only — oracle is the ODE vs Nomoto step segments).
pub fn zigzag_10_10(
    model: NomotoKT,
    u: f64,
    delta_amp: f64,
    psi_switch: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<(f64, f64, f64, f64)> {
    let mut s = NomotoState::new(model, u);
    let mut delta = delta_amp;
    let mut out = Vec::with_capacity(n_steps + 1);
    out.push((s.t, s.psi, s.r, delta));
    for _ in 0..n_steps {
        // Switch when heading crosses ±psi_switch with matching sign of δ.
        if delta > 0.0 && s.psi >= psi_switch {
            delta = -delta_amp;
        } else if delta < 0.0 && s.psi <= -psi_switch {
            delta = delta_amp;
        }
        s.step(dt, delta);
        out.push((s.t, s.psi, s.r, delta));
    }
    out
}

/// Linear MMG-style hydrodynamic derivatives (dimensional SI).
///
/// All fields that are not derived from a cited table must be treated as
/// **unverified** when using the bundled [`nominal_linear_mmg_unverified`].
#[derive(Clone, Copy, Debug)]
pub struct LinearMmgDerivatives {
    /// Surge added mass `m_x` [kg].
    pub m_x: f64,
    /// Sway added mass `m_y` [kg].
    pub m_y: f64,
    /// Yaw added inertia `J_z` [kg·m²].
    pub j_z: f64,
    /// `X_u` [kg/s] (usually negative).
    pub x_u: f64,
    /// `Y_v` [kg/s].
    pub y_v: f64,
    /// `Y_r` [kg·m/s].
    pub y_r: f64,
    /// `Y_δ` [N/rad].
    pub y_delta: f64,
    /// `N_v` [kg·m/s].
    pub n_v: f64,
    /// `N_r` [kg·m²/s].
    pub n_r: f64,
    /// `N_δ` [N·m/rad].
    pub n_delta: f64,
    /// Thrust deduction factor `t_p` ∈ [0,1] (**unverified** if default).
    pub t_p: f64,
}

/// **Unverified** nominal derivatives scaled from mass / length for demos.
///
/// Not from KVLCC2/SIMMAN. Signs follow conventional stable course-keeping
/// (`Y_v < 0`, `N_r < 0`, `N_δ < 0` for δ>0 → bow to starboard in +r frame
/// depends on convention: here +δ → positive `N_δ` producing +r).
pub fn nominal_linear_mmg_unverified(
    mass: f64,
    length: f64,
    iz: f64,
    u0: f64,
) -> LinearMmgDerivatives {
    let _ = u0;
    LinearMmgDerivatives {
        m_x: 0.05 * mass,
        m_y: 0.20 * mass,
        j_z: 0.10 * iz,
        x_u: -0.02 * mass, // weak surge damping
        y_v: -0.30 * mass * (u0 / length.max(1.0)),
        y_r: 0.05 * mass * u0,
        y_delta: 0.15 * mass * u0 * u0 / length.max(1.0),
        n_v: -0.05 * mass * u0,
        n_r: -0.10 * iz * (u0 / length.max(1.0)),
        n_delta: -0.20 * mass * u0 * u0,
        t_p: 0.2,
    }
}

/// Extract equivalent Nomoto `(K, T)` from linear sway–yaw by eliminating `v`
/// under quasi-steady sway (`v̇ ≈ 0`) at speed `U`.
///
/// ```text
/// T = (I_z+J_z) / (−N_r + N_v Y_r / Y_v)     (when Y_v ≠ 0)
/// K = N_δ_eff / (−N_r + …) with rudder coupling through Y_δ
/// ```
///
/// Exact block reduction (Fossen / Nomoto textbooks). Returns `None` if the
/// sway damping vanishes.
pub fn nomoto_from_linear_mmg(
    d: &LinearMmgDerivatives,
    mass: f64,
    iz: f64,
    u: f64,
) -> Option<NomotoKT> {
    let _ = u;
    let y_v = d.y_v;
    if y_v.abs() < 1e-18 {
        return None;
    }
    // Quasi-steady: 0 = Y_v v + Y_r r + Y_δ δ  ⇒  v = −(Y_r r + Y_δ δ)/Y_v
    // N: (Iz+Jz) ṙ = N_v v + N_r r + N_δ δ
    //   = [N_r − N_v Y_r/Y_v] r + [N_δ − N_v Y_δ/Y_v] δ
    let n_r_eff = d.n_r - d.n_v * d.y_r / y_v;
    let n_d_eff = d.n_delta - d.n_v * d.y_delta / y_v;
    let iz_tot = iz + d.j_z;
    // (Iz) ṙ − n_r_eff r = n_d_eff δ
    // T ṙ + r = K δ  with T = −Iz/n_r_eff, K = −n_d_eff/n_r_eff  when n_r_eff<0
    if n_r_eff >= 0.0 {
        return None; // unstable / non-Nomoto sign
    }
    let t = -iz_tot / n_r_eff;
    let k = -n_d_eff / n_r_eff;
    let _ = mass;
    Some(NomotoKT::new(k, t))
}

/// Planar MMG state (surge, sway, yaw).
#[derive(Clone, Debug)]
pub struct MmgPlanar {
    /// Mass [kg].
    pub mass: f64,
    /// Yaw inertia [kg·m²].
    pub iz: f64,
    /// Derivatives.
    pub d: LinearMmgDerivatives,
    /// Surge velocity [m/s] (body).
    pub u: f64,
    /// Sway velocity [m/s] (body).
    pub v: f64,
    /// Yaw rate [rad/s].
    pub r: f64,
    /// Heading [rad].
    pub psi: f64,
    /// Earth x [m].
    pub x: f64,
    /// Earth y [m].
    pub y: f64,
    /// Time [s].
    pub t: f64,
}

impl MmgPlanar {
    /// Start at steady surge `u0`, zero sway/yaw.
    pub fn new(mass: f64, iz: f64, d: LinearMmgDerivatives, u0: f64) -> Self {
        Self {
            mass,
            iz,
            d,
            u: u0,
            v: 0.0,
            r: 0.0,
            psi: 0.0,
            x: 0.0,
            y: 0.0,
            t: 0.0,
        }
    }

    /// One semi-implicit step with propeller thrust [N] and rudder [rad].
    pub fn step(&mut self, dt: f64, thrust: f64, delta: f64) {
        let d = &self.d;
        let m_u = self.mass + d.m_x;
        let m_v = self.mass + d.m_y;
        let i_r = self.iz + d.j_z;
        // Forces / moments (linear)
        let x_force = d.x_u * self.u + (1.0 - d.t_p) * thrust + m_v * self.v * self.r;
        let y_force = d.y_v * self.v + d.y_r * self.r + d.y_delta * delta - m_u * self.u * self.r;
        let n_mom = d.n_v * self.v + d.n_r * self.r + d.n_delta * delta;
        let u_dot = x_force / m_u;
        let v_dot = y_force / m_v;
        let r_dot = n_mom / i_r;
        self.u += dt * u_dot;
        self.v += dt * v_dot;
        self.r += dt * r_dot;
        self.psi += dt * self.r;
        // η̇ = R(ψ) ν
        let c = libm_ops::cos(self.psi);
        let s = libm_ops::sin(self.psi);
        self.x += dt * (c * self.u - s * self.v);
        self.y += dt * (s * self.u + c * self.v);
        self.t += dt;
    }
}

/// Propeller thrust [N] from a simple open-water surrogate
/// `T = ρ n² D⁴ K_T` with **unverified** `K_T` (caller should pass a cited map).
pub fn propeller_thrust_unverified(rho: f64, n_rps: f64, diameter: f64, k_t: f64) -> f64 {
    let n = n_rps;
    let d = diameter;
    rho * n * n * d * d * d * d * k_t
}

/// Rudder lift-style side force [N]: `F = ½ ρ A_r V_r² C_Lα α_r` (**unverified**
/// `C_Lα` unless supplied from a cited foil).
pub fn rudder_force_unverified(
    rho: f64,
    area: f64,
    flow_speed: f64,
    cl_alpha: f64,
    alpha_rad: f64,
) -> f64 {
    0.5 * rho * area * flow_speed * flow_speed * cl_alpha * alpha_rad
}

// ---------------------------------------------------------------------------
// KVLCC2 — Yasukawa & Yoshimura (2015) MMG standard method
// ---------------------------------------------------------------------------

/// KVLCC2 L7-model geometry (Yasukawa & Yoshimura 2015, Table 1).
#[derive(Clone, Copy, Debug)]
pub struct Kvlcc2Geom {
    /// Length between perpendiculars [m].
    pub lpp: f64,
    /// Beam [m].
    pub beam: f64,
    /// Draft [m].
    pub draft: f64,
    /// Displacement volume [m³].
    pub volume: f64,
    /// Longitudinal COM from midship [m] (+ forward).
    pub x_g: f64,
    /// Propeller diameter [m].
    pub d_p: f64,
    /// Rudder span [m].
    pub h_r: f64,
    /// Rudder movable-area [m²].
    pub a_r: f64,
}

/// L7-model particulars from Yasukawa & Yoshimura (2015) Table 1.
pub fn kvlcc2_l7_geom() -> Kvlcc2Geom {
    Kvlcc2Geom {
        lpp: 7.00,
        beam: 1.27,
        draft: 0.46,
        volume: 3.27,
        x_g: 0.25,
        d_p: 0.216,
        h_r: 0.345,
        a_r: 0.0539,
    }
}

/// Nondimensional MMG coefficients for KVLCC2 (Yasukawa & Yoshimura 2015 Table 3).
///
/// Force/moment nondimensionalisation follows the paper:
/// `F' = F / (½ ρ L d U²)`, `N' = N / (½ ρ L² d U²)`,
/// `m' = m / (½ ρ L² d)`, rates `v' = v/U`, `r' = r L / U`.
#[derive(Clone, Copy, Debug)]
pub struct Kvlcc2MmgCoeffs {
    /// Straight-line resistance `R₀'`.
    pub r0: f64,
    /// `X'vv`.
    pub x_vv: f64,
    /// `X'vr`.
    pub x_vr: f64,
    /// `X'rr`.
    pub x_rr: f64,
    /// `X'vvvv`.
    pub x_vvvv: f64,
    /// `Y'v`.
    pub y_v: f64,
    /// `Y'r` (paper `Y'R`).
    pub y_r: f64,
    /// `Y'vvv`.
    pub y_vvv: f64,
    /// `Y'vvr`.
    pub y_vvr: f64,
    /// `Y'vrr`.
    pub y_vrr: f64,
    /// `Y'rrr`.
    pub y_rrr: f64,
    /// `N'v`.
    pub n_v: f64,
    /// `N'r` (paper `N'R`).
    pub n_r: f64,
    /// `N'vvv`.
    pub n_vvv: f64,
    /// `N'vvr`.
    pub n_vvr: f64,
    /// `N'vrr`.
    pub n_vrr: f64,
    /// `N'rrr`.
    pub n_rrr: f64,
    /// `m'x`.
    pub m_x: f64,
    /// `m'y`.
    pub m_y: f64,
    /// `J'z`.
    pub j_z: f64,
    /// Thrust deduction `t_P`.
    pub t_p: f64,
    /// Rudder resistance deduction `t_R`.
    pub t_r: f64,
    /// Rudder force increase factor `a_H`.
    pub a_h: f64,
    /// Additional lateral-force point `x'H`.
    pub x_h: f64,
    /// Wake-change `C₁`.
    pub c1: f64,
    /// Wake `C₂` for `β_P > 0`.
    pub c2_pos: f64,
    /// Wake `C₂` for `β_P < 0`.
    pub c2_neg: f64,
    /// Flow straightening `γ_R` (`β_R < 0`).
    pub gamma_r_neg: f64,
    /// Flow straightening `γ_R` (`β_R > 0`).
    pub gamma_r_pos: f64,
    /// Effective rudder lever `ℓ'R`.
    pub l_r: f64,
    /// Wake ratio `ε`.
    pub epsilon: f64,
    /// Propeller–rudder interaction `κ`.
    pub kappa: f64,
    /// Rudder lift gradient `f_α` (Fujii).
    pub f_alpha: f64,
    /// Straight-motion wake `w_P0` (L7-model: 0.40).
    pub w_p0: f64,
    /// Open-water `K_T` polynomial `k₀ + k₁ J + k₂ J²`.
    pub k_t: [f64; 3],
    /// Rudder station `x'R` (= −0.5).
    pub x_r: f64,
}

/// Yasukawa & Yoshimura (2015) Table 3 + stated `w_P0=0.40`,
/// `(k₀,k₁,k₂)=(0.2931,−0.2753,−0.1385)`, `x'R=−0.5`.
pub fn kvlcc2_yasukawa2015_coeffs() -> Kvlcc2MmgCoeffs {
    Kvlcc2MmgCoeffs {
        r0: 0.022,
        x_vv: -0.040,
        x_vr: 0.002,
        x_rr: 0.011,
        x_vvvv: 0.771,
        y_v: -0.315,
        y_r: 0.083,
        y_vvv: -1.607,
        y_vvr: 0.379,
        y_vrr: -0.391,
        y_rrr: 0.008,
        n_v: -0.137,
        n_r: -0.049,
        n_vvv: -0.030,
        n_vvr: -0.294,
        n_vrr: 0.055,
        n_rrr: -0.013,
        m_x: 0.022,
        m_y: 0.223,
        j_z: 0.011,
        t_p: 0.220,
        t_r: 0.387,
        a_h: 0.312,
        x_h: -0.464,
        c1: 2.0,
        c2_pos: 1.6,
        c2_neg: 1.1,
        gamma_r_neg: 0.395,
        gamma_r_pos: 0.640,
        l_r: -0.710,
        epsilon: 1.09,
        kappa: 0.50,
        f_alpha: 2.747,
        w_p0: 0.40,
        k_t: [0.2931, -0.2753, -0.1385],
        x_r: -0.5,
    }
}

/// Published L7 free-running turning indices (Yasukawa & Yoshimura 2015 Table 4 Exp.).
///
/// Source experiment: MARIN / SIMMAN L7-model, reported as Exp. in that table.
#[derive(Clone, Copy, Debug)]
pub struct TurningIndices {
    /// Advance / `Lpp` at `δ = +35°`.
    pub ad_pos35: f64,
    /// Tactical diameter / `Lpp` at `δ = +35°`.
    pub dt_pos35: f64,
    /// Advance / `Lpp` at `δ = −35°`.
    pub ad_neg35: f64,
    /// Tactical diameter / `Lpp` at `δ = −35°`.
    pub dt_neg35: f64,
}

/// MARIN FRMT turning indices from Yasukawa & Yoshimura (2015) Table 4 Exp.
pub fn kvlcc2_marin_frmt_turning_35() -> TurningIndices {
    TurningIndices {
        ad_pos35: 3.25,
        dt_pos35: 3.34,
        ad_neg35: 3.11,
        dt_neg35: 3.08,
    }
}

/// 3-DOF MMG state for KVLCC2 (Yasukawa standard method).
#[derive(Clone, Debug)]
pub struct Kvlcc2Mmg {
    /// Geometry.
    pub geom: Kvlcc2Geom,
    /// Coefficients.
    pub c: Kvlcc2MmgCoeffs,
    /// Fluid density [kg/m³].
    pub rho: f64,
    /// Ship mass [kg].
    pub mass: f64,
    /// Yaw inertia about midship [kg·m²] (`I_zG + m x_G²`).
    pub i_z: f64,
    /// Propeller revolutions [1/s] (held constant).
    pub n_p: f64,
    /// Surge [m/s].
    pub u: f64,
    /// Sway [m/s].
    pub v: f64,
    /// Yaw rate [rad/s].
    pub r: f64,
    /// Heading [rad].
    pub psi: f64,
    /// Earth x [m].
    pub x: f64,
    /// Earth y [m].
    pub y: f64,
    /// Time [s].
    pub t: f64,
}

impl Kvlcc2Mmg {
    /// L7-model at approach speed `u0` with self-propulsion `n_p` solved from
    /// straight-line balance `X_P = R₀` at `U=u0`, `v=r=δ=0`.
    pub fn l7_at_speed(rho: f64, u0: f64) -> Self {
        let geom = kvlcc2_l7_geom();
        let c = kvlcc2_yasukawa2015_coeffs();
        let mass = rho * geom.volume;
        // Radius of yaw gyration 0.25 Lpp about COM (paper §5); IzG = m k²,
        // midship inertia Iz = IzG + m xG².
        let k_yy = 0.25 * geom.lpp;
        let i_zg = mass * k_yy * k_yy;
        let i_z = i_zg + mass * geom.x_g * geom.x_g;
        let n_p = self_propulsion_rps(&geom, &c, rho, u0);
        Self {
            geom,
            c,
            rho,
            mass,
            i_z,
            n_p,
            u: u0,
            v: 0.0,
            r: 0.0,
            psi: 0.0,
            x: 0.0,
            y: 0.0,
            t: 0.0,
        }
    }

    /// Semi-implicit Euler step with rudder `delta` [rad].
    pub fn step(&mut self, dt: f64, delta: f64) {
        let (x_f, y_f, n_m) = self.forces(delta);
        let g = &self.geom;
        let c = &self.c;
        let half_rho_l2_d = 0.5 * self.rho * g.lpp * g.lpp * g.draft;
        let m = self.mass;
        let m_x = c.m_x * half_rho_l2_d;
        let m_y = c.m_y * half_rho_l2_d;
        let j_z = c.j_z * half_rho_l2_d * g.lpp * g.lpp;
        let x_g = g.x_g;

        // (m+mx) ú − (m+my) v r − xG m r² = X
        // (m+my) v̇ + (m+mx) u r + xG m ṙ = Y
        // (Iz + Jz) ṙ + xG m (v̇ + u r) = N
        let m11 = m + m_x;
        let m22 = m + m_y;
        let m33 = self.i_z + j_z;
        // Solve the coupled (v̇, ṙ) 2×2; ú is explicit.
        // m22 v̇ + xG m ṙ = Y − (m+mx) u r
        // xG m v̇ + m33 ṙ = N − xG m u r
        let rhs_v = y_f - (m + m_x) * self.u * self.r;
        let rhs_r = n_m - x_g * m * self.u * self.r;
        let det = m22 * m33 - (x_g * m) * (x_g * m);
        let (v_dot, r_dot) = if libm_ops::fabs(det) < 1e-30 {
            (rhs_v / m22, rhs_r / m33)
        } else {
            let vd = (rhs_v * m33 - rhs_r * (x_g * m)) / det;
            let rd = (m22 * rhs_r - (x_g * m) * rhs_v) / det;
            (vd, rd)
        };
        let u_dot = (x_f + (m + m_y) * self.v * self.r + x_g * m * self.r * self.r) / m11;

        self.u += dt * u_dot;
        self.v += dt * v_dot;
        self.r += dt * r_dot;
        self.psi += dt * self.r;
        let ch = libm_ops::cos(self.psi);
        let sh = libm_ops::sin(self.psi);
        self.x += dt * (ch * self.u - sh * self.v);
        self.y += dt * (sh * self.u + ch * self.v);
        self.t += dt;
    }

    /// Hull + propeller + rudder forces/moments [N], [N], [N·m] at midship.
    pub fn forces(&self, delta: f64) -> (f64, f64, f64) {
        let g = &self.geom;
        let c = &self.c;
        let u = self.u;
        let v = self.v;
        let r = self.r;
        let u_sq = u * u + v * v;
        let u_abs = libm_ops::sqrt(u_sq).max(1e-9);
        let half = 0.5 * self.rho * g.lpp * g.draft * u_abs * u_abs;
        let half_n = half * g.lpp;
        let v_p = v / u_abs;
        let r_p = r * g.lpp / u_abs;

        // Hull (Yasukawa Eq. 3 / Table 3)
        let x_h = -c.r0
            + c.x_vv * v_p * v_p
            + c.x_vr * v_p * r_p
            + c.x_rr * r_p * r_p
            + c.x_vvvv * v_p * v_p * v_p * v_p;
        let y_h = c.y_v * v_p
            + c.y_r * r_p
            + c.y_vvv * v_p * v_p * v_p
            + c.y_vvr * v_p * v_p * r_p
            + c.y_vrr * v_p * r_p * r_p
            + c.y_rrr * r_p * r_p * r_p;
        let n_h = c.n_v * v_p
            + c.n_r * r_p
            + c.n_vvv * v_p * v_p * v_p
            + c.n_vvr * v_p * v_p * r_p
            + c.n_vrr * v_p * r_p * r_p
            + c.n_rrr * r_p * r_p * r_p;

        // Propeller (wake Eq. 16; open-water KT polynomial from the paper).
        let beta = libm_ops::atan2(-v, u.max(1e-9));
        let beta_p = libm_ops::atan2(-(v - 0.5 * g.lpp * r), u.max(1e-9));
        let c2 = if beta_p > 0.0 { c.c2_pos } else { c.c2_neg };
        let wake_ratio = 1.0 + (1.0 - libm_ops::exp(-c.c1 * libm_ops::fabs(beta_p))) * (c2 - 1.0);
        let w_p = 1.0 - (1.0 - c.w_p0) * wake_ratio;
        let j_p = if libm_ops::fabs(self.n_p) < 1e-12 {
            0.0
        } else {
            u * (1.0 - w_p) / (self.n_p * g.d_p)
        };
        let k_t = c.k_t[0] + c.k_t[1] * j_p + c.k_t[2] * j_p * j_p;
        let thrust = self.rho * self.n_p * self.n_p * g.d_p.powi(4) * k_t;
        let x_p = (1.0 - c.t_p) * thrust;

        // Rudder: u_R = ε u (1−w_P) √{η[1+κ(√(1+8K_T/(πJ²))−1)]²+(1−η)}
        let eta = g.d_p / g.h_r;
        let s_prop = {
            let jp2 = j_p * j_p + 1e-30;
            let root = libm_ops::sqrt((1.0 + 8.0 * k_t / (core::f64::consts::PI * jp2)).max(0.0));
            let term = 1.0 + c.kappa * (root - 1.0);
            eta * term * term + (1.0 - eta)
        };
        let u_r = c.epsilon * u * (1.0 - w_p) * libm_ops::sqrt(s_prop.max(0.0));
        let beta_r = beta - c.l_r * r_p;
        let gamma = if beta_r > 0.0 {
            c.gamma_r_pos
        } else {
            c.gamma_r_neg
        };
        let v_r = u_abs * gamma * beta_r;
        let alpha_r = delta - libm_ops::atan2(v_r, u_r.max(1e-9));
        let f_n =
            0.5 * self.rho * g.a_r * (u_r * u_r + v_r * v_r) * c.f_alpha * libm_ops::sin(alpha_r);
        let x_rud = -(1.0 - c.t_r) * f_n * libm_ops::sin(delta);
        let y_rud = -(1.0 + c.a_h) * f_n * libm_ops::cos(delta);
        let n_rud = -(c.x_r + c.a_h * c.x_h) * g.lpp * f_n * libm_ops::cos(delta);

        (
            half * x_h + x_p + x_rud,
            half * y_h + y_rud,
            half_n * n_h + n_rud,
        )
    }

    /// Simulate a constant-rudder turning manoeuvre; returns advance & tactical
    /// diameter in units of `Lpp` (first time heading changes by 90° / 180°).
    pub fn turning_indices(&self, delta: f64, dt: f64, max_steps: usize) -> Option<(f64, f64)> {
        let mut s = self.clone();
        let x0 = s.x;
        let y0 = s.y;
        let mut ad = None;
        let mut dt_idx = None;
        for _ in 0..max_steps {
            s.step(dt, delta);
            let dpsi = libm_ops::fabs(s.psi);
            if ad.is_none() && dpsi >= 0.5 * core::f64::consts::PI {
                // Advance: longitudinal travel of midship in initial axes.
                ad = Some((s.x - x0) / self.geom.lpp);
            }
            if dt_idx.is_none() && dpsi >= core::f64::consts::PI {
                dt_idx = Some(libm_ops::fabs(s.y - y0) / self.geom.lpp);
                break;
            }
        }
        match (ad, dt_idx) {
            (Some(a), Some(d)) => Some((a, d)),
            _ => None,
        }
    }
}

fn self_propulsion_rps(geom: &Kvlcc2Geom, c: &Kvlcc2MmgCoeffs, rho: f64, u0: f64) -> f64 {
    // R0 = R0' * ½ ρ L d U² ; need (1−tP) ρ n² D⁴ KT(J) = R0 with
    // J = U(1−wP0)/(n D). Solve by a few Newton / secure iterations on n.
    let r0 = c.r0 * 0.5 * rho * geom.lpp * geom.draft * u0 * u0;
    let target = r0; // XP = (1−tP) T = R0
    let mut n = u0 * (1.0 - c.w_p0) / (0.5 * geom.d_p); // guess J≈0.5
    for _ in 0..30 {
        let j = u0 * (1.0 - c.w_p0) / (n * geom.d_p + 1e-30);
        let kt = c.k_t[0] + c.k_t[1] * j + c.k_t[2] * j * j;
        let xp = (1.0 - c.t_p) * rho * n * n * geom.d_p.powi(4) * kt;
        let err = xp - target;
        if libm_ops::fabs(err) < 1e-6 * target.abs().max(1.0) {
            break;
        }
        // dXP/dn ≈ 2 (1−tP) ρ n D⁴ KT + (1−tP) ρ n² D⁴ (k1+2k2 J) dJ/dn
        // dJ/dn = −U(1−w)/ (n² D)
        let d_j_dn = -u0 * (1.0 - c.w_p0) / (n * n * geom.d_p + 1e-30);
        let d_kt_dn = (c.k_t[1] + 2.0 * c.k_t[2] * j) * d_j_dn;
        let d_xp = (1.0 - c.t_p) * rho * geom.d_p.powi(4) * (2.0 * n * kt + n * n * d_kt_dn);
        if libm_ops::fabs(d_xp) < 1e-18 {
            break;
        }
        n -= err / d_xp;
        if n < 1e-3 {
            n = 1e-3;
        }
    }
    n
}
