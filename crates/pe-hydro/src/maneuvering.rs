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
//! Validation target: **analytic Nomoto K–T** (closed form). Published
//! KVLCC2 / SIMMAN hydrodynamic derivatives are **not** transcribed here
//! (would require a verified primary table). The linear MMG block below uses
//! **nominal / unverified** nondimensional derivatives for demonstration and
//! reduces to Nomoto when sway is quasi-steady.
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
//! - MMG standard method — Yasukawa & Yoshimura (2015) *J. Mar. Sci. Technol.*
//!   (structure only; no KVLCC2 table copied).

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
