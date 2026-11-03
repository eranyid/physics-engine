//! Six-DOF Cummins equation with matrix retardation kernels.
//!
//! Coupled time-domain seakeeping (Cummins 1962 / Ogilvie 1964) for
//! surge, sway, heave, roll, pitch, yaw:
//!
//! ```text
//! (M + A_∞) ξ̈ + ∫_0^t K(t−τ) ξ̇(τ) dτ + C ξ = F(t)
//! ```
//!
//! with matrix-valued infinite-frequency added mass `A_∞`, retardation
//! `K_ij(t) = (2/π) ∫_0^{ω_max} B_ij(ω) cos(ω t) dω`, and hydrostatic
//! restoring `C` from the GM/GZ model (waterplane + metacentric).
//!
//! # DOF order
//!
//! `| 0 surge | 1 sway | 2 heave | 3 roll | 4 pitch | 5 yaw |`
//!
//! # Hydrodynamics source (box barge)
//!
//! Port–starboard and (approx.) fore–aft symmetry ⇒ radiation couples in
//! two blocks: longitudinal `(0,2,4)` and lateral `(1,3,5)`. Strip `B_ij(ω)`
//! and `A_∞` come from Lewis-circular sectional formulas (same nominal
//! circle form as [`crate::seakeeping`]; **not** Vugts tables).
//!
//! # Wave excitation
//!
//! Froude–Krylov from Airy pressure on strips **plus** STF Haskind
//! diffraction `−iω ∫ b₃₃ e^{ikx} dx` for heave/pitch (head seas). Lateral
//! excitation in beam seas uses the analogous sway strip Haskind form.
//! Diffraction class: **STF strip Haskind approximation** (not 3-D panel).
//!
//! # Bounded memory / truncation
//!
//! Kernels are truncated at horizon `T = (n−1) Δt`. The omitted tail bound
//! for a monotone-decaying positive `K_ii` is documented via
//! [`truncation_tail_l1`]: `E_trunc ≤ ‖v‖_∞ ∫_{T}^{∞} |K| dt` estimated from
//! the last sample under exponential decay assumption
//! `K(t) ≈ K(T) e^{−(t−T)/τ}` with `τ` from the late-time slope (**engineering
//! estimate**). Convolution uses trapezoidal weights on the truncated grid
//! (deterministic, fixed length history).
//!
//! # Determinism
//!
//! Portable [`pe_math::libm_ops`] only; no FMA; no `HashMap`; fixed DOF and
//! strip order.
//!
//! # References
//! - Cummins (1962); Ogilvie (1964); Salvesen–Tuck–Faltinsen (1970).

use pe_math::libm_ops;

use crate::cummins::retardation_from_damping;
use crate::seakeeping::{
    box_barge_stations, excitation_fk_diffraction, sectional_a33_inf, sectional_b33, StripStation,
};

/// Number of rigid-body DOFs.
pub const N_DOF: usize = 6;

/// Matrix retardation kernel on a uniform time grid.
#[derive(Clone, Debug)]
pub struct MatrixRetardationKernel {
    /// Sample spacing `Δt` [s].
    pub dt: f64,
    /// Number of time samples per `K_ij`.
    pub n_time: usize,
    /// Flat storage `K[i][j][t]` with index `((i * 6 + j) * n_time + t)`.
    pub k: Vec<f64>,
    /// Infinite-frequency added mass `A_∞` [SI mixed].
    pub a_inf: [[f64; N_DOF]; N_DOF],
    /// Hydrostatic restoring `C` [SI mixed].
    pub c: [[f64; N_DOF]; N_DOF],
}

impl MatrixRetardationKernel {
    /// Zero kernel with given `dt` and length.
    pub fn zeros(dt: f64, n_time: usize) -> Self {
        Self {
            dt,
            n_time,
            k: vec![0.0; N_DOF * N_DOF * n_time],
            a_inf: [[0.0; N_DOF]; N_DOF],
            c: [[0.0; N_DOF]; N_DOF],
        }
    }

    /// `K_ij` at time index `t`.
    pub fn get(&self, i: usize, j: usize, t: usize) -> f64 {
        self.k[(i * N_DOF + j) * self.n_time + t]
    }

    /// Set `K_ij` at time index `t`.
    pub fn set(&mut self, i: usize, j: usize, t: usize, val: f64) {
        self.k[(i * N_DOF + j) * self.n_time + t] = val;
    }

    /// Memory horizon [s].
    pub fn horizon(&self) -> f64 {
        if self.n_time <= 1 {
            0.0
        } else {
            (self.n_time - 1) as f64 * self.dt
        }
    }

    /// Write a full `K_ij(t)` sample vector.
    pub fn set_channel(&mut self, i: usize, j: usize, samples: &[f64]) {
        assert_eq!(samples.len(), self.n_time);
        for (t, &v) in samples.iter().enumerate() {
            self.set(i, j, t, v);
        }
    }
}

/// L1 tail estimate for diagonal kernel `K_ii` beyond the stored horizon.
///
/// Assumes late-time exponential decay from the last two positive samples:
/// `τ = −Δt / ln(K_n / K_{n−1})`, then `∫_T^∞ |K| ≈ |K_n| τ`.
/// Returns `0` if the slope is non-decaying. **Engineering truncation-error
/// estimate**, not a rigorous spectral bound.
pub fn truncation_tail_l1(kernel: &MatrixRetardationKernel, i: usize) -> f64 {
    let n = kernel.n_time;
    if n < 2 {
        return 0.0;
    }
    let k_n = kernel.get(i, i, n - 1);
    let k_nm = kernel.get(i, i, n - 2);
    if k_n <= 0.0 || k_nm <= k_n {
        return libm_ops::fabs(k_n) * kernel.dt; // one-step fallback
    }
    let ratio = k_n / k_nm;
    if ratio <= 0.0 || ratio >= 1.0 {
        return libm_ops::fabs(k_n) * kernel.dt;
    }
    let tau = -kernel.dt / libm_ops::log(ratio);
    libm_ops::fabs(k_n) * tau
}

/// Structural mass matrix (diagonal) for a box barge at the COM.
#[derive(Clone, Copy, Debug)]
pub struct RigidMass6 {
    /// Mass [kg].
    pub mass: f64,
    /// Roll inertia `I_xx` [kg·m²].
    pub ixx: f64,
    /// Pitch inertia `I_yy` [kg·m²].
    pub iyy: f64,
    /// Yaw inertia `I_zz` [kg·m²].
    pub izz: f64,
}

impl RigidMass6 {
    /// Diagonal mass matrix.
    pub fn matrix(&self) -> [[f64; N_DOF]; N_DOF] {
        let mut m = [[0.0; N_DOF]; N_DOF];
        m[0][0] = self.mass;
        m[1][1] = self.mass;
        m[2][2] = self.mass;
        m[3][3] = self.ixx;
        m[4][4] = self.iyy;
        m[5][5] = self.izz;
        m
    }
}

/// Hydrostatic restoring for a wall-sided box about upright equilibrium.
///
/// ```text
/// C₃₃ = ρ g A_wp
/// C₃₅ = C₅₃ = −ρ g ∫ x B dx   (0 for F–A symmetric box about midship)
/// C₅₅ = ρ g ∫ x² B dx        (= ρ g L³ B / 12 for uniform box)
/// C₄₄ = ρ g ∇ GM_T
/// C₅₅ also += ρ g ∇ GM_L residual already in waterplane for barges with
///            KG on the waterplane centroid — we use waterplane pitch restoring
///            from strips plus optional GM_L override.
/// ```
///
/// Surge/sway/yaw hydrostatic restoring are zero in deep water (no soft
/// mooring). Roll uses `GM_T`; pitch waterplane term from strips.
pub fn box_hydrostatic_restoring(
    length: f64,
    beam: f64,
    draft: f64,
    rho: f64,
    g: f64,
    gm_t: f64,
    n_strips: usize,
) -> [[f64; N_DOF]; N_DOF] {
    let stations = box_barge_stations(length, beam, draft, n_strips);
    let mut c = [[0.0; N_DOF]; N_DOF];
    for s in &stations {
        let c33 = rho * g * s.beam * s.dx;
        c[2][2] += c33;
        c[2][4] += -s.x * c33;
        c[4][2] += -s.x * c33;
        c[4][4] += s.x * s.x * c33;
    }
    let volume = length * beam * draft;
    c[3][3] = rho * g * volume * gm_t;
    c
}

/// Sectional sway added mass per unit length (circular Lewis substitute).
///
/// Same circle limit as heave: `a₂₂∞ = ρ π (B/2)² σ` (Newman / Faltinsen
/// floating-circle; beam is the waterline breadth). **Nominal** for rectangles.
pub fn sectional_a22_inf(rho: f64, beam: f64, area_coeff: f64) -> f64 {
    sectional_a33_inf(rho, beam, area_coeff)
}

/// Sectional sway damping (nominal circle form, same β as heave).
pub fn sectional_b22(rho: f64, omega: f64, beam: f64, draft: f64, area_coeff: f64, g: f64) -> f64 {
    sectional_b33(rho, omega, beam, draft, area_coeff, g)
}

/// Build 6×6 `A_∞` for a box barge from strip integrals (symmetry blocks).
///
/// - Heave–pitch from `a₃₃∞`
/// - Sway–roll–yaw from `a₂₂∞` with lever `z_b = −draft/2` (section centroid
///   below waterplane; roll about waterplane COM approximation)
/// - Surge: `A₁₁∞ = C_x ρ ∇` with **nominal** `C_x = 0.5` (unverified; same
///   order as [`crate::helpers::box_added_mass_surge`])
pub fn box_a_inf_matrix(
    length: f64,
    beam: f64,
    draft: f64,
    rho: f64,
    n_strips: usize,
) -> [[f64; N_DOF]; N_DOF] {
    let stations = box_barge_stations(length, beam, draft, n_strips);
    let mut a = [[0.0; N_DOF]; N_DOF];
    let z_b = -0.5 * draft;
    for s in &stations {
        let a33 = sectional_a33_inf(rho, s.beam, s.area_coeff);
        let a22 = sectional_a22_inf(rho, s.beam, s.area_coeff);
        let dx = s.dx;
        let x = s.x;
        // Heave–pitch block
        a[2][2] += a33 * dx;
        a[2][4] += -x * a33 * dx;
        a[4][2] += -x * a33 * dx;
        a[4][4] += x * x * a33 * dx;
        // Sway–roll–yaw block (strip a22; roll lever z_b; yaw lever x)
        a[1][1] += a22 * dx;
        a[1][3] += z_b * a22 * dx;
        a[3][1] += z_b * a22 * dx;
        a[3][3] += z_b * z_b * a22 * dx;
        a[1][5] += -x * a22 * dx;
        a[5][1] += -x * a22 * dx;
        a[3][5] += -x * z_b * a22 * dx;
        a[5][3] += -x * z_b * a22 * dx;
        a[5][5] += x * x * a22 * dx;
    }
    // Surge: nominal Cx=0.5 (unverified)
    const CX_NOMINAL: f64 = 0.5;
    let volume = length * beam * draft;
    a[0][0] = CX_NOMINAL * rho * volume;
    a
}

/// Sample strip damping matrices `B_ij(ω)` for coupled channels used in K.
fn sample_b_matrix(
    stations: &[StripStation],
    rho: f64,
    omega: f64,
    g: f64,
    draft: f64,
) -> [[f64; N_DOF]; N_DOF] {
    let mut b = [[0.0; N_DOF]; N_DOF];
    let z_b = -0.5 * draft;
    for s in stations {
        let b33 = sectional_b33(rho, omega, s.beam, s.draft, s.area_coeff, g);
        let b22 = sectional_b22(rho, omega, s.beam, s.draft, s.area_coeff, g);
        let dx = s.dx;
        let x = s.x;
        b[2][2] += b33 * dx;
        b[2][4] += -x * b33 * dx;
        b[4][2] += -x * b33 * dx;
        b[4][4] += x * x * b33 * dx;
        b[1][1] += b22 * dx;
        b[1][3] += z_b * b22 * dx;
        b[3][1] += z_b * b22 * dx;
        b[3][3] += z_b * z_b * b22 * dx;
        b[1][5] += -x * b22 * dx;
        b[5][1] += -x * b22 * dx;
        b[3][5] += -x * z_b * b22 * dx;
        b[5][3] += -x * z_b * b22 * dx;
        b[5][5] += x * x * b22 * dx;
        // Surge radiation damping left at 0 (potential soft mode).
    }
    b
}

/// Channels `(i,j)` that receive non-zero retardation for a symmetric box.
const ACTIVE_CHANNELS: &[(usize, usize)] = &[
    (1, 1),
    (1, 3),
    (1, 5),
    (2, 2),
    (2, 4),
    (3, 1),
    (3, 3),
    (3, 5),
    (4, 2),
    (4, 4),
    (5, 1),
    (5, 3),
    (5, 5),
];

/// Build a 6-DOF retardation kernel for a box barge.
#[allow(clippy::too_many_arguments)]
pub fn box_retardation_6dof(
    length: f64,
    beam: f64,
    draft: f64,
    rho: f64,
    g: f64,
    gm_t: f64,
    n_strips: usize,
    omega_min: f64,
    omega_max: f64,
    n_omega: usize,
    dt: f64,
    n_time: usize,
) -> MatrixRetardationKernel {
    assert!(n_omega >= 8 && n_time >= 8 && dt > 0.0 && omega_max > omega_min);
    let stations = box_barge_stations(length, beam, draft, n_strips);
    let mut omega = Vec::with_capacity(n_omega);
    for i in 0..n_omega {
        omega.push(omega_min + (omega_max - omega_min) * (i as f64) / (n_omega as f64 - 1.0));
    }
    let t_samples: Vec<f64> = (0..n_time).map(|i| i as f64 * dt).collect();
    let mut kernel = MatrixRetardationKernel::zeros(dt, n_time);
    kernel.a_inf = box_a_inf_matrix(length, beam, draft, rho, n_strips);
    kernel.c = box_hydrostatic_restoring(length, beam, draft, rho, g, gm_t, n_strips);

    for &(ii, jj) in ACTIVE_CHANNELS {
        let mut b_w = Vec::with_capacity(n_omega);
        for &w in &omega {
            let bm = sample_b_matrix(&stations, rho, w, g, draft);
            b_w.push(bm[ii][jj]);
        }
        let k_ij = retardation_from_damping(&omega, &b_w, &t_samples);
        kernel.set_channel(ii, jj, &k_ij);
    }
    kernel
}

/// Head-seas excitation amplitudes for unit wave amplitude (FK + Haskind).
///
/// Returns complex `(re, im)` for each DOF. Non-zero primarily heave (2) and
/// pitch (4); surge FK neglected (soft). Diffraction: **STF strip Haskind**.
pub fn excitation_head_seas_6(
    stations: &[StripStation],
    rho: f64,
    omega: f64,
    k: f64,
    depth: f64,
    g: f64,
) -> [[f64; 2]; N_DOF] {
    let (f3, f5) = excitation_fk_diffraction(stations, rho, omega, k, depth, g);
    let mut out = [[0.0; 2]; N_DOF];
    out[2] = f3;
    out[4] = f5;
    out
}

/// Instantaneous head-seas force vector `Re{ F̂ A e^{-iωt} }`.
pub fn excitation_head_seas_time(
    f_hat: &[[f64; 2]; N_DOF],
    amp: f64,
    omega: f64,
    t: f64,
) -> [f64; N_DOF] {
    let c = libm_ops::cos(omega * t);
    let s = libm_ops::sin(omega * t);
    let mut f = [0.0; N_DOF];
    for i in 0..N_DOF {
        f[i] = amp * (f_hat[i][0] * c + f_hat[i][1] * s);
    }
    f
}

/// Six-DOF Cummins time-domain state with bounded trapezoidal convolution.
#[derive(Clone, Debug)]
pub struct Cummins6Dof {
    /// Structural mass matrix.
    pub mass: RigidMass6,
    /// Retardation + `A_∞` + `C`.
    pub kernel: MatrixRetardationKernel,
    /// Displacements / angles `ξ` [m, m, m, rad, rad, rad].
    pub xi: [f64; N_DOF],
    /// Rates `ξ̇`.
    pub xi_dot: [f64; N_DOF],
    /// Time [s].
    pub t: f64,
    /// Velocity history: `hist[lag][dof]`, lag 0 = oldest among stored past
    /// (not including current). Length ≤ `n_time`.
    hist: Vec<[f64; N_DOF]>,
    /// Cached `M + A_∞`.
    m_tot: [[f64; N_DOF]; N_DOF],
}

impl Cummins6Dof {
    /// At-rest initial condition with optional initial `ξ`.
    pub fn new(mass: RigidMass6, kernel: MatrixRetardationKernel, xi0: [f64; N_DOF]) -> Self {
        let mut m_tot = mass.matrix();
        for i in 0..N_DOF {
            for j in 0..N_DOF {
                m_tot[i][j] += kernel.a_inf[i][j];
            }
        }
        Self {
            mass,
            kernel,
            xi: xi0,
            xi_dot: [0.0; N_DOF],
            t: 0.0,
            hist: Vec::new(),
            m_tot,
        }
    }

    /// Box-barge convenience constructor (upright GM_T required for roll).
    #[allow(clippy::too_many_arguments)]
    pub fn box_barge(
        length: f64,
        beam: f64,
        draft: f64,
        mass: RigidMass6,
        rho: f64,
        g: f64,
        gm_t: f64,
        n_strips: usize,
        omega_max: f64,
        n_omega: usize,
        dt: f64,
        n_time: usize,
        xi0: [f64; N_DOF],
    ) -> Self {
        let kernel = box_retardation_6dof(
            length, beam, draft, rho, g, gm_t, n_strips, 1e-3, omega_max, n_omega, dt, n_time,
        );
        Self::new(mass, kernel, xi0)
    }

    fn memory_force(&self, v_now: &[f64; N_DOF]) -> [f64; N_DOF] {
        let dt = self.kernel.dt;
        let n_k = self.kernel.n_time;
        let n_hist = self.hist.len();
        let mut acc = [0.0; N_DOF];
        for i in 0..N_DOF {
            let mut s = 0.0;
            for j in 0..N_DOF {
                for lag in 0..n_k {
                    let v_lag = if lag == 0 {
                        v_now[j]
                    } else if lag <= n_hist {
                        self.hist[n_hist - lag][j]
                    } else {
                        0.0
                    };
                    let weight = if lag == 0 || lag + 1 == n_k { 0.5 } else { 1.0 };
                    s += weight * self.kernel.get(i, j, lag) * v_lag * dt;
                }
            }
            acc[i] = s;
        }
        acc
    }

    /// One semi-implicit step with external force/moment `f` [N, N·m].
    pub fn step(&mut self, f: &[f64; N_DOF]) {
        let dt = self.kernel.dt;
        let mem = self.memory_force(&self.xi_dot);
        let mut rhs = [0.0; N_DOF];
        for i in 0..N_DOF {
            let mut cx = 0.0;
            for j in 0..N_DOF {
                cx += self.kernel.c[i][j] * self.xi[j];
            }
            rhs[i] = f[i] - mem[i] - cx;
        }
        let acc = solve6(&self.m_tot, &rhs);
        for i in 0..N_DOF {
            self.xi_dot[i] += dt * acc[i];
            self.xi[i] += dt * self.xi_dot[i];
        }
        self.t += dt;
        self.hist.push(self.xi_dot);
        let max_hist = self.kernel.n_time;
        if self.hist.len() > max_hist {
            let drop = self.hist.len() - max_hist;
            self.hist.drain(0..drop);
        }
    }

    /// Free decay for `n_steps` (zero excitation); returns heave samples.
    pub fn free_decay_heave(&mut self, n_steps: usize) -> Vec<f64> {
        let mut samples = Vec::with_capacity(n_steps + 1);
        samples.push(self.xi[2]);
        let zero = [0.0; N_DOF];
        for _ in 0..n_steps {
            self.step(&zero);
            samples.push(self.xi[2]);
        }
        samples
    }

    /// Instantaneous mechanical energy proxy (excludes hereditary reservoir):
    /// `½ ξ̇ᵀ (M+A_∞) ξ̇ + ½ ξᵀ C ξ`.
    pub fn energy_proxy(&self) -> f64 {
        let mut ke = 0.0;
        let mut pe = 0.0;
        for i in 0..N_DOF {
            for j in 0..N_DOF {
                ke += self.xi_dot[i] * self.m_tot[i][j] * self.xi_dot[j];
                pe += self.xi[i] * self.kernel.c[i][j] * self.xi[j];
            }
        }
        0.5 * ke + 0.5 * pe
    }
}

/// Solve `A x = b` for SPD-ish 6×6 via Gaussian elimination with partial pivot.
pub fn solve6(a_in: &[[f64; N_DOF]; N_DOF], b_in: &[f64; N_DOF]) -> [f64; N_DOF] {
    let mut a = *a_in;
    let mut b = *b_in;
    for col in 0..N_DOF {
        let mut piv = col;
        let mut best = libm_ops::fabs(a[col][col]);
        for r in (col + 1)..N_DOF {
            let v = libm_ops::fabs(a[r][col]);
            if v > best {
                best = v;
                piv = r;
            }
        }
        if piv != col {
            a.swap(col, piv);
            b.swap(col, piv);
        }
        let diag = a[col][col];
        if libm_ops::fabs(diag) < 1e-30 {
            continue;
        }
        for r in (col + 1)..N_DOF {
            let f = a[r][col] / diag;
            for c in col..N_DOF {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0; N_DOF];
    for i in (0..N_DOF).rev() {
        let mut s = b[i];
        for j in (i + 1)..N_DOF {
            s -= a[i][j] * x[j];
        }
        x[i] = if libm_ops::fabs(a[i][i]) < 1e-30 {
            0.0
        } else {
            s / a[i][i]
        };
    }
    x
}

/// Analytic roll free-decay under constant linear damping (oracle helper).
///
/// `(I₄₄+A₄₄) φ̈ + B₄₄ φ̇ + C₄₄ φ = 0` with `β = B₄₄/(2(I+A))`,
/// `ω_d = √(ω₀² − β²)`, `ω₀ = √(C/(I+A))`, and `φ̇(0)=0`:
///
/// `φ(t) = φ₀ e^{−βt} (cos ω_d t + (β/ω_d) sin ω_d t)`.
pub fn roll_decay_analytic(phi0: f64, beta: f64, omega_d: f64, t: f64) -> f64 {
    let e = libm_ops::exp(-beta * t);
    let c = libm_ops::cos(omega_d * t);
    let s = libm_ops::sin(omega_d * t);
    phi0 * e * (c + (beta / omega_d) * s)
}

/// Integrate constant-coefficient roll ODE (RK4) for oracle comparison.
pub fn roll_decay_rk4(
    i_plus_a: f64,
    b44: f64,
    c44: f64,
    phi0: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<f64> {
    let mut phi = phi0;
    let mut om = 0.0;
    let mut out = Vec::with_capacity(n_steps + 1);
    out.push(phi);
    let accel = |p: f64, w: f64| (-b44 * w - c44 * p) / i_plus_a;
    for _ in 0..n_steps {
        let k1p = om;
        let k1w = accel(phi, om);
        let k2p = om + 0.5 * dt * k1w;
        let k2w = accel(phi + 0.5 * dt * k1p, om + 0.5 * dt * k1w);
        let k3p = om + 0.5 * dt * k2w;
        let k3w = accel(phi + 0.5 * dt * k2p, om + 0.5 * dt * k2w);
        let k4p = om + dt * k3w;
        let k4w = accel(phi + dt * k3p, om + dt * k3w);
        phi += (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
        om += (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
        out.push(phi);
    }
    out
}

/// Heave & pitch undamped natural frequencies from `(M+A_∞)` and `C` 2×2 block.
///
/// Solves the generalized eigenproblem on the heave–pitch subspace (indices 2,4).
/// Returns `(ω₃, ω₅)` [rad/s].
pub fn heave_pitch_natural_omegas(
    m_tot: &[[f64; N_DOF]; N_DOF],
    c: &[[f64; N_DOF]; N_DOF],
) -> (f64, f64) {
    // 2×2: det(C − ω² M) = 0 on indices (2,4)
    let m11 = m_tot[2][2];
    let m12 = m_tot[2][4];
    let m22 = m_tot[4][4];
    let c11 = c[2][2];
    let c12 = c[2][4];
    let c22 = c[4][4];
    // Characteristic: det(C − λ M) = 0, λ = ω²
    // (c11−λ m11)(c22−λ m22) − (c12−λ m12)² = 0
    let a = m11 * m22 - m12 * m12;
    let b = -(c11 * m22 + c22 * m11 - 2.0 * c12 * m12);
    let cc = c11 * c22 - c12 * c12;
    let disc = b * b - 4.0 * a * cc;
    if a.abs() < 1e-30 || disc < 0.0 {
        let w3 = libm_ops::sqrt((c11 / m11).max(0.0));
        let w5 = libm_ops::sqrt((c22 / m22).max(0.0));
        return (w3, w5);
    }
    let s = libm_ops::sqrt(disc);
    let l1 = (-b - s) / (2.0 * a);
    let l2 = (-b + s) / (2.0 * a);
    let (lo, hi) = if l1 < l2 { (l1, l2) } else { (l2, l1) };
    (libm_ops::sqrt(lo.max(0.0)), libm_ops::sqrt(hi.max(0.0)))
}

/// Canonical 6-DOF Cummins hash (FNV-1a over packed `ξ` samples).
///
/// Locked scenario (`docs/DETERMINISM.md`):
/// - Box 40×8×2 m, freshwater, mass = ρ L B T
/// - `I = diag` box inertias about COM; `GM_T` from BM−KG with KG=draft
/// - Kernel: ω∈[1e-3,4], 48 freq, dt=0.05, 120 time samples, 12 strips
/// - Free decay from heave `z0=0.08`, 300 steps; hash heave trace
pub fn canonical_cummins6_hash() -> u64 {
    use crate::hull::RHO_FRESHWATER;
    use pe_math::STANDARD_GRAVITY;

    let length = 40.0;
    let beam = 8.0;
    let draft = 2.0;
    let rho = RHO_FRESHWATER;
    let g = STANDARD_GRAVITY;
    let mass_kg = rho * length * beam * draft;
    let ixx = mass_kg * (beam * beam + draft * draft) / 12.0;
    let iyy = mass_kg * (length * length + draft * draft) / 12.0;
    let izz = mass_kg * (length * length + beam * beam) / 12.0;
    let mass = RigidMass6 {
        mass: mass_kg,
        ixx,
        iyy,
        izz,
    };
    // KG at draft (waterplane) ⇒ KB=T/2, BM=B²/(12T), GM = KB+BM−KG
    let kb = 0.5 * draft;
    let bm = beam * beam / (12.0 * draft);
    let kg = draft;
    let gm_t = kb + bm - kg;
    let dt = 0.05;
    let mut xi0 = [0.0; N_DOF];
    xi0[2] = 0.08;
    let mut td = Cummins6Dof::box_barge(
        length, beam, draft, mass, rho, g, gm_t, 12, 4.0, 48, dt, 120, xi0,
    );
    let samples = td.free_decay_heave(300);
    fnv1a64_f64_slice(&samples)
}

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
