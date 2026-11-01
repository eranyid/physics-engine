//! Strip-theory seakeeping (heave + pitch) after Salvesen, Tuck & Faltinsen (1970).
//!
//! Frequency-domain RAOs for regular **head seas**, plus a constant-coefficient
//! time-domain integrator. Cummins retardation memory lives in [`crate::cummins`].
//!
//! # Sectional hydrodynamics
//!
//! Lewis-form / simple-section formulas (cited):
//! - High-frequency heave added mass per unit length for a half-immersed circle
//!   of radius `R = B/2`: `a₃₃∞ = ρ π R²` (Faltinsen, *Sea Loads on Ships and
//!   Offshore Structures*, §3; equivalent Newman floating-circle limit).
//! - Rectangular wall-sided strip of beam `B`, draft `T`: use the **circular
//!   Lewis substitute** of equal beam (`R = B/2`) with a draft/area correction
//!   factor `σ = A/(B T)` capped to `[0.5, 1]` — **nominal engineering
//!   approximation**, not a tabulated Vugts/Frank close-fit value.
//! - Sectional radiation damping (2-D): floating-circle scaling
//!   `b₃₃ ≈ ρ π R² ω · β · exp(−2 k T_eq)` with **nominal** `β = 0.5`
//!   (unverified vs Ursell/Vugts). Used also in the STF Haskind diffraction
//!   strip approximation.
//!
//! # Coupled heave–pitch (head seas)
//!
//! Strip integrals (origin at midship, +x forward, +z up):
//! `A₃₃ = ∫ a₃₃ dx`, `A₃₅ = A₅₃ = −∫ x a₃₃ dx`, `A₅₅ = ∫ x² a₃₃ dx`
//! (same pattern for `Bᵢⱼ`). Restoring:
//! `C₃₃ = ρ g ∫ B(x) dx`, `C₃₅ = C₅₃ = −ρ g ∫ x B(x) dx`,
//! `C₅₅ = ρ g ∫ x² B(x) dx` (pitch about COM; longitudinal GM term omitted —
//! box barges with COM on the waterplane centroid are the validation target).
//!
//! Excitation: Froude–Krylov from Airy pressure head on the section plus
//! diffraction via Haskind (`F_D ~ −iω ∫ b₃₃ e^{i k x} dx` style, STF §3).
//!
//! # Limits / deferred
//!
//! - This module's [`HeavePitchTd`] freezes matrices at one `ω_ref`; for
//!   retardation convolution see [`crate::cummins`].
//! - No forward speed (`U = 0`); no roll/sway/yaw; no viscous roll damping.
//! - Lewis/circular substitute — **not** ITTC/Vugts tabulated sections.
//! - Diffraction is the STF strip Haskind approximation, not a 3-D panel code.
//!
//! # References
//! - Salvesen, N., Tuck, E. O. & Faltinsen, O. (1970). Ship motions and sea
//!   loads. *Trans. SNAME* **78**.
//! - Lewis, F. M. (1929). The inertia of water surrounding a pitching vessel.
//! - Faltinsen (1990); Newman, *Marine Hydrodynamics*.

use pe_math::libm_ops;
use pe_math::STANDARD_GRAVITY;

use crate::waves::{dispersion_omega, solve_wave_number};

/// One strip station along the hull.
#[derive(Clone, Debug)]
pub struct StripStation {
    /// Longitudinal position from midship [m] (+ forward).
    pub x: f64,
    /// Local beam [m].
    pub beam: f64,
    /// Local draft [m].
    pub draft: f64,
    /// Sectional area coefficient `σ = A/(B T)` ∈ (0, 1]; default 1 (rectangle).
    pub area_coeff: f64,
    /// Station length weight `dx` [m] for strip integrals.
    pub dx: f64,
}

impl StripStation {
    /// Rectangular strip (`σ = 1`).
    pub fn rectangular(x: f64, beam: f64, draft: f64, dx: f64) -> Self {
        Self {
            x,
            beam,
            draft,
            area_coeff: 1.0,
            dx,
        }
    }
}

/// Sectional heave added mass per unit length [kg/m] — circular Lewis substitute.
///
/// `a₃₃∞ = ρ π (B/2)² · σ_clamp` with `σ_clamp = clamp(σ, 0.5, 1)`.
/// **Nominal** for non-circular sections (see module docs).
pub fn sectional_a33_inf(rho: f64, beam: f64, area_coeff: f64) -> f64 {
    let r = 0.5 * beam;
    let sigma = area_coeff.clamp(0.5, 1.0);
    rho * core::f64::consts::PI * r * r * sigma
}

/// Sectional radiation damping per unit length [kg/(m·s)] — nominal circle form.
///
/// Uses the order-of-magnitude floating-circle scaling
/// `b₃₃ ≈ ρ π R² ω · β · exp(−2 k T_eq)` with `R = B/2`, deep-water
/// `k = ω²/g`, and **nominal** peak factor `β = 0.5` (unverified vs Ursell /
/// Vugts tables; see Faltinsen / Newman non-dimensional damping charts).
/// The classical thin-strut Havelock `ρ g² A²/ω³` form is **not** used here
/// (that `A` is a potential amplitude, not the beam).
pub fn sectional_b33(rho: f64, omega: f64, beam: f64, draft: f64, area_coeff: f64, g: f64) -> f64 {
    if omega <= 1e-12 {
        return 0.0;
    }
    let r = 0.5 * beam;
    let k = omega * omega / g;
    let t_eq = area_coeff.clamp(0.5, 1.0) * draft;
    let decay = libm_ops::exp(-2.0 * k * t_eq);
    // Nominal non-dimensional damping peak (unverified vs Ursell/Vugts).
    const BETA_NOMINAL: f64 = 0.5;
    rho * core::f64::consts::PI * r * r * omega * BETA_NOMINAL * decay
}

/// Frequency-dependent sectional a₃₃: blend toward 0 as `ω → 0` (free-surface).
///
/// `a₃₃(ω) = a₃₃∞ · (ω/ω₀)² / (1 + (ω/ω₀)²)` with `ω₀ = √(g / (B/2))` —
/// smooth surrogate, **not** a published Lewis frequency table.
pub fn sectional_a33(rho: f64, omega: f64, beam: f64, draft: f64, area_coeff: f64, g: f64) -> f64 {
    let _ = draft;
    let a_inf = sectional_a33_inf(rho, beam, area_coeff);
    let r = (0.5 * beam).max(1e-9);
    let w0 = libm_ops::sqrt(g / r);
    let r2 = (omega / w0) * (omega / w0);
    a_inf * r2 / (1.0 + r2)
}

/// Box barge discretised into equal strips (origin midship).
pub fn box_barge_stations(length: f64, beam: f64, draft: f64, n: usize) -> Vec<StripStation> {
    assert!(n >= 2 && length > 0.0 && beam > 0.0 && draft > 0.0);
    let dx = length / n as f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = -0.5 * length + (i as f64 + 0.5) * dx;
        out.push(StripStation::rectangular(x, beam, draft, dx));
    }
    out
}

/// 2×2 heave–pitch hydrodynamic matrices at one encounter frequency.
#[derive(Clone, Debug)]
pub struct StripMatrices {
    /// Added mass `[A33, A35; A53, A55]` [kg, kg·m, kg·m, kg·m²].
    pub a: [[f64; 2]; 2],
    /// Damping `[B33, …]` [kg/s, …].
    pub b: [[f64; 2]; 2],
    /// Restoring `[C33, …]` [N/m, …].
    pub c: [[f64; 2]; 2],
}

/// Assemble strip matrices for head-seas heave (index 0) and pitch (index 1).
pub fn assemble_strip_matrices(
    stations: &[StripStation],
    rho: f64,
    omega: f64,
    g: f64,
) -> StripMatrices {
    let mut a = [[0.0; 2]; 2];
    let mut b = [[0.0; 2]; 2];
    let mut c = [[0.0; 2]; 2];
    for s in stations {
        let a33 = sectional_a33(rho, omega, s.beam, s.draft, s.area_coeff, g);
        let b33 = sectional_b33(rho, omega, s.beam, s.draft, s.area_coeff, g);
        let dx = s.dx;
        let x = s.x;
        a[0][0] += a33 * dx;
        a[0][1] += -x * a33 * dx;
        a[1][0] += -x * a33 * dx;
        a[1][1] += x * x * a33 * dx;
        b[0][0] += b33 * dx;
        b[0][1] += -x * b33 * dx;
        b[1][0] += -x * b33 * dx;
        b[1][1] += x * x * b33 * dx;
        // Restoring from waterplane: dF = −ρ g B dx · η_rel
        let c33 = rho * g * s.beam * dx;
        c[0][0] += c33;
        c[0][1] += -x * c33;
        c[1][0] += -x * c33;
        c[1][1] += x * x * c33;
    }
    StripMatrices { a, b, c }
}

/// Complex excitation amplitudes `(F3, F5)` [N], [N·m] for unit wave amplitude.
///
/// Froude–Krylov: vertical force density `ρ g B e^{i(kx)}` with depth decay
/// `e^{−k T}` (deep) / `cosh(k(−T+h))/cosh(kh)` (finite). Diffraction: STF
/// Haskind strip form `−iω ∫ b₃₃ e^{ikx} dx` (complex via `(re, im)`).
pub fn excitation_fk_diffraction(
    stations: &[StripStation],
    rho: f64,
    omega: f64,
    k: f64,
    depth: f64,
    g: f64,
) -> ([f64; 2], [f64; 2]) {
    // Returns ((F3_re, F3_im), (F5_re, F5_im)) for A=1 m.
    let mut f3 = [0.0; 2];
    let mut f5 = [0.0; 2];
    for s in stations {
        let kx = k * s.x;
        let c = libm_ops::cos(kx);
        let sn = libm_ops::sin(kx);
        // Depth attenuation of pressure at section centroid depth T/2.
        let z = -0.5 * s.draft;
        let atten = pressure_atten(k, depth, z);
        let fk = rho * g * s.beam * s.dx * atten;
        // FK vertical force ~ ρ g B dx · atten · e^{ikx} (complex)
        f3[0] += fk * c;
        f3[1] += fk * sn;
        f5[0] += -s.x * fk * c;
        f5[1] += -s.x * fk * sn;
        // Diffraction (Haskind): −iω b33 dx e^{ikx}
        let b33 = sectional_b33(rho, omega, s.beam, s.draft, s.area_coeff, g);
        let d = omega * b33 * s.dx;
        // −i · (c + i s) = −i c + s = s − i c
        f3[0] += d * sn;
        f3[1] += -d * c;
        f5[0] += -s.x * d * sn;
        f5[1] += s.x * d * c;
    }
    (f3, f5)
}

fn pressure_atten(k: f64, depth: f64, z: f64) -> f64 {
    let kh = k * depth;
    if kh > 30.0 {
        libm_ops::exp(k * z)
    } else {
        libm_ops::cosh(k * (z + depth)) / libm_ops::cosh(kh)
    }
}

/// Heave / pitch RAO magnitudes for amplitude-A waves: `|ξ|/A` [1], [1/m]→[rad/m] as `|ξ5|/A`.
#[derive(Clone, Copy, Debug)]
pub struct HeavePitchRao {
    /// Encounter / wave frequency [rad/s].
    pub omega: f64,
    /// Heave RAO `|ξ₃|/A` [dimensionless].
    pub heave: f64,
    /// Pitch RAO `|ξ₅|/A` [rad/m].
    pub pitch: f64,
    /// Heave phase [rad] (ξ₃ vs η).
    pub heave_phase: f64,
    /// Pitch phase [rad].
    pub pitch_phase: f64,
}

/// Solve the 2×2 frequency-domain system for unit-amplitude head seas.
pub fn heave_pitch_rao(
    stations: &[StripStation],
    mass: f64,
    iyy: f64,
    rho: f64,
    omega: f64,
    depth: f64,
    g: f64,
) -> HeavePitchRao {
    let k = solve_wave_number(omega, depth, g);
    let _ = dispersion_omega(k, depth, g); // consistency
    let m = assemble_strip_matrices(stations, rho, omega, g);
    let (f3, f5) = excitation_fk_diffraction(stations, rho, omega, k, depth, g);

    // (−ω²(M+A) + iω B + C) ξ = F
    let w2 = omega * omega;
    // Complex matrix Z = R + i I, 2×2
    let mut zr = [[0.0; 2]; 2];
    let mut zi = [[0.0; 2]; 2];
    let mass_diag = [mass, iyy];
    for r in 0..2 {
        for c in 0..2 {
            let md = if r == c { mass_diag[r] } else { 0.0 };
            zr[r][c] = -w2 * (md + m.a[r][c]) + m.c[r][c];
            zi[r][c] = omega * m.b[r][c];
        }
    }
    let (xi_re, xi_im) = solve2x2_complex(zr, zi, [f3[0], f5[0]], [f3[1], f5[1]]);
    let heave = libm_ops::hypot(xi_re[0], xi_im[0]);
    let pitch = libm_ops::hypot(xi_re[1], xi_im[1]);
    HeavePitchRao {
        omega,
        heave,
        pitch,
        heave_phase: libm_ops::atan2(xi_im[0], xi_re[0]),
        pitch_phase: libm_ops::atan2(xi_im[1], xi_re[1]),
    }
}

fn solve2x2_complex(
    ar: [[f64; 2]; 2],
    ai: [[f64; 2]; 2],
    br: [f64; 2],
    bi: [f64; 2],
) -> ([f64; 2], [f64; 2]) {
    // Invert 2×2 complex via real 4×4 block (adjugate).
    // [Ar, -Ai; Ai, Ar] [xr; xi] = [br; bi]
    let a = [
        [ar[0][0], -ai[0][0], ar[0][1], -ai[0][1]],
        [ai[0][0], ar[0][0], ai[0][1], ar[0][1]],
        [ar[1][0], -ai[1][0], ar[1][1], -ai[1][1]],
        [ai[1][0], ar[1][0], ai[1][1], ar[1][1]],
    ];
    let rhs = [br[0], bi[0], br[1], bi[1]];
    let x = solve4(a, rhs);
    ([x[0], x[2]], [x[1], x[3]])
}

fn solve4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> [f64; 4] {
    // Partial-pivot Gaussian elimination (deterministic row order).
    for col in 0..4 {
        let mut piv = col;
        let mut best = libm_ops::fabs(a[col][col]);
        for r in (col + 1)..4 {
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
        for r in (col + 1)..4 {
            let f = a[r][col] / diag;
            for c in col..4 {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0; 4];
    for i in (0..4).rev() {
        let mut s = b[i];
        for j in (i + 1)..4 {
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

/// Analytic heave natural frequency [rad/s] for a box barge (no coupling).
///
/// `ω₃ = √(ρ g A_wp / (m + A₃₃))` with `A₃₃` from strip `a₃₃∞` (high-frequency).
pub fn box_heave_natural_omega(
    length: f64,
    beam: f64,
    draft: f64,
    mass: f64,
    rho: f64,
    g: f64,
    n_strips: usize,
) -> f64 {
    let stations = box_barge_stations(length, beam, draft, n_strips);
    // Use a large omega so a33 → a33∞
    let m = assemble_strip_matrices(&stations, rho, 1e3, g);
    let a_wp = length * beam;
    libm_ops::sqrt(rho * g * a_wp / (mass + m.a[0][0]))
}

/// Analytic roll natural period [s]: `T = 2π √((I₄₄+A₄₄)/(ρ g ∇ GM))`.
///
/// For a box barge, `A₄₄` defaults to `0` unless supplied (added inertia in roll
/// is **not** predicted by this heave/pitch strip module — pass a documented
/// value or 0).
pub fn roll_natural_period(inertia_plus_a44: f64, rho: f64, volume: f64, gm: f64, g: f64) -> f64 {
    assert!(gm > 0.0 && volume > 0.0 && inertia_plus_a44 > 0.0);
    2.0 * core::f64::consts::PI * libm_ops::sqrt(inertia_plus_a44 / (rho * g * volume * gm))
}

/// Constant-coefficient heave–pitch time-domain state.
///
/// Integrates `(M+A) ξ̈ + B ξ̇ + C ξ = F(t)` with matrices frozen at a single
/// `omega_ref` (documented limit: no Cummins memory). Excitation from a regular
/// Airy head wave of amplitude `A` and frequency `omega`.
#[derive(Clone, Debug)]
pub struct HeavePitchTd {
    /// Mass + added-mass matrix (2×2).
    pub m: [[f64; 2]; 2],
    /// Damping.
    pub b: [[f64; 2]; 2],
    /// Restoring.
    pub c: [[f64; 2]; 2],
    /// Heave [m], pitch [rad].
    pub xi: [f64; 2],
    /// Heave rate [m/s], pitch rate [rad/s].
    pub xi_dot: [f64; 2],
    /// Wave amplitude [m].
    pub wave_amp: f64,
    /// Wave frequency [rad/s].
    pub omega: f64,
    /// Wave number [1/m].
    pub k: f64,
    /// Depth [m].
    pub depth: f64,
    /// Density [kg/m³].
    pub rho: f64,
    /// Gravity [m/s²].
    pub g: f64,
    /// Stations (owned copy for excitation).
    pub stations: Vec<StripStation>,
    /// Time [s].
    pub t: f64,
}

impl HeavePitchTd {
    /// Build from barge geometry; matrices frozen at `omega`.
    #[allow(clippy::too_many_arguments)]
    pub fn box_barge(
        length: f64,
        beam: f64,
        draft: f64,
        mass: f64,
        iyy: f64,
        rho: f64,
        wave_amp: f64,
        omega: f64,
        depth: f64,
        g: f64,
        n_strips: usize,
    ) -> Self {
        let stations = box_barge_stations(length, beam, draft, n_strips);
        let mats = assemble_strip_matrices(&stations, rho, omega, g);
        let k = solve_wave_number(omega, depth, g);
        let mut m = mats.a;
        m[0][0] += mass;
        m[1][1] += iyy;
        Self {
            m,
            b: mats.b,
            c: mats.c,
            xi: [0.0; 2],
            xi_dot: [0.0; 2],
            wave_amp,
            omega,
            k,
            depth,
            rho,
            g,
            stations,
            t: 0.0,
        }
    }

    /// Instantaneous excitation `(F3, F5)` [N], [N·m] at current `t`.
    pub fn excitation(&self) -> [f64; 2] {
        let (f3c, f5c) = excitation_fk_diffraction(
            &self.stations,
            self.rho,
            self.omega,
            self.k,
            self.depth,
            self.g,
        );
        // F(t) = Re{ F̂ A e^{−iωt} } = A (Fre cos ωt + Fim sin ωt)
        // with e^{−iωt} = cos(ωt) − i sin(ωt): Re((Fr+iFi)(c−is)) = Fr c + Fi s
        let c = libm_ops::cos(self.omega * self.t);
        let s = libm_ops::sin(self.omega * self.t);
        let a = self.wave_amp;
        [a * (f3c[0] * c + f3c[1] * s), a * (f5c[0] * c + f5c[1] * s)]
    }

    /// One semi-implicit Euler step of size `dt` [s].
    pub fn step(&mut self, dt: f64) {
        let f = self.excitation();
        // M a = F − B v − C x
        let mut rhs = [0.0; 2];
        for i in 0..2 {
            rhs[i] = f[i]
                - (self.b[i][0] * self.xi_dot[0] + self.b[i][1] * self.xi_dot[1])
                - (self.c[i][0] * self.xi[0] + self.c[i][1] * self.xi[1]);
        }
        let acc = solve2_real(self.m, rhs);
        self.xi_dot[0] += dt * acc[0];
        self.xi_dot[1] += dt * acc[1];
        self.xi[0] += dt * self.xi_dot[0];
        self.xi[1] += dt * self.xi_dot[1];
        self.t += dt;
    }
}

fn solve2_real(a: [[f64; 2]; 2], b: [f64; 2]) -> [f64; 2] {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if libm_ops::fabs(det) < 1e-30 {
        return [0.0, 0.0];
    }
    [
        (b[0] * a[1][1] - b[1] * a[0][1]) / det,
        (a[0][0] * b[1] - a[1][0] * b[0]) / det,
    ]
}

/// Free-decay heave with constant damping: returns samples, analytic `β`, and
/// a log-decrement fit from successive **positive** peaks.
///
/// Analytic underdamped: `β = B₃₃ / (2 (m+A₃₃))`. Integration uses RK4 on the
/// linear oscillator (portable `libm` only). Peak times use parabolic
/// interpolation; `β_fit` averages log-decrements over available cycles:
/// `β = ln(A_i/A_{i+1}) / (t_{i+1}−t_i)`.
pub fn free_decay_heave(
    mass: f64,
    a33: f64,
    b33: f64,
    c33: f64,
    z0: f64,
    dt: f64,
    n_steps: usize,
) -> (Vec<f64>, f64, f64) {
    let m = mass + a33;
    let beta_analytic = b33 / (2.0 * m);
    let mut z = z0;
    let mut v = 0.0;
    let mut samples = Vec::with_capacity(n_steps + 1);
    samples.push(z);
    let accel = |z: f64, v: f64| (-b33 * v - c33 * z) / m;
    for _ in 0..n_steps {
        // Classical RK4 for ż=v, v̇=a(z,v).
        let k1z = v;
        let k1v = accel(z, v);
        let k2z = v + 0.5 * dt * k1v;
        let k2v = accel(z + 0.5 * dt * k1z, v + 0.5 * dt * k1v);
        let k3z = v + 0.5 * dt * k2v;
        let k3v = accel(z + 0.5 * dt * k2z, v + 0.5 * dt * k2v);
        let k4z = v + dt * k3v;
        let k4v = accel(z + dt * k3z, v + dt * k3v);
        z += (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
        v += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        samples.push(z);
    }
    // Positive peaks only (full-period log-decrement).
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for i in 1..samples.len() - 1 {
        if samples[i] > samples[i - 1] && samples[i] > samples[i + 1] && samples[i] > 1e-12 {
            // Parabolic peak time/amplitude about samples[i-1],i,i+1.
            let y0 = samples[i - 1];
            let y1 = samples[i];
            let y2 = samples[i + 1];
            let denom = y0 - 2.0 * y1 + y2;
            let (t_pk, a_pk) = if libm_ops::fabs(denom) > 1e-30 {
                let delta = 0.5 * (y0 - y2) / denom; // in [-0.5,0.5] typically
                let d = delta.clamp(-0.5, 0.5);
                let t = (i as f64 + d) * dt;
                let a = y1 - 0.25 * (y0 - y2) * d;
                (t, a)
            } else {
                (i as f64 * dt, y1)
            };
            peaks.push((t_pk, a_pk));
        }
    }
    let beta_fit = if peaks.len() >= 3 {
        // Average log-decrement over consecutive positive peaks.
        let mut sum = 0.0;
        let mut n = 0usize;
        for i in 0..peaks.len() - 1 {
            let (t0, a0) = peaks[i];
            let (t1, a1) = peaks[i + 1];
            if a0 > 0.0 && a1 > 0.0 && t1 > t0 {
                sum += libm_ops::log(a0 / a1) / (t1 - t0);
                n += 1;
            }
        }
        if n > 0 {
            sum / n as f64
        } else {
            beta_analytic
        }
    } else if peaks.len() == 2 {
        let (t0, a0) = peaks[0];
        let (t1, a1) = peaks[1];
        libm_ops::log(a0 / a1) / (t1 - t0)
    } else {
        beta_analytic
    };
    (samples, beta_analytic, beta_fit)
}

/// Earth-g convenience for [`box_heave_natural_omega`].
pub fn box_heave_natural_omega_earth(
    length: f64,
    beam: f64,
    draft: f64,
    mass: f64,
    rho: f64,
    n_strips: usize,
) -> f64 {
    box_heave_natural_omega(length, beam, draft, mass, rho, STANDARD_GRAVITY, n_strips)
}
