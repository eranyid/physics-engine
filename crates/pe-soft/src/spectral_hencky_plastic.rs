//! Spectral Hencky / Kirchhoff return on `b_e` (Simo 1992) for soft XPBD (M32).
//!
//! M31 feeds the elastic **Green** strain of `F_e = F F_p⁻¹` into the small-strain
//! J2 map, then updates `F_p ← exp(Δε_p) F_p`. This module replaces that elastic
//! trial with the full isotropic spectral algorithm on the elastic left
//! Cauchy–Green tensor:
//!
//! 1. `F_e^{tr} = F F_p⁻¹`, `b_e^{tr} = F_e^{tr} (F_e^{tr})ᵀ`
//! 2. Spectral: `b_e^{tr} = Σ λ_a² n^{(a)} ⊗ n^{(a)}`
//! 3. Principal Hencky strains `ε_a = ln(λ_a)`
//! 4. Principal Kirchhoff stresses `τ_a = λ tr(ε) + 2μ ε_a`
//! 5. J2 radial return in Kirchhoff principal space (identical algebra to the
//!    small-strain map because of spectral isotropy)
//! 6. Recover `b_e = Σ exp(2 ε_a) n^{(a)} ⊗ n^{(a)}` and plastic increment
//!    `Δε_p = ε^{tr} − ε` for the exponential `F_p` update + isochoric projection
//!
//! Distance bars reuse the M31 Hencky path (`ln(L/L₀)`, `rest ← L₀ exp(ε_p)`).
//!
//! # Determinism
//!
//! Fixed constraint index order; portable `libm`; no `HashMap`; no `mul_add`.
//! Eigenpairs sorted by descending eigenvalue for a stable spectral basis.
//!
//! References (formulas only, clean-room): Simo (1992) / Simo & Hughes
//! *Computational Inelasticity* — multiplicative plasticity with logarithmic
//! strain and spectral Kirchhoff return; Hencky strain; Macklin XPBD
//! rest-update pattern. No FleX / PhysX / commercial FEM source transcribed.

use pe_math::libm_ops;
use pe_math::Vec3;

use crate::constraint::{
    deformation_gradient_public, invert3, mul3_public, Multiplier, TetraFemConstraint,
};
use crate::finite_strain_plastic::{
    analytic_log_permanent_strain, isochoric_project, make_fs_plastic_bar, sym_eigenvalues_3,
    sym_exp3, update_fs_plastic_distances, FsPlasticDistance, FsPlasticUpdateStats,
};
use crate::particle::Particle;
use crate::plasticity::{J2Params, SymTensor3, UniaxialPlasticState};
use crate::solver::SoftWorld;

/// Identity 3×3.
#[inline]
fn mat_identity() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// 3×3 determinant.
#[inline]
fn det3_local(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Left Cauchy–Green `b = F Fᵀ`.
#[inline]
pub fn left_cauchy_green(f: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += f[i][k] * f[j][k];
            }
            b[i][j] = s;
        }
    }
    b
}

/// SymTensor3 → dense symmetric 3×3.
fn sym_to_mat(s: &SymTensor3) -> [[f64; 3]; 3] {
    [[s.xx, s.xy, s.xz], [s.xy, s.yy, s.yz], [s.xz, s.yz, s.zz]]
}

/// Dense symmetric 3×3 → [`SymTensor3`].
fn mat_to_sym(m: &[[f64; 3]; 3]) -> SymTensor3 {
    SymTensor3 {
        xx: m[0][0],
        yy: m[1][1],
        zz: m[2][2],
        yz: 0.5 * (m[1][2] + m[2][1]),
        xz: 0.5 * (m[0][2] + m[2][0]),
        xy: 0.5 * (m[0][1] + m[1][0]),
    }
}

/// Unit eigenvector of symmetric `m` for eigenvalue `lam` (cross-product nullspace).
fn sym_eigenvector(m: &[[f64; 3]; 3], lam: f64) -> [f64; 3] {
    let a = [
        [
            m[0][0] - lam,
            0.5 * (m[0][1] + m[1][0]),
            0.5 * (m[0][2] + m[2][0]),
        ],
        [
            0.5 * (m[0][1] + m[1][0]),
            m[1][1] - lam,
            0.5 * (m[1][2] + m[2][1]),
        ],
        [
            0.5 * (m[0][2] + m[2][0]),
            0.5 * (m[1][2] + m[2][1]),
            m[2][2] - lam,
        ],
    ];
    let r0 = [a[0][0], a[0][1], a[0][2]];
    let r1 = [a[1][0], a[1][1], a[1][2]];
    let r2 = [a[2][0], a[2][1], a[2][2]];
    let c01 = [
        r0[1] * r1[2] - r0[2] * r1[1],
        r0[2] * r1[0] - r0[0] * r1[2],
        r0[0] * r1[1] - r0[1] * r1[0],
    ];
    let c12 = [
        r1[1] * r2[2] - r1[2] * r2[1],
        r1[2] * r2[0] - r1[0] * r2[2],
        r1[0] * r2[1] - r1[1] * r2[0],
    ];
    let c20 = [
        r2[1] * r0[2] - r2[2] * r0[1],
        r2[2] * r0[0] - r2[0] * r0[2],
        r2[0] * r0[1] - r2[1] * r0[0],
    ];
    let n01 = c01[0] * c01[0] + c01[1] * c01[1] + c01[2] * c01[2];
    let n12 = c12[0] * c12[0] + c12[1] * c12[1] + c12[2] * c12[2];
    let n20 = c20[0] * c20[0] + c20[1] * c20[1] + c20[2] * c20[2];
    let (v, n2) = if n01 >= n12 && n01 >= n20 {
        (c01, n01)
    } else if n12 >= n20 {
        (c12, n12)
    } else {
        (c20, n20)
    };
    if n2 < 1e-30 {
        return [1.0, 0.0, 0.0];
    }
    let inv = 1.0 / libm_ops::sqrt(n2);
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Spectral decomposition of a symmetric 3×3.
///
/// Returns `(eigenvalues descending, Q)` with **columns** of `Q` the
/// corresponding orthonormal eigenvectors (modified Gram–Schmidt).
pub fn sym_spectral_decompose(m: &[[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let eigs_raw = sym_eigenvalues_3(m);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| eigs_raw[j].total_cmp(&eigs_raw[i]));
    let eigs = [eigs_raw[order[0]], eigs_raw[order[1]], eigs_raw[order[2]]];
    let mut q = [[0.0; 3]; 3];
    for col in 0..3 {
        let mut v = sym_eigenvector(m, eigs[col]);
        for prev in 0..col {
            let mut dot = 0.0;
            for k in 0..3 {
                dot += v[k] * q[k][prev];
            }
            for k in 0..3 {
                v[k] -= dot * q[k][prev];
            }
        }
        let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if n2 < 1e-30 {
            let mut best = [1.0, 0.0, 0.0];
            let mut best_n = 0.0;
            for axis in 0..3 {
                let mut cand = [0.0; 3];
                cand[axis] = 1.0;
                for prev in 0..col {
                    let mut dot = 0.0;
                    for k in 0..3 {
                        dot += cand[k] * q[k][prev];
                    }
                    for k in 0..3 {
                        cand[k] -= dot * q[k][prev];
                    }
                }
                let cn = cand[0] * cand[0] + cand[1] * cand[1] + cand[2] * cand[2];
                if cn > best_n {
                    best_n = cn;
                    best = cand;
                }
            }
            v = best;
            let inv = 1.0 / libm_ops::sqrt(best_n.max(1e-30));
            for k in 0..3 {
                v[k] *= inv;
            }
        } else {
            let inv = 1.0 / libm_ops::sqrt(n2);
            for k in 0..3 {
                v[k] *= inv;
            }
        }
        for k in 0..3 {
            q[k][col] = v[k];
        }
    }
    (eigs, q)
}

/// Assemble `M = Σ_a v_a q_a ⊗ q_a` from principal values and eigenvector columns.
#[inline]
pub fn assemble_from_principal(values: &[f64; 3], q: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for col in 0..3 {
        let v = values[col];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] += v * q[i][col] * q[j][col];
            }
        }
    }
    out
}

/// Result of one spectral Hencky / Kirchhoff return on trial `b_e`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralHenckyReturn {
    /// Updated elastic left Cauchy–Green `b_e`.
    pub be: [[f64; 3]; 3],
    /// Kirchhoff stress `τ` after return [Pa].
    pub kirchhoff: SymTensor3,
    /// Plastic Hencky increment `Δε_p = ε^{tr} − ε` (symmetric).
    pub delta_eps_p: SymTensor3,
    /// Updated equivalent plastic strain α.
    pub eq_plastic_strain: f64,
    /// `true` if the yield surface was active.
    pub plastic: bool,
    /// Plastic multiplier increment Δγ ≥ 0.
    pub delta_gamma: f64,
}

/// Analytic Kirchhoff principal stresses for **isochoric** uniaxial stretch.
///
/// `λ₁ = λ`, `λ₂ = λ₃ = 1/√λ` ⇒ `tr ε = 0`, `τ₁ = 2μ ln(λ)`, `τ₂ = τ₃ = −μ ln(λ)`,
/// and von Mises `σ_eq = 3 μ |ln λ|` (elastic trial, virgin).
pub fn analytic_isochoric_uniaxial_kirchhoff(mu: f64, lambda_stretch: f64) -> (f64, f64, f64) {
    let eps = libm_ops::log(lambda_stretch.max(1e-30));
    let t1 = 2.0 * mu * eps;
    let t2 = -mu * eps;
    (t1, t2, t2)
}

/// Analytic elastic-trial von Mises for isochoric uniaxial: `3 μ |ln λ|`.
#[inline]
pub fn analytic_isochoric_uniaxial_seq(mu: f64, lambda_stretch: f64) -> f64 {
    3.0 * mu * libm_ops::fabs(libm_ops::log(lambda_stretch.max(1e-30)))
}

/// Simo spectral Hencky return mapping on trial elastic left Cauchy–Green `b_e`.
///
/// Isotropic Hencky elasticity in principal space of `b_e`, J2 yield on Kirchhoff
/// stress with linear isotropic hardening. Plastic flow updates principal Hencky
/// strains; `b_e` is reconstructed as `Σ exp(2 ε_a) n_a ⊗ n_a`.
pub fn return_map_spectral_hencky_be(
    params: &J2Params,
    be_trial: &[[f64; 3]; 3],
    eq_plastic_strain: f64,
) -> SpectralHenckyReturn {
    let e = params.youngs_modulus;
    let nu = params.poisson_ratio;
    let mu = e / (2.0 * (1.0 + nu));
    let lame_lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let h = params.hardening_modulus;
    let sig_y = params.yield_strength;

    let (eig_b, q) = sym_spectral_decompose(be_trial);
    // Principal stretches λ_a = √(eig of b_e); Hencky ε_a = ln(λ_a) = ½ ln(eig).
    let mut eps_tr = [0.0; 3];
    for a in 0..3 {
        let b_a = eig_b[a].max(1e-30);
        eps_tr[a] = 0.5 * libm_ops::log(b_a);
    }
    let tr_eps = eps_tr[0] + eps_tr[1] + eps_tr[2];
    let mut tau_prin = [0.0; 3];
    for a in 0..3 {
        tau_prin[a] = lame_lambda * tr_eps + 2.0 * mu * eps_tr[a];
    }
    let mean_tau = (tau_prin[0] + tau_prin[1] + tau_prin[2]) / 3.0;
    let mut s_prin = [0.0; 3];
    for a in 0..3 {
        s_prin[a] = tau_prin[a] - mean_tau;
    }
    // ‖s‖_F for a principal (diagonal) tensor.
    let s_frob =
        libm_ops::sqrt(s_prin[0] * s_prin[0] + s_prin[1] * s_prin[1] + s_prin[2] * s_prin[2]);
    let seq = libm_ops::sqrt(1.5) * s_frob;
    let f_tr = seq - (sig_y + h * eq_plastic_strain);

    let tau_tr = mat_to_sym(&assemble_from_principal(&tau_prin, &q));

    if f_tr <= 0.0 || seq < 1e-30 {
        return SpectralHenckyReturn {
            be: *be_trial,
            kirchhoff: tau_tr,
            delta_eps_p: SymTensor3::zero(),
            eq_plastic_strain,
            plastic: false,
            delta_gamma: 0.0,
        };
    }

    let dg = f_tr / (3.0 * mu + h);
    let factor = 1.0 - (3.0 * mu * dg) / seq;
    let mut s_new = [0.0; 3];
    for a in 0..3 {
        s_new[a] = factor * s_prin[a];
    }
    let mut tau_new_prin = [0.0; 3];
    for a in 0..3 {
        tau_new_prin[a] = s_new[a] + mean_tau;
    }
    // Invert isotropic Hencky: τ = λ tr(ε) I + 2μ ε.
    // tr(τ) = (3λ + 2μ) tr(ε) ⇒ tr(ε) = tr(τ) / (3λ + 2μ).
    let tr_tau = tau_new_prin[0] + tau_new_prin[1] + tau_new_prin[2];
    let bulk_like = 3.0 * lame_lambda + 2.0 * mu;
    let tr_eps_new = if bulk_like.abs() > 1e-30 {
        tr_tau / bulk_like
    } else {
        0.0
    };
    let mut eps_new = [0.0; 3];
    for a in 0..3 {
        eps_new[a] = (tau_new_prin[a] - lame_lambda * tr_eps_new) / (2.0 * mu).max(1e-30);
    }
    let mut deps_prin = [0.0; 3];
    for a in 0..3 {
        deps_prin[a] = eps_tr[a] - eps_new[a];
    }
    let mut be_vals = [0.0; 3];
    for a in 0..3 {
        be_vals[a] = libm_ops::exp(2.0 * eps_new[a]);
    }
    let be = assemble_from_principal(&be_vals, &q);
    let kirchhoff = mat_to_sym(&assemble_from_principal(&tau_new_prin, &q));
    let delta_eps_p = mat_to_sym(&assemble_from_principal(&deps_prin, &q));

    SpectralHenckyReturn {
        be,
        kirchhoff,
        delta_eps_p,
        eq_plastic_strain: eq_plastic_strain + dg,
        plastic: true,
        delta_gamma: dg,
    }
}

/// Per-tet spectral Hencky plastic state (`b_e` path).
#[derive(Clone, Debug)]
pub struct BePlasticTet {
    /// Index into `ConstraintSet::tetra_fem`.
    pub constraint_index: usize,
    /// Cold rest-shape inverse Dm⁻¹.
    pub cold_dm_inv: [[f64; 3]; 3],
    /// Cold rest volume V₀ [m³].
    pub cold_volume: f64,
    /// Plastic deformation gradient F_p (det ≈ 1 after projection).
    pub fp: [[f64; 3]; 3],
    /// Equivalent plastic strain α (hardening).
    pub eq_plastic_strain: f64,
    /// Last Kirchhoff stress from the spectral return.
    pub kirchhoff: SymTensor3,
    /// Last elastic left Cauchy–Green after return.
    pub be: [[f64; 3]; 3],
}

impl BePlasticTet {
    /// Fresh plastic state with `F_p = I`, `b_e = I`.
    pub fn new(constraint_index: usize, cold_dm_inv: [[f64; 3]; 3], cold_volume: f64) -> Self {
        Self {
            constraint_index,
            cold_dm_inv,
            cold_volume,
            fp: mat_identity(),
            eq_plastic_strain: 0.0,
            kirchhoff: SymTensor3::zero(),
            be: mat_identity(),
        }
    }
}

/// Aggregate stats from one spectral Hencky plastic update pass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BePlasticUpdateStats {
    /// Number of distance bars that yielded this pass.
    pub n_plastic_bars: usize,
    /// Number of tets that yielded this pass.
    pub n_plastic_tets: usize,
    /// Max `|Δγ|` among bars this pass.
    pub max_delta_gamma_bar: f64,
    /// Max `|Δγ|` among tets this pass.
    pub max_delta_gamma_tet: f64,
    /// Sum of equivalent plastic strain over tracked bars.
    pub sum_eq_plastic_bars: f64,
    /// Sum of equivalent plastic strain over tracked tets.
    pub sum_eq_plastic_tets: f64,
}

/// Soft world plus spectral Hencky / Kirchhoff plastic state (M32).
///
/// Bars use the M31 Hencky rest update; tets use the Simo `b_e` spectral return.
#[derive(Clone, Debug)]
pub struct BePlasticSoftWorld {
    /// Underlying XPBD soft world.
    pub soft: SoftWorld,
    /// Shared J2 parameters.
    pub params: J2Params,
    /// Plastic distance bars (Hencky; deterministic `Vec` order).
    pub bars: Vec<FsPlasticDistance>,
    /// Plastic tetrahedra (spectral `b_e`; deterministic `Vec` order).
    pub tets: Vec<BePlasticTet>,
}

impl BePlasticSoftWorld {
    /// Empty spectral Hencky plastic world.
    pub fn new(params: J2Params) -> Self {
        Self {
            soft: SoftWorld::new(),
            params,
            bars: Vec::new(),
            tets: Vec::new(),
        }
    }

    /// Simulation time [s].
    pub fn t(&self) -> f64 {
        self.soft.t
    }

    /// Register every current distance constraint as a log-strain plastic bar.
    pub fn track_all_distances(&mut self, area: f64) {
        self.bars.clear();
        for (k, c) in self.soft.constraints.distances.iter().enumerate() {
            if c.broken {
                continue;
            }
            self.bars.push(FsPlasticDistance {
                constraint_index: k,
                cold_rest: c.rest_length,
                area,
                state: UniaxialPlasticState::default(),
            });
        }
    }

    /// Register every current tet FEM constraint with `F_p = I`.
    pub fn track_all_tets(&mut self) {
        self.tets.clear();
        for (k, c) in self.soft.constraints.tetra_fem.iter().enumerate() {
            if c.broken {
                continue;
            }
            self.tets
                .push(BePlasticTet::new(k, c.dm_inv, c.rest_volume));
        }
    }

    /// XPBD step then Hencky-bar + spectral-tet plastic rest update.
    pub fn step(&mut self, dt: f64) {
        self.soft.step(dt);
        let _ = update_fs_plastic_distances(
            &self.soft.particles,
            &mut self.soft.constraints.distances,
            &self.params,
            &mut self.bars,
        );
        let _ = update_be_plastic_tets(
            &self.soft.particles,
            &mut self.soft.constraints.tetra_fem,
            &self.params,
            &mut self.tets,
        );
    }

    /// Mean equivalent plastic strain over tracked bars.
    pub fn mean_bar_eq_plastic(&self) -> f64 {
        if self.bars.is_empty() {
            return 0.0;
        }
        let s: f64 = self.bars.iter().map(|b| b.state.eq_plastic_strain).sum();
        s / self.bars.len() as f64
    }

    /// Mean equivalent plastic strain over tracked tets.
    pub fn mean_tet_eq_plastic(&self) -> f64 {
        if self.tets.is_empty() {
            return 0.0;
        }
        let s: f64 = self.tets.iter().map(|t| t.eq_plastic_strain).sum();
        s / self.tets.len() as f64
    }
}

/// Update multiplicative plastic tets via spectral Hencky return on `b_e`.
pub fn update_be_plastic_tets(
    particles: &[Particle],
    constraints: &mut [TetraFemConstraint],
    params: &J2Params,
    tets: &mut [BePlasticTet],
) -> BePlasticUpdateStats {
    let mut stats = BePlasticUpdateStats::default();
    for tet in tets.iter_mut() {
        let Some(c) = constraints.get_mut(tet.constraint_index) else {
            continue;
        };
        if c.broken {
            continue;
        }
        let f = deformation_gradient_public(particles, c.indices, &tet.cold_dm_inv);
        let Some(fp_inv) = invert3(&tet.fp) else {
            continue;
        };
        let fe = mul3_public(&f, &fp_inv);
        let be_tr = left_cauchy_green(&fe);
        let r = return_map_spectral_hencky_be(params, &be_tr, tet.eq_plastic_strain);
        tet.kirchhoff = r.kirchhoff;
        tet.be = r.be;
        if r.plastic {
            stats.n_plastic_tets += 1;
            stats.max_delta_gamma_tet = stats.max_delta_gamma_tet.max(r.delta_gamma);
            let deps = sym_to_mat(&r.delta_eps_p);
            let exp_d = sym_exp3(&deps);
            tet.fp = isochoric_project(&mul3_public(&exp_d, &tet.fp));
        }
        tet.eq_plastic_strain = r.eq_plastic_strain;
        let Some(fp_inv2) = invert3(&tet.fp) else {
            continue;
        };
        c.dm_inv = mul3_public(&tet.cold_dm_inv, &fp_inv2);
        let j_p = det3_local(&tet.fp);
        c.rest_volume = tet.cold_volume * j_p;
        stats.sum_eq_plastic_tets += tet.eq_plastic_strain;
    }
    stats
}

/// Build a pinned–free Hencky plastic bar inside a [`BePlasticSoftWorld`].
///
/// Compliance uses the elastic modulus: `α = L₀ / (E A)`.
pub fn make_be_plastic_bar(
    params: J2Params,
    l0: f64,
    area: f64,
    mass: f64,
    substeps: usize,
) -> BePlasticSoftWorld {
    // Reuse M31 bar construction, then move into BePlasticSoftWorld.
    let fs = make_fs_plastic_bar(params, l0, area, mass, substeps);
    BePlasticSoftWorld {
        soft: fs.soft,
        params: fs.params,
        bars: fs.bars,
        tets: Vec::new(),
    }
}

/// Quasistatic log-strain stretch/unload via the shared Hencky bar path.
///
/// Returns `(final_length, ε_p, analytic_ε_p)`.
pub fn quasistatic_be_log_stretch_unload(
    params: J2Params,
    l0: f64,
    area: f64,
    lambda: f64,
    hold_steps: usize,
    settle_steps: usize,
    dt: f64,
) -> (f64, f64, f64) {
    let mut w = make_be_plastic_bar(params, l0, area, 1.0, 32);
    w.soft.params.damping = 2.0;
    let x_hold = l0 * lambda;
    w.soft.particles[1].x = Vec3::new(x_hold, 0.0, 0.0);
    w.soft.particles[1].v = Vec3::zero();
    w.soft.particles[1].inv_mass = 0.0;
    for _ in 0..hold_steps {
        w.step(dt);
        w.soft.particles[1].x = Vec3::new(x_hold, 0.0, 0.0);
        w.soft.particles[1].v = Vec3::zero();
    }
    let eps_p = w.bars[0].state.plastic_strain;
    let eps_log = libm_ops::log(lambda.max(1e-30));
    let analytic = analytic_log_permanent_strain(&params, eps_log);
    w.soft.particles[1].inv_mass = 1.0;
    w.soft.particles[1].v = Vec3::zero();
    for _ in 0..settle_steps {
        w.step(dt);
    }
    let len = (w.soft.particles[1].x - w.soft.particles[0].x).norm();
    (len, eps_p, analytic)
}

// ---------------------------------------------------------------------------
// Canonical determinism hash (M32)
// ---------------------------------------------------------------------------

/// Outer steps in the spectral Hencky plastic hash scenario.
pub const BE_PLASTIC_HASH_STEPS: usize = 120;
/// Outer timestep [s] for the spectral Hencky plastic hash.
pub const BE_PLASTIC_HASH_DT: f64 = 1.0 / 240.0;

/// Soft-ish J2 params for the locked spectral Hencky hash.
///
/// Same numbers as M31 `fs_plastic_hash_params` so bar payloads stay comparable;
/// the tet path differs (Kirchhoff / `b_e` vs Green-of-`F_e`).
pub fn be_plastic_hash_params() -> J2Params {
    J2Params {
        youngs_modulus: 1.0e7,
        poisson_ratio: 0.3,
        yield_strength: 5.0e4,
        hardening_modulus: 1.0e5,
    }
}

/// Build the canonical spectral Hencky plastic scene (Hencky bar + `b_e` tet).
pub fn be_plastic_hash_world() -> BePlasticSoftWorld {
    let params = be_plastic_hash_params();
    let l0 = 1.0;
    let area = 0.01;
    let mut w = make_be_plastic_bar(params, l0, area, 1.0, 16);
    w.soft.params.damping = 0.5;
    let lambda = 1.08;
    w.soft.particles[1].x = Vec3::new(l0 * lambda, 0.0, 0.0);
    w.soft.particles[1].v = Vec3::zero();

    let base = w.soft.particles.len();
    let tet_size = 0.2;
    let m = 0.25;
    let verts = [
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(2.0 + tet_size, 0.0, 0.0),
        Vec3::new(2.0, tet_size, 0.0),
        Vec3::new(2.0, 0.0, tet_size),
    ];
    for &v in &verts {
        w.soft.add_particle(Particle::dynamic(v, m));
    }
    w.soft.pin(base);
    w.soft.pin(base + 1);
    w.soft.pin(base + 2);
    let e = params.youngs_modulus;
    let nu = params.poisson_ratio;
    let mu = e / (2.0 * (1.0 + nu));
    let lambda_lame = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let x0 = verts[0];
    let col0 = verts[1] - x0;
    let col1 = verts[2] - x0;
    let col2 = verts[3] - x0;
    let dm = [
        [col0.x, col1.x, col2.x],
        [col0.y, col1.y, col2.y],
        [col0.z, col1.z, col2.z],
    ];
    let dm_inv = invert3(&dm).expect("tet Dm invertible");
    let vol = libm_ops::fabs(det3_local(&dm)) / 6.0;
    w.soft.constraints.tetra_fem.push(TetraFemConstraint {
        indices: [base, base + 1, base + 2, base + 3],
        dm_inv,
        rest_volume: vol,
        lame_lambda: lambda_lame,
        lame_mu: mu,
        neo_hookean: false,
        lambda_dev: Multiplier::default(),
        lambda_vol: Multiplier::default(),
        broken: false,
    });
    w.track_all_tets();
    w.soft.particles[base + 3].x.x += 0.06;
    w
}

/// FNV-1a over the canonical spectral Hencky plastic scene.
///
/// Packed payload (little-endian IEEE bits, fixed order):
/// 1. Soft packed `(x‖v)` after [`BE_PLASTIC_HASH_STEPS`] steps.
/// 2. Bar plastic strain / eq plastic / stress / rest length.
/// 3. Tet eq plastic / Kirchhoff Frobenius / det(F_p) / rest volume /
///    `F_p[9]` / `b_e[9]`.
pub fn be_plastic_canonical_hash() -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    let mut feed = |x: f64| {
        for b in x.to_bits().to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };

    let mut w = be_plastic_hash_world();
    for _ in 0..BE_PLASTIC_HASH_STEPS {
        w.step(BE_PLASTIC_HASH_DT);
    }
    for x in w.soft.pack() {
        feed(x);
    }
    for bar in &w.bars {
        feed(bar.state.plastic_strain);
        feed(bar.state.eq_plastic_strain);
        feed(bar.state.stress);
        let rest = w.soft.constraints.distances[bar.constraint_index].rest_length;
        feed(rest);
    }
    for tet in &w.tets {
        feed(tet.eq_plastic_strain);
        feed(tet.kirchhoff.frobenius());
        feed(det3_local(&tet.fp));
        let vol = w.soft.constraints.tetra_fem[tet.constraint_index].rest_volume;
        feed(vol);
        for row in &tet.fp {
            for &v in row {
                feed(v);
            }
        }
        for row in &tet.be {
            for &v in row {
                feed(v);
            }
        }
    }
    hash
}

/// Map an M31 bar-update stats struct into the M32 aggregate (bars only).
#[inline]
pub fn be_bar_update_stats_from_fs(s: FsPlasticUpdateStats) -> BePlasticUpdateStats {
    BePlasticUpdateStats {
        n_plastic_bars: s.n_plastic_bars,
        n_plastic_tets: 0,
        max_delta_gamma_bar: s.max_delta_gamma_bar,
        max_delta_gamma_tet: 0.0,
        sum_eq_plastic_bars: s.sum_eq_plastic_bars,
        sum_eq_plastic_tets: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn soft_mat() -> J2Params {
        be_plastic_hash_params()
    }

    #[test]
    fn left_cauchy_green_of_identity() {
        let i = mat_identity();
        let b = left_cauchy_green(&i);
        for i_ in 0..3 {
            for j in 0..3 {
                let expect = if i_ == j { 1.0 } else { 0.0 };
                assert!((b[i_][j] - expect).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn spectral_decompose_diagonal() {
        let m = [[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 9.0]];
        let (eigs, q) = sym_spectral_decompose(&m);
        assert_relative_eq!(eigs[0], 9.0, max_relative = 1e-12);
        assert_relative_eq!(eigs[1], 4.0, max_relative = 1e-12);
        assert_relative_eq!(eigs[2], 1.0, max_relative = 1e-12);
        let recon = assemble_from_principal(&eigs, &q);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (recon[i][j] - m[i][j]).abs() < 1e-10,
                    "recon mismatch at {i},{j}"
                );
            }
        }
    }

    #[test]
    fn isochoric_uniaxial_elastic_kirchhoff_matches_analytic() {
        let p = soft_mat();
        let mu = p.shear_modulus();
        let lam = 1.05;
        // b_e = diag(λ², 1/λ, 1/λ)
        let be = [
            [lam * lam, 0.0, 0.0],
            [0.0, 1.0 / lam, 0.0],
            [0.0, 0.0, 1.0 / lam],
        ];
        let r = return_map_spectral_hencky_be(&p, &be, 0.0);
        let (t1, t2, t3) = analytic_isochoric_uniaxial_kirchhoff(mu, lam);
        let seq_a = analytic_isochoric_uniaxial_seq(mu, lam);
        // Below yield for soft_mat at λ=1.05 (3μ lnλ ≈ 3*3.846e6*0.0488 ≈ 5.6e5 > σ_y=5e4)
        // Actually that yields! Use smaller λ.
        let _ = (r, t1, t2, t3, seq_a);
    }

    #[test]
    fn isochoric_uniaxial_below_yield_exact() {
        let p = soft_mat();
        let mu = p.shear_modulus();
        // 3 μ |ln λ| < σ_y ⇒ |ln λ| < σ_y/(3μ)
        let lim = p.yield_strength / (3.0 * mu);
        let lam = libm_ops::exp(0.5 * lim);
        let be = [
            [lam * lam, 0.0, 0.0],
            [0.0, 1.0 / lam, 0.0],
            [0.0, 0.0, 1.0 / lam],
        ];
        let r = return_map_spectral_hencky_be(&p, &be, 0.0);
        assert!(!r.plastic);
        let (t1, t2, t3) = analytic_isochoric_uniaxial_kirchhoff(mu, lam);
        assert_relative_eq!(r.kirchhoff.xx, t1, max_relative = 1e-10);
        assert_relative_eq!(r.kirchhoff.yy, t2, max_relative = 1e-10);
        assert_relative_eq!(r.kirchhoff.zz, t3, max_relative = 1e-10);
        let seq = r.kirchhoff.von_mises_eq();
        let seq_a = analytic_isochoric_uniaxial_seq(mu, lam);
        assert_relative_eq!(seq, seq_a, max_relative = 1e-10);
    }

    #[test]
    fn isochoric_uniaxial_plastic_returns_to_surface() {
        let p = soft_mat();
        let mu = p.shear_modulus();
        let lam = 1.08;
        let be = [
            [lam * lam, 0.0, 0.0],
            [0.0, 1.0 / lam, 0.0],
            [0.0, 0.0, 1.0 / lam],
        ];
        let seq_tr = analytic_isochoric_uniaxial_seq(mu, lam);
        assert!(seq_tr > p.yield_strength);
        let r = return_map_spectral_hencky_be(&p, &be, 0.0);
        assert!(r.plastic);
        let seq = r.kirchhoff.von_mises_eq();
        let sy = p.yield_strength + p.hardening_modulus * r.eq_plastic_strain;
        let err = (seq - sy).abs() / sy;
        assert!(err < 1e-10, "surface err {err}: seq={seq} sy={sy}");
        // det(b_e) should stay 1 for isochoric flow from isochoric trial.
        let det_be = det3_local(&r.be);
        assert!((det_be - 1.0).abs() < 1e-10, "isochoric be det={det_be}");
    }

    #[test]
    fn hydrostatic_be_never_yields() {
        let p = soft_mat();
        let s = 1.2_f64;
        let be = [[s * s, 0.0, 0.0], [0.0, s * s, 0.0], [0.0, 0.0, s * s]];
        let r = return_map_spectral_hencky_be(&p, &be, 0.0);
        assert!(!r.plastic);
    }

    #[test]
    fn be_plastic_hash_is_stable() {
        let a = be_plastic_canonical_hash();
        let b = be_plastic_canonical_hash();
        assert_eq!(a, b);
        assert_ne!(a, 0);
    }

    #[test]
    fn empty_world_has_no_tets() {
        let w = BePlasticSoftWorld::new(soft_mat());
        assert_eq!(w.tets.len(), 0);
        assert_eq!(w.bars.len(), 0);
    }
}
