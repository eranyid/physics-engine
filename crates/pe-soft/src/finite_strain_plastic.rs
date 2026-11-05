//! Finite-strain multiplicative F_p plasticity for soft XPBD (M31).
//!
//! Extends the M30 rest-update coupling beyond the small-strain
//! `F_p ≈ I + ε_p` approximation:
//!
//! # 1-D distance bars (Hencky / logarithmic strain)
//!
//! `ε = ln(L / L₀_cold)`. After the uniaxial J2 return map,
//! `rest_length ← L₀_cold · exp(ε_p)`. At small strain this recovers the
//! M30 engineering-strain rest update (`ln(1+e)≈e`, `exp(ε_p)≈1+ε_p`).
//!
//! # Tetra FEM (multiplicative decomposition)
//!
//! `F = F_e F_p` with `F = Ds Dm_cold⁻¹`. Elastic Green strain is taken from
//! `F_e = F F_p⁻¹` and fed to the existing Simo–Hughes J2 map with a virgin
//! plastic-strain tensor (hardening `α` carries over). The plastic increment
//! `Δε_p` updates the intermediate configuration via the exponential map
//! `F_p ← exp(Δε_p) F_p`, then an isochoric projection
//! `F_p ← F_p / ∛det(F_p)` so J2 flow stays volume-preserving.
//!
//! XPBD rest: `Dm_inv ← Dm_cold⁻¹ F_p⁻¹`, `V₀ ← V₀_cold · det(F_p)`.
//!
//! # Determinism
//!
//! Fixed constraint index order; portable `libm`; no `HashMap`; no `mul_add`.
//!
//! References (formulas only, clean-room): Simo & Hughes multiplicative
//! plasticity / exponential map; Hencky (logarithmic) strain; Macklin XPBD
//! rest-update pattern. No FleX / PhysX / PBD source transcribed.

use pe_math::libm_ops;
use pe_math::Vec3;

use crate::constraint::{
    deformation_gradient_public, green_strain_public, invert3, mul3_public, DistanceConstraint,
    Multiplier, TetraFemConstraint,
};
use crate::particle::Particle;
use crate::plasticity::{
    return_map_j2_3d, return_map_uniaxial, J2Params, J2PlasticState3, SymTensor3,
    UniaxialPlasticState,
};
use crate::solver::{SoftParams, SoftWorld};

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

/// Convert Green strain matrix to [`SymTensor3`].
fn green_to_sym(e: &[[f64; 3]; 3]) -> SymTensor3 {
    SymTensor3 {
        xx: e[0][0],
        yy: e[1][1],
        zz: e[2][2],
        yz: 0.5 * (e[1][2] + e[2][1]),
        xz: 0.5 * (e[0][2] + e[2][0]),
        xy: 0.5 * (e[0][1] + e[1][0]),
    }
}

/// SymTensor3 → dense symmetric 3×3.
fn sym_to_mat(s: &SymTensor3) -> [[f64; 3]; 3] {
    [[s.xx, s.xy, s.xz], [s.xy, s.yy, s.yz], [s.xz, s.yz, s.zz]]
}

/// Analytic eigenvalues of a symmetric 3×3 (Kopp / Wikipedia closed form).
///
/// Returns `(λ0, λ1, λ2)` unsorted. Matches the cubic used by
/// [`crate::fracture::max_principal_sym`].
pub fn sym_eigenvalues_3(m: &[[f64; 3]; 3]) -> [f64; 3] {
    let a = m[0][0];
    let b = 0.5 * (m[0][1] + m[1][0]);
    let c = 0.5 * (m[0][2] + m[2][0]);
    let d = m[1][1];
    let e = 0.5 * (m[1][2] + m[2][1]);
    let f = m[2][2];
    let p1 = b * b + c * c + e * e;
    if p1 < 1e-30 {
        return [a, d, f];
    }
    let q = (a + d + f) / 3.0;
    let p2 = (a - q) * (a - q) + (d - q) * (d - q) + (f - q) * (f - q) + 2.0 * p1;
    let p = libm_ops::sqrt(p2 / 6.0);
    let inv_p = 1.0 / p;
    let b00 = (a - q) * inv_p;
    let b01 = b * inv_p;
    let b02 = c * inv_p;
    let b11 = (d - q) * inv_p;
    let b12 = e * inv_p;
    let b22 = (f - q) * inv_p;
    let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
        + b02 * (b01 * b12 - b11 * b02);
    let r = (det_b * 0.5).clamp(-1.0, 1.0);
    let phi = libm_ops::acos(r) / 3.0;
    let eig0 = q + 2.0 * p * libm_ops::cos(phi);
    let eig1 = q + 2.0 * p * libm_ops::cos(phi + 2.0 * core::f64::consts::PI / 3.0);
    let eig2 = q + 2.0 * p * libm_ops::cos(phi + 4.0 * core::f64::consts::PI / 3.0);
    [eig0, eig1, eig2]
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
    // Cross products of rows → nullspace candidates.
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
        // Degenerate: pick axis of largest |diagonal − λ| residual fallback.
        return [1.0, 0.0, 0.0];
    }
    let inv = 1.0 / libm_ops::sqrt(n2);
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Matrix exponential of a symmetric 3×3 via spectral decomposition.
///
/// `exp(A) = Σ_i exp(λ_i) q_i ⊗ q_i`. Portable `libm::exp` only.
pub fn sym_exp3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let eigs = sym_eigenvalues_3(m);
    // Sort eigenvalues descending for a deterministic eigenvector build order.
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| eigs[j].total_cmp(&eigs[i]));
    let mut q = [[0.0; 3]; 3]; // columns = eigenvectors
    for (col, &idx) in order.iter().enumerate() {
        let mut v = sym_eigenvector(m, eigs[idx]);
        // Re-orthogonalize against previous columns (modified Gram–Schmidt).
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
            // Pick a coordinate axis least aligned with previous columns.
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
    // exp(A) = Q diag(exp λ) Qᵀ — rebuild in sorted column order.
    let mut out = [[0.0; 3]; 3];
    for col in 0..3 {
        let lam = eigs[order[col]];
        let e = libm_ops::exp(lam);
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] += e * q[i][col] * q[j][col];
            }
        }
    }
    out
}

/// Isochoric projection `F ← F / ∛det(F)` (J2 plastic flow is volume-preserving).
pub fn isochoric_project(fp: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let j = det3_local(fp);
    if j.abs() < 1e-30 {
        return mat_identity();
    }
    let s = 1.0 / libm_ops::cbrt(j);
    [
        [fp[0][0] * s, fp[0][1] * s, fp[0][2] * s],
        [fp[1][0] * s, fp[1][1] * s, fp[1][2] * s],
        [fp[2][0] * s, fp[2][1] * s, fp[2][2] * s],
    ]
}

/// Per-distance finite-strain plastic bar (log-strain rest update).
#[derive(Clone, Debug)]
pub struct FsPlasticDistance {
    /// Index into `ConstraintSet::distances`.
    pub constraint_index: usize,
    /// Cold (undeformed) rest length L₀ [m].
    pub cold_rest: f64,
    /// Cross-section area A [m²].
    pub area: f64,
    /// Accumulated uniaxial plastic state (Hencky `ε_p`).
    pub state: UniaxialPlasticState,
}

/// Per-tet multiplicative plastic state.
#[derive(Clone, Debug)]
pub struct FsPlasticTet {
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
    /// Last Cauchy/PK2-proxy stress from the elastic return map.
    pub stress: SymTensor3,
}

impl FsPlasticTet {
    /// Fresh plastic state with `F_p = I`.
    pub fn new(constraint_index: usize, cold_dm_inv: [[f64; 3]; 3], cold_volume: f64) -> Self {
        Self {
            constraint_index,
            cold_dm_inv,
            cold_volume,
            fp: mat_identity(),
            eq_plastic_strain: 0.0,
            stress: SymTensor3::zero(),
        }
    }
}

/// Aggregate stats from one finite-strain plastic update pass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FsPlasticUpdateStats {
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

/// Soft world plus finite-strain / multiplicative plastic state.
#[derive(Clone, Debug)]
pub struct FsPlasticSoftWorld {
    /// Underlying XPBD soft world.
    pub soft: SoftWorld,
    /// Shared J2 parameters.
    pub params: J2Params,
    /// Plastic distance bars (deterministic `Vec` order).
    pub bars: Vec<FsPlasticDistance>,
    /// Plastic tetrahedra (deterministic `Vec` order).
    pub tets: Vec<FsPlasticTet>,
}

impl FsPlasticSoftWorld {
    /// Empty finite-strain plastic world.
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
                .push(FsPlasticTet::new(k, c.dm_inv, c.rest_volume));
        }
    }

    /// XPBD step then finite-strain plastic rest update.
    pub fn step(&mut self, dt: f64) {
        self.soft.step(dt);
        let _ = update_fs_plastic_distances(
            &self.soft.particles,
            &mut self.soft.constraints.distances,
            &self.params,
            &mut self.bars,
        );
        let _ = update_fs_plastic_tets(
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

/// Logarithmic (Hencky) axial strain `ln(L / L₀_cold)`.
#[inline]
pub fn bar_log_strain(particles: &[Particle], c: &DistanceConstraint, cold_rest: f64) -> f64 {
    if cold_rest <= 1e-30 {
        return 0.0;
    }
    let len = (particles[c.i].x - particles[c.j].x).norm();
    if len <= 1e-30 {
        return libm_ops::log(1e-30 / cold_rest);
    }
    libm_ops::log(len / cold_rest)
}

/// Analytic permanent Hencky strain after a virgin stretch to `ε_log`.
pub fn analytic_log_permanent_strain(params: &J2Params, eps_log: f64) -> f64 {
    let r = return_map_uniaxial(params, UniaxialPlasticState::default(), eps_log);
    r.state.plastic_strain
}

/// Analytic uniaxial stress after a virgin log-strain stretch.
pub fn analytic_log_uniaxial_stress(params: &J2Params, eps_log: f64) -> f64 {
    let r = return_map_uniaxial(params, UniaxialPlasticState::default(), eps_log);
    r.state.stress
}

/// Update log-strain plastic bars; write `rest = L₀ exp(ε_p)`.
pub fn update_fs_plastic_distances(
    particles: &[Particle],
    constraints: &mut [DistanceConstraint],
    params: &J2Params,
    bars: &mut [FsPlasticDistance],
) -> FsPlasticUpdateStats {
    let mut stats = FsPlasticUpdateStats::default();
    for bar in bars.iter_mut() {
        let Some(c) = constraints.get_mut(bar.constraint_index) else {
            continue;
        };
        if c.broken {
            continue;
        }
        let eps = bar_log_strain(particles, c, bar.cold_rest);
        let r = return_map_uniaxial(params, bar.state, eps);
        if r.plastic {
            stats.n_plastic_bars += 1;
            stats.max_delta_gamma_bar = stats.max_delta_gamma_bar.max(r.delta_gamma);
        }
        bar.state = r.state;
        c.rest_length = bar.cold_rest * libm_ops::exp(bar.state.plastic_strain);
        stats.sum_eq_plastic_bars += bar.state.eq_plastic_strain;
    }
    stats
}

/// Update multiplicative plastic tets from elastic Green strain of `F_e`.
pub fn update_fs_plastic_tets(
    particles: &[Particle],
    constraints: &mut [TetraFemConstraint],
    params: &J2Params,
    tets: &mut [FsPlasticTet],
) -> FsPlasticUpdateStats {
    let mut stats = FsPlasticUpdateStats::default();
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
        let e = green_strain_public(&fe);
        let strain = green_to_sym(&e);
        // Elastic trial in the intermediate configuration: ε_p = 0, α carried.
        let trial = J2PlasticState3 {
            plastic_strain: SymTensor3::zero(),
            eq_plastic_strain: tet.eq_plastic_strain,
            stress: SymTensor3::zero(),
        };
        let r = return_map_j2_3d(params, trial, strain);
        tet.stress = r.state.stress;
        if r.plastic {
            stats.n_plastic_tets += 1;
            stats.max_delta_gamma_tet = stats.max_delta_gamma_tet.max(r.delta_gamma);
            let deps = sym_to_mat(&r.state.plastic_strain);
            let exp_d = sym_exp3(&deps);
            tet.fp = isochoric_project(&mul3_public(&exp_d, &tet.fp));
        }
        tet.eq_plastic_strain = r.state.eq_plastic_strain;
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

/// Build a pinned–free finite-strain plastic bar.
///
/// Compliance uses the elastic modulus: `α = L₀ / (E A)`.
pub fn make_fs_plastic_bar(
    params: J2Params,
    l0: f64,
    area: f64,
    mass: f64,
    substeps: usize,
) -> FsPlasticSoftWorld {
    let mut w = FsPlasticSoftWorld::new(params);
    w.soft.params = SoftParams {
        substeps: substeps.max(1),
        iterations: 2,
        gravity: Vec3::zero(),
        damping: 0.0,
    };
    let i0 = w.soft.add_particle(Particle::pinned(Vec3::zero()));
    let i1 = w
        .soft
        .add_particle(Particle::dynamic(Vec3::new(l0, 0.0, 0.0), mass));
    let alpha = l0 / (params.youngs_modulus * area);
    w.soft.constraints.distances.push(DistanceConstraint {
        i: i0,
        j: i1,
        rest_length: l0,
        compliance: alpha,
        lambda: Multiplier::default(),
        broken: false,
    });
    w.track_all_distances(area);
    w
}

/// Quasistatic log-strain stretch: pin free end at stretch ratio `lambda`,
/// step XPBD+plastic, then release.
///
/// Returns `(final_length, ε_p, analytic_ε_p)` where analytic uses
/// `ln(lambda)` in the virgin return map.
pub fn quasistatic_log_stretch_unload(
    params: J2Params,
    l0: f64,
    area: f64,
    lambda: f64,
    hold_steps: usize,
    settle_steps: usize,
    dt: f64,
) -> (f64, f64, f64) {
    let mut w = make_fs_plastic_bar(params, l0, area, 1.0, 32);
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

/// Axial force proxy `σ A` from the finite-strain bar state.
#[inline]
pub fn fs_bar_axial_force(bar: &FsPlasticDistance) -> f64 {
    bar.state.stress * bar.area
}

// ---------------------------------------------------------------------------
// Canonical determinism hash (M31)
// ---------------------------------------------------------------------------

/// Outer steps in the finite-strain plastic hash scenario.
pub const FS_PLASTIC_HASH_STEPS: usize = 120;
/// Outer timestep [s] for the finite-strain plastic hash.
pub const FS_PLASTIC_HASH_DT: f64 = 1.0 / 240.0;

/// Soft-ish J2 params for the locked finite-strain hash (large log strain OK).
///
/// Not a verified alloy row — chosen so `λ ≈ 1.15` is well past yield while
/// remaining numerically gentle for XPBD.
pub fn fs_plastic_hash_params() -> J2Params {
    J2Params {
        youngs_modulus: 1.0e7,
        poisson_ratio: 0.3,
        yield_strength: 5.0e4,
        hardening_modulus: 1.0e5,
    }
}

/// Build the canonical finite-strain plastic scene (log bar + multiplicative tet).
pub fn fs_plastic_hash_world() -> FsPlasticSoftWorld {
    let params = fs_plastic_hash_params();
    let l0 = 1.0;
    let area = 0.01;
    let mut w = make_fs_plastic_bar(params, l0, area, 1.0, 16);
    w.soft.params.damping = 0.5;
    // Stretch ratio past yield (ε_y = σ_y/E = 0.005; λ=1.08 ⇒ ε_log≈0.077).
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
    // Simple shear of the apex (finite strain, volume-preserving to first order).
    w.soft.particles[base + 3].x.x += 0.06;
    w
}

/// FNV-1a over the canonical finite-strain plastic scene.
///
/// Packed payload (little-endian IEEE bits, fixed order):
/// 1. Soft packed `(x‖v)` after [`FS_PLASTIC_HASH_STEPS`] steps.
/// 2. Bar plastic strain / eq plastic / stress / rest length.
/// 3. Tet eq plastic / stress Frobenius / det(F_p) / rest volume / F_p entries.
pub fn fs_plastic_canonical_hash() -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    let mut feed = |x: f64| {
        for b in x.to_bits().to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };

    let mut w = fs_plastic_hash_world();
    for _ in 0..FS_PLASTIC_HASH_STEPS {
        w.step(FS_PLASTIC_HASH_DT);
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
        feed(tet.stress.frobenius());
        feed(det3_local(&tet.fp));
        let vol = w.soft.constraints.tetra_fem[tet.constraint_index].rest_volume;
        feed(vol);
        for row in &tet.fp {
            for &v in row {
                feed(v);
            }
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sym_exp_of_zero_is_identity() {
        let z = [[0.0; 3]; 3];
        let e = sym_exp3(&z);
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((e[i][j] - expect).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn sym_exp_diagonal_matches_componentwise() {
        let m = [[0.1, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.05]];
        let e = sym_exp3(&m);
        assert_relative_eq!(e[0][0], libm_ops::exp(0.1), max_relative = 1e-10);
        assert_relative_eq!(e[1][1], libm_ops::exp(-0.2), max_relative = 1e-10);
        assert_relative_eq!(e[2][2], libm_ops::exp(0.05), max_relative = 1e-10);
        assert!(e[0][1].abs() < 1e-12 && e[0][2].abs() < 1e-12);
    }

    #[test]
    fn isochoric_projection_unit_det() {
        let m = [[1.2, 0.1, 0.0], [0.0, 0.9, 0.05], [0.0, 0.0, 1.1]];
        let p = isochoric_project(&m);
        assert!((det3_local(&p) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn log_rest_update_matches_return_map() {
        let p = fs_plastic_hash_params();
        let l0 = 1.0;
        let mut w = make_fs_plastic_bar(p, l0, 0.01, 1.0, 8);
        let lambda = 1.05;
        w.soft.particles[1].x = Vec3::new(l0 * lambda, 0.0, 0.0);
        let stats = update_fs_plastic_distances(
            &w.soft.particles,
            &mut w.soft.constraints.distances,
            &p,
            &mut w.bars,
        );
        assert!(stats.n_plastic_bars >= 1);
        let eps = libm_ops::log(lambda);
        let expect = analytic_log_permanent_strain(&p, eps);
        assert_relative_eq!(w.bars[0].state.plastic_strain, expect, max_relative = 1e-12);
        assert_relative_eq!(
            w.soft.constraints.distances[0].rest_length,
            l0 * libm_ops::exp(expect),
            max_relative = 1e-12
        );
    }

    #[test]
    fn fs_plastic_hash_is_stable() {
        let a = fs_plastic_canonical_hash();
        let b = fs_plastic_canonical_hash();
        assert_eq!(a, b);
        assert_ne!(a, 0);
    }
}
