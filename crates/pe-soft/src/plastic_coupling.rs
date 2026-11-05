//! Soft XPBD ↔ J2 plasticity coupling (M30).
//!
//! Clean-room rest-update plasticity on distance bars and tetra FEM:
//! measure total strain from the cold rest configuration, apply the
//! existing Simo–Hughes J2 return map, then permanently update XPBD rest
//! lengths / rest shapes so the elastic residual matches `ε − ε_p`.
//!
//! # 1-D distance bars
//!
//! Engineering strain `ε = (L − L₀_cold) / L₀_cold`. After the return map,
//! `rest_length ← L₀_cold (1 + ε_p)`. XPBD then enforces the elastic strain
//! only. Compliance stays `α = L₀_cold / (E A)` (elastic modulus); plastic
//! flow lives entirely in the rest update.
//!
//! # Tetra FEM
//!
//! Green–Lagrange `E = ½(FᵀF − I)` from `F = Ds Dm_cold⁻¹` is fed to the
//! 3-D J2 map. Plastic deformation is applied multiplicatively:
//! `F_p ≈ I + ε_p` (small-strain), `Dm_inv ← Dm_cold⁻¹ F_p⁻¹`,
//! `V₀ ← V₀_cold det(F_p)`. J2 flow is isochoric so `det(F_p) ≈ 1`.
//!
//! # Determinism
//!
//! Fixed constraint index order; portable `libm`; no `HashMap`; no `mul_add`.
//!
//! References (formulas only): Simo & Hughes return mapping; Macklin XPBD
//! rest-update plasticity pattern (algorithms only — no FleX/PhysX source).

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

/// Per-distance plastic bookkeeping (parallel to a selected constraint index).
#[derive(Clone, Debug)]
pub struct PlasticDistance {
    /// Index into `ConstraintSet::distances`.
    pub constraint_index: usize,
    /// Cold (undeformed) rest length L₀ [m].
    pub cold_rest: f64,
    /// Cross-section area A [m²] (for stress queries / compliance).
    pub area: f64,
    /// Accumulated uniaxial plastic state.
    pub state: UniaxialPlasticState,
}

/// Per-tet plastic bookkeeping (parallel to a selected `tetra_fem` index).
#[derive(Clone, Debug)]
pub struct PlasticTet {
    /// Index into `ConstraintSet::tetra_fem`.
    pub constraint_index: usize,
    /// Cold rest-shape inverse Dm⁻¹.
    pub cold_dm_inv: [[f64; 3]; 3],
    /// Cold rest volume V₀ [m³].
    pub cold_volume: f64,
    /// Accumulated 3-D J2 plastic state.
    pub state: J2PlasticState3,
}

/// Aggregate stats from one plastic update pass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlasticUpdateStats {
    /// Number of distance bars that yielded this pass.
    pub n_plastic_bars: usize,
    /// Number of tets that yielded this pass.
    pub n_plastic_tets: usize,
    /// Max `|Δγ|` among bars this pass.
    pub max_delta_gamma_bar: f64,
    /// Max `|Δγ|` among tets this pass.
    pub max_delta_gamma_tet: f64,
    /// Sum of equivalent plastic strain over all tracked bars.
    pub sum_eq_plastic_bars: f64,
    /// Sum of equivalent plastic strain over all tracked tets.
    pub sum_eq_plastic_tets: f64,
}

/// Soft world plus J2 plastic state for selected constraints.
#[derive(Clone, Debug)]
pub struct PlasticSoftWorld {
    /// Underlying XPBD soft world.
    pub soft: SoftWorld,
    /// Shared J2 parameters for all tracked constraints.
    pub params: J2Params,
    /// Plastic distance bars (deterministic `Vec` order).
    pub bars: Vec<PlasticDistance>,
    /// Plastic tetrahedra (deterministic `Vec` order).
    pub tets: Vec<PlasticTet>,
}

impl PlasticSoftWorld {
    /// Empty plastic world with given J2 params.
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

    /// Register every current distance constraint as a plastic bar with area `area`.
    pub fn track_all_distances(&mut self, area: f64) {
        self.bars.clear();
        for (k, c) in self.soft.constraints.distances.iter().enumerate() {
            if c.broken {
                continue;
            }
            self.bars.push(PlasticDistance {
                constraint_index: k,
                cold_rest: c.rest_length,
                area,
                state: UniaxialPlasticState::default(),
            });
        }
    }

    /// Register every current tet FEM constraint as a plastic tet.
    pub fn track_all_tets(&mut self) {
        self.tets.clear();
        for (k, c) in self.soft.constraints.tetra_fem.iter().enumerate() {
            if c.broken {
                continue;
            }
            self.tets.push(PlasticTet {
                constraint_index: k,
                cold_dm_inv: c.dm_inv,
                cold_volume: c.rest_volume,
                state: J2PlasticState3::default(),
            });
        }
    }

    /// XPBD step then plastic rest update.
    pub fn step(&mut self, dt: f64) {
        self.soft.step(dt);
        let _ = update_plastic_distances(
            &self.soft.particles,
            &mut self.soft.constraints.distances,
            &self.params,
            &mut self.bars,
        );
        let _ = update_plastic_tets(
            &self.soft.particles,
            &mut self.soft.constraints.tetra_fem,
            &self.params,
            &mut self.tets,
        );
    }

    /// Integrate until `t_end` with fixed outer `dt`.
    pub fn run_for(&mut self, t_end: f64, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        while self.soft.t + dt <= t_end + 1e-15 {
            self.step(dt);
        }
    }

    /// Mean equivalent plastic strain over tracked bars (0 if none).
    pub fn mean_bar_eq_plastic(&self) -> f64 {
        if self.bars.is_empty() {
            return 0.0;
        }
        let s: f64 = self.bars.iter().map(|b| b.state.eq_plastic_strain).sum();
        s / self.bars.len() as f64
    }

    /// Mean equivalent plastic strain over tracked tets (0 if none).
    pub fn mean_tet_eq_plastic(&self) -> f64 {
        if self.tets.is_empty() {
            return 0.0;
        }
        let s: f64 = self.tets.iter().map(|t| t.state.eq_plastic_strain).sum();
        s / self.tets.len() as f64
    }
}

/// Engineering strain of a distance bar relative to cold rest.
#[inline]
pub fn bar_engineering_strain(
    particles: &[Particle],
    c: &DistanceConstraint,
    cold_rest: f64,
) -> f64 {
    if cold_rest.abs() < 1e-30 {
        return 0.0;
    }
    let len = (particles[c.j].x - particles[c.i].x).norm();
    (len - cold_rest) / cold_rest
}

/// Analytic permanent elongation after stretch to `ε_peak` then unload to σ=0:
/// `ΔL/L₀ = ε_p` from the uniaxial return map.
#[inline]
pub fn analytic_permanent_strain(params: &J2Params, eps_peak: f64) -> f64 {
    let r = return_map_uniaxial(params, UniaxialPlasticState::default(), eps_peak);
    r.state.plastic_strain
}

/// Analytic uniaxial stress at total strain `ε` (single return-map call from virgin).
#[inline]
pub fn analytic_uniaxial_stress(params: &J2Params, eps: f64) -> f64 {
    return_map_uniaxial(params, UniaxialPlasticState::default(), eps)
        .state
        .stress
}

/// Update plastic distance bars from current particle positions; write new rests.
pub fn update_plastic_distances(
    particles: &[Particle],
    constraints: &mut [DistanceConstraint],
    params: &J2Params,
    bars: &mut [PlasticDistance],
) -> PlasticUpdateStats {
    let mut stats = PlasticUpdateStats::default();
    for bar in bars.iter_mut() {
        let Some(c) = constraints.get_mut(bar.constraint_index) else {
            continue;
        };
        if c.broken {
            continue;
        }
        let eps = bar_engineering_strain(particles, c, bar.cold_rest);
        let r = return_map_uniaxial(params, bar.state, eps);
        if r.plastic {
            stats.n_plastic_bars += 1;
            stats.max_delta_gamma_bar = stats.max_delta_gamma_bar.max(r.delta_gamma);
        }
        bar.state = r.state;
        c.rest_length = bar.cold_rest * (1.0 + bar.state.plastic_strain);
        stats.sum_eq_plastic_bars += bar.state.eq_plastic_strain;
    }
    stats
}

/// Convert a 3×3 Green strain matrix to [`SymTensor3`].
fn green_to_sym(e: &[[f64; 3]; 3]) -> SymTensor3 {
    SymTensor3 {
        xx: e[0][0],
        yy: e[1][1],
        zz: e[2][2],
        // Symmetrize off-diagonals for robustness.
        yz: 0.5 * (e[1][2] + e[2][1]),
        xz: 0.5 * (e[0][2] + e[2][0]),
        xy: 0.5 * (e[0][1] + e[1][0]),
    }
}

/// Small-strain plastic deformation gradient `F_p ≈ I + ε_p`.
fn fp_from_plastic_strain(eps_p: &SymTensor3) -> [[f64; 3]; 3] {
    [
        [1.0 + eps_p.xx, eps_p.xy, eps_p.xz],
        [eps_p.xy, 1.0 + eps_p.yy, eps_p.yz],
        [eps_p.xz, eps_p.yz, 1.0 + eps_p.zz],
    ]
}

/// 3×3 determinant (local copy; avoids depending on private `det3`).
fn det3_local(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Update plastic tets from Green strain relative to cold rest; write new rests.
pub fn update_plastic_tets(
    particles: &[Particle],
    constraints: &mut [TetraFemConstraint],
    params: &J2Params,
    tets: &mut [PlasticTet],
) -> PlasticUpdateStats {
    let mut stats = PlasticUpdateStats::default();
    for tet in tets.iter_mut() {
        let Some(c) = constraints.get_mut(tet.constraint_index) else {
            continue;
        };
        if c.broken {
            continue;
        }
        let f = deformation_gradient_public(particles, c.indices, &tet.cold_dm_inv);
        let e = green_strain_public(&f);
        let strain = green_to_sym(&e);
        let r = return_map_j2_3d(params, tet.state, strain);
        if r.plastic {
            stats.n_plastic_tets += 1;
            stats.max_delta_gamma_tet = stats.max_delta_gamma_tet.max(r.delta_gamma);
        }
        tet.state = r.state;
        let fp = fp_from_plastic_strain(&tet.state.plastic_strain);
        let Some(fp_inv) = invert3(&fp) else {
            continue;
        };
        // Dm_inv_new = Dm_cold_inv * F_p^{-1}  ⇒  F_e = Ds Dm_inv_new = F_total F_p^{-1}
        c.dm_inv = mul3_public(&tet.cold_dm_inv, &fp_inv);
        let j_p = det3_local(&fp);
        c.rest_volume = tet.cold_volume * j_p;
        stats.sum_eq_plastic_tets += tet.state.eq_plastic_strain;
    }
    stats
}

/// Build a pinned–free plastic bar of length `l0`, area `area`, mass `mass`.
///
/// Compliance uses the elastic modulus: `α = L₀ / (E A)`.
pub fn make_plastic_bar(
    params: J2Params,
    l0: f64,
    area: f64,
    mass: f64,
    substeps: usize,
) -> PlasticSoftWorld {
    let mut w = PlasticSoftWorld::new(params);
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

/// Quasistatic stretch: pin free end at `ε_target`, step XPBD+plastic, then release.
///
/// After release + damping settle, the free-end length should approach
/// `L₀ (1 + ε_p(ε_target))`. Returns `(final_length, ε_p, analytic_ε_p)`.
pub fn quasistatic_stretch_unload(
    params: J2Params,
    l0: f64,
    area: f64,
    eps_target: f64,
    hold_steps: usize,
    settle_steps: usize,
    dt: f64,
) -> (f64, f64, f64) {
    let mut w = make_plastic_bar(params, l0, area, 1.0, 32);
    w.soft.params.damping = 2.0;
    // Hold: pin free end at target stretch.
    let x_hold = l0 * (1.0 + eps_target);
    w.soft.particles[1].x = Vec3::new(x_hold, 0.0, 0.0);
    w.soft.particles[1].v = Vec3::zero();
    w.soft.particles[1].inv_mass = 0.0; // temporarily pin
    for _ in 0..hold_steps {
        w.step(dt);
        // Keep the held position (pinned) — plastic rest updates still run.
        w.soft.particles[1].x = Vec3::new(x_hold, 0.0, 0.0);
        w.soft.particles[1].v = Vec3::zero();
    }
    let eps_p = w.bars[0].state.plastic_strain;
    let analytic = analytic_permanent_strain(&params, eps_target);
    // Release: restore mass and let XPBD settle to the plastic rest length.
    w.soft.particles[1].inv_mass = 1.0;
    w.soft.particles[1].v = Vec3::zero();
    for _ in 0..settle_steps {
        w.step(dt);
    }
    let len = (w.soft.particles[1].x - w.soft.particles[0].x).norm();
    (len, eps_p, analytic)
}

/// Axial force proxy `σ A` from the plastic bar state (after update).
#[inline]
pub fn bar_axial_force(bar: &PlasticDistance) -> f64 {
    bar.state.stress * bar.area
}

// ---------------------------------------------------------------------------
// Canonical determinism hash (M30)
// ---------------------------------------------------------------------------

/// Outer steps in the plastic-coupling hash scenario.
pub const PLASTIC_HASH_STEPS: usize = 120;
/// Outer timestep [s] for the plastic hash.
pub const PLASTIC_HASH_DT: f64 = 1.0 / 240.0;

/// Steel-like J2 params used by the locked plastic hash (not a verified alloy row).
pub fn plastic_hash_params() -> J2Params {
    J2Params {
        youngs_modulus: 2.0e11,
        poisson_ratio: 0.3,
        yield_strength: 2.5e8,
        hardening_modulus: 2.0e9,
    }
}

/// Build the canonical plastic-coupling scene (bar stretch + soft tet squash).
pub fn plastic_hash_world() -> PlasticSoftWorld {
    let params = plastic_hash_params();
    let l0 = 1.0;
    let area = 0.01;
    let mut w = make_plastic_bar(params, l0, area, 1.0, 16);
    w.soft.params.damping = 0.5;
    // Prescribe a stretch past yield and keep the free end free so XPBD + plastic
    // evolve together (no temporary pin — fully determined by initial condition).
    let eps_y = params.yield_strength / params.youngs_modulus;
    let eps_peak = eps_y + 1.5e-3;
    w.soft.particles[1].x = Vec3::new(l0 * (1.0 + eps_peak), 0.0, 0.0);
    w.soft.particles[1].v = Vec3::zero();

    // Append a single plastic tet (unit reference) undergoing compression.
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
    // Pin first three verts; move the apex to compress.
    w.soft.pin(base);
    w.soft.pin(base + 1);
    w.soft.pin(base + 2);
    let e = params.youngs_modulus;
    let nu = params.poisson_ratio;
    let mu = e / (2.0 * (1.0 + nu));
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    // Rest Dm from initial (uncompressed) positions.
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
        lame_lambda: lambda,
        lame_mu: mu,
        neo_hookean: false,
        lambda_dev: Multiplier::default(),
        lambda_vol: Multiplier::default(),
        broken: false,
    });
    // Compress apex along z before tracking so cold_dm_inv is the undeformed rest
    // already stored on the constraint — track now, then displace.
    w.track_all_tets();
    w.soft.particles[base + 3].x.z *= 0.85;

    w
}

/// FNV-1a over the canonical plastic XPBD scene.
///
/// Packed payload (little-endian IEEE bits, fixed order):
/// 1. Soft packed `(x‖v)` after [`PLASTIC_HASH_STEPS`] plastic steps.
/// 2. Bar plastic strain / eq plastic / stress / rest length.
/// 3. Tet eq plastic / stress Frobenius / rest volume.
pub fn plastic_canonical_hash() -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    let mut feed = |x: f64| {
        for b in x.to_bits().to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };

    let mut w = plastic_hash_world();
    for _ in 0..PLASTIC_HASH_STEPS {
        w.step(PLASTIC_HASH_DT);
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
        feed(tet.state.eq_plastic_strain);
        feed(tet.state.stress.frobenius());
        let vol = w.soft.constraints.tetra_fem[tet.constraint_index].rest_volume;
        feed(vol);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn soft_steel() -> J2Params {
        plastic_hash_params()
    }

    #[test]
    fn rest_update_matches_return_map_at_fixed_stretch() {
        let p = soft_steel();
        let l0 = 1.0;
        let area = 0.01;
        let mut w = make_plastic_bar(p, l0, area, 1.0, 8);
        let eps_y = p.yield_strength / p.youngs_modulus;
        let eps = eps_y + 1e-3;
        w.soft.particles[1].x = Vec3::new(l0 * (1.0 + eps), 0.0, 0.0);
        let stats = update_plastic_distances(
            &w.soft.particles,
            &mut w.soft.constraints.distances,
            &p,
            &mut w.bars,
        );
        assert!(stats.n_plastic_bars >= 1);
        let expect = analytic_permanent_strain(&p, eps);
        assert_relative_eq!(w.bars[0].state.plastic_strain, expect, max_relative = 1e-12);
        assert_relative_eq!(
            w.soft.constraints.distances[0].rest_length,
            l0 * (1.0 + expect),
            max_relative = 1e-12
        );
    }

    #[test]
    fn below_yield_no_permanent_set() {
        let p = soft_steel();
        let l0 = 1.0;
        let mut w = make_plastic_bar(p, l0, 0.01, 1.0, 8);
        let eps_y = p.yield_strength / p.youngs_modulus;
        w.soft.particles[1].x = Vec3::new(l0 * (1.0 + 0.5 * eps_y), 0.0, 0.0);
        let stats = update_plastic_distances(
            &w.soft.particles,
            &mut w.soft.constraints.distances,
            &p,
            &mut w.bars,
        );
        assert_eq!(stats.n_plastic_bars, 0);
        assert_eq!(w.bars[0].state.plastic_strain, 0.0);
        assert_relative_eq!(
            w.soft.constraints.distances[0].rest_length,
            l0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn plastic_hash_is_stable() {
        let a = plastic_canonical_hash();
        let b = plastic_canonical_hash();
        assert_eq!(a, b);
    }
}
