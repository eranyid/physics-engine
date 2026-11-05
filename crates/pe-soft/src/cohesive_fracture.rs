//! Cohesive-zone fracture with bilinear traction–separation (M33).
//!
//! Completes the soft fracture track beyond max-principal snap (`fracture.rs`):
//! irreversible damage, progressive softening, and deterministic crack-path
//! records on cohesive bonds between XPBD particles.
//!
//! # Bilinear CZM (Mode I, tension only)
//!
//! Parameters: peak traction `σ_c` [Pa], critical opening `δ_c` [m], failure
//! opening `δ_f` [m] with `0 < δ_c < δ_f`.
//!
//! - Elastic: `0 ≤ δ ≤ δ_c` → `t = (σ_c/δ_c) δ`
//! - Softening: `δ_c < δ < δ_f` → `t = σ_c (δ_f − δ)/(δ_f − δ_c)`
//! - Failed: `δ ≥ δ_f` → `t = 0`, bond broken
//!
//! Fracture energy (area of the triangle): `G_c = ½ σ_c δ_f` [J/m²].
//!
//! Irreversible damage from historical max opening `δ* = max_history δ`:
//!
//! ```text
//! d = 0                                           if δ* ≤ δ_c
//! d = 1 − (δ_c/δ*) (δ_f − δ*)/(δ_f − δ_c)          if δ_c < δ* < δ_f
//! d = 1                                           if δ* ≥ δ_f
//! ```
//!
//! On unload / reload below `δ*`, traction follows the secant
//! `t = (1−d) (σ_c/δ_c) δ` (standard bilinear CZM).
//!
//! # XPBD coupling
//!
//! Each cohesive bond owns a distance constraint. After XPBD, the opening
//! `δ = max(0, L − L₀)` updates damage; compliance is rewritten so the
//! effective axial stiffness matches the secant CZM stiffness for the current
//! damage state (`k_eff = (1−d) σ_c/δ_c · A`, `α = 1/(k_eff L₀)` for the
//! engineering distance constraint). Fully damaged bonds are marked `broken`
//! (same flag as max-principal fracture) and appended to the crack path.
//!
//! # Determinism
//!
//! Fixed bond index order; portable `libm`; no `HashMap`; no `mul_add`.
//! Crack events sorted by `(bond_index)` (already ascending by construction).
//!
//! References (formulas only, clean-room): bilinear cohesive-zone /
//! traction–separation identities (Needleman; Camacho–Ortiz; continuum
//! fracture textbooks); XPBD rest/compliance update pattern (Macklin).
//! No FleX / PhysX / commercial CZM source transcribed.

use pe_math::Vec3;

use crate::constraint::{DistanceConstraint, Multiplier};
use crate::fracture::constraint_islands;
use crate::particle::Particle;
use crate::solver::{SoftParams, SoftWorld};

/// Bilinear Mode-I cohesive-zone parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BilinearCzmParams {
    /// Peak cohesive traction `σ_c` [Pa].
    pub sigma_c: f64,
    /// Opening at peak traction `δ_c` [m] (`0 < δ_c < δ_f`).
    pub delta_c: f64,
    /// Opening at complete failure `δ_f` [m].
    pub delta_f: f64,
}

impl BilinearCzmParams {
    /// Validate `σ_c > 0` and `0 < δ_c < δ_f`.
    #[inline]
    pub fn is_valid(self) -> bool {
        self.sigma_c > 0.0 && self.delta_c > 0.0 && self.delta_f > self.delta_c
    }

    /// Initial (undamaged) cohesive stiffness `k = σ_c / δ_c` [Pa/m].
    #[inline]
    pub fn initial_stiffness(self) -> f64 {
        self.sigma_c / self.delta_c
    }

    /// Fracture energy `G_c = ½ σ_c δ_f` [J/m²] (exact for bilinear triangle).
    #[inline]
    pub fn fracture_energy(self) -> f64 {
        0.5 * self.sigma_c * self.delta_f
    }
}

/// Result of evaluating the bilinear CZM at a current opening with history.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CzmEval {
    /// Traction `t` [Pa] (tension positive; compression → 0).
    pub traction: f64,
    /// Updated historical max opening `δ*` [m].
    pub delta_max: f64,
    /// Irreversible damage `d ∈ [0, 1]`.
    pub damage: f64,
    /// True when `δ* ≥ δ_f` (fully failed).
    pub failed: bool,
    /// Secant stiffness `(1−d) σ_c/δ_c` [Pa/m] (0 when failed).
    pub secant_stiffness: f64,
}

/// Analytic damage from historical max opening (textbook bilinear CZM).
#[inline]
pub fn damage_from_delta_max(params: &BilinearCzmParams, delta_max: f64) -> f64 {
    if delta_max <= params.delta_c {
        return 0.0;
    }
    if delta_max >= params.delta_f {
        return 1.0;
    }
    // d = 1 − (δ_c/δ*) (δ_f − δ*)/(δ_f − δ_c)
    let ratio = (params.delta_f - delta_max) / (params.delta_f - params.delta_c);
    1.0 - (params.delta_c / delta_max) * ratio
}

/// Evaluate bilinear CZM traction at opening `δ` given prior `δ_max`.
///
/// Compression (`δ ≤ 0`) returns zero traction and leaves history unchanged.
#[inline]
pub fn eval_bilinear_czm(params: &BilinearCzmParams, delta: f64, delta_max_prev: f64) -> CzmEval {
    let delta_pos = if delta > 0.0 { delta } else { 0.0 };
    let delta_max = if delta_pos > delta_max_prev {
        delta_pos
    } else {
        delta_max_prev
    };
    let damage = damage_from_delta_max(params, delta_max);
    let failed = damage >= 1.0 - 1e-15;
    let k0 = params.initial_stiffness();
    let secant = if failed { 0.0 } else { (1.0 - damage) * k0 };
    let traction = if delta_pos <= 0.0 || failed {
        0.0
    } else {
        // Secant response covers both first loading and unload/reload.
        secant * delta_pos
    };
    CzmEval {
        traction,
        delta_max,
        damage,
        failed,
        secant_stiffness: secant,
    }
}

/// Analytic traction on the *virgin* loading envelope (no prior damage).
///
/// Useful for peak / G_c oracles. Returns `t(δ)` for `δ ≥ 0`.
#[inline]
pub fn traction_virgin_envelope(params: &BilinearCzmParams, delta: f64) -> f64 {
    if delta <= 0.0 {
        return 0.0;
    }
    if delta <= params.delta_c {
        return params.initial_stiffness() * delta;
    }
    if delta >= params.delta_f {
        return 0.0;
    }
    params.sigma_c * (params.delta_f - delta) / (params.delta_f - params.delta_c)
}

/// Numeric integral of virgin envelope `∫_0^{δ_f} t(δ) dδ` via trapezoid rule.
///
/// Should recover `G_c = ½ σ_c δ_f` within quadrature error.
pub fn integrate_virgin_envelope(params: &BilinearCzmParams, n_segments: usize) -> f64 {
    let n = n_segments.max(2);
    let df = params.delta_f;
    let h = df / n as f64;
    let mut sum = 0.0;
    for i in 0..=n {
        let d = h * i as f64;
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        sum += w * traction_virgin_envelope(params, d);
    }
    sum * h
}

/// Work dissipated along a monotonic virgin open `0 → δ_f` at unit area [J/m²].
///
/// Exact closed form for bilinear: `G_c`.
#[inline]
pub fn analytic_gc(params: &BilinearCzmParams) -> f64 {
    params.fracture_energy()
}

/// One cohesive bond between two soft particles (Mode I).
#[derive(Clone, Debug)]
pub struct CohesiveBond {
    /// Index into `SoftWorld` distance constraints.
    pub constraint_index: usize,
    /// Cross-section area `A` [m²] converting traction ↔ force.
    pub area: f64,
    /// Historical max opening `δ*` [m].
    pub delta_max: f64,
    /// Current irreversible damage `d ∈ [0, 1]`.
    pub damage: f64,
    /// Last evaluated traction [Pa].
    pub traction: f64,
    /// Last measured opening [m].
    pub opening: f64,
    /// True after complete failure (`d = 1`).
    pub failed: bool,
    /// Cold rest length [m] (never updated — cohesive opening uses cold L₀).
    pub cold_rest: f64,
}

impl CohesiveBond {
    /// Fresh undamaged bond.
    pub fn new(constraint_index: usize, area: f64, cold_rest: f64) -> Self {
        Self {
            constraint_index,
            area,
            delta_max: 0.0,
            damage: 0.0,
            traction: 0.0,
            opening: 0.0,
            failed: false,
            cold_rest,
        }
    }
}

/// Deterministic crack-path event (one newly failed bond).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrackEvent {
    /// Bond index into [`CohesiveSoftWorld::bonds`].
    pub bond_index: usize,
    /// Distance-constraint index.
    pub constraint_index: usize,
}

/// Aggregate stats from one cohesive update pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct CohesiveUpdateStats {
    /// Bonds that newly failed this pass.
    pub n_new_failures: usize,
    /// Bonds currently failed (cumulative).
    pub n_failed: usize,
    /// Max damage among active (not-yet-failed) bonds.
    pub max_damage_active: f64,
    /// Max opening among all bonds.
    pub max_opening: f64,
}

/// Soft world with bilinear cohesive bonds on selected distance constraints.
#[derive(Clone, Debug)]
pub struct CohesiveSoftWorld {
    /// Underlying XPBD soft world.
    pub soft: SoftWorld,
    /// Cohesive-zone parameters (shared by all bonds).
    pub params: BilinearCzmParams,
    /// Tracked cohesive bonds (fixed index order).
    pub bonds: Vec<CohesiveBond>,
    /// Ordered crack path: bond indices in the order they failed.
    pub crack_path: Vec<CrackEvent>,
    /// Simulation time [s].
    pub t: f64,
}

impl CohesiveSoftWorld {
    /// Empty world with given CZM params (default soft params).
    pub fn new(czm: BilinearCzmParams) -> Self {
        Self {
            soft: SoftWorld::new(),
            params: czm,
            bonds: Vec::new(),
            crack_path: Vec::new(),
            t: 0.0,
        }
    }

    /// Simulation time [s].
    #[inline]
    pub fn time(&self) -> f64 {
        self.t
    }

    /// Track every current distance constraint as a cohesive bond of area `A`.
    pub fn track_all_distances(&mut self, area: f64) {
        self.bonds.clear();
        for (k, c) in self.soft.constraints.distances.iter().enumerate() {
            self.bonds.push(CohesiveBond::new(k, area, c.rest_length));
        }
        self.apply_compliance_from_damage();
    }

    /// Track a single distance constraint by index.
    pub fn track_distance(&mut self, constraint_index: usize, area: f64) {
        let rest = self.soft.constraints.distances[constraint_index].rest_length;
        self.bonds
            .push(CohesiveBond::new(constraint_index, area, rest));
        self.apply_compliance_from_damage();
    }

    /// Rewrite distance compliances from current damage (call after construction).
    pub fn apply_compliance_from_damage(&mut self) {
        for bond in &self.bonds {
            let c = &mut self.soft.constraints.distances[bond.constraint_index];
            if bond.failed || bond.damage >= 1.0 {
                c.broken = true;
                continue;
            }
            let k_trac = (1.0 - bond.damage) * self.params.initial_stiffness();
            // Axial spring: F = (k_trac A) ΔL ⇒ k_spring = k_trac A, α = 1/k_spring.
            // Distance XPBD uses α with C = L − L₀, so α = 1/(k_trac A).
            let k_spring = k_trac * bond.area;
            if k_spring > 0.0 {
                c.compliance = 1.0 / k_spring;
            } else {
                c.broken = true;
            }
        }
    }

    /// One XPBD step then cohesive damage / compliance / crack update.
    pub fn step(&mut self, dt: f64) {
        self.soft.step(dt);
        let _ = update_cohesive_bonds(
            &self.soft.particles,
            &mut self.soft.constraints.distances,
            &mut self.bonds,
            &mut self.crack_path,
            &self.params,
        );
        self.t += dt;
    }

    /// Island labels over remaining active (non-broken) distance + tet edges.
    pub fn islands(&self) -> (Vec<usize>, usize) {
        constraint_islands(self.soft.particles.len(), &self.soft.constraints)
    }
}

/// Measure openings, update damage / compliance, append new crack events.
///
/// Processes bonds in ascending index order (deterministic).
pub fn update_cohesive_bonds(
    particles: &[Particle],
    distances: &mut [DistanceConstraint],
    bonds: &mut [CohesiveBond],
    crack_path: &mut Vec<CrackEvent>,
    params: &BilinearCzmParams,
) -> CohesiveUpdateStats {
    let mut stats = CohesiveUpdateStats::default();
    for (bi, bond) in bonds.iter_mut().enumerate() {
        if bond.failed {
            stats.n_failed += 1;
            continue;
        }
        let c = &distances[bond.constraint_index];
        if c.broken {
            // Externally broken — treat as failed.
            bond.failed = true;
            bond.damage = 1.0;
            bond.traction = 0.0;
            stats.n_failed += 1;
            stats.n_new_failures += 1;
            crack_path.push(CrackEvent {
                bond_index: bi,
                constraint_index: bond.constraint_index,
            });
            continue;
        }
        let i = c.i;
        let j = c.j;
        let len = (particles[i].x - particles[j].x).norm();
        let opening = if len > bond.cold_rest {
            len - bond.cold_rest
        } else {
            0.0
        };
        bond.opening = opening;
        if opening > stats.max_opening {
            stats.max_opening = opening;
        }
        let ev = eval_bilinear_czm(params, opening, bond.delta_max);
        bond.delta_max = ev.delta_max;
        bond.damage = ev.damage;
        bond.traction = ev.traction;
        if ev.failed {
            bond.failed = true;
            distances[bond.constraint_index].broken = true;
            stats.n_new_failures += 1;
            stats.n_failed += 1;
            crack_path.push(CrackEvent {
                bond_index: bi,
                constraint_index: bond.constraint_index,
            });
        } else {
            if bond.damage > stats.max_damage_active {
                stats.max_damage_active = bond.damage;
            }
            let k_spring = ev.secant_stiffness * bond.area;
            if k_spring > 0.0 {
                distances[bond.constraint_index].compliance = 1.0 / k_spring;
            } else {
                distances[bond.constraint_index].broken = true;
                bond.failed = true;
                stats.n_new_failures += 1;
                stats.n_failed += 1;
                crack_path.push(CrackEvent {
                    bond_index: bi,
                    constraint_index: bond.constraint_index,
                });
            }
        }
    }
    stats
}

/// Build a pinned–free single cohesive bar (two particles, one bond).
///
/// Initial compliance matches undamaged CZM: `α = 1/( (σ_c/δ_c) A )`.
pub fn make_cohesive_bar(
    czm: BilinearCzmParams,
    l0: f64,
    area: f64,
    mass: f64,
    substeps: usize,
) -> CohesiveSoftWorld {
    let mut w = CohesiveSoftWorld::new(czm);
    w.soft.params = SoftParams {
        substeps,
        iterations: 2,
        gravity: Vec3::zero(),
        damping: 0.0,
    };
    w.soft.add_particle(Particle::pinned(Vec3::zero()));
    w.soft
        .add_particle(Particle::dynamic(Vec3::new(l0, 0.0, 0.0), mass));
    let k0 = czm.initial_stiffness() * area;
    let compliance = 1.0 / k0;
    w.soft.constraints.distances.push(DistanceConstraint {
        i: 0,
        j: 1,
        rest_length: l0,
        compliance,
        lambda: Multiplier::default(),
        broken: false,
    });
    w.track_all_distances(area);
    w
}

/// Build a multi-segment cohesive chain (n_seg bonds, n_seg+1 particles).
///
/// Particle 0 pinned; free end is the last particle. Used for crack-path
/// propagation oracles.
pub fn make_cohesive_chain(
    czm: BilinearCzmParams,
    n_seg: usize,
    l0_total: f64,
    area: f64,
    mass_per: f64,
    substeps: usize,
) -> CohesiveSoftWorld {
    let n_seg = n_seg.max(1);
    let mut w = CohesiveSoftWorld::new(czm);
    w.soft.params = SoftParams {
        substeps,
        iterations: 2,
        gravity: Vec3::zero(),
        damping: 0.0,
    };
    let seg = l0_total / n_seg as f64;
    let k0 = czm.initial_stiffness() * area;
    let compliance = 1.0 / k0;
    for i in 0..=n_seg {
        let x = Vec3::new(seg * i as f64, 0.0, 0.0);
        if i == 0 {
            w.soft.add_particle(Particle::pinned(x));
        } else {
            w.soft.add_particle(Particle::dynamic(x, mass_per));
        }
    }
    for i in 0..n_seg {
        w.soft.constraints.distances.push(DistanceConstraint {
            i,
            j: i + 1,
            rest_length: seg,
            compliance,
            lambda: Multiplier::default(),
            broken: false,
        });
    }
    w.track_all_distances(area);
    w
}

/// Quasistatic displace free end, step XPBD+CZM, return final opening / damage / failed.
pub fn quasistatic_cohesive_open(
    czm: BilinearCzmParams,
    l0: f64,
    area: f64,
    target_opening: f64,
    settle_steps: usize,
    dt: f64,
) -> (f64, f64, f64, bool) {
    let mut w = make_cohesive_bar(czm, l0, area, 1.0, 32);
    w.soft.params.damping = 2.0;
    w.soft.particles[1].x = Vec3::new(l0 + target_opening, 0.0, 0.0);
    w.soft.particles[1].v = Vec3::zero();
    w.soft.particles[1].inv_mass = 0.0; // hold
    for _ in 0..settle_steps {
        w.step(dt);
    }
    let bond = &w.bonds[0];
    (bond.opening, bond.damage, bond.traction, bond.failed)
}

/// Dissipated work per unit area for a virgin monotonic open via discrete sum.
///
/// Uses the analytic envelope at sample openings (not the XPBD path) so the
/// oracle is exact up to quadrature — compares to `G_c`.
pub fn discrete_work_to_failure(params: &BilinearCzmParams, n: usize) -> f64 {
    integrate_virgin_envelope(params, n)
}

/// Outer steps in the cohesive hash scenario.
pub const COHESIVE_HASH_STEPS: usize = 160;
/// Outer timestep [s] for the cohesive hash.
pub const COHESIVE_HASH_DT: f64 = 1.0 / 240.0;

/// Locked-hash CZM parameters (soft demo scale).
pub fn cohesive_hash_params() -> BilinearCzmParams {
    BilinearCzmParams {
        sigma_c: 1.0e5,
        delta_c: 2.0e-3,
        delta_f: 1.0e-2,
    }
}

/// Canonical cohesive scene: single bar held past peak + 4-seg chain mid-open.
pub fn cohesive_hash_world() -> CohesiveSoftWorld {
    let czm = cohesive_hash_params();
    let l0 = 1.0;
    let area = 0.01;
    let mut w = make_cohesive_bar(czm, l0, area, 1.0, 16);
    w.soft.params.damping = 0.5;
    // Hold into softening: δ = 0.6 δ_f
    let open = 0.6 * czm.delta_f;
    w.soft.particles[1].x = Vec3::new(l0 + open, 0.0, 0.0);
    w.soft.particles[1].v = Vec3::zero();

    // Append a short cohesive chain (separate particles) partially opened.
    let base = w.soft.particles.len();
    let n_seg = 4;
    let seg = 0.25;
    let mass = 0.25;
    let k0 = czm.initial_stiffness() * area;
    let compliance = 1.0 / k0;
    for i in 0..=n_seg {
        let x = Vec3::new(3.0 + seg * i as f64, 0.0, 0.0);
        if i == 0 {
            w.soft.add_particle(Particle::pinned(x));
        } else {
            w.soft.add_particle(Particle::dynamic(x, mass));
        }
    }
    for i in 0..n_seg {
        w.soft.constraints.distances.push(DistanceConstraint {
            i: base + i,
            j: base + i + 1,
            rest_length: seg,
            compliance,
            lambda: Multiplier::default(),
            broken: false,
        });
        let ci = w.soft.constraints.distances.len() - 1;
        w.bonds.push(CohesiveBond::new(ci, area, seg));
    }
    // Stretch free end of chain into softening.
    let last = base + n_seg;
    w.soft.particles[last].x.x += 0.5 * czm.delta_f;
    w.apply_compliance_from_damage();
    w
}

/// FNV-1a over the canonical cohesive fracture scene.
///
/// Packed payload (little-endian IEEE bits, fixed order):
/// 1. Soft packed `(x‖v)` after [`COHESIVE_HASH_STEPS`] steps.
/// 2. Per bond: `(opening, delta_max, damage, traction, failed_as_f64)`.
/// 3. Crack path length as f64, then each `(bond_index, constraint_index)` as f64.
pub fn cohesive_canonical_hash() -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    let mut feed = |x: f64| {
        for b in x.to_bits().to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };

    let mut w = cohesive_hash_world();
    for _ in 0..COHESIVE_HASH_STEPS {
        w.step(COHESIVE_HASH_DT);
    }
    for x in w.soft.pack() {
        feed(x);
    }
    for bond in &w.bonds {
        feed(bond.opening);
        feed(bond.delta_max);
        feed(bond.damage);
        feed(bond.traction);
        feed(if bond.failed { 1.0 } else { 0.0 });
    }
    feed(w.crack_path.len() as f64);
    for ev in &w.crack_path {
        feed(ev.bond_index as f64);
        feed(ev.constraint_index as f64);
    }
    hash
}

/// Work along a prescribed opening history using secant CZM (unit area).
///
/// Trapezoidal ∫ t(δ) dδ with irreversible damage — for unload/reload oracles.
pub fn path_work_unit_area(params: &BilinearCzmParams, openings: &[f64]) -> (f64, f64, f64, bool) {
    let mut delta_max = 0.0;
    let mut work = 0.0;
    let mut prev_d = 0.0;
    let mut prev_t = 0.0;
    let mut damage = 0.0;
    let mut failed = false;
    for (k, &d) in openings.iter().enumerate() {
        let ev = eval_bilinear_czm(params, d, delta_max);
        delta_max = ev.delta_max;
        damage = ev.damage;
        failed = ev.failed;
        if k > 0 {
            work += 0.5 * (prev_t + ev.traction) * (d - prev_d);
        }
        prev_d = d;
        prev_t = ev.traction;
    }
    (work, damage, delta_max, failed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn demo_czm() -> BilinearCzmParams {
        BilinearCzmParams {
            sigma_c: 1.0e6,
            delta_c: 1.0e-3,
            delta_f: 5.0e-3,
        }
    }

    #[test]
    fn gc_identity_exact() {
        let p = demo_czm();
        assert_relative_eq!(
            p.fracture_energy(),
            0.5 * p.sigma_c * p.delta_f,
            epsilon = 0.0
        );
    }

    #[test]
    fn virgin_peak_at_delta_c() {
        let p = demo_czm();
        let t = traction_virgin_envelope(&p, p.delta_c);
        assert_relative_eq!(t, p.sigma_c, max_relative = 1e-15);
    }

    #[test]
    fn damage_zero_below_delta_c() {
        let p = demo_czm();
        assert_eq!(damage_from_delta_max(&p, 0.5 * p.delta_c), 0.0);
    }

    #[test]
    fn damage_one_at_delta_f() {
        let p = demo_czm();
        assert_relative_eq!(damage_from_delta_max(&p, p.delta_f), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn unload_keeps_damage() {
        let p = demo_czm();
        let d_star = 0.5 * (p.delta_c + p.delta_f);
        let ev1 = eval_bilinear_czm(&p, d_star, 0.0);
        assert!(ev1.damage > 0.0 && !ev1.failed);
        let ev2 = eval_bilinear_czm(&p, 0.1 * p.delta_c, ev1.delta_max);
        assert_relative_eq!(ev2.damage, ev1.damage, epsilon = 0.0);
        // Secant: t = (1-d) k0 δ
        let expect = (1.0 - ev1.damage) * p.initial_stiffness() * (0.1 * p.delta_c);
        assert_relative_eq!(ev2.traction, expect, max_relative = 1e-12);
    }

    #[test]
    fn cohesive_hash_stable() {
        let a = cohesive_canonical_hash();
        let b = cohesive_canonical_hash();
        assert_eq!(a, b);
        assert_ne!(a, 0);
    }

    #[test]
    fn integrate_recovers_gc() {
        let p = demo_czm();
        let w = integrate_virgin_envelope(&p, 4096);
        let gc = analytic_gc(&p);
        let err = (w - gc).abs() / gc;
        assert!(err < 1e-6, "quad err {err}");
    }

    #[test]
    fn cohesive_chain_builds() {
        let p = cohesive_hash_params();
        let w = make_cohesive_chain(p, 3, 0.75, 0.01, 0.2, 8);
        assert_eq!(w.bonds.len(), 3);
        assert_eq!(w.soft.particles.len(), 4);
    }
}
