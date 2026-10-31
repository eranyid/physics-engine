//! Constraint-breaking fracture by max principal stress / strain threshold.
//!
//! Deterministic ordering: candidates are collected, sorted by
//! `(kind_rank, constraint_index)`, then broken in that order. No `HashMap`
//! iteration. Island labels are connected components over remaining active
//! constraints (union-find on particle indices, processed in ascending edge
//! order).

use pe_math::libm_ops;
use pe_math::Vec3;

use crate::constraint::{
    deformation_gradient_public, green_strain_public, ConstraintSet, DistanceConstraint,
    TetraFemConstraint,
};
use crate::particle::Particle;

/// Fracture criterion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FractureCriterion {
    /// Break when max principal Cauchy/PK2 proxy stress ≥ threshold [Pa].
    MaxPrincipalStress(f64),
    /// Break when max principal Green strain ≥ threshold [dimensionless].
    MaxPrincipalStrain(f64),
}

/// Which constraint family was examined / broken.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum FractureKind {
    /// Distance / spring / rope segment.
    Distance = 0,
    /// Tetrahedral FEM element.
    TetraFem = 1,
}

/// One break event (deterministic record).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FractureEvent {
    /// Constraint family.
    pub kind: FractureKind,
    /// Index into the corresponding `ConstraintSet` vector.
    pub index: usize,
    /// Measured scalar that crossed the threshold (stress [Pa] or strain [-]).
    pub value: f64,
    /// Threshold used.
    pub threshold: f64,
}

/// Axial engineering stress in a distance constraint: `σ = E (L − L₀) / L₀`.
///
/// For a bar segment this is the uniaxial stress proxy used by the fracture oracle.
#[inline]
pub fn distance_axial_stress(
    particles: &[Particle],
    c: &DistanceConstraint,
    youngs_modulus: f64,
) -> f64 {
    let d = particles[c.i].x - particles[c.j].x;
    let len = d.norm();
    if c.rest_length <= 1e-30 {
        return 0.0;
    }
    youngs_modulus * (len - c.rest_length) / c.rest_length
}

/// Engineering axial strain `(L − L₀)/L₀`.
#[inline]
pub fn distance_axial_strain(particles: &[Particle], c: &DistanceConstraint) -> f64 {
    let d = particles[c.i].x - particles[c.j].x;
    let len = d.norm();
    if c.rest_length <= 1e-30 {
        return 0.0;
    }
    (len - c.rest_length) / c.rest_length
}

/// Max principal value of a symmetric 3×3 matrix (analytic cubic / Jacobi-free).
///
/// Uses the closed-form eigenvalue formula for symmetric 3×3.
pub fn max_principal_sym(m: &[[f64; 3]; 3]) -> f64 {
    // Ensure symmetry by averaging off-diagonals.
    let a = m[0][0];
    let b = 0.5 * (m[0][1] + m[1][0]);
    let c = 0.5 * (m[0][2] + m[2][0]);
    let d = m[1][1];
    let e = 0.5 * (m[1][2] + m[2][1]);
    let f = m[2][2];
    // Matrix [[a,b,c],[b,d,e],[c,e,f]]
    let p1 = b * b + c * c + e * e;
    if p1 < 1e-30 {
        return a.max(d).max(f);
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
    let r = det_b * 0.5;
    let r_clamped = r.clamp(-1.0, 1.0);
    let phi = libm_ops::acos(r_clamped) / 3.0;
    // Eigenvalues: q + 2 p cos(φ), q + 2 p cos(φ + 2π/3), q + 2 p cos(φ + 4π/3)
    let eig0 = q + 2.0 * p * libm_ops::cos(phi);
    let eig1 = q + 2.0 * p * libm_ops::cos(phi + 2.0 * core::f64::consts::PI / 3.0);
    let eig2 = q + 2.0 * p * libm_ops::cos(phi + 4.0 * core::f64::consts::PI / 3.0);
    eig0.max(eig1).max(eig2)
}

/// PK2 stress for StVK: `S = 2μ E + λ tr(E) I` from Green strain E.
pub fn stvk_pk2(e_green: &[[f64; 3]; 3], lame_lambda: f64, lame_mu: f64) -> [[f64; 3]; 3] {
    let tr = e_green[0][0] + e_green[1][1] + e_green[2][2];
    [
        [
            2.0 * lame_mu * e_green[0][0] + lame_lambda * tr,
            2.0 * lame_mu * e_green[0][1],
            2.0 * lame_mu * e_green[0][2],
        ],
        [
            2.0 * lame_mu * e_green[1][0],
            2.0 * lame_mu * e_green[1][1] + lame_lambda * tr,
            2.0 * lame_mu * e_green[1][2],
        ],
        [
            2.0 * lame_mu * e_green[2][0],
            2.0 * lame_mu * e_green[2][1],
            2.0 * lame_mu * e_green[2][2] + lame_lambda * tr,
        ],
    ]
}

/// Max principal PK2 stress on a tet FEM element (StVK proxy; works for SNH too
/// as an ordering signal — absolute threshold calibrated for StVK/bar tests).
pub fn tetra_max_principal_stress(particles: &[Particle], c: &TetraFemConstraint) -> f64 {
    let f = deformation_gradient_public(particles, c.indices, &c.dm_inv);
    let e = green_strain_public(&f);
    let s = stvk_pk2(&e, c.lame_lambda, c.lame_mu);
    max_principal_sym(&s)
}

/// Max principal Green strain on a tet.
pub fn tetra_max_principal_strain(particles: &[Particle], c: &TetraFemConstraint) -> f64 {
    let f = deformation_gradient_public(particles, c.indices, &c.dm_inv);
    let e = green_strain_public(&f);
    max_principal_sym(&e)
}

/// Scan active constraints and break those exceeding the criterion.
///
/// Returns events in **deterministic sorted order**. Already-broken constraints
/// are skipped. Energy of broken distance springs is reported via
/// [`fracture_released_spring_energy`].
pub fn apply_fracture(
    particles: &[Particle],
    constraints: &mut ConstraintSet,
    criterion: FractureCriterion,
    youngs_for_distance: f64,
) -> Vec<FractureEvent> {
    let mut candidates: Vec<FractureEvent> = Vec::new();

    let (thresh, is_stress) = match criterion {
        FractureCriterion::MaxPrincipalStress(t) => (t, true),
        FractureCriterion::MaxPrincipalStrain(t) => (t, false),
    };

    for (index, c) in constraints.distances.iter().enumerate() {
        if c.broken {
            continue;
        }
        let value = if is_stress {
            distance_axial_stress(particles, c, youngs_for_distance)
        } else {
            distance_axial_strain(particles, c)
        };
        if value >= thresh {
            candidates.push(FractureEvent {
                kind: FractureKind::Distance,
                index,
                value,
                threshold: thresh,
            });
        }
    }
    for (index, c) in constraints.tetra_fem.iter().enumerate() {
        if c.broken {
            continue;
        }
        let value = if is_stress {
            tetra_max_principal_stress(particles, c)
        } else {
            tetra_max_principal_strain(particles, c)
        };
        if value >= thresh {
            candidates.push(FractureEvent {
                kind: FractureKind::TetraFem,
                index,
                value,
                threshold: thresh,
            });
        }
    }

    // Deterministic order: kind then index (stable for equal keys).
    candidates.sort_by_key(|a| (a.kind, a.index));

    for ev in &candidates {
        match ev.kind {
            FractureKind::Distance => {
                constraints.distances[ev.index].broken = true;
            }
            FractureKind::TetraFem => {
                constraints.tetra_fem[ev.index].broken = true;
            }
        }
    }
    candidates
}

/// Elastic energy stored in a distance spring `½ k (ΔL)²` with `k = 1/α`
/// (infinite k ⇒ 0 contribution when α=0 and ΔL≈0; otherwise use α).
pub fn distance_spring_energy(particles: &[Particle], c: &DistanceConstraint) -> f64 {
    let d = particles[c.i].x - particles[c.j].x;
    let len = d.norm();
    let dl = len - c.rest_length;
    if c.compliance <= 0.0 {
        // Rigid: treat as zero stored energy when satisfied; if violated, report
        // a large but finite sentinel based on a stiff surrogate.
        return 0.5 * 1.0e12 * dl * dl;
    }
    let k = 1.0 / c.compliance;
    0.5 * k * dl * dl
}

/// Sum of spring energy that was held by constraints marked `broken`
/// (call **before** positions relax, or pass a snapshot). Used for energy
/// accounting: broken energy is considered released (dissipated / converted).
pub fn fracture_released_spring_energy(particles: &[Particle], constraints: &ConstraintSet) -> f64 {
    let mut e = 0.0;
    for c in &constraints.distances {
        if c.broken {
            e += distance_spring_energy(particles, c);
        }
    }
    e
}

/// Count remaining active (not broken) distance + tet constraints.
pub fn active_constraint_count(constraints: &ConstraintSet) -> usize {
    let d = constraints.distances.iter().filter(|c| !c.broken).count();
    let t = constraints.tetra_fem.iter().filter(|c| !c.broken).count();
    d + t
}

/// Connected-component island ids for particles linked by **active** distance
/// and tet constraints. Isolated particles get their own island.
///
/// Returns `(island_of_particle, island_count)`. Island ids are dense in
/// `0..island_count`, assigned by first-seen root order (deterministic).
pub fn constraint_islands(n_particles: usize, constraints: &ConstraintSet) -> (Vec<usize>, usize) {
    let mut parent: Vec<usize> = (0..n_particles).collect();
    let mut rank: Vec<u8> = vec![0; n_particles];

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    fn union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
        let mut ra = find(parent, a);
        let mut rb = find(parent, b);
        if ra == rb {
            return;
        }
        // Deterministic: attach higher index under lower index on equal rank.
        if rank[ra] < rank[rb] {
            core::mem::swap(&mut ra, &mut rb);
        }
        parent[rb] = ra;
        if rank[ra] == rank[rb] {
            rank[ra] = rank[ra].saturating_add(1);
        }
    }

    for c in &constraints.distances {
        if c.broken {
            continue;
        }
        union(&mut parent, &mut rank, c.i, c.j);
    }
    for c in &constraints.tetra_fem {
        if c.broken {
            continue;
        }
        let i0 = c.indices[0];
        for k in 1..4 {
            union(&mut parent, &mut rank, i0, c.indices[k]);
        }
    }

    // Assign dense ids in ascending particle order of first appearance of a root.
    let mut root_to_island = vec![usize::MAX; n_particles];
    let mut islands = Vec::with_capacity(n_particles);
    let mut next = 0usize;
    for i in 0..n_particles {
        let r = find(&mut parent, i);
        if root_to_island[r] == usize::MAX {
            root_to_island[r] = next;
            next += 1;
        }
        islands.push(root_to_island[r]);
    }
    (islands, next)
}

/// True if any particle state or constraint metric is non-finite.
pub fn soft_state_has_nan(particles: &[Particle], constraints: &ConstraintSet) -> bool {
    for p in particles {
        let a = p.x.to_array();
        let b = p.v.to_array();
        if a.iter().any(|x| !x.is_finite()) || b.iter().any(|x| !x.is_finite()) {
            return true;
        }
    }
    for c in &constraints.distances {
        if !c.rest_length.is_finite() || !c.compliance.is_finite() {
            return true;
        }
    }
    false
}

/// Helper: stretch a single distance bar quasistatically and return the axial
/// stress at which the first break occurs (or `None` if never).
pub fn find_break_stress(
    particles: &mut [Particle],
    constraints: &mut ConstraintSet,
    youngs: f64,
    threshold: f64,
    free_end: usize,
    axis: Vec3,
    step: f64,
    max_steps: usize,
) -> Option<(f64, FractureEvent)> {
    let dir = axis.normalize();
    for _ in 0..max_steps {
        particles[free_end].x += dir * step;
        // Project once with high stiffness interpretation (caller sets compliance).
        for c in &mut constraints.distances {
            if !c.broken {
                crate::constraint::project_distance(particles, c, 1.0 / 60.0);
            }
        }
        let events = apply_fracture(
            particles,
            constraints,
            FractureCriterion::MaxPrincipalStress(threshold),
            youngs,
        );
        if let Some(&ev) = events.first() {
            return Some((ev.value, ev));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::{DistanceConstraint, Multiplier};
    use crate::particle::Particle;
    use approx::assert_relative_eq;

    #[test]
    fn max_principal_of_diag() {
        let m = [[3.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 2.0]];
        assert_relative_eq!(max_principal_sym(&m), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn islands_split_after_middle_break() {
        let n = 4;
        let mut cs = ConstraintSet::default();
        for i in 0..3 {
            cs.distances.push(DistanceConstraint {
                i,
                j: i + 1,
                rest_length: 1.0,
                compliance: 1e-6,
                lambda: Multiplier::default(),
                broken: false,
            });
        }
        let (_, n0) = constraint_islands(n, &cs);
        assert_eq!(n0, 1);
        cs.distances[1].broken = true;
        let (ids, n1) = constraint_islands(n, &cs);
        assert_eq!(n1, 2);
        assert_eq!(ids[0], ids[1]);
        assert_eq!(ids[2], ids[3]);
        assert_ne!(ids[0], ids[2]);
    }

    #[test]
    fn fracture_order_is_deterministic() {
        let mut particles = vec![
            Particle::pinned(Vec3::zero()),
            Particle::dynamic(Vec3::new(1.1, 0.0, 0.0), 1.0),
            Particle::dynamic(Vec3::new(2.2, 0.0, 0.0), 1.0),
        ];
        let mut cs = ConstraintSet::default();
        for i in 0..2 {
            cs.distances.push(DistanceConstraint {
                i,
                j: i + 1,
                rest_length: 1.0,
                compliance: 1e-8,
                lambda: Multiplier::default(),
                broken: false,
            });
        }
        // Stretch both past threshold.
        particles[1].x.x = 1.5;
        particles[2].x.x = 3.0;
        let e = 1.0e6;
        let events = apply_fracture(
            &particles,
            &mut cs,
            FractureCriterion::MaxPrincipalStress(1.0e5),
            e,
        );
        assert!(events.len() >= 2);
        assert!(events[0].index <= events[1].index);
        assert!(cs.distances.iter().all(|c| c.broken));
    }
}
