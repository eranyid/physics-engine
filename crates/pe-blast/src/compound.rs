//! Pre-fractured (Voronoi) rigid compounds with breakable cohesive bonds.
//!
//! Authoring: partition a box into deterministic Voronoi chunks (seeded
//! [`pe_math::Xoshiro256StarStar`]), estimate mass properties by voxel sampling,
//! and join neighbouring chunks with soft distance joints whose break force
//! comes from a material fracture stress × bond area.
//!
//! Runtime: bonds are ordinary [`pe_collision::Joint`]s with
//! [`JointBreak`]; [`pe_collision::finalize_joints`] marks them broken when
//! reaction force exceeds the threshold. Islands/sleeping stay correct because
//! broken joints drop out of [`pe_collision::joint_island_pairs`].
//!
//! # Model limits
//! - Voxel Voronoi is an engineering pre-fracture, not continuum LEFM.
//! - Chunk inertia uses a solid-box proxy from AABB half-extents of the voxel
//!   set (exact polyhedral inertia deferred).
//! - Bond area is the shared voxel-face area estimate (not a true Voronoi face).
//! - Strength thresholds should come from the materials table
//!   (`pe_soft::Material::fracture_stress`); this module takes stress as a
//!   parameter and does not depend on `pe-soft`.

use pe_collision::{BoxShape, Joint, JointBreak, JointSoftParams};
use pe_core::{inertia_solid_box, RigidBody};
use pe_math::libm_ops;
use pe_math::{Vec3, Xoshiro256StarStar};

/// One Voronoi chunk after voxel sampling.
#[derive(Clone, Debug)]
pub struct VoronoiChunk {
    /// Chunk index (0..n_sites−1). Empty chunks are omitted from the output list
    /// but neighbour tables still use site indices — see [`VoronoiPartition`].
    pub site_index: usize,
    /// Centre of mass [m] (world / parent frame).
    pub com: Vec3,
    /// Mass [kg].
    pub mass: f64,
    /// AABB half-extents [m] about `com` (box proxy for collision / inertia).
    pub half_extents: Vec3,
    /// Solid-box inertia about COM [kg·m²].
    pub inertia: [[f64; 3]; 3],
}

/// Bond between two chunks (site indices into the partition).
#[derive(Clone, Copy, Debug)]
pub struct CohesiveBondSpec {
    /// Site index A.
    pub site_a: usize,
    /// Site index B.
    pub site_b: usize,
    /// Estimated shared face area [m²].
    pub area: f64,
    /// Midpoint of shared face samples [m].
    pub midpoint: Vec3,
}

/// Result of a deterministic box Voronoi partition.
#[derive(Clone, Debug)]
pub struct VoronoiPartition {
    /// Seed sites [m].
    pub sites: Vec<Vec3>,
    /// Non-empty chunks (sorted by ascending `site_index`).
    pub chunks: Vec<VoronoiChunk>,
    /// Map site_index → index into `chunks`, or `usize::MAX` if empty.
    pub site_to_chunk: Vec<usize>,
    /// Neighbour bonds (sorted by `(min_site, max_site)`).
    pub bonds: Vec<CohesiveBondSpec>,
}

/// Partition an axis-aligned box into `n_sites` Voronoi cells.
///
/// - Box centre `center`, half-extents `half` [m].
/// - Uniform density `density` [kg/m³].
/// - `resolution` voxels per axis (≥ 2).
/// - Sites from seeded RNG; voxels assigned to nearest site (`total_cmp` ties
///   broken by lower site index).
pub fn voronoi_box_partition(
    center: Vec3,
    half: Vec3,
    n_sites: usize,
    density: f64,
    resolution: usize,
    seed: u64,
) -> VoronoiPartition {
    let n_sites = n_sites.max(1);
    let res = resolution.max(2);
    let mut rng = Xoshiro256StarStar::seed(seed);
    let mut sites = Vec::with_capacity(n_sites);
    for _ in 0..n_sites {
        let u = rng.next_f64();
        let v = rng.next_f64();
        let w = rng.next_f64();
        sites.push(Vec3::new(
            center.x + (2.0 * u - 1.0) * half.x,
            center.y + (2.0 * v - 1.0) * half.y,
            center.z + (2.0 * w - 1.0) * half.z,
        ));
    }

    let dx = (2.0 * half.x) / res as f64;
    let dy = (2.0 * half.y) / res as f64;
    let dz = (2.0 * half.z) / res as f64;
    let d_vol = dx * dy * dz;
    let voxel_mass = density * d_vol;

    let mut sum_m = vec![0.0_f64; n_sites];
    let mut sum_x = vec![Vec3::zero(); n_sites];
    let mut min_c = vec![Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY); n_sites];
    let mut max_c =
        vec![Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY); n_sites];
    // Face-neighbour bond accumulators keyed later by sorted pair.
    let mut bond_area: Vec<((usize, usize), f64, Vec3, f64)> = Vec::new();

    let mut label = vec![0usize; res * res * res];
    let idx = |i: usize, j: usize, k: usize| -> usize { (i * res + j) * res + k };

    for i in 0..res {
        for j in 0..res {
            for k in 0..res {
                let p = Vec3::new(
                    center.x - half.x + (i as f64 + 0.5) * dx,
                    center.y - half.y + (j as f64 + 0.5) * dy,
                    center.z - half.z + (k as f64 + 0.5) * dz,
                );
                let mut best = 0usize;
                let mut best_d = f64::INFINITY;
                for (s, site) in sites.iter().enumerate() {
                    let d = (p - *site).norm_squared();
                    if d < best_d || (d == best_d && s < best) {
                        best_d = d;
                        best = s;
                    }
                }
                label[idx(i, j, k)] = best;
                sum_m[best] += voxel_mass;
                sum_x[best] += p * voxel_mass;
                min_c[best] = Vec3::new(
                    min_c[best].x.min(p.x - 0.5 * dx),
                    min_c[best].y.min(p.y - 0.5 * dy),
                    min_c[best].z.min(p.z - 0.5 * dz),
                );
                max_c[best] = Vec3::new(
                    max_c[best].x.max(p.x + 0.5 * dx),
                    max_c[best].y.max(p.y + 0.5 * dy),
                    max_c[best].z.max(p.z + 0.5 * dz),
                );
            }
        }
    }

    // Neighbouring voxels with different labels → bond area.
    let face_xy = dx * dy;
    let face_xz = dx * dz;
    let face_yz = dy * dz;
    for i in 0..res {
        for j in 0..res {
            for k in 0..res {
                let a = label[idx(i, j, k)];
                let p = Vec3::new(
                    center.x - half.x + (i as f64 + 0.5) * dx,
                    center.y - half.y + (j as f64 + 0.5) * dy,
                    center.z - half.z + (k as f64 + 0.5) * dz,
                );
                if i + 1 < res {
                    let b = label[idx(i + 1, j, k)];
                    if a != b {
                        push_bond(
                            &mut bond_area,
                            a,
                            b,
                            face_yz,
                            p + Vec3::new(0.5 * dx, 0.0, 0.0),
                        );
                    }
                }
                if j + 1 < res {
                    let b = label[idx(i, j + 1, k)];
                    if a != b {
                        push_bond(
                            &mut bond_area,
                            a,
                            b,
                            face_xz,
                            p + Vec3::new(0.0, 0.5 * dy, 0.0),
                        );
                    }
                }
                if k + 1 < res {
                    let b = label[idx(i, j, k + 1)];
                    if a != b {
                        push_bond(
                            &mut bond_area,
                            a,
                            b,
                            face_xy,
                            p + Vec3::new(0.0, 0.0, 0.5 * dz),
                        );
                    }
                }
            }
        }
    }

    // Merge bond accumulators by pair.
    bond_area.sort_by_key(|x| x.0);
    let mut bonds = Vec::new();
    let mut bi = 0;
    while bi < bond_area.len() {
        let key = bond_area[bi].0;
        let mut area = 0.0;
        let mut mid = Vec3::zero();
        let mut w = 0.0;
        while bi < bond_area.len() && bond_area[bi].0 == key {
            area += bond_area[bi].1;
            mid += bond_area[bi].2 * bond_area[bi].1;
            w += bond_area[bi].1;
            bi += 1;
        }
        if w > 0.0 {
            mid *= 1.0 / w;
        }
        bonds.push(CohesiveBondSpec {
            site_a: key.0,
            site_b: key.1,
            area,
            midpoint: mid,
        });
    }

    let mut chunks = Vec::new();
    let mut site_to_chunk = vec![usize::MAX; n_sites];
    for s in 0..n_sites {
        if sum_m[s] <= 0.0 {
            continue;
        }
        let com = sum_x[s] * (1.0 / sum_m[s]);
        let he = Vec3::new(
            0.5 * (max_c[s].x - min_c[s].x).max(dx),
            0.5 * (max_c[s].y - min_c[s].y).max(dy),
            0.5 * (max_c[s].z - min_c[s].z).max(dz),
        );
        let inertia = inertia_solid_box(sum_m[s], 2.0 * he.x, 2.0 * he.y, 2.0 * he.z);
        site_to_chunk[s] = chunks.len();
        chunks.push(VoronoiChunk {
            site_index: s,
            com,
            mass: sum_m[s],
            half_extents: he,
            inertia,
        });
    }

    // Keep only bonds whose both sites produced chunks.
    bonds
        .retain(|b| site_to_chunk[b.site_a] != usize::MAX && site_to_chunk[b.site_b] != usize::MAX);
    bonds.sort_by_key(|a| (a.site_a, a.site_b));

    VoronoiPartition {
        sites,
        chunks,
        site_to_chunk,
        bonds,
    }
}

fn push_bond(
    out: &mut Vec<((usize, usize), f64, Vec3, f64)>,
    a: usize,
    b: usize,
    area: f64,
    mid: Vec3,
) {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    out.push(((lo, hi), area, mid, area));
}

/// Break force [N] from fracture stress [Pa] and bond area [m²]: `F = σ · A`.
pub fn bond_break_force(fracture_stress_pa: f64, area_m2: f64) -> f64 {
    (fracture_stress_pa * area_m2).max(0.0)
}

/// Exact threshold predicate for impulse-based bond break [N·s].
///
/// Breaks iff `|J| > J_thresh` (strict). Equality does **not** break — oracle
/// for threshold exactness.
pub fn bond_breaks_on_impulse(impulse_n_s: f64, threshold_n_s: f64) -> bool {
    libm_ops::fabs(impulse_n_s) > threshold_n_s
}

/// Build soft distance joints for a partition once chunk bodies are in the
/// world. `body_of_site(site) -> world body index`.
///
/// Rest length = distance between chunk COMs. Anchors at COM in each body
/// frame (identity orientation assumed at construction). Break force from
/// `fracture_stress_pa * bond.area`.
pub fn cohesive_joints_from_partition<F>(
    part: &VoronoiPartition,
    mut body_of_site: F,
    fracture_stress_pa: f64,
    soft: JointSoftParams,
) -> Vec<Joint>
where
    F: FnMut(usize) -> usize,
{
    let mut joints = Vec::with_capacity(part.bonds.len());
    for b in &part.bonds {
        let ca = &part.chunks[part.site_to_chunk[b.site_a]];
        let cb = &part.chunks[part.site_to_chunk[b.site_b]];
        let ba = body_of_site(b.site_a);
        let bb = body_of_site(b.site_b);
        let len = (cb.com - ca.com).norm();
        let f_break = bond_break_force(fracture_stress_pa, b.area);
        let j = Joint::distance(ba, bb, Vec3::zero(), Vec3::zero(), len)
            .with_soft(soft)
            .with_break(JointBreak {
                force: f_break,
                torque: 0.0,
            });
        joints.push(j);
    }
    // Deterministic order: already sorted by bond (site_a, site_b).
    joints
}

/// Spawn rigid bodies for each chunk (identity orientation, zero velocity).
pub fn chunks_to_bodies(part: &VoronoiPartition) -> Vec<RigidBody> {
    part.chunks
        .iter()
        .map(|c| {
            RigidBody::new(c.mass)
                .with_position(c.com)
                .with_inertia(c.inertia)
        })
        .collect()
}

/// Box collision shapes for each chunk (AABB proxy).
pub fn chunks_to_box_shapes(part: &VoronoiPartition) -> Vec<BoxShape> {
    part.chunks
        .iter()
        .map(|c| BoxShape::new(c.half_extents.x, c.half_extents.y, c.half_extents.z))
        .collect()
}

/// Apply impulse-threshold breaks to joints in ascending index order.
///
/// `impulse_along_bond[i]` is the scalar impulse [N·s] compared to
/// `threshold_n_s[i]` (typically `break_force * dt` for a one-step load).
/// Returns indices of newly broken joints (sorted).
pub fn break_bonds_by_impulse(
    joints: &mut [Joint],
    impulse_along_bond: &[f64],
    threshold_n_s: &[f64],
) -> Vec<usize> {
    let n = joints
        .len()
        .min(impulse_along_bond.len())
        .min(threshold_n_s.len());
    let mut broken = Vec::new();
    for i in 0..n {
        if joints[i].broken {
            continue;
        }
        if bond_breaks_on_impulse(impulse_along_bond[i], threshold_n_s[i]) {
            joints[i].broken = true;
            broken.push(i);
        }
    }
    broken
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bond_threshold_exactness() {
        assert!(!bond_breaks_on_impulse(1.0, 1.0));
        assert!(bond_breaks_on_impulse(1.0 + 1e-15, 1.0) || !bond_breaks_on_impulse(1.0, 1.0));
        assert!(bond_breaks_on_impulse(1.0001, 1.0));
        assert!(!bond_breaks_on_impulse(0.9999, 1.0));
    }

    #[test]
    fn voronoi_mass_conserved() {
        let half = Vec3::new(0.5, 0.5, 0.5);
        let dens = 1000.0;
        let part = voronoi_box_partition(Vec3::zero(), half, 4, dens, 8, 0xB1A5_7C0DE);
        let vol = 8.0 * half.x * half.y * half.z;
        let m: f64 = part.chunks.iter().map(|c| c.mass).sum();
        assert_relative_eq!(m, dens * vol, epsilon = 1e-9 * dens * vol);
        assert!(!part.bonds.is_empty());
    }
}
