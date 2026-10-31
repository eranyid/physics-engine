//! Soft-body collisions against static planes/spheres/boxes and self-collision.
//!
//! Spatial hash uses a dense grid of `Vec` buckets (never `HashMap` iteration).
//! Contact pairs are sorted by `(min_id, max_id)` before resolution for determinism.

use pe_math::libm_ops;
use pe_math::Vec3;

use crate::particle::Particle;

/// Infinite static plane: `n · x = d` with unit normal `n` (half-space `n·x ≥ d` is free).
#[derive(Clone, Copy, Debug)]
pub struct StaticPlane {
    /// Unit normal.
    pub normal: Vec3,
    /// Plane offset such that `n · x = offset` on the plane.
    pub offset: f64,
    /// Coulomb friction μ.
    pub friction: f64,
    /// Restitution e (applied in velocity update post-contact).
    pub restitution: f64,
}

impl StaticPlane {
    /// Construct and normalize the normal.
    pub fn new(normal: Vec3, offset: f64, friction: f64, restitution: f64) -> Self {
        let n = if normal.norm_squared() > 0.0 {
            normal.normalize()
        } else {
            Vec3::unit_z()
        };
        Self {
            normal: n,
            offset,
            friction,
            restitution,
        }
    }
}

/// Static sphere obstacle (solid exterior free).
#[derive(Clone, Copy, Debug)]
pub struct StaticSphere {
    /// Centre [m].
    pub center: Vec3,
    /// Radius [m].
    pub radius: f64,
    /// Friction μ.
    pub friction: f64,
    /// Restitution e.
    pub restitution: f64,
}

/// Axis-aligned static box (solid interior; particles pushed outside).
#[derive(Clone, Copy, Debug)]
pub struct StaticBox {
    /// Centre [m].
    pub center: Vec3,
    /// Half-extents [m].
    pub half_extents: Vec3,
    /// Friction μ.
    pub friction: f64,
    /// Restitution e.
    pub restitution: f64,
}

/// Collection of static colliders (fixed `Vec` order).
#[derive(Clone, Debug, Default)]
pub struct StaticColliders {
    /// Planes.
    pub planes: Vec<StaticPlane>,
    /// Spheres.
    pub spheres: Vec<StaticSphere>,
    /// Boxes.
    pub boxes: Vec<StaticBox>,
}

/// Self-collision / particle radius settings.
#[derive(Clone, Copy, Debug)]
pub struct SelfCollisionParams {
    /// Particle collision radius [m].
    pub radius: f64,
    /// Enable self-collision.
    pub enabled: bool,
    /// Friction between particles.
    pub friction: f64,
}

impl Default for SelfCollisionParams {
    fn default() -> Self {
        Self {
            radius: 0.02,
            enabled: false,
            friction: 0.1,
        }
    }
}

/// Push particles out of static colliders (positional XPBD-style unilateral).
pub fn collide_statics(particles: &mut [Particle], colliders: &StaticColliders) {
    for p in particles.iter_mut() {
        if p.inv_mass == 0.0 {
            continue;
        }
        for plane in &colliders.planes {
            let pen = plane.offset - p.x.dot(plane.normal);
            if pen > 0.0 {
                p.x += plane.normal * pen;
                // Kill normal velocity component into the plane (inelastic resting).
                let vn = p.v.dot(plane.normal);
                if vn < 0.0 {
                    let v_n = plane.normal * vn;
                    let v_t = p.v - v_n;
                    p.v = v_t * (1.0 - plane.friction).max(0.0) - v_n * plane.restitution;
                }
            }
        }
        for sph in &colliders.spheres {
            let d = p.x - sph.center;
            let dist = d.norm();
            if dist < sph.radius && dist > 1e-15 {
                let n = d * (1.0 / dist);
                p.x = sph.center + n * sph.radius;
                let vn = p.v.dot(n);
                if vn < 0.0 {
                    let v_n = n * vn;
                    let v_t = p.v - v_n;
                    p.v = v_t * (1.0 - sph.friction).max(0.0) - v_n * sph.restitution;
                }
            }
        }
        for b in &colliders.boxes {
            let local = p.x - b.center;
            let hx = b.half_extents.x;
            let hy = b.half_extents.y;
            let hz = b.half_extents.z;
            // Inside test
            if local.x.abs() <= hx && local.y.abs() <= hy && local.z.abs() <= hz {
                // Push out along axis of least penetration
                let px = hx - local.x.abs();
                let py = hy - local.y.abs();
                let pz = hz - local.z.abs();
                let (n, pen) = if px <= py && px <= pz {
                    (
                        Vec3::new(if local.x >= 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0),
                        px,
                    )
                } else if py <= pz {
                    (
                        Vec3::new(0.0, if local.y >= 0.0 { 1.0 } else { -1.0 }, 0.0),
                        py,
                    )
                } else {
                    (
                        Vec3::new(0.0, 0.0, if local.z >= 0.0 { 1.0 } else { -1.0 }),
                        pz,
                    )
                };
                p.x += n * pen;
                let vn = p.v.dot(n);
                if vn < 0.0 {
                    let v_n = n * vn;
                    let v_t = p.v - v_n;
                    p.v = v_t * (1.0 - b.friction).max(0.0) - v_n * b.restitution;
                }
            }
        }
    }
}

#[inline]
fn cell_hash(ix: i32, iy: i32, iz: i32, n_buckets: usize) -> usize {
    // Deterministic integer hash (not HashMap).
    let mut h = (ix as i64).wrapping_mul(73856093)
        ^ (iy as i64).wrapping_mul(19349663)
        ^ (iz as i64).wrapping_mul(83492791);
    if h < 0 {
        h = -h;
    }
    (h as usize) % n_buckets
}

/// Self-collision via spatial hash; pairs sorted by `(min,max)` before resolve.
pub fn collide_self(particles: &mut [Particle], params: &SelfCollisionParams) {
    if !params.enabled || particles.len() < 2 {
        return;
    }
    let r = params.radius;
    let diam = 2.0 * r;
    if diam <= 0.0 {
        return;
    }
    let inv_cell = 1.0 / diam;
    let n = particles.len();
    let n_buckets = (n * 2).next_power_of_two().max(16);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_buckets];

    for i in 0..n {
        let p = particles[i].x;
        let ix = libm_ops::floor(p.x * inv_cell) as i32;
        let iy = libm_ops::floor(p.y * inv_cell) as i32;
        let iz = libm_ops::floor(p.z * inv_cell) as i32;
        let h = cell_hash(ix, iy, iz, n_buckets);
        buckets[h].push(i);
    }

    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        if particles[i].inv_mass == 0.0 {
            // Still collide against dynamic neighbors.
        }
        let p = particles[i].x;
        let ix = libm_ops::floor(p.x * inv_cell) as i32;
        let iy = libm_ops::floor(p.y * inv_cell) as i32;
        let iz = libm_ops::floor(p.z * inv_cell) as i32;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let h = cell_hash(ix + dx, iy + dy, iz + dz, n_buckets);
                    for &j in &buckets[h] {
                        if j <= i {
                            continue;
                        }
                        pairs.push((i, j));
                    }
                }
            }
        }
    }
    // Deterministic order
    pairs.sort_unstable();
    pairs.dedup();

    let min_dist = diam;
    for (i, j) in pairs {
        let wi = particles[i].inv_mass;
        let wj = particles[j].inv_mass;
        if wi == 0.0 && wj == 0.0 {
            continue;
        }
        let d = particles[i].x - particles[j].x;
        let dist = d.norm();
        if dist >= min_dist || dist < 1e-15 {
            continue;
        }
        let nrm = d * (1.0 / dist);
        let pen = min_dist - dist;
        let w_sum = wi + wj;
        if w_sum < 1e-30 {
            continue;
        }
        let corr_i = nrm * (pen * wi / w_sum);
        let corr_j = nrm * (-pen * wj / w_sum);
        particles[i].x += corr_i;
        particles[j].x += corr_j;
        // Tangential friction damp
        let v_rel = particles[i].v - particles[j].v;
        let vn = v_rel.dot(nrm);
        let v_t = v_rel - nrm * vn;
        let friction = params.friction;
        if wi != 0.0 {
            particles[i].v -= v_t * (friction * wi / w_sum);
        }
        if wj != 0.0 {
            particles[j].v += v_t * (friction * wj / w_sum);
        }
    }
}
