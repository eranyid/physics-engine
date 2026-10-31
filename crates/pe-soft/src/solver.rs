//! XPBD soft-body world with substepping (Macklin 2019 Small Steps).

use pe_math::Vec3;

use crate::collision::{collide_self, collide_statics, SelfCollisionParams, StaticColliders};
use crate::constraint::{project_all, ConstraintSet};
use crate::particle::Particle;

/// Soft-body simulation parameters.
#[derive(Clone, Copy, Debug)]
pub struct SoftParams {
    /// Number of XPBD substeps per `step(dt)` call (Macklin 2019).
    pub substeps: usize,
    /// Constraint iterations per substep (usually 1–2 with many substeps).
    pub iterations: usize,
    /// Gravity acceleration [m/s²] (world frame).
    pub gravity: Vec3,
    /// Global velocity damping per second (0 = none); applied as `v *= 1/(1+d·dt)`.
    pub damping: f64,
}

impl Default for SoftParams {
    fn default() -> Self {
        Self {
            substeps: 20,
            iterations: 1,
            gravity: Vec3::new(0.0, 0.0, -9.80665),
            damping: 0.0,
        }
    }
}

/// Soft-body world: particles + XPBD constraints + static/self collisions.
#[derive(Clone, Debug)]
pub struct SoftWorld {
    /// Particles (SoA-friendly `Vec`, fixed index order).
    pub particles: Vec<Particle>,
    /// Constraints.
    pub constraints: ConstraintSet,
    /// Static colliders.
    pub colliders: StaticColliders,
    /// Self-collision parameters.
    pub self_collision: SelfCollisionParams,
    /// Solver parameters.
    pub params: SoftParams,
    /// Simulation time [s].
    pub t: f64,
}

impl SoftWorld {
    /// Empty world with default parameters.
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            constraints: ConstraintSet::default(),
            colliders: StaticColliders::default(),
            self_collision: SelfCollisionParams::default(),
            params: SoftParams::default(),
            t: 0.0,
        }
    }

    /// Number of particles.
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// True if no particles.
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Add a particle; returns its index.
    pub fn add_particle(&mut self, p: Particle) -> usize {
        let i = self.particles.len();
        self.particles.push(p);
        i
    }

    /// Pin particle `i` (set inverse mass to 0 and zero velocity).
    pub fn pin(&mut self, i: usize) {
        self.particles[i].inv_mass = 0.0;
        self.particles[i].v = Vec3::zero();
    }

    /// One outer step of size `dt` [s] using `substeps` equal XPBD substeps.
    pub fn step(&mut self, dt: f64) {
        if dt <= 0.0 || self.particles.is_empty() {
            return;
        }
        let n_sub = self.params.substeps.max(1);
        let h = dt / n_sub as f64;
        for _ in 0..n_sub {
            self.substep(h);
        }
        self.t += dt;
    }

    /// Integrate until `t_end` with fixed outer `dt`.
    pub fn run_for(&mut self, t_end: f64, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        while self.t + dt <= t_end + 1e-15 {
            self.step(dt);
        }
    }

    fn substep(&mut self, h: f64) {
        let g = self.params.gravity;
        let damp = self.params.damping;
        // Predict
        for p in &mut self.particles {
            p.x_prev = p.x;
            if p.inv_mass == 0.0 {
                continue;
            }
            p.v += g * h;
            if damp > 0.0 {
                p.v *= 1.0 / (1.0 + damp * h);
            }
            p.x += p.v * h;
        }

        self.constraints.reset_lambdas();
        let iters = self.params.iterations.max(1);
        for _ in 0..iters {
            project_all(&mut self.particles, &mut self.constraints, h);
            collide_statics(&mut self.particles, &self.colliders);
            collide_self(&mut self.particles, &self.self_collision);
        }

        // Update velocities from positions
        let inv_h = 1.0 / h;
        for p in &mut self.particles {
            if p.inv_mass == 0.0 {
                p.v = Vec3::zero();
                p.x_prev = p.x;
                continue;
            }
            p.v = (p.x - p.x_prev) * inv_h;
        }
    }

    /// Pack positions then velocities as flat `f64` (3N + 3N) for hashing / NumPy.
    pub fn pack(&self) -> Vec<f64> {
        let n = self.particles.len();
        let mut out = Vec::with_capacity(n * 6);
        for p in &self.particles {
            out.extend_from_slice(&p.x.to_array());
        }
        for p in &self.particles {
            out.extend_from_slice(&p.v.to_array());
        }
        out
    }

    /// Positions as flat length-3N.
    pub fn positions_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.particles.len() * 3);
        for p in &self.particles {
            out.extend_from_slice(&p.x.to_array());
        }
        out
    }

    /// Velocities as flat length-3N.
    pub fn velocities_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.particles.len() * 3);
        for p in &self.particles {
            out.extend_from_slice(&p.v.to_array());
        }
        out
    }
}

impl Default for SoftWorld {
    fn default() -> Self {
        Self::new()
    }
}

/// FNV-1a 64-bit hash of packed particle state (IEEE bits, little-endian).
pub fn soft_state_hash64(world: &SoftWorld) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for x in world.pack() {
        for b in x.to_bits().to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}
