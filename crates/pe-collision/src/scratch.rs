//! Reusable per-step scratch buffers for the rigid-body contact pipeline.
//!
//! Avoids allocating pose / AABB / pair / solver working sets every step.
//! Layout and values are identical to fresh `Vec`s; only capacity is reused.

use pe_math::Vec3;

use crate::manifold::ContactPoint;
use crate::shapes::Aabb;

/// Working memory cleared and reused across [`crate::ContactWorld::step`] calls.
#[derive(Clone, Debug, Default)]
pub struct StepScratch {
    /// World positions (dynamics then statics).
    pub x: Vec<Vec3>,
    /// World rotation matrices (dynamics then statics).
    pub r: Vec<[[f64; 3]; 3]>,
    /// Tight AABBs matching `x`/`r`.
    pub aabbs: Vec<Aabb>,
    /// Broad-phase pairs `(min_id, max_id)`.
    pub pairs: Vec<(usize, usize)>,
    /// Narrow-phase contacts for this detect.
    pub contacts: Vec<ContactPoint>,
    /// Inverse masses (dynamics then zeros for statics).
    pub im: Vec<f64>,
    /// World-frame inverse inertia.
    pub iinv: Vec<[[f64; 3]; 3]>,
    /// Linear velocities (extended with zeros for statics).
    pub v: Vec<Vec3>,
    /// Angular momenta (extended with zeros for statics).
    pub l: Vec<Vec3>,
}

impl StepScratch {
    /// Empty scratch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear lengths without freeing capacity.
    pub fn clear_detect(&mut self) {
        self.x.clear();
        self.r.clear();
        self.aabbs.clear();
        self.pairs.clear();
        self.contacts.clear();
    }

    /// Clear solver working sets (keeps detect buffers).
    pub fn clear_solver(&mut self) {
        self.im.clear();
        self.iinv.clear();
        self.v.clear();
        self.l.clear();
    }
}

/// Optional wall-clock phase timers for profiling (not used by default CI).
#[derive(Clone, Copy, Debug, Default)]
pub struct StepPhaseTimes {
    /// Broad phase [s].
    pub broad: f64,
    /// Narrow phase [s].
    pub narrow: f64,
    /// Velocity solver (contacts + joints) [s].
    pub solver: f64,
    /// Full step wall time [s].
    pub total: f64,
}

impl StepPhaseTimes {
    /// Steps per second implied by `total` for one step.
    #[inline]
    pub fn steps_per_sec(&self) -> f64 {
        if self.total > 0.0 {
            1.0 / self.total
        } else {
            f64::INFINITY
        }
    }
}
