//! External force/torque buffers for domain coupling (fluids, blast, …).
//!
//! [`ExternalLoads`] is the **stable** continuous-load interface into
//! [`crate::World`]: a coupler writes `force` / `torque` each step; the
//! generator adds them during [`crate::World::forces`]. Impulsive loads
//! continue to use [`crate::World::apply_impulse`] /
//! [`crate::World::apply_wrench`].
//!
//! Share one instance via [`std::sync::Arc`]: keep a clone in the coupler and
//! `Box` another clone into the world's generator list (fixed order).
//! Interior mutability uses `Mutex` (required by `ForceGenerator: Send + Sync`);
//! the deterministic core is single-threaded so the lock is uncontended.
//! Determinism: buffers are plain `Vec`s, never hash-map iteration.

use std::sync::{Arc, Mutex};

use pe_math::Vec3;

use crate::forces::ForceGenerator;
use crate::world::{State, World};

/// Per-body world-frame force [N] and torque about COM [N·m].
#[derive(Clone, Debug, Default)]
pub struct LoadBuffers {
    /// Force on each body [N].
    pub force: Vec<Vec3>,
    /// Torque about each COM [N·m].
    pub torque: Vec<Vec3>,
}

impl LoadBuffers {
    /// Zero buffers for `n` bodies.
    pub fn zeros(n: usize) -> Self {
        Self {
            force: vec![Vec3::zero(); n],
            torque: vec![Vec3::zero(); n],
        }
    }

    /// Clear all entries to zero (keeps length).
    pub fn clear(&mut self) {
        for f in &mut self.force {
            *f = Vec3::zero();
        }
        for t in &mut self.torque {
            *t = Vec3::zero();
        }
    }

    /// Ensure length `n` (extends with zeros or truncates).
    pub fn resize(&mut self, n: usize) {
        self.force.resize(n, Vec3::zero());
        self.torque.resize(n, Vec3::zero());
    }
}

/// Force generator that injects caller-updated external loads into the world.
#[derive(Debug, Default)]
pub struct ExternalLoads {
    /// Mutable load buffers (written by the coupler, read in `apply`).
    pub buffers: Mutex<LoadBuffers>,
}

impl ExternalLoads {
    /// Empty buffers.
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(LoadBuffers::default()),
        }
    }

    /// Pre-sized zero buffers for `n` bodies.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            buffers: Mutex::new(LoadBuffers::zeros(n)),
        }
    }

    /// Replace force/torque for body `i` (extends if needed).
    pub fn set(&self, i: usize, force: Vec3, torque: Vec3) {
        let mut b = self.buffers.lock().expect("ExternalLoads lock");
        if i >= b.force.len() {
            b.resize(i + 1);
        }
        b.force[i] = force;
        b.torque[i] = torque;
    }

    /// Add into body `i` (extends if needed).
    pub fn add(&self, i: usize, force: Vec3, torque: Vec3) {
        let mut b = self.buffers.lock().expect("ExternalLoads lock");
        if i >= b.force.len() {
            b.resize(i + 1);
        }
        b.force[i] += force;
        b.torque[i] += torque;
    }

    /// Copy an entire force/torque pair of slices (lengths must match).
    pub fn set_all(&self, force: &[Vec3], torque: &[Vec3]) {
        debug_assert_eq!(force.len(), torque.len());
        let mut b = self.buffers.lock().expect("ExternalLoads lock");
        b.resize(force.len());
        b.force.copy_from_slice(force);
        b.torque.copy_from_slice(torque);
    }

    /// Zero all stored loads.
    pub fn clear(&self) {
        self.buffers.lock().expect("ExternalLoads lock").clear();
    }
}

impl ForceGenerator for ExternalLoads {
    fn apply(&self, world: &World, _s: &State, _t: f64, f: &mut [Vec3], torque: &mut [Vec3]) {
        let b = self.buffers.lock().expect("ExternalLoads lock");
        let n = world.n.min(f.len()).min(torque.len());
        for i in 0..n {
            if i < b.force.len() {
                f[i] += b.force[i];
            }
            if i < b.torque.len() {
                torque[i] += b.torque[i];
            }
        }
    }
}

impl ForceGenerator for Arc<ExternalLoads> {
    fn apply(&self, world: &World, s: &State, t: f64, f: &mut [Vec3], torque: &mut [Vec3]) {
        (**self).apply(world, s, t, f, torque);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;
    use crate::forces::UniformGravity;

    #[test]
    fn external_loads_apply_adds_force() {
        let loads = ExternalLoads::with_capacity(1);
        loads.set(0, Vec3::new(10.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.5));
        let body = RigidBody::new(2.0);
        let world = World::new(vec![body], vec![Box::new(UniformGravity::earth())]);
        let mut f = vec![Vec3::zero()];
        let mut t = vec![Vec3::zero()];
        loads.apply(&world, &world.state, 0.0, &mut f, &mut t);
        assert!((f[0].x - 10.0).abs() < 1e-15);
        assert!((t[0].z - 0.5).abs() < 1e-15);
    }

    #[test]
    fn arc_external_loads_is_force_generator() {
        let loads = Arc::new(ExternalLoads::with_capacity(1));
        loads.set(0, Vec3::new(1.0, 0.0, 0.0), Vec3::zero());
        let body = RigidBody::new(1.0);
        let world = World::new(
            vec![body],
            vec![
                Box::new(UniformGravity::new(Vec3::zero())),
                Box::new(Arc::clone(&loads)),
            ],
        );
        let (f, _) = world.forces(&world.state, 0.0);
        assert!((f[0].x - 1.0).abs() < 1e-15);
    }
}
