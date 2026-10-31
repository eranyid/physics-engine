//! N-body translational state (positions + velocities) with optional compensated
//! position updates and origin rebasing.

use pe_math::Vec3;

use crate::compensated::CompensatedVec3;

/// Policy for recentring the coordinate origin during long integrations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OriginRebase {
    /// Never rebase.
    Never,
    /// Subtract the centre of mass after every `every_steps` accepted steps.
    ComEvery {
        /// Step interval between rebases.
        every_steps: usize,
    },
    /// Rebase when max |r_i| exceeds `threshold` [m].
    WhenFar {
        /// Distance threshold [m].
        threshold: f64,
    },
}

/// Translational N-body state in an inertial frame.
///
/// Masses are stored alongside state so gravity fields and diagnostics share one
/// SoA layout. Order is fixed for determinism (never a hash map).
#[derive(Clone, Debug)]
pub struct NBodyState {
    /// Masses [kg] (or dimensionless in geometric units when G=1, m=1 demos).
    pub mass: Vec<f64>,
    /// Positions [m].
    pub x: Vec<Vec3>,
    /// Velocities [m/s].
    pub v: Vec<Vec3>,
    /// Optional compensated position accumulators (same length as `x` when enabled).
    compensated: Option<Vec<CompensatedVec3>>,
    /// Steps since last origin rebase.
    steps_since_rebase: usize,
    /// Rebase policy.
    pub rebase: OriginRebase,
}

impl NBodyState {
    /// Construct from masses, positions, velocities. Lengths must match.
    pub fn new(mass: Vec<f64>, x: Vec<Vec3>, v: Vec<Vec3>) -> Self {
        assert_eq!(mass.len(), x.len());
        assert_eq!(x.len(), v.len());
        Self {
            mass,
            x,
            v,
            compensated: None,
            steps_since_rebase: 0,
            rebase: OriginRebase::Never,
        }
    }

    /// Number of bodies.
    #[inline]
    pub fn n(&self) -> usize {
        self.mass.len()
    }

    /// Enable Neumaier-compensated position updates (initialised from current `x`).
    pub fn enable_compensated_positions(&mut self) {
        self.compensated = Some(self.x.iter().copied().map(CompensatedVec3::new).collect());
    }

    /// Disable compensation (keeps current `x`).
    pub fn disable_compensated_positions(&mut self) {
        if let Some(c) = self.compensated.take() {
            for (i, ci) in c.into_iter().enumerate() {
                self.x[i] = ci.value;
            }
        }
    }

    /// Set origin-rebase policy.
    pub fn with_rebase(mut self, rebase: OriginRebase) -> Self {
        self.rebase = rebase;
        self
    }

    /// Centre of mass position [m].
    pub fn com(&self) -> Vec3 {
        let mut mtot = 0.0;
        let mut r = Vec3::zero();
        for i in 0..self.n() {
            mtot += self.mass[i];
            r += self.x[i] * self.mass[i];
        }
        if mtot == 0.0 {
            Vec3::zero()
        } else {
            r / mtot
        }
    }

    /// Centre-of-mass velocity [m/s].
    pub fn com_velocity(&self) -> Vec3 {
        let mut mtot = 0.0;
        let mut p = Vec3::zero();
        for i in 0..self.n() {
            mtot += self.mass[i];
            p += self.v[i] * self.mass[i];
        }
        if mtot == 0.0 {
            Vec3::zero()
        } else {
            p / mtot
        }
    }

    /// Apply a position increment `dx` [m] (uses compensation when enabled).
    pub fn add_position(&mut self, i: usize, dx: Vec3) {
        if let Some(ref mut c) = self.compensated {
            c[i].add(dx);
            self.x[i] = c[i].value;
        } else {
            self.x[i] += dx;
        }
    }

    /// Replace position `i` (resets compensation channel for that body).
    pub fn set_position(&mut self, i: usize, x: Vec3) {
        self.x[i] = x;
        if let Some(ref mut c) = self.compensated {
            c[i] = CompensatedVec3::new(x);
        }
    }

    /// Pack little-endian IEEE bits of all `x` then `v` (for hashing / FFI).
    pub fn pack(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n() * 6);
        for i in 0..self.n() {
            out.extend_from_slice(&self.x[i].to_array());
        }
        for i in 0..self.n() {
            out.extend_from_slice(&self.v[i].to_array());
        }
        out
    }

    /// Notify that one accepted integrator step completed; may rebase origin.
    pub fn after_step(&mut self) {
        self.steps_since_rebase = self.steps_since_rebase.saturating_add(1);
        let do_rebase = match self.rebase {
            OriginRebase::Never => false,
            OriginRebase::ComEvery { every_steps } => {
                every_steps > 0 && self.steps_since_rebase >= every_steps
            }
            OriginRebase::WhenFar { threshold } => self.x.iter().any(|r| r.norm() > threshold),
        };
        if do_rebase {
            self.rebase_to_com();
            self.steps_since_rebase = 0;
        }
    }

    /// Subtract COM position (and optionally COM velocity) so the origin sits at the COM.
    pub fn rebase_to_com(&mut self) {
        let com = self.com();
        let vcom = self.com_velocity();
        for i in 0..self.n() {
            let x = self.x[i] - com;
            self.set_position(i, x);
            self.v[i] -= vcom;
        }
    }
}
