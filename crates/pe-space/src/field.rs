//! Gravity field traits and point-mass N-body / central-μ implementations.
//!
//! Accelerations are returned directly [m/s²]. Fields that know μ use that
//! parameter rather than evaluating `G * M` repeatedly.

use pe_math::libm_ops;
use pe_math::{Vec3, G as G_SI};

use crate::state::NBodyState;

/// Acceleration field for an N-body (or reduced) system.
///
/// Implementations must be deterministic: fixed body loops, no unordered maps,
/// portable libm only, no FMA.
pub trait GravityField: Send + Sync {
    /// Write accelerations [m/s²] for every body into `out` (length = `state.n()`).
    fn accelerations(&self, state: &NBodyState, t: f64, out: &mut [Vec3]);

    /// Optional gravitational potential energy [J]. Default: none.
    fn potential(&self, _state: &NBodyState) -> Option<f64> {
        None
    }
}

/// Central two-body field: `a = −μ r / |r|³` for body 0 about a fixed origin.
///
/// Body 0 is the only accelerated mass; additional bodies are ignored unless the
/// caller uses [`NBodyPointMass`]. Useful for Kepler tests with μ given directly.
#[derive(Clone, Debug)]
pub struct CentralMu {
    /// Gravitational parameter μ [m³/s²].
    pub mu: f64,
}

impl CentralMu {
    /// Construct with μ [m³/s²].
    pub fn new(mu: f64) -> Self {
        Self { mu }
    }
}

impl GravityField for CentralMu {
    fn accelerations(&self, state: &NBodyState, _t: f64, out: &mut [Vec3]) {
        for a in out.iter_mut() {
            *a = Vec3::zero();
        }
        if state.n() == 0 {
            return;
        }
        let r = state.x[0];
        let r2 = r.norm_squared();
        if r2 == 0.0 {
            return;
        }
        let inv_r3 = 1.0 / (r2 * libm_ops::sqrt(r2));
        out[0] = r * (-self.mu * inv_r3);
    }

    fn potential(&self, state: &NBodyState) -> Option<f64> {
        if state.n() == 0 {
            return Some(0.0);
        }
        let r = state.x[0].norm();
        if r == 0.0 {
            return None;
        }
        // Specific potential times mass of the satellite.
        Some(-state.mass[0] * self.mu / r)
    }
}

/// Pairwise Newtonian point-mass N-body field.
///
/// `F_ij = G m_i m_j (x_j − x_i) / (|r|² + ε²)^{3/2}` with antisymmetric `i < j`
/// loops (linear momentum conserved). Prefer [`Self::from_mu_primary`] when a
/// dominant body has a published μ.
#[derive(Clone, Debug)]
pub struct NBodyPointMass {
    /// Gravitational constant [m³ kg⁻¹ s⁻²] (or 1.0 in geometric units).
    pub g_const: f64,
    /// Softening length squared [m²].
    pub eps2: f64,
}

impl NBodyPointMass {
    /// SI `G`, no softening.
    pub fn si() -> Self {
        Self {
            g_const: G_SI,
            eps2: 0.0,
        }
    }

    /// Geometric units `G = 1` (figure-eight / Pythagorean demos).
    pub fn geometric() -> Self {
        Self {
            g_const: 1.0,
            eps2: 0.0,
        }
    }

    /// Custom `G` and softening length [m].
    pub fn with_params(g_const: f64, softening: f64) -> Self {
        Self {
            g_const,
            eps2: softening * softening,
        }
    }
}

impl GravityField for NBodyPointMass {
    fn accelerations(&self, state: &NBodyState, _t: f64, out: &mut [Vec3]) {
        let n = state.n();
        for a in out.iter_mut().take(n) {
            *a = Vec3::zero();
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let d = state.x[j] - state.x[i];
                let r2 = d.norm_squared() + self.eps2;
                let inv_r3 = 1.0 / (r2 * libm_ops::sqrt(r2));
                let g = self.g_const * inv_r3;
                // a_i += G m_j r_ij / r³ ; a_j -= G m_i r_ij / r³
                out[i] += d * (g * state.mass[j]);
                out[j] -= d * (g * state.mass[i]);
            }
        }
    }

    fn potential(&self, state: &NBodyState) -> Option<f64> {
        let n = state.n();
        let mut u = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = state.x[j] - state.x[i];
                let r2 = d.norm_squared() + self.eps2;
                let r = libm_ops::sqrt(r2);
                if r == 0.0 {
                    continue;
                }
                u -= self.g_const * state.mass[i] * state.mass[j] / r;
            }
        }
        Some(u)
    }
}

/// Per-body extra acceleration callback `(state, t, body_index) -> a [m/s²]`.
pub type ExtraAccel = Box<dyn Fn(&NBodyState, f64, usize) -> Vec3 + Send + Sync>;

/// Composite field: primary gravity + additive acceleration models (J2, drag, …).
pub struct CompositeField {
    /// Primary gravity model.
    pub primary: Box<dyn GravityField>,
    /// Extra acceleration callbacks.
    extras: Vec<ExtraAccel>,
}

impl CompositeField {
    /// Wrap a primary field.
    pub fn new(primary: Box<dyn GravityField>) -> Self {
        Self {
            primary,
            extras: Vec::new(),
        }
    }

    /// Push an extra per-body acceleration model.
    pub fn push_extra<F>(&mut self, f: F)
    where
        F: Fn(&NBodyState, f64, usize) -> Vec3 + Send + Sync + 'static,
    {
        self.extras.push(Box::new(f));
    }
}

impl GravityField for CompositeField {
    fn accelerations(&self, state: &NBodyState, t: f64, out: &mut [Vec3]) {
        self.primary.accelerations(state, t, out);
        let n = state.n();
        for i in 0..n {
            for extra in &self.extras {
                out[i] += extra(state, t, i);
            }
        }
    }

    fn potential(&self, state: &NBodyState) -> Option<f64> {
        self.primary.potential(state)
    }
}
