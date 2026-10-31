//! Artificial viscosity (Monaghan) and XSPH smoothing.
//!
//! # Monaghan artificial viscosity
//!
//! For approaching pairs (`v_ij · r_ij < 0`):
//!
//! ```text
//! μ_ij = h (v_ij · r_ij) / (r_ij² + ε h²)
//! Π_ij = (−α c̄_ij μ_ij + β μ_ij²) / ρ̄_ij
//! ```
//!
//! with `v_ij = v_i − v_j`, `r_ij = x_i − x_j`, `ε = 0.01`.
//! Force contribution on i from j: `− m_j Π_ij ∇_i W`.
//!
//! # XSPH
//!
//! ```text
//! v_i ← v_i + ε_xsph Σ_j m_j (v_j − v_i) / ρ̄_ij  W_ij
//! ```
//!
//! # References
//! - Monaghan, J. J. (1992 / 1997). SPH artificial viscosity.
//! - Monaghan, J. J. (1989). On the problem of turbulence in SPH (XSPH).

use pe_math::libm_ops::sqrt;
use pe_math::Vec3;

/// Monaghan viscosity parameters.
#[derive(Clone, Copy, Debug)]
pub struct MonaghanViscosity {
    /// Linear viscosity coefficient α (typical 0.01–0.1 for fluids).
    pub alpha: f64,
    /// Quadratic coefficient β (often 0 for weakly compressible liquids).
    pub beta: f64,
    /// ε in the μ denominator (`ε h²`); default 0.01.
    pub eps: f64,
}

impl Default for MonaghanViscosity {
    fn default() -> Self {
        Self {
            alpha: 0.01,
            beta: 0.0,
            eps: 0.01,
        }
    }
}

/// Pairwise Π_ij [m²/s²] (pressure-like viscosity term). Returns 0 if separating.
#[inline]
pub fn pi_ij(
    visc: MonaghanViscosity,
    h: f64,
    sound_speed: f64,
    xi: Vec3,
    xj: Vec3,
    vi: Vec3,
    vj: Vec3,
    rho_i: f64,
    rho_j: f64,
) -> f64 {
    let rij = xi - xj; // r_ij = x_i − x_j
    let vij = vi - vj;
    let vij_rij = vij.dot(rij);
    if vij_rij >= 0.0 {
        return 0.0;
    }
    let r2 = rij.norm_squared();
    let mu = h * vij_rij / (r2 + visc.eps * h * h);
    let rho_bar = 0.5 * (rho_i + rho_j);
    let c_bar = sound_speed; // common WCSPH simplification
    (-visc.alpha * c_bar * mu + visc.beta * mu * mu) / rho_bar
}

/// XSPH coefficient ε (0 = off; typical 0.25–0.5).
#[derive(Clone, Copy, Debug)]
pub struct Xsph {
    /// Mixing coefficient.
    pub epsilon: f64,
}

impl Default for Xsph {
    fn default() -> Self {
        Self { epsilon: 0.25 }
    }
}

/// Max particle speed [m/s] (fluid only if `fluid_mask[i]`).
pub fn max_speed(v: &[Vec3], fluid: &[bool]) -> f64 {
    let mut m = 0.0;
    for (i, vi) in v.iter().enumerate() {
        if fluid.get(i).copied().unwrap_or(true) {
            let s = vi.norm_squared();
            if s > m {
                m = s;
            }
        }
    }
    sqrt(m)
}
