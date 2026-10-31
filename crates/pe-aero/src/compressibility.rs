//! Compressible-flow corrections and gas-dynamic relations (clean-room).
//!
//! # References
//! - Prandtl–Glauert rule: Anderson, *Fundamentals of Aerodynamics*;
//!   Glauert, H. (1928). The effect of compressibility on the lift of aerofoils.
//! - Kármán–Tsien: von Kármán, T. & Tsien, H. S. (1939).
//! - Ackeret / linearized supersonic: Ackeret, J. (1925); `C_L = 4 α / √(M²−1)`.
//! - Normal shock / isentropic: NASA / Anderson textbook identities
//!   (γ-gas perfect).

use pe_math::libm_ops;

/// Mach proximity clamp for Prandtl–Glauert (`|M−1|` floor ≈ 0.05).
pub const PG_MACH_CLAMP: f64 = 0.05;

/// Prandtl–Glauert factor `1/β` with `β = √|1 − M²|`, clamped near M=1.
///
/// Valid for subsonic use with `M < 0.7` typically; the clamp prevents the
/// singularity at M→1. Returns a multiplicative correction for `C_L`, `C_p`.
pub fn prandtl_glauert_factor(mach: f64) -> f64 {
    let m = libm_ops::fabs(mach);
    let m_eff = if libm_ops::fabs(m - 1.0) < PG_MACH_CLAMP {
        if m >= 1.0 {
            1.0 + PG_MACH_CLAMP
        } else {
            1.0 - PG_MACH_CLAMP
        }
    } else {
        m
    };
    let beta = libm_ops::sqrt(libm_ops::fabs(1.0 - m_eff * m_eff));
    1.0 / beta.max(1e-6)
}

/// Apply Prandtl–Glauert to an incompressible coefficient: `C_c = C_i / β`.
pub fn prandtl_glauert(c_incomp: f64, mach: f64) -> f64 {
    c_incomp * prandtl_glauert_factor(mach)
}

/// Kármán–Tsien compressibility correction for pressure coefficient.
///
/// `C_p = C_{p,i} / (√(1−M²) + M²/(1+√(1−M²)) · C_{p,i}/2)` (subsonic).
/// Near M=1 the same clamp as Prandtl–Glauert is applied to `√(1−M²)`.
pub fn karman_tsien(cp_incomp: f64, mach: f64) -> f64 {
    let m = libm_ops::fabs(mach).min(1.0 - PG_MACH_CLAMP);
    let beta = libm_ops::sqrt(1.0 - m * m);
    let denom = beta + (m * m) / (1.0 + beta) * (cp_incomp * 0.5);
    cp_incomp / denom.max(1e-12)
}

/// Ackeret linearized-supersonic lift-curve: `C_L = 4 α / √(M² − 1)` (`α` [rad]).
///
/// Requires `M > 1`. Returns `None` if `M ≤ 1`.
pub fn ackeret_cl(alpha: f64, mach: f64) -> Option<f64> {
    if mach <= 1.0 {
        return None;
    }
    Some(4.0 * alpha / libm_ops::sqrt(mach * mach - 1.0))
}

/// Normal-shock ratios across a stationary shock (perfect gas).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalShock {
    /// Downstream Mach `M₂`.
    pub m2: f64,
    /// Static pressure ratio `p₂/p₁`.
    pub p_ratio: f64,
    /// Density ratio `ρ₂/ρ₁`.
    pub rho_ratio: f64,
    /// Temperature ratio `T₂/T₁`.
    pub t_ratio: f64,
    /// Stagnation pressure ratio `p₀₂/p₀₁`.
    pub p0_ratio: f64,
}

/// Normal-shock relations for upstream Mach `m1 > 1`, ratio of specific heats `gamma`.
///
/// Anderson / NASA:
/// - `M₂² = (1 + ½(γ−1)M₁²) / (γ M₁² − ½(γ−1))`
/// - `p₂/p₁ = 1 + 2γ/(γ+1) (M₁² − 1)`
/// - `ρ₂/ρ₁ = (γ+1)M₁² / ((γ−1)M₁² + 2)`
/// - `T₂/T₁ = (p₂/p₁)/(ρ₂/ρ₁)`
/// - `p₀₂/p₀₁` from isentropic relations on each side.
pub fn normal_shock(m1: f64, gamma: f64) -> Option<NormalShock> {
    if m1 <= 1.0 || gamma <= 1.0 {
        return None;
    }
    let gm1 = gamma - 1.0;
    let gp1 = gamma + 1.0;
    let m1sq = m1 * m1;
    let m2sq = (1.0 + 0.5 * gm1 * m1sq) / (gamma * m1sq - 0.5 * gm1);
    let m2 = libm_ops::sqrt(m2sq);
    let p_ratio = 1.0 + (2.0 * gamma / gp1) * (m1sq - 1.0);
    let rho_ratio = (gp1 * m1sq) / (gm1 * m1sq + 2.0);
    let t_ratio = p_ratio / rho_ratio;
    // Stagnation pressure ratio:
    // p0/p = (1 + ½(γ−1)M²)^{γ/(γ−1)}
    let p0_p1 = libm_ops::pow(1.0 + 0.5 * gm1 * m1sq, gamma / gm1);
    let p0_p2 = libm_ops::pow(1.0 + 0.5 * gm1 * m2sq, gamma / gm1);
    let p0_ratio = (p0_p2 / p0_p1) * p_ratio;
    Some(NormalShock {
        m2,
        p_ratio,
        rho_ratio,
        t_ratio,
        p0_ratio,
    })
}

/// Isentropic freestream / stagnation ratios at Mach `m`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Isentropic {
    /// `p/p₀`.
    pub p_over_p0: f64,
    /// `T/T₀`.
    pub t_over_t0: f64,
    /// `ρ/ρ₀`.
    pub rho_over_rho0: f64,
}

/// Isentropic relations: `T/T₀ = [1 + ½(γ−1)M²]⁻¹`, etc.
pub fn isentropic(mach: f64, gamma: f64) -> Isentropic {
    let gm1 = gamma - 1.0;
    let t_over_t0 = 1.0 / (1.0 + 0.5 * gm1 * mach * mach);
    let p_over_p0 = libm_ops::pow(t_over_t0, gamma / gm1);
    let rho_over_rho0 = libm_ops::pow(t_over_t0, 1.0 / gm1);
    Isentropic {
        p_over_p0,
        t_over_t0,
        rho_over_rho0,
    }
}
