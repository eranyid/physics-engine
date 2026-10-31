//! Rankine–Hugoniot jump relations for an ideal-gas normal shock.
//!
//! # References
//! - Rankine, W. J. M. (1870); Hugoniot, P. H. (1887) — classical shock relations.
//! - Anderson, J. D. *Modern Compressible Flow*; Kinney & Graham (1985) Ch. 2
//!   for blast-wave applications (incident / reflected overpressure).
//!
//! Perfect gas with constant `γ`. Overpressure `p_s = p₁ − p₀` where `p₁` is
//! post-shock static pressure and `p₀` ambient.

use pe_math::libm_ops;

/// Ideal-gas ratio of specific heats for dry air.
pub const GAMMA_AIR: f64 = 1.4;

/// Shock strength from overpressure ratio `p_s / p_0` (dimensionless).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShockState {
    /// Pressure ratio `p₁/p₀` [= 1 + p_s/p₀].
    pub p_ratio: f64,
    /// Density ratio `ρ₁/ρ₀`.
    pub rho_ratio: f64,
    /// Temperature ratio `T₁/T₀`.
    pub t_ratio: f64,
    /// Shock Mach number `M_s` relative to ambient sound speed.
    pub mach: f64,
    /// Particle velocity behind the shock / ambient sound speed.
    pub u_over_a0: f64,
    /// Dynamic pressure `q = ½ ρ₁ u²` / `p₀` (dimensionless).
    pub dynamic_pressure_over_p0: f64,
    /// Normally reflected overpressure ratio `p_r / p_s` (ideal gas).
    pub reflected_overpressure_ratio: f64,
}

/// Rankine–Hugoniot state from incident overpressure ratio `ps_over_p0 = p_s/p_0`.
///
/// Reflected overpressure for a **normal** reflection on a rigid wall uses the
/// standard γ-gas closed form
/// `p_r/p_s = ((3γ−1) ξ + 4γ) / ((γ−1) ξ + 2γ)` with `ξ = p_s/p_0`.
/// For `γ = 1.4` this → **2** as `ξ → 0` (weak) and **8** as `ξ → ∞` (strong).
pub fn from_overpressure_ratio(ps_over_p0: f64, gamma: f64) -> ShockState {
    let xi = ps_over_p0.max(0.0);
    let p_ratio = 1.0 + xi;
    // p₁/p₀ = 1 + 2γ/(γ+1) (M² − 1)  ⇒  M² = 1 + (γ+1)/(2γ) ξ
    let mach_sq = 1.0 + (gamma + 1.0) / (2.0 * gamma) * xi;
    let mach = libm_ops::sqrt(mach_sq.max(1.0));
    // ρ₁/ρ₀ = (γ+1) M² / ((γ−1) M² + 2)
    let rho_ratio = ((gamma + 1.0) * mach_sq) / ((gamma - 1.0) * mach_sq + 2.0);
    let t_ratio = p_ratio / rho_ratio;
    // u = M_s a₀ (1 − ρ₀/ρ₁)
    let u_over_a0 = mach * (1.0 - 1.0 / rho_ratio);
    let q_over_p0 = 0.5 * rho_ratio * u_over_a0 * u_over_a0;
    let reflected_overpressure_ratio = if xi < 1e-30 {
        2.0
    } else {
        ((3.0 * gamma - 1.0) * xi + 4.0 * gamma) / ((gamma - 1.0) * xi + 2.0 * gamma)
    };
    ShockState {
        p_ratio,
        rho_ratio,
        t_ratio,
        mach,
        u_over_a0,
        dynamic_pressure_over_p0: q_over_p0,
        reflected_overpressure_ratio,
    }
}

/// Dynamic pressure behind the shock [Pa]: `q = (q/p₀)·p₀`.
pub fn dynamic_pressure(ps_over_p0: f64, p0: f64, gamma: f64) -> f64 {
    from_overpressure_ratio(ps_over_p0, gamma).dynamic_pressure_over_p0 * p0
}

/// Normally reflected overpressure [Pa] from incident overpressure [Pa] and `p0` [Pa].
pub fn reflected_overpressure(p_s: f64, p0: f64, gamma: f64) -> f64 {
    let xi = p_s / p0.max(1e-30);
    from_overpressure_ratio(xi, gamma).reflected_overpressure_ratio * p_s
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn weak_shock_reflection_limit_2() {
        let s = from_overpressure_ratio(1e-6, GAMMA_AIR);
        assert_relative_eq!(s.reflected_overpressure_ratio, 2.0, epsilon = 1e-5);
    }

    #[test]
    fn strong_shock_reflection_limit_8() {
        let s = from_overpressure_ratio(1e12, GAMMA_AIR);
        assert_relative_eq!(s.reflected_overpressure_ratio, 8.0, epsilon = 1e-6);
    }

    #[test]
    fn mach_unity_at_zero_overpressure() {
        let s = from_overpressure_ratio(0.0, GAMMA_AIR);
        assert_relative_eq!(s.mach, 1.0, epsilon = 1e-14);
    }
}
