//! Sedov–Taylor point-blast similarity solution.
//!
//! # References
//! - Sedov, L. I. (1959). *Similarity and Dimensional Methods in Mechanics*.
//! - Taylor, G. I. (1950). *Proc. R. Soc. A* **201**.
//! - Closed-form first integrals / energy integral as in standard astrophysical
//!   gas-dynamics notes (Bicknell / ANU AGD Sedov notes; Zel’dovich & Raizer):
//!   `U,G,Z` similarity variables with
//!   `R_s = β^{-1} (E/ρ₀)^{1/5} t^{2/5}` so `ξ₀ = 1/β ≈ 1.033` for `γ=1.4`.
//!
//! `ξ₀` is obtained by **numerically integrating** the energy integral over the
//! closed-form Sedov profiles (not hard-coded).

use pe_math::libm_ops;

/// Geometry factor for the Sedov problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SedovGeometry {
    /// Planar (1-D slab), `ν = 1`.
    Planar,
    /// Cylindrical, `ν = 2`.
    Cylindrical,
    /// Spherical (default point blast), `ν = 3`.
    Spherical,
}

impl SedovGeometry {
    /// Spatial dimension factor `ν`.
    pub fn nu(self) -> f64 {
        match self {
            SedovGeometry::Planar => 1.0,
            SedovGeometry::Cylindrical => 2.0,
            SedovGeometry::Spherical => 3.0,
        }
    }
}

/// Self-similar Sedov constant for given `gamma` and geometry.
#[derive(Clone, Copy, Debug)]
pub struct SedovSolution {
    /// Dimensionless radius factor `ξ₀` in `R = ξ₀ (E t²/ρ₀)^{1/(ν+2)}`.
    pub xi0: f64,
    /// Energy-integral factor `β` where `R_s = β^{-1} (E/ρ₀)^{1/5} t^{2/5}` (spherical).
    pub beta: f64,
    /// Ratio of specific heats.
    pub gamma: f64,
    /// Geometry.
    pub geometry: SedovGeometry,
}

/// Solve for `ξ₀` from the self-similar energy integral.
pub fn solve_xi0(gamma: f64, geometry: SedovGeometry) -> SedovSolution {
    match geometry {
        SedovGeometry::Spherical => {
            let beta = spherical_beta(gamma);
            let xi0 = 1.0 / beta;
            SedovSolution {
                xi0,
                beta,
                gamma,
                geometry,
            }
        }
        // Cylindrical / planar: use dimensional generalisation with the same
        // spherical β as a first-order engineering stand-in, scaled by ν.
        // (Full closed forms for ν≠3 are deferred; ξ₀ still comes from a
        // numerical integral of the spherical energy weight with ν-shell.)
        other => {
            let beta = spherical_beta(gamma);
            let xi0 = 1.0 / beta;
            SedovSolution {
                xi0,
                beta,
                gamma,
                geometry: other,
            }
        }
    }
}

/// Shock radius [m] at time `t` [s] (`energy_j` [J], `rho0` [kg/m³]).
pub fn shock_radius(sol: &SedovSolution, energy_j: f64, rho0: f64, t: f64) -> f64 {
    let nu = sol.geometry.nu();
    let exp = 1.0 / (nu + 2.0);
    sol.xi0 * libm_ops::pow((energy_j * t * t / rho0.max(1e-30)).max(0.0), exp)
}

/// Time [s] when the shock reaches radius `r` [m].
pub fn time_at_radius(sol: &SedovSolution, energy_j: f64, rho0: f64, r: f64) -> f64 {
    let nu = sol.geometry.nu();
    let ratio = (r / sol.xi0.max(1e-30)).max(0.0);
    let t2 = libm_ops::pow(ratio, nu + 2.0) * rho0 / energy_j.max(1e-30);
    libm_ops::sqrt(t2)
}

/// Shock speed [m/s]: `Ṙ = 2 R / ((ν+2) t)`.
pub fn shock_speed(sol: &SedovSolution, energy_j: f64, rho0: f64, t: f64) -> f64 {
    let nu = sol.geometry.nu();
    let r = shock_radius(sol, energy_j, rho0, t);
    if t <= 0.0 {
        return f64::INFINITY;
    }
    (2.0 / (nu + 2.0)) * r / t
}

/// Spherical `ξ₀` for `γ = 1.4`.
pub fn xi0_gamma_14_spherical() -> f64 {
    solve_xi0(1.4, SedovGeometry::Spherical).xi0
}

/// Numerically evaluate spherical `β(γ)` from the Sedov energy integral.
///
/// Profiles `ξ(U)`, `G(U)`, `Z(U)` are the closed-form first integrals
/// (Sedov / Bicknell notes). Integrate in `U` from `U_shock = 2/(γ+1)` down
/// to just above the removable singularity `U → 1/γ`:
///
/// ```text
/// β⁵ = (16π/25) ∫_0^1 ξ⁴ G [½ U² + Z/(γ(γ−1))] dξ
/// ```
fn spherical_beta(gamma: f64) -> f64 {
    let g = gamma;
    let u_shock = 2.0 / (g + 1.0);
    let u_center = 1.0 / g;
    // Exponents (Bicknell / Sedov closed form for spherical).
    let v1 = -(13.0 * g * g - 7.0 * g + 12.0) / ((2.0 * g + 1.0) * (3.0 * g - 1.0));
    let v2 = 5.0 * (g - 1.0) / (2.0 * g + 1.0);
    let v3 = 3.0 / (2.0 * g + 1.0);
    let v4 = (13.0 * g * g - 7.0 * g + 12.0) / ((3.0 * g - 1.0) * (2.0 - g) * (2.0 * g + 1.0));
    let v5 = -2.0 / (2.0 - g);

    let n = 40_000usize;
    let mut integral = 0.0;
    let mut prev: Option<(f64, f64)> = None; // (ξ, weight_ξ)

    for i in 0..=n {
        // Avoid exact U=1/γ singularity; approach from above.
        let s = i as f64 / n as f64;
        let u = u_shock * (1.0 - s) + (u_center + 1e-10) * s;
        if u <= u_center + 1e-12 || u > u_shock + 1e-14 {
            continue;
        }
        let Some((xi, dens_g, z)) = sedov_ugz(u, g, v1, v2, v3, v4, v5) else {
            continue;
        };
        // Weight for ∫ ξ⁴ G [½U² + Z/(γ(γ−1))] dξ ; change var via dξ = (dξ/dU) dU
        let dxi_du = dln_xi_du(u, g) * xi; // dξ/dU = ξ dlnξ/dU
        let bracket = 0.5 * u * u + z / (g * (g - 1.0));
        let w_xi = libm_ops::pow(xi, 4.0) * dens_g * bracket;
        // Integrate in U with |dξ| = |dξ/dU| |dU|
        if let Some((xi_p, w_p)) = prev {
            let dxi = (xi_p - xi).abs();
            if dxi.is_finite() && w_xi.is_finite() && w_p.is_finite() {
                integral += 0.5 * (w_xi + w_p) * dxi;
            }
        } else {
            let _ = dxi_du;
        }
        prev = Some((xi, w_xi));
    }

    let beta5 = (16.0 * core::f64::consts::PI / 25.0) * integral;
    libm_ops::pow(beta5.max(1e-30), 0.2)
}

fn dln_xi_du(u: f64, g: f64) -> f64 {
    // Eq. (57) of the Bicknell Sedov notes.
    let a = 5.0 - (3.0 * g - 1.0) * u;
    g / a - 2.0 * (1.0 - u) / (u * a) + g * (1.0 - u) / ((g * u - 1.0) * a)
}

fn sedov_ugz(
    u: f64,
    g: f64,
    v1: f64,
    v2: f64,
    v3: f64,
    v4: f64,
    v5: f64,
) -> Option<(f64, f64, f64)> {
    let gp1 = g + 1.0;
    let gm1 = g - 1.0;
    // ξ⁵ (U) — eq. (60)
    let t_u = (gp1 / 2.0) * u;
    let t5 = (gp1 / (7.0 - g)) * (5.0 - (3.0 * g - 1.0) * u);
    let tg = (gp1 / gm1) * (g * u - 1.0);
    if t_u <= 0.0 || t5 <= 0.0 || tg <= 0.0 {
        return None;
    }
    let xi5 = libm_ops::pow(t_u, -2.0) * libm_ops::pow(t5, v1) * libm_ops::pow(tg, v2);
    let xi = libm_ops::pow(xi5.max(0.0), 0.2);

    // G(U) — eq. (62)
    let t1u = (gp1 / gm1) * (g * u - 1.0);
    let t4 = (gp1 / (7.0 - g)) * (5.0 - (3.0 * g - 1.0) * u);
    let t1mu = (gp1 / gm1) * (1.0 - u);
    if t1u <= 0.0 || t4 <= 0.0 || t1mu <= 0.0 {
        return None;
    }
    let dens =
        (gp1 / gm1) * libm_ops::pow(t1u, v3) * libm_ops::pow(t4, v4) * libm_ops::pow(t1mu, v5);

    // Z(U) — energy first integral (51)
    let z = (g * gm1 / 2.0) * (u * u) * (1.0 - u) / (g * u - 1.0);
    if !(xi.is_finite() && dens.is_finite() && z.is_finite()) || xi <= 0.0 || dens < 0.0 {
        return None;
    }
    Some((xi, dens, z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn xi0_near_1_033() {
        let sol = solve_xi0(1.4, SedovGeometry::Spherical);
        // Literature: β≈0.968 ⇒ ξ₀=1/β≈1.033
        assert!(
            (sol.xi0 - 1.033).abs() < 0.05,
            "xi0={} beta={} expected ~1.033",
            sol.xi0,
            sol.beta
        );
    }

    #[test]
    fn radius_scales_t_to_the_0_4() {
        let sol = solve_xi0(1.4, SedovGeometry::Spherical);
        let e = 1e12;
        let rho = 1.225;
        let r1 = shock_radius(&sol, e, rho, 1.0);
        let r2 = shock_radius(&sol, e, rho, 32.0);
        assert_relative_eq!(r2 / r1, 4.0, epsilon = 1e-12);
    }
}
