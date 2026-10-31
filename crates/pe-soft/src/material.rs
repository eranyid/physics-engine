//! Continuum / contact material parameters and XPBD compliance mapping.
//!
//! Every table entry is marked `verified: false` unless a primary source was
//! actually fetched for that constant. Values are **nominal** engineering
//! placeholders suitable for demos and unit tests — not certified material data.

use pe_math::libm_ops;

/// Continuum + contact material parameters (SI).
///
/// Fields used by the soft solver today: `density`, `youngs_modulus`,
/// `poisson_ratio`, `friction`, `restitution`. Yield / thermal fields are
/// stored for future plasticity / heat modules and are unused by XPBD v1.
#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    /// Display name (table key).
    pub name: &'static str,
    /// Mass density ρ [kg/m³].
    pub density: f64,
    /// Young's modulus E [Pa].
    pub youngs_modulus: f64,
    /// Poisson's ratio ν [dimensionless], typically in (−1, 0.5).
    pub poisson_ratio: f64,
    /// Coulomb friction μ [dimensionless].
    pub friction: f64,
    /// Coefficient of restitution e [dimensionless].
    pub restitution: f64,
    /// Yield strength σ_y [Pa] (unused in XPBD v1).
    pub yield_strength: f64,
    /// Thermal conductivity k [W/(m·K)] (unused in XPBD v1).
    pub thermal_conductivity: f64,
    /// Specific heat c_p [J/(kg·K)] (unused in XPBD v1).
    pub specific_heat: f64,
    /// `true` only when a primary source was fetched for these numbers.
    pub verified: bool,
    /// Short provenance note (e.g. "nominal rubber demo values").
    pub source_note: &'static str,
}

impl Material {
    /// First Lamé parameter λ = E ν / ((1+ν)(1−2ν)) [Pa].
    ///
    /// Returns `None` if ν → 0.5 (denominator vanishes) or inputs are non-physical.
    pub fn lame_lambda(&self) -> Option<f64> {
        lame_lambda(self.youngs_modulus, self.poisson_ratio)
    }

    /// Second Lamé parameter (shear modulus) μ = E / (2(1+ν)) [Pa].
    pub fn lame_mu(&self) -> Option<f64> {
        lame_mu(self.youngs_modulus, self.poisson_ratio)
    }

    /// XPBD stretch compliance α = 1/k for a uniaxial bar segment of length `rest_length`
    /// [m] and cross-section `area` [m²], with k = E A / L₀.
    ///
    /// Units of α: m/N (= s²/kg).
    pub fn stretch_compliance(&self, rest_length: f64, area: f64) -> Option<f64> {
        stretch_compliance(self.youngs_modulus, rest_length, area)
    }

    /// XPBD volume / volumetric compliance from bulk modulus K = λ + 2μ/3.
    ///
    /// α_vol ≈ 1/(K V₀) with V₀ the rest volume [m³]. Units: 1/N (= s²/(kg·m²))
    /// when used as a volume constraint compliance (C in m³).
    pub fn volume_compliance(&self, rest_volume: f64) -> Option<f64> {
        let lambda = self.lame_lambda()?;
        let mu = self.lame_mu()?;
        let k_bulk = lambda + (2.0 / 3.0) * mu;
        if k_bulk <= 0.0 || rest_volume <= 0.0 {
            return None;
        }
        Some(1.0 / (k_bulk * rest_volume))
    }
}

/// λ = E ν / ((1+ν)(1−2ν)).
pub fn lame_lambda(e: f64, nu: f64) -> Option<f64> {
    if e <= 0.0 || !nu.is_finite() {
        return None;
    }
    let den = (1.0 + nu) * (1.0 - 2.0 * nu);
    if den.abs() < 1e-15 {
        return None;
    }
    Some(e * nu / den)
}

/// μ = E / (2(1+ν)).
pub fn lame_mu(e: f64, nu: f64) -> Option<f64> {
    if e <= 0.0 || !nu.is_finite() || (1.0 + nu).abs() < 1e-15 {
        return None;
    }
    Some(e / (2.0 * (1.0 + nu)))
}

/// Stretch spring stiffness k = E A / L₀ [N/m].
pub fn stretch_stiffness(e: f64, rest_length: f64, area: f64) -> Option<f64> {
    if e <= 0.0 || rest_length <= 0.0 || area <= 0.0 {
        return None;
    }
    Some(e * area / rest_length)
}

/// XPBD compliance α = 1/k [m/N] for a stretch spring with k = E A / L₀.
pub fn stretch_compliance(e: f64, rest_length: f64, area: f64) -> Option<f64> {
    stretch_stiffness(e, rest_length, area).map(|k| 1.0 / k)
}

/// α̃ = α / dt² used inside the XPBD multiplier update.
#[inline]
pub fn alpha_tilde(compliance: f64, dt: f64) -> f64 {
    if dt == 0.0 {
        return f64::INFINITY;
    }
    compliance / (dt * dt)
}

/// Inverse: stiffness from XPBD compliance (α = 1/k ⇒ k = 1/α).
#[inline]
pub fn stiffness_from_compliance(compliance: f64) -> Option<f64> {
    if compliance <= 0.0 {
        // Zero compliance ⇒ rigid (infinite stiffness).
        if compliance == 0.0 {
            return Some(f64::INFINITY);
        }
        return None;
    }
    Some(1.0 / compliance)
}

fn entry(
    name: &'static str,
    density: f64,
    e: f64,
    nu: f64,
    friction: f64,
    restitution: f64,
    yield_strength: f64,
    k_th: f64,
    cp: f64,
    note: &'static str,
) -> Material {
    Material {
        name,
        density,
        youngs_modulus: e,
        poisson_ratio: nu,
        friction,
        restitution,
        yield_strength,
        thermal_conductivity: k_th,
        specific_heat: cp,
        verified: false,
        source_note: note,
    }
}

/// Small built-in table of **nominal** materials (`verified: false` for all).
///
/// Order is fixed for determinism. Do not treat numbers as certified handbook data.
pub fn builtin_materials() -> Vec<Material> {
    vec![
        entry(
            "rubber_soft",
            1100.0,
            1.0e6,
            0.49,
            0.8,
            0.3,
            5.0e6,
            0.15,
            1800.0,
            "nominal soft rubber demo; verified=false",
        ),
        entry(
            "rubber_hard",
            1200.0,
            1.0e7,
            0.48,
            0.9,
            0.4,
            2.0e7,
            0.16,
            1700.0,
            "nominal hard rubber demo; verified=false",
        ),
        entry(
            "silicone",
            1100.0,
            5.0e5,
            0.49,
            0.6,
            0.2,
            3.0e6,
            0.2,
            1400.0,
            "nominal silicone elastomer demo; verified=false",
        ),
        entry(
            "cloth_cotton",
            400.0,
            5.0e8,
            0.3,
            0.4,
            0.05,
            1.0e7,
            0.04,
            1300.0,
            "nominal woven cloth demo (effective continuum E); verified=false",
        ),
        entry(
            "rope_nylon",
            1140.0,
            2.0e9,
            0.4,
            0.3,
            0.1,
            7.0e7,
            0.25,
            1700.0,
            "nominal nylon rope axial demo; verified=false",
        ),
        entry(
            "jelly",
            1050.0,
            5.0e4,
            0.45,
            0.5,
            0.1,
            1.0e5,
            0.5,
            3500.0,
            "nominal gelatin-like soft cube demo; verified=false",
        ),
        entry(
            "steel_nominal",
            7850.0,
            2.0e11,
            0.30,
            0.4,
            0.2,
            2.5e8,
            50.0,
            500.0,
            "nominal structural steel order-of-magnitude; verified=false",
        ),
        entry(
            "aluminum_nominal",
            2700.0,
            7.0e10,
            0.33,
            0.5,
            0.3,
            1.0e8,
            200.0,
            900.0,
            "nominal aluminum alloy order-of-magnitude; verified=false",
        ),
        entry(
            "wood_pine_nominal",
            500.0,
            1.0e10,
            0.35,
            0.5,
            0.2,
            4.0e7,
            0.12,
            1700.0,
            "nominal softwood along-grain order-of-magnitude; verified=false",
        ),
        entry(
            "concrete_nominal",
            2400.0,
            3.0e10,
            0.20,
            0.7,
            0.1,
            3.0e7,
            1.4,
            880.0,
            "nominal concrete order-of-magnitude; verified=false",
        ),
    ]
}

/// Look up a built-in material by exact name.
pub fn material_by_name(name: &str) -> Option<Material> {
    builtin_materials().into_iter().find(|m| m.name == name)
}

/// Bulk modulus K = λ + 2μ/3 from E, ν.
pub fn bulk_modulus(e: f64, nu: f64) -> Option<f64> {
    let lambda = lame_lambda(e, nu)?;
    let mu = lame_mu(e, nu)?;
    Some(lambda + (2.0 / 3.0) * mu)
}

/// Wave speed estimate √(E/ρ) [m/s] (1-D bar; order-of-magnitude only).
pub fn bar_wave_speed(e: f64, density: f64) -> Option<f64> {
    if e <= 0.0 || density <= 0.0 {
        return None;
    }
    Some(libm_ops::sqrt(e / density))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn lame_from_e_nu_steel_nominal() {
        let e = 2.0e11;
        let nu = 0.3;
        let mu = lame_mu(e, nu).unwrap();
        let lambda = lame_lambda(e, nu).unwrap();
        assert_relative_eq!(mu, e / (2.0 * 1.3), epsilon = 1e-12);
        assert_relative_eq!(
            lambda,
            e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
            epsilon = 1e-9
        );
        // Invert: E = μ(3λ+2μ)/(λ+μ), ν = λ/(2(λ+μ))
        let e_back = mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
        let nu_back = lambda / (2.0 * (lambda + mu));
        assert_relative_eq!(e_back, e, epsilon = 1e-6);
        assert_relative_eq!(nu_back, nu, epsilon = 1e-12);
    }

    #[test]
    fn stretch_compliance_is_inverse_stiffness() {
        let e = 1.0e6;
        let l = 0.5;
        let a = 0.01;
        let k = stretch_stiffness(e, l, a).unwrap();
        let alpha = stretch_compliance(e, l, a).unwrap();
        assert_relative_eq!(alpha, 1.0 / k, epsilon = 1e-15);
        assert_relative_eq!(k, e * a / l, epsilon = 1e-12);
    }

    #[test]
    fn zero_compliance_is_rigid() {
        assert!(stiffness_from_compliance(0.0).unwrap().is_infinite());
    }

    #[test]
    fn poisson_half_rejects_lambda() {
        assert!(lame_lambda(1e6, 0.5).is_none());
    }

    #[test]
    fn all_builtin_unverified() {
        for m in builtin_materials() {
            assert!(!m.verified, "{} should be unverified", m.name);
            assert!(m.density > 0.0);
            assert!(m.youngs_modulus > 0.0);
        }
    }

    #[test]
    fn alpha_tilde_scales_with_dt() {
        let a = alpha_tilde(1e-3, 0.01);
        assert_relative_eq!(a, 1e-3 / 1e-4, epsilon = 1e-12);
    }
}
