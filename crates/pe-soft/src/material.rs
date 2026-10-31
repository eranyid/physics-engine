//! Continuum / contact material parameters and XPBD compliance mapping.
//!
//! Every table entry is marked `verified: false` unless a primary source was
//! actually fetched for that constant. Values are **nominal** engineering
//! placeholders suitable for demos and unit tests — not certified material data.
//! See [`docs/MATERIALS.md`](../../../docs/MATERIALS.md).

use pe_math::libm_ops;

/// Continuum + contact + thermal / plasticity / fracture material parameters (SI).
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
    /// Initial yield strength σ_y [Pa] (J2 plasticity).
    pub yield_strength: f64,
    /// Linear isotropic hardening modulus H [Pa] (`σ_y_eff = σ_y + H α`).
    pub hardening_modulus: f64,
    /// Thermal conductivity k [W/(m·K)].
    pub thermal_conductivity: f64,
    /// Specific heat c_p [J/(kg·K)].
    pub specific_heat: f64,
    /// Linear thermal expansion coefficient α [1/K].
    pub thermal_expansion: f64,
    /// Fracture max-principal stress threshold [Pa] (0 = unused).
    pub fracture_stress: f64,
    /// Fracture max-principal strain threshold [-] (0 = unused).
    pub fracture_strain: f64,
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

    /// J2 parameter view.
    pub fn j2_params(&self) -> crate::plasticity::J2Params {
        crate::plasticity::J2Params {
            youngs_modulus: self.youngs_modulus,
            poisson_ratio: self.poisson_ratio,
            yield_strength: self.yield_strength,
            hardening_modulus: self.hardening_modulus,
        }
    }

    /// Thermal property view.
    pub fn thermal_props(&self) -> crate::thermal::ThermalProps {
        crate::thermal::ThermalProps {
            density: self.density,
            specific_heat: self.specific_heat,
            conductivity: self.thermal_conductivity,
            expansion: self.thermal_expansion,
        }
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
    hardening: f64,
    k_th: f64,
    cp: f64,
    alpha: f64,
    frac_stress: f64,
    frac_strain: f64,
    verified: bool,
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
        hardening_modulus: hardening,
        thermal_conductivity: k_th,
        specific_heat: cp,
        thermal_expansion: alpha,
        fracture_stress: frac_stress,
        fracture_strain: frac_strain,
        verified,
        source_note: note,
    }
}

/// Small built-in table of materials. Order is fixed for determinism.
///
/// Unless `verified: true`, treat numbers as **nominal** demo orders of magnitude.
/// Citations in `source_note` / [`docs/MATERIALS.md`](../../../docs/MATERIALS.md).
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
            1.0e5,
            0.15,
            1800.0,
            2.0e-4,
            1.0e7,
            1.0,
            false,
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
            5.0e5,
            0.16,
            1700.0,
            1.5e-4,
            3.0e7,
            0.8,
            false,
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
            5.0e4,
            0.2,
            1400.0,
            3.0e-4,
            5.0e6,
            1.2,
            false,
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
            1.0e6,
            0.04,
            1300.0,
            1.0e-5,
            2.0e7,
            0.05,
            false,
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
            5.0e7,
            0.25,
            1700.0,
            8.0e-5,
            8.0e7,
            0.15,
            false,
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
            1.0e4,
            0.5,
            3500.0,
            1.0e-4,
            2.0e5,
            0.5,
            false,
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
            2.0e9,
            50.0,
            500.0,
            12.0e-6,
            4.0e8,
            0.2,
            false,
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
            1.0e9,
            200.0,
            900.0,
            23.0e-6,
            1.5e8,
            0.15,
            false,
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
            5.0e8,
            0.12,
            1700.0,
            5.0e-6,
            5.0e7,
            0.01,
            false,
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
            0.0,
            1.4,
            880.0,
            10.0e-6,
            3.0e6,
            1e-4,
            false,
            "nominal concrete order-of-magnitude; verified=false",
        ),
        // --- additions (M10 materials track) ---
        entry(
            "copper_nominal",
            8960.0,
            1.1e11,
            0.34,
            0.4,
            0.2,
            7.0e7,
            1.0e9,
            400.0,
            385.0,
            16.5e-6,
            2.0e8,
            0.4,
            false,
            "nominal Cu order-of-magnitude (k≈400 W/m·K, α≈16.5e-6 textbook); verified=false",
        ),
        entry(
            "glass_soda_lime_nominal",
            2500.0,
            7.0e10,
            0.23,
            0.5,
            0.05,
            5.0e7,
            0.0,
            1.0,
            840.0,
            9.0e-6,
            5.0e7,
            1e-3,
            false,
            "nominal soda-lime glass order-of-magnitude; verified=false",
        ),
        entry(
            "titanium_ti6al4v_nominal",
            4430.0,
            1.14e11,
            0.34,
            0.4,
            0.2,
            8.8e8,
            1.0e9,
            6.7,
            526.0,
            9.0e-6,
            9.5e8,
            0.1,
            false,
            "nominal Ti-6Al-4V order-of-magnitude; verified=false",
        ),
        entry(
            "lead_nominal",
            11340.0,
            1.6e10,
            0.44,
            0.4,
            0.05,
            1.2e7,
            1.0e8,
            35.0,
            130.0,
            29.0e-6,
            1.5e7,
            0.3,
            false,
            "nominal lead order-of-magnitude; verified=false",
        ),
        entry(
            "ice_nominal",
            917.0,
            9.0e9,
            0.33,
            0.05,
            0.1,
            1.0e6,
            0.0,
            2.2,
            2100.0,
            5.0e-5,
            1.5e6,
            1e-3,
            false,
            "nominal ice Ih order-of-magnitude near 0 °C; verified=false",
        ),
        entry(
            "polycarbonate_nominal",
            1200.0,
            2.3e9,
            0.37,
            0.4,
            0.4,
            6.0e7,
            5.0e7,
            0.20,
            1200.0,
            65.0e-6,
            7.0e7,
            0.8,
            false,
            "nominal polycarbonate order-of-magnitude; verified=false",
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
    fn all_builtin_unverified_unless_flagged() {
        for m in builtin_materials() {
            if !m.verified {
                assert!(m.density > 0.0);
                assert!(m.youngs_modulus > 0.0);
            }
            assert!(!m.name.is_empty());
        }
        assert!(material_by_name("copper_nominal").is_some());
        assert!(material_by_name("polycarbonate_nominal").is_some());
    }

    #[test]
    fn alpha_tilde_scales_with_dt() {
        let a = alpha_tilde(1e-3, 0.01);
        assert_relative_eq!(a, 1e-3 / 1e-4, epsilon = 1e-12);
    }
}
