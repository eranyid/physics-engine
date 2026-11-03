//! Hopkinson–Cranz cube-root scaled distance.
//!
//! # References
//! - Hopkinson, B. (1915). British Ordnance Board Minutes 13565 (cube-root scaling).
//! - Cranz, C. (1926). *Lehrbuch der Ballistik*. Springer.
//! - Kinney, G. F. & Graham, K. J. (1985). *Explosive Shocks in Air* (2nd ed.).
//!   Springer — engineering use of `Z = R / W^{1/3}` with TNT-equivalent mass.
//!
//! TNT-equivalence is a **caller-supplied factor** only. This module does not
//! contain explosive recipes, chemistry, or manufacture details.

use pe_math::libm_ops;

use crate::error::{require_non_negative, require_positive, BlastError};

/// Hopkinson–Cranz scaled distance `Z = R / W_eq^{1/3}` [m · kg^{-1/3}].
///
/// - `range_m`: standoff distance `R` [m] (≥ 0).
/// - `charge_mass_kg`: actual charge mass `W` [kg] (> 0).
/// - `tnt_equivalence`: dimensionless factor so `W_eq = tnt_equivalence · W`
///   (TNT-equivalent mass [kg]). Must be > 0.
///
/// Returns `None` if inputs are non-physical (legacy API). Prefer [`try_scaled_distance`].
pub fn scaled_distance(range_m: f64, charge_mass_kg: f64, tnt_equivalence: f64) -> Option<f64> {
    try_scaled_distance(range_m, charge_mass_kg, tnt_equivalence).ok()
}

/// Fallible Hopkinson–Cranz scaled distance with typed [`BlastError`].
pub fn try_scaled_distance(
    range_m: f64,
    charge_mass_kg: f64,
    tnt_equivalence: f64,
) -> Result<f64, BlastError> {
    require_non_negative(range_m, "range_m")?;
    require_positive(charge_mass_kg, "charge_mass_kg")?;
    require_positive(tnt_equivalence, "tnt_equivalence")?;
    let w_eq = charge_mass_kg * tnt_equivalence;
    Ok(range_m / libm_ops::cbrt(w_eq))
}

/// Equivalent TNT mass [kg]: `W_eq = tnt_equivalence · W`.
///
/// - `charge_mass_kg` [kg], `tnt_equivalence` [dimensionless].
pub fn equivalent_tnt_mass(charge_mass_kg: f64, tnt_equivalence: f64) -> Option<f64> {
    try_equivalent_tnt_mass(charge_mass_kg, tnt_equivalence).ok()
}

/// Fallible TNT-equivalent mass with typed [`BlastError`].
pub fn try_equivalent_tnt_mass(
    charge_mass_kg: f64,
    tnt_equivalence: f64,
) -> Result<f64, BlastError> {
    require_positive(charge_mass_kg, "charge_mass_kg")?;
    require_positive(tnt_equivalence, "tnt_equivalence")?;
    Ok(charge_mass_kg * tnt_equivalence)
}

/// Standoff [m] from scaled distance: `R = Z · W_eq^{1/3}`.
pub fn range_from_scaled(z: f64, w_eq_kg: f64) -> Option<f64> {
    try_range_from_scaled(z, w_eq_kg).ok()
}

/// Fallible standoff from scaled distance with typed [`BlastError`].
pub fn try_range_from_scaled(z: f64, w_eq_kg: f64) -> Result<f64, BlastError> {
    require_non_negative(z, "z")?;
    require_positive(w_eq_kg, "w_eq_kg")?;
    Ok(z * libm_ops::cbrt(w_eq_kg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn unit_charge_unit_range() {
        let z = scaled_distance(1.0, 1.0, 1.0).unwrap();
        assert_relative_eq!(z, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn cube_root_scaling() {
        // 8 kg → cbrt = 2; R=2 → Z=1
        let z = scaled_distance(2.0, 8.0, 1.0).unwrap();
        assert_relative_eq!(z, 1.0, epsilon = 1e-14);
    }
}
