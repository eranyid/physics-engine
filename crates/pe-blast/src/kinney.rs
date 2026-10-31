//! Kinney & Graham (1985) closed-form free-air peak overpressure.
//!
//! # Reference
//! Kinney, G. F. & Graham, K. J. (1985). *Explosive Shocks in Air*, 2nd ed.
//! Springer-Verlag. Empirical fit (SI form used widely in engineering blast
//! load estimation):
//!
//! ```text
//! p_s / p_0 = 808 · [1 + (Z/4.5)²]
//!           / √{ [1+(Z/0.048)²] [1+(Z/0.32)²] [1+(Z/1.35)²] }
//! ```
//!
//! with Hopkinson–Cranz `Z` [m·kg^{-1/3}] and ambient pressure `p_0` [Pa].
//! `p_s` is **incident peak overpressure** [Pa] (gauge).

use pe_math::libm_ops;

/// Default sea-level ambient pressure [Pa] (ISA).
pub const P0_DEFAULT: f64 = 101_325.0;

/// Peak overpressure ratio `p_s / p_0` (dimensionless) at scaled distance `z`
/// [m·kg^{-1/3}] from the Kinney–Graham formula.
pub fn peak_overpressure_ratio(z: f64) -> f64 {
    let z = z.max(0.0);
    let num = 808.0 * (1.0 + (z / 4.5) * (z / 4.5));
    let d1 = 1.0 + (z / 0.048) * (z / 0.048);
    let d2 = 1.0 + (z / 0.32) * (z / 0.32);
    let d3 = 1.0 + (z / 1.35) * (z / 1.35);
    num / libm_ops::sqrt(d1 * d2 * d3)
}

/// Peak overpressure `p_s` [Pa] at scaled distance `z` [m·kg^{-1/3}] with
/// ambient `p0` [Pa].
pub fn peak_overpressure(z: f64, p0: f64) -> f64 {
    peak_overpressure_ratio(z) * p0
}

/// Peak overpressure [Pa] from range and TNT-equivalent mass.
///
/// - `range_m` [m], `w_eq_kg` [kg], `p0` [Pa].
pub fn peak_overpressure_at(range_m: f64, w_eq_kg: f64, p0: f64) -> Option<f64> {
    let z = crate::scaling::scaled_distance(range_m, w_eq_kg, 1.0)?;
    Some(peak_overpressure(z, p0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn formula_z1_and_z10() {
        // Recomputed from the printed formula (not from a chart).
        // Z=1 → ≈ 1008.79 kPa; Z=10 → ≈ 9.985 kPa at p0 = 101.325 kPa.
        let p0_kpa = 101.325;
        let p0 = p0_kpa * 1e3;
        let ps1 = peak_overpressure(1.0, p0) / 1e3; // kPa
        let ps10 = peak_overpressure(10.0, p0) / 1e3;
        assert_relative_eq!(ps1, 1008.7895037932, epsilon = 1e-8);
        assert_relative_eq!(ps10, 9.9853637227, epsilon = 1e-8);
    }
}
