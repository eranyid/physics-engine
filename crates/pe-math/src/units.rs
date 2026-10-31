//! SI constants (CODATA / scipy.constants values). No unit conversion in the engine.

/// Newtonian gravitational constant `G` [m³ kg⁻¹ s⁻²] (CODATA 2018 / scipy).
pub const G: f64 = 6.674_30e-11;

/// Standard acceleration of gravity `g₀` [m/s²] (exact by definition for this project: 9.80665).
pub const STANDARD_GRAVITY: f64 = 9.806_65;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_codata_scipy() {
        assert!((G - 6.67430e-11).abs() / 6.67430e-11 < 1e-6);
        assert_eq!(STANDARD_GRAVITY, 9.80665);
    }
}
