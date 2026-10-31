//! Gurney fragment velocity and Mott fragment-mass distribution.
//!
//! # References
//! - Gurney, R. W. (1943). The initial velocities of fragments from bombs,
//!   shells, and grenades. BRL Report 405 — energy methods with caller-supplied
//!   Gurney constant `√(2E)` [m/s]. **No explosive chemistry or recipes.**
//! - Mott, N. F. (1947). Fragmentation of shell cases. *Proc. R. Soc. A* —
//!   exponential mass distribution sampling.
//!
//! RNG: [`pe_math::Xoshiro256StarStar`] (shared with pe-aero / pe-hydro).

use pe_math::libm_ops;
use pe_math::Xoshiro256StarStar;

/// Charge-to-metal mass ratio geometries for Gurney.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GurneyGeometry {
    /// Flat sandwich / open-faced slab: `V = √(2E) · √(C/M / (1 + C/M))`.
    Flat,
    /// Cylinder: `V = √(2E) · √(C/M / (1 + ½ C/M))`.
    Cylindrical,
    /// Sphere: `V = √(2E) · √(C/M / (1 + ⅗ C/M))`.
    Spherical,
}

/// Gurney fragment velocity [m/s].
///
/// - `gurney_sqrt_2e` = `√(2E)` [m/s] — **caller-supplied** Gurney energy
///   parameter (not derived from compositions here).
/// - `charge_to_mass` = `C/M` [dimensionless] (> 0).
pub fn gurney_velocity(
    gurney_sqrt_2e: f64,
    charge_to_mass: f64,
    geometry: GurneyGeometry,
) -> Option<f64> {
    if !(gurney_sqrt_2e > 0.0 && charge_to_mass > 0.0) {
        return None;
    }
    let cm = charge_to_mass;
    let factor = match geometry {
        GurneyGeometry::Flat => cm / (1.0 + cm),
        GurneyGeometry::Cylindrical => cm / (1.0 + 0.5 * cm),
        GurneyGeometry::Spherical => cm / (1.0 + 0.6 * cm),
    };
    Some(gurney_sqrt_2e * libm_ops::sqrt(factor))
}

/// Mott fragment mass distribution: `N(>m) ∝ exp(−√(m/μ))` style sampling.
///
/// Samples a fragment mass [kg] given characteristic mass `mu_kg` [kg] using
/// the inverse-CDF of an exponential-of-sqrt law:
/// `m = μ · [−ln(1 − U)]²` with `U ~ Uniform(0,1)`.
///
/// Reference: Mott (1947); engineering form used in fragmenting-case models.
pub fn mott_sample_mass(rng: &mut Xoshiro256StarStar, mu_kg: f64) -> f64 {
    let u = rng.next_f64().clamp(1e-12, 1.0 - 1e-12);
    let x = -libm_ops::log(1.0 - u);
    mu_kg * x * x
}

/// Sample `n` fragment masses [kg] (deterministic given seed).
pub fn mott_sample_many(seed: u64, mu_kg: f64, n: usize) -> Vec<f64> {
    let mut rng = Xoshiro256StarStar::seed(seed);
    (0..n).map(|_| mott_sample_mass(&mut rng, mu_kg)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn flat_equals_sqrt_half_when_cm_one() {
        // C/M=1 → √(1/2)=√0.5
        let v = gurney_velocity(1000.0, 1.0, GurneyGeometry::Flat).unwrap();
        assert_relative_eq!(v, 1000.0 * libm_ops::sqrt(0.5), epsilon = 1e-12);
    }

    #[test]
    fn mott_seed_deterministic() {
        let a = mott_sample_many(7, 0.01, 16);
        let b = mott_sample_many(7, 0.01, 16);
        assert_eq!(a, b);
    }
}
