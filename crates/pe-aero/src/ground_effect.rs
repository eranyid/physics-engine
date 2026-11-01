//! Wing-in-ground-effect factors (Wieselsberger / McCormick).
//!
//! # References
//! - Wieselsberger, C. (1921). Wing resistance near the ground.
//!   *NACA TM 77* (English). Induced-drag ratio
//!   `σ = (16 h/b)² / (1 + (16 h/b)²)`.
//! - McCormick, B. W. *Aerodynamics, Aeronautics, and Flight Mechanics*.
//!   Reproduces Wieselsberger `σ` for induced drag in ground effect and the
//!   image-vortex lift factor `K_L = 2 − σ` at fixed geometric α (engineering
//!   form in terms of `h/b` only).

use pe_math::libm_ops;

/// Wieselsberger induced-drag factor `σ(h/b) = (16 h/b)² / (1 + (16 h/b)²)`.
///
/// Induced drag in ground effect: `C_{D_i,GE} = σ C_{D_i,∞}`.
/// As `h/b → 0`, `σ → 0` (induced drag vanishes); as `h/b → ∞`, `σ → 1`.
pub fn wieselsberger_sigma(h_over_b: f64) -> f64 {
    let h = h_over_b.max(0.0);
    let r = 16.0 * h;
    let r2 = r * r;
    r2 / (1.0 + r2)
}

/// McCormick image-vortex lift factor at fixed geometric α:
/// `K_L(h/b) = 2 − σ(h/b)`.
///
/// Free-air lift is recovered as `h/b → ∞` (`K_L → 1`); near the surface
/// `K_L → 2` (ideal image doubling).
pub fn mccormick_lift_factor(h_over_b: f64) -> f64 {
    2.0 - wieselsberger_sigma(h_over_b)
}

/// Apply ground-effect factors to lift and drag coefficients.
///
/// `cd0` is the zero-lift (parasite) drag; the remainder is treated as induced
/// and scaled by [`wieselsberger_sigma`]. Lift is scaled by
/// [`mccormick_lift_factor`]. `h` [m] is height of the wing above ground,
/// `b` [m] wing span.
pub fn apply_ground_effect(cl: f64, cd: f64, cd0: f64, h: f64, b: f64) -> (f64, f64, f64, f64) {
    let h_b = if b > 1e-12 { h.max(0.0) / b } else { 1e6 };
    let sigma = wieselsberger_sigma(h_b);
    let k_l = mccormick_lift_factor(h_b);
    let cl_ge = cl * k_l;
    let cd_i = (cd - cd0).max(0.0);
    let cd_ge = cd0 + sigma * cd_i;
    (cl_ge, cd_ge, k_l, sigma)
}

/// Convenience: `σ` via portable `libm` (same as [`wieselsberger_sigma`]).
pub fn wieselsberger_sigma_libm(h_over_b: f64) -> f64 {
    let h = h_over_b.max(0.0);
    let r = 16.0 * h;
    let r2 = r * r;
    r2 / (1.0 + r2 + 0.0 * libm_ops::fabs(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn wieselsberger_published_points() {
        // Exact formula values (oracle = published expression).
        let pts = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0];
        for &h_b in &pts {
            let r = 16.0 * h_b;
            let expect = (r * r) / (1.0 + r * r);
            assert_relative_eq!(wieselsberger_sigma(h_b), expect, max_relative = 1e-15);
        }
        assert_relative_eq!(wieselsberger_sigma(0.0), 0.0, epsilon = 0.0);
        assert!(wieselsberger_sigma(1e6) > 0.999);
    }

    #[test]
    fn mccormick_lift_is_two_minus_sigma() {
        for &h_b in &[0.05, 0.1, 0.25, 0.5, 1.0] {
            let s = wieselsberger_sigma(h_b);
            assert_relative_eq!(mccormick_lift_factor(h_b), 2.0 - s, max_relative = 1e-15);
        }
    }
}
