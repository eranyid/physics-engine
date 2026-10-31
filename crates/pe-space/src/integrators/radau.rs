//! Gauss–Radau abscissae for the 15th-order IAS15 scheme.
//!
//! # Derivation (clean-room)
//!
//! Rein & Spiegel (2015), arXiv:1409.4779 / MNRAS 446, 1424, §2.1: eight force
//! evaluations on `[0, dt]` including the left endpoint `h = 0`, with the seven
//! free abscissae taken from **Gauss–Radau** quadrature.
//!
//! On the reference interval `[-1, 1]`, the `n`-point Radau nodes that include
//! the left endpoint `x = −1` are the roots of
//!
//! ```text
//! P_n(x) + P_{n−1}(x) = 0
//! ```
//!
//! where `P_k` is the degree-`k` Legendre polynomial (Hildebrand, *Introduction
//! to Numerical Analysis*; standard Gauss–Radau construction). Mapping
//! `h = (x + 1) / 2` sends these roots to `[0, 1]`.
//!
//! For IAS15, `n = 8` (seven free nodes). We compute the roots by Newton
//! iteration on `P_8 + P_7` (portable `libm` only) and expose them as
//! [`RADAU_H`]. Unit tests re-solve the node equations and check agreement to
//! `1e-15`.
//!
//! **Do not** copy constants or code from REBOUND (GPL).

use pe_math::libm_ops;

/// Number of free Radau abscissae (excluding `h = 0`).
pub const N_RADAU_FREE: usize = 7;

/// Total force evaluations per accepted step (`h = 0` plus free nodes).
pub const N_RADAU: usize = N_RADAU_FREE + 1;

/// Gauss–Radau spacings `h_n ∈ [0, 1]` for IAS15, including `h_0 = 0`.
///
/// Indexed `0..8`. Values are computed once via [`compute_radau_nodes`] at
/// compile-friendly precision and locked by the node-equation tests.
pub const RADAU_H: [f64; N_RADAU] = [
    0.0,
    0.056_262_560_536_922_15,
    0.180_240_691_736_892_37,
    0.352_624_717_113_169_6,
    0.547_153_626_330_555_4,
    0.734_210_177_215_410_5,
    0.885_320_946_839_095_8,
    0.977_520_613_561_287_5,
];

/// Legendre `P_n(x)` and `P_n'(x)` via the three-term recurrence.
fn legendre_p_dp(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    let mut p0 = 1.0;
    let mut p1 = x;
    let mut dp0 = 0.0;
    let mut dp1 = 1.0;
    for k in 1..n {
        let kf = k as f64;
        let p2 = (((2.0 * kf + 1.0) * x * p1) - (kf * p0)) / (kf + 1.0);
        let dp2 = (((2.0 * kf + 1.0) * (p1 + x * dp1)) - (kf * dp0)) / (kf + 1.0);
        p0 = p1;
        p1 = p2;
        dp0 = dp1;
        dp1 = dp2;
    }
    (p1, dp1)
}

fn radau_f_df(n: usize, x: f64) -> (f64, f64) {
    let (pn, dpn) = legendre_p_dp(n, x);
    let (pn1, dpn1) = legendre_p_dp(n - 1, x);
    (pn + pn1, dpn + dpn1)
}

/// Compute the eight Gauss–Radau nodes on `[0, 1]` for `n = 8` (including 0).
///
/// Solves `P_8(x) + P_7(x) = 0` on `(-1, 1]` by Newton, then maps
/// `h = (x + 1) / 2`. Used by tests to verify [`RADAU_H`] from the defining
/// equations rather than from remembered constants.
pub fn compute_radau_nodes() -> [f64; N_RADAU] {
    const N: usize = 8;
    // Chebyshev-like starters in (-1, 1).
    let mut roots_vec: Vec<f64> = Vec::with_capacity(N_RADAU_FREE);
    for i in 0..N_RADAU_FREE {
        let seed = libm_ops::cos(core::f64::consts::PI * ((i as f64) + 0.75) / (N as f64));
        let mut x = seed;
        for _ in 0..200 {
            let (f, df) = radau_f_df(N, x);
            if libm_ops::fabs(df) < 1e-30 {
                break;
            }
            let dx = f / df;
            x -= dx;
            if libm_ops::fabs(dx) < 1e-18 {
                break;
            }
        }
        if x > -1.0 + 1e-12 {
            let mut insert_at = roots_vec.len();
            let mut duplicate = false;
            for (j, &rj) in roots_vec.iter().enumerate() {
                if libm_ops::fabs(rj - x) < 1e-12 {
                    duplicate = true;
                    break;
                }
                if x < rj {
                    insert_at = j;
                    break;
                }
            }
            if !duplicate {
                roots_vec.insert(insert_at, x);
            }
        }
    }
    debug_assert_eq!(roots_vec.len(), N_RADAU_FREE);
    let mut h = [0.0; N_RADAU];
    h[0] = 0.0;
    for i in 0..N_RADAU_FREE {
        h[i + 1] = 0.5 * (roots_vec[i] + 1.0);
    }
    h
}

/// Evaluate `P_n(x) + P_{n-1}(x)` (for tests).
pub fn radau_polynomial(n: usize, x: f64) -> f64 {
    radau_f_df(n, x).0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn radau_nodes_match_node_equations_to_1e15() {
        let h = compute_radau_nodes();
        // Left endpoint
        assert_eq!(h[0], 0.0);
        // Interior / right nodes satisfy P_8(x)+P_7(x)=0 with x = 2h-1
        for i in 1..N_RADAU {
            let x = 2.0 * h[i] - 1.0;
            let f = radau_polynomial(8, x);
            assert!(libm_ops::fabs(f) < 1e-14, "node {i}: P8+P7({x}) = {f}");
        }
        // Locked table agrees with freshly computed nodes to 1e-15
        for i in 0..N_RADAU {
            assert_abs_diff_eq!(h[i], RADAU_H[i], epsilon = 1e-15);
        }
    }

    #[test]
    fn radau_nodes_monotone_in_unit_interval() {
        for i in 0..N_RADAU - 1 {
            assert!(RADAU_H[i] < RADAU_H[i + 1]);
            assert!(RADAU_H[i] >= 0.0);
            assert!(RADAU_H[i + 1] <= 1.0);
        }
    }
}
