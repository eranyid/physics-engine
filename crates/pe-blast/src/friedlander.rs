//! Modified Friedlander blast pressure–time waveform.
//!
//! # References
//! - Friedlander, F. G. (1946). The diffraction of sound pulses. *Proc. R. Soc. A*.
//! - Kinney & Graham (1985). *Explosive Shocks in Air* — engineering form
//!   `p(t) = p_s (1 − t/t_d) exp(−b t / t_d)` for the positive phase.
//!
//! Units: `p_s` [Pa], `t`/`t_d` [s], `b` dimensionless decay.

use pe_math::libm_ops;

/// Friedlander positive-phase overpressure [Pa] at time `t` [s] after arrival.
///
/// - `p_s`: peak overpressure [Pa]
/// - `t_d`: positive-phase duration [s] (> 0)
/// - `b`: decay parameter [dimensionless] (≥ 0); `b = 0` → linear triangle
///
/// Returns 0 for `t < 0` or `t > t_d` (positive phase only).
pub fn pressure(t: f64, p_s: f64, t_d: f64, b: f64) -> f64 {
    if t < 0.0 || t_d <= 0.0 || t > t_d {
        return 0.0;
    }
    let tau = t / t_d;
    p_s * (1.0 - tau) * libm_ops::exp(-b * tau)
}

/// Analytic positive impulse `I⁺ = ∫_0^{t_d} p(t) dt` [Pa·s].
///
/// Closed form:
/// - `b = 0`: `I = p_s t_d / 2`
/// - `b ≠ 0`: `I = p_s t_d (b − 1 + e^{−b}) / b²`
pub fn positive_impulse(p_s: f64, t_d: f64, b: f64) -> f64 {
    if t_d <= 0.0 {
        return 0.0;
    }
    if libm_ops::fabs(b) < 1e-14 {
        return 0.5 * p_s * t_d;
    }
    // ∫_0^1 (1−τ) e^{−b τ} dτ = (b − 1 + e^{−b}) / b²
    let integral_tau = (b - 1.0 + libm_ops::exp(-b)) / (b * b);
    p_s * t_d * integral_tau
}

/// Sample the waveform on a uniform grid; returns `(times_s, pressures_pa)`.
pub fn sample(p_s: f64, t_d: f64, b: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let n = n.max(2);
    let mut ts = Vec::with_capacity(n);
    let mut ps = Vec::with_capacity(n);
    for i in 0..n {
        let t = t_d * (i as f64) / ((n - 1) as f64);
        ts.push(t);
        ps.push(pressure(t, p_s, t_d, b));
    }
    (ts, ps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn impulse_b0_triangle() {
        let i = positive_impulse(1000.0, 0.01, 0.0);
        assert_relative_eq!(i, 5.0, epsilon = 1e-12);
    }

    #[test]
    fn impulse_matches_numeric_quad() {
        let p_s = 2.5e5;
        let t_d = 0.02;
        let b = 1.3;
        let analytic = positive_impulse(p_s, t_d, b);
        // Trapezoid on fine grid
        let n = 10_000;
        let dt = t_d / (n as f64);
        let mut num = 0.0;
        for k in 0..=n {
            let t = k as f64 * dt;
            let w = if k == 0 || k == n { 0.5 } else { 1.0 };
            num += w * pressure(t, p_s, t_d, b);
        }
        num *= dt;
        assert_relative_eq!(analytic, num, epsilon = 1e-6 * analytic.abs().max(1.0));
    }

    #[test]
    fn peak_at_arrival() {
        assert_relative_eq!(pressure(0.0, 1e5, 0.01, 1.0), 1e5, epsilon = 0.0);
        assert_eq!(pressure(0.01, 1e5, 0.01, 1.0), 0.0);
    }
}
