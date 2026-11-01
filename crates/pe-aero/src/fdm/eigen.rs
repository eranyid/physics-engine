//! Eigenvalues of small real matrices (longitudinal 4×4) via characteristic
//! polynomial + Bairstow quadratic factoring.
//!
//! Portable `libm` only; no external linear-algebra crates.

use pe_math::libm_ops;

/// Complex eigenvalue `re + i im`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexEigen {
    /// Real part [1/s for dynamics matrices].
    pub re: f64,
    /// Imaginary part [1/s].
    pub im: f64,
}

impl ComplexEigen {
    /// Undamped natural frequency `ω_n = |λ|` [rad/s] for oscillatory modes.
    pub fn omega_n(self) -> f64 {
        libm_ops::sqrt(self.re * self.re + self.im * self.im)
    }

    /// Damping ratio `ζ = −re / ω_n` (positive ⇒ stable spiral-in).
    pub fn zeta(self) -> f64 {
        let w = self.omega_n();
        if w < 1e-15 {
            return 0.0;
        }
        -self.re / w
    }
}

/// Classified longitudinal modes from a 4×4 `A` matrix.
#[derive(Clone, Copy, Debug)]
pub struct LongitudinalModes {
    /// Phugoid (longer period, lighter damping).
    pub phugoid: ComplexEigen,
    /// Short-period (higher frequency, heavier damping).
    pub short_period: ComplexEigen,
    /// All four eigenvalues.
    pub all: [ComplexEigen; 4],
}

/// Faddeev–LeVerrier: monic char poly `det(sI − A) = s⁴ + c₃ s³ + c₂ s² + c₁ s + c₀`.
fn charpoly_4(a: &[[f64; 4]; 4]) -> [f64; 4] {
    // Returns [c3, c2, c1, c0].
    let mut m = *a;
    let mut c = [0.0; 4];
    for k in 1..=4 {
        let mut tr = 0.0;
        for i in 0..4 {
            tr += m[i][i];
        }
        c[k - 1] = -tr / k as f64;
        // M ← A (M + c_{k-1} I)
        let mut next = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for t in 0..4 {
                    let mid = m[t][j] + if t == j { c[k - 1] } else { 0.0 };
                    s += a[i][t] * mid;
                }
                next[i][j] = s;
            }
        }
        m = next;
    }
    c
}

/// Roots of `s² − p s − q = 0` (note sign convention of Bairstow factor).
fn quad_roots(p: f64, q: f64) -> (ComplexEigen, ComplexEigen) {
    // s² - p s - q = 0 ⇒ s = (p ± √(p²+4q))/2
    let disc = p * p + 4.0 * q;
    if disc >= 0.0 {
        let s = libm_ops::sqrt(disc);
        (
            ComplexEigen {
                re: 0.5 * (p + s),
                im: 0.0,
            },
            ComplexEigen {
                re: 0.5 * (p - s),
                im: 0.0,
            },
        )
    } else {
        let s = libm_ops::sqrt((-disc).max(0.0));
        (
            ComplexEigen {
                re: 0.5 * p,
                im: 0.5 * s,
            },
            ComplexEigen {
                re: 0.5 * p,
                im: -0.5 * s,
            },
        )
    }
}

/// Bairstow: factor monic quartic `s⁴ + a s³ + b s² + c s + d` into two real quadratics.
fn bairstow_quartic(a: f64, b: f64, c: f64, d: f64) -> [ComplexEigen; 4] {
    // Factor as (s² − p s − q)(s² − r s − t) with p+r=−a, …
    // Iterate on (u,v) for divisor s² − u s − v.
    let mut u = -a / 2.0;
    let mut v = if b.abs() > 1e-12 { b / 2.0 } else { 1.0 };
    for _ in 0..80 {
        // Synthetic division of s⁴+a s³+b s²+c s+d by s² − u s − v.
        // b0 = 1
        // b1 = a + u
        // b2 = b + u b1 + v
        // b3 = c + u b2 + v b1
        // b4 = d + u b3 + v b2   (remainder b3 s + b4 if not exact; for quadratic
        // remainder we use the standard Bairstow b,c arrays)
        let b0 = 1.0;
        let b1 = a + u * b0;
        let b2 = b + u * b1 + v * b0;
        let b3 = c + u * b2 + v * b1;
        let b4 = d + u * b3 + v * b2;
        // Partial derivatives for Newton on (b3,b4)→0
        let c0 = 1.0;
        let c1 = b1 + u * c0;
        let c2 = b2 + u * c1 + v * c0;
        let c3 = b3 + u * c2 + v * c1;
        // ∂(b3,b4)/∂(u,v):
        // J = [[c2, c1], [c3, c2]]
        let det = c2 * c2 - c1 * c3;
        if det.abs() < 1e-30 {
            break;
        }
        let du = (c2 * (-b3) - c1 * (-b4)) / det;
        let dv = (c2 * (-b4) - c3 * (-b3)) / det;
        u += du;
        v += dv;
        if du.abs() + dv.abs() < 1e-14 {
            break;
        }
        let _ = (b0, b4);
    }
    // Quotient s² + (a+u)s + (b + u(a+u) + v)  and divisor s² − u s − v.
    // From synthetic division the quotient coeffs are b0,b1,b2 for s²+b1 s+b2
    // with our b0=1, so quotient = s² + b1 s + b2, i.e. s² − (−b1)s − (−b2).
    let b1 = a + u;
    let b2 = b + u * b1 + v;
    let (r1, r2) = quad_roots(u, v); // s² − u s − v
    let (r3, r4) = quad_roots(-b1, -b2); // s² + b1 s + b2 = s² − (−b1)s − (−b2)
    [r1, r2, r3, r4]
}

/// Eigenvalues of a real 4×4 matrix.
pub fn eigenvalues_4x4(a: &[[f64; 4]; 4]) -> [ComplexEigen; 4] {
    let c = charpoly_4(a); // [c3,c2,c1,c0] for s⁴ + c3 s³ + …
    bairstow_quartic(c[0], c[1], c[2], c[3])
}

/// Classify the two oscillatory pairs of a longitudinal `A` into phugoid /
/// short-period (by ascending natural frequency).
pub fn classify_longitudinal_modes(a: &[[f64; 4]; 4]) -> Option<LongitudinalModes> {
    let mut eigs = eigenvalues_4x4(a);
    eigs.sort_by(|p, q| {
        p.omega_n()
            .partial_cmp(&q.omega_n())
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(
                p.im.abs()
                    .partial_cmp(&q.im.abs())
                    .unwrap_or(core::cmp::Ordering::Equal),
            )
    });

    let mut pairs: Vec<ComplexEigen> = Vec::new();
    let mut used = [false; 4];
    for i in 0..4 {
        if used[i] {
            continue;
        }
        if eigs[i].im.abs() < 1e-9 {
            pairs.push(eigs[i]);
            used[i] = true;
            continue;
        }
        let mut found_c = false;
        for j in i + 1..4 {
            if used[j] {
                continue;
            }
            if (eigs[j].re - eigs[i].re).abs() < 1e-6 && (eigs[j].im + eigs[i].im).abs() < 1e-6 {
                let rep = if eigs[i].im >= 0.0 { eigs[i] } else { eigs[j] };
                pairs.push(rep);
                used[i] = true;
                used[j] = true;
                found_c = true;
                break;
            }
        }
        if !found_c {
            pairs.push(eigs[i]);
            used[i] = true;
        }
    }
    if pairs.len() < 2 {
        return None;
    }
    pairs.sort_by(|p, q| {
        p.omega_n()
            .partial_cmp(&q.omega_n())
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    Some(LongitudinalModes {
        phugoid: pairs[0],
        short_period: pairs[1],
        all: eigs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn eig_oscillator_and_decays() {
        // Block diag: rotation [[0,-2],[2,0]] → ±2i, and diag(-1,-3).
        let a = [
            [0.0, -2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -3.0],
        ];
        let e = eigenvalues_4x4(&a);
        let mut got: Vec<(f64, f64)> = e.iter().map(|z| (z.re, z.im)).collect();
        got.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then(a.1.partial_cmp(&b.1).unwrap())
        });
        // Expect (-3,0), (-1,0), (0,-2), (0,2) unordered.
        assert!(got
            .iter()
            .any(|z| (z.0 + 3.0).abs() < 1e-6 && z.1.abs() < 1e-6));
        assert!(got
            .iter()
            .any(|z| (z.0 + 1.0).abs() < 1e-6 && z.1.abs() < 1e-6));
        assert!(got
            .iter()
            .any(|z| z.0.abs() < 1e-5 && (z.1 - 2.0).abs() < 1e-5));
        assert!(got
            .iter()
            .any(|z| z.0.abs() < 1e-5 && (z.1 + 2.0).abs() < 1e-5));
    }

    #[test]
    fn navion_like_a_has_phugoid() {
        // Snapshot A from the Navion linearisation (z-up, verified against NumPy).
        let a = [
            [
                -0.03286698459892268,
                -0.22358401504052328,
                -4.99991726654778,
                -9.76148849276548,
            ],
            [
                0.1695828349390638,
                -2.035553920505251,
                -51.91823202668322,
                0.9400673513404455,
            ],
            [
                0.015707521201966862,
                0.1631040448913419,
                -2.0757154494919723,
                0.0,
            ],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let modes = classify_longitudinal_modes(&a).expect("modes");
        assert_relative_eq!(modes.phugoid.omega_n(), 0.2132, epsilon = 5e-3);
        assert_relative_eq!(modes.phugoid.zeta(), 0.078, epsilon = 2e-2);
        assert!(modes.short_period.omega_n() > 2.0);
    }
}
