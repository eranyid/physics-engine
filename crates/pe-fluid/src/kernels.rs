//! SPH smoothing kernels (clean-room textbook / paper formulas).
//!
//! # References
//! - Monaghan, J. J. (1992). Smoothed particle hydrodynamics. *ARA&A* **30**.
//! - Wendland, H. (1995). Piecewise polynomial, positive definite and compactly
//!   supported radial functions of minimal degree. *Adv. Comput. Math.* **4**.
//! - Müller, M., Charypar, D. & Gross, M. (2003). Particle-based fluid
//!   simulation for interactive applications. *SCA*.
//!
//! All elementary functions go through [`pe_math::libm_ops`]. No FMA.

use pe_math::libm_ops::{fabs, pow, sqrt};
use pe_math::Vec3;

use crate::Dim;

/// Kernel family used by density / force evaluations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelKind {
    /// Monaghan cubic spline (support radius `2 h`).
    CubicSpline,
    /// Wendland C2 (support radius `2 h`).
    WendlandC2,
    /// Müller poly6 (support radius `h`) — density / colour field.
    Poly6,
    /// Müller spiky (support radius `h`) — pressure gradient.
    Spiky,
}

/// Compact support radius [m] for the given kernel and smoothing length `h` [m].
#[inline]
pub fn support_radius(kind: KernelKind, h: f64) -> f64 {
    match kind {
        KernelKind::CubicSpline | KernelKind::WendlandC2 => 2.0 * h,
        KernelKind::Poly6 | KernelKind::Spiky => h,
    }
}

/// Scalar kernel value `W(r, h)` [1/m^d] for dimension `dim`.
pub fn w(kind: KernelKind, r: f64, h: f64, dim: Dim) -> f64 {
    if h <= 0.0 || r < 0.0 {
        return 0.0;
    }
    match kind {
        KernelKind::CubicSpline => cubic_spline_w(r, h, dim),
        KernelKind::WendlandC2 => wendland_c2_w(r, h, dim),
        KernelKind::Poly6 => poly6_w(r, h, dim),
        KernelKind::Spiky => spiky_w(r, h, dim),
    }
}

/// Gradient `∇W` evaluated at displacement `rij = rj − ri` (points from i toward j).
///
/// Returns the vector `∇_i W(|rij|, h)` so pressure forces using
/// `−(p_i/ρ_i² + p_j/ρ_j²) m_j ∇_i W` have the conventional SPH sign.
pub fn grad_w(kind: KernelKind, rij: Vec3, h: f64, dim: Dim) -> Vec3 {
    if h <= 0.0 {
        return Vec3::zero();
    }
    let r2 = rij.norm_squared();
    let r = sqrt(r2);
    let dwdr = match kind {
        KernelKind::CubicSpline => cubic_spline_dwdr(r, h, dim),
        KernelKind::WendlandC2 => wendland_c2_dwdr(r, h, dim),
        KernelKind::Poly6 => poly6_dwdr(r, h, dim),
        KernelKind::Spiky => spiky_dwdr(r, h, dim),
    };
    if r < 1e-30 {
        // Antisymmetric limit: zero contribution on self; random direction unused.
        return Vec3::zero();
    }
    // ∇_i W(r) = (dW/dr) * (ri − rj)/r = −(dW/dr) * rij/r
    // with rij = rj − ri, so ∇_i W = −(dW/dr / r) * rij.
    rij * (-dwdr / r)
}

/// Monaghan cubic-spline normalisation constants (support `2h`).
#[inline]
fn cubic_alpha(h: f64, dim: Dim) -> f64 {
    match dim {
        Dim::D2 => 10.0 / (7.0 * std::f64::consts::PI * h * h),
        Dim::D3 => 1.0 / (std::f64::consts::PI * h * h * h),
    }
}

fn cubic_spline_w(r: f64, h: f64, dim: Dim) -> f64 {
    // Monaghan & Lattanzio form (integrates to 1 with α above):
    // W = α { 1 − (3/2)q² + (3/4)q³ , (1/4)(2−q)³ }
    let q = r / h;
    if q >= 2.0 {
        return 0.0;
    }
    let a = cubic_alpha(h, dim);
    if q < 1.0 {
        a * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    } else {
        let t = 2.0 - q;
        a * 0.25 * t * t * t
    }
}

fn cubic_spline_dwdr(r: f64, h: f64, dim: Dim) -> f64 {
    let q = r / h;
    if q >= 2.0 || q <= 0.0 {
        return 0.0;
    }
    let a = cubic_alpha(h, dim);
    // dW/dq then / h
    let dwdq = if q < 1.0 {
        a * (-3.0 * q + 2.25 * q * q)
    } else {
        let t = 2.0 - q;
        a * (-0.75 * t * t)
    };
    dwdq / h
}

/// Wendland C2 with support `2h` (Dehnen & Aly / SPH literature form).
#[inline]
fn wendland_alpha(h: f64, dim: Dim) -> f64 {
    match dim {
        // α = 7/(4π h²) for q=r/h ∈ [0,2]
        Dim::D2 => 7.0 / (4.0 * std::f64::consts::PI * h * h),
        // α = 21/(16π h³)
        Dim::D3 => 21.0 / (16.0 * std::f64::consts::PI * h * h * h),
    }
}

fn wendland_c2_w(r: f64, h: f64, dim: Dim) -> f64 {
    let q = r / h;
    if q >= 2.0 {
        return 0.0;
    }
    let a = wendland_alpha(h, dim);
    let t = 1.0 - 0.5 * q;
    a * t * t * t * t * (2.0 * q + 1.0)
}

fn wendland_c2_dwdr(r: f64, h: f64, dim: Dim) -> f64 {
    let q = r / h;
    if q >= 2.0 || q <= 0.0 {
        return 0.0;
    }
    let a = wendland_alpha(h, dim);
    // W = α (1 − q/2)^4 (2q + 1)
    // dW/dq = α [ −2 (1−q/2)^3 (2q+1) + 2 (1−q/2)^4 ]
    //       = α (−5 q) (1 − q/2)^3
    let t = 1.0 - 0.5 * q;
    let dwdq = a * (-5.0 * q) * t * t * t;
    dwdq / h
}

/// Poly6 normalisation (Müller 2003), support `h`.
#[inline]
fn poly6_alpha(h: f64, dim: Dim) -> f64 {
    match dim {
        Dim::D2 => 4.0 / (std::f64::consts::PI * pow(h, 8.0)),
        Dim::D3 => 315.0 / (64.0 * std::f64::consts::PI * pow(h, 9.0)),
    }
}

fn poly6_w(r: f64, h: f64, dim: Dim) -> f64 {
    if r >= h {
        return 0.0;
    }
    let a = poly6_alpha(h, dim);
    let t = h * h - r * r;
    a * t * t * t
}

fn poly6_dwdr(r: f64, h: f64, dim: Dim) -> f64 {
    if r >= h || r <= 0.0 {
        return 0.0;
    }
    let a = poly6_alpha(h, dim);
    let t = h * h - r * r;
    // d/dr (t³) = 3 t² (−2 r) = −6 r t²
    a * (-6.0 * r * t * t)
}

/// Spiky kernel (Müller 2003), support `h`.
#[inline]
fn spiky_alpha(h: f64, dim: Dim) -> f64 {
    match dim {
        Dim::D2 => 10.0 / (std::f64::consts::PI * pow(h, 5.0)),
        Dim::D3 => 15.0 / (std::f64::consts::PI * pow(h, 6.0)),
    }
}

fn spiky_w(r: f64, h: f64, dim: Dim) -> f64 {
    if r >= h {
        return 0.0;
    }
    let a = spiky_alpha(h, dim);
    let t = h - r;
    a * t * t * t
}

fn spiky_dwdr(r: f64, h: f64, dim: Dim) -> f64 {
    if r >= h || r <= 0.0 {
        return 0.0;
    }
    let a = spiky_alpha(h, dim);
    let t = h - r;
    // d/dr (t³) = −3 t²
    a * (-3.0 * t * t)
}

/// Numerical partition-of-unity check: ∫ W dV ≈ 1 on a fine Cartesian grid.
///
/// Returns absolute error `|∫W − 1|`. Used by analytic tests — not a runtime path.
pub fn partition_of_unity_error(kind: KernelKind, h: f64, dim: Dim, n_half: usize) -> f64 {
    let support = support_radius(kind, h);
    let dx = support / n_half as f64;
    let mut sum = 0.0;
    match dim {
        Dim::D2 => {
            let n = 2 * n_half;
            for ix in 0..=n {
                for iy in 0..=n {
                    let x = -support + ix as f64 * dx;
                    let y = -support + iy as f64 * dx;
                    let r = sqrt(x * x + y * y);
                    // Simple rectangle / Riemann sum over the support square.
                    sum += w(kind, r, h, dim) * dx * dx;
                }
            }
        }
        Dim::D3 => {
            let n = 2 * n_half;
            for ix in 0..=n {
                for iy in 0..=n {
                    for iz in 0..=n {
                        let x = -support + ix as f64 * dx;
                        let y = -support + iy as f64 * dx;
                        let z = -support + iz as f64 * dx;
                        let r = sqrt(x * x + y * y + z * z);
                        sum += w(kind, r, h, dim) * dx * dx * dx;
                    }
                }
            }
        }
    }
    fabs(sum - 1.0)
}

/// Discrete zeroth-order gradient consistency: `|Σ_j V_j ∇_i W|` on a lattice.
///
/// For a symmetric kernel on a complete lattice this should → 0. Returns the
/// Euclidean norm of the summed gradient (not divided by particle count).
pub fn gradient_sum_norm(kind: KernelKind, h: f64, dim: Dim, dx: f64) -> f64 {
    let support = support_radius(kind, h);
    let n = ((support / dx).ceil() as i32) + 1;
    let mut gsum = Vec3::zero();
    for ix in -n..=n {
        for iy in -n..=n {
            let jz_range = match dim {
                Dim::D2 => 0..=0,
                Dim::D3 => -n..=n,
            };
            for iz in jz_range {
                if ix == 0 && iy == 0 && iz == 0 {
                    continue;
                }
                let rij = Vec3::new(ix as f64 * dx, iy as f64 * dx, iz as f64 * dx);
                if rij.norm() >= support {
                    continue;
                }
                let vol = match dim {
                    Dim::D2 => dx * dx,
                    Dim::D3 => dx * dx * dx,
                };
                gsum += grad_w(kind, rij, h, dim) * vol;
            }
        }
    }
    gsum.norm()
}

/// First-order gradient consistency on a lattice: `Σ_j V_j (xj−xi) ⊗ ∇_i W ≈ I`.
///
/// With [`grad_w`] returning `∇_i W` (points toward neighbour j for falling
/// kernels), the discrete tensor approaches **+I**. Returns the max absolute
/// entry-wise error versus `+I`.
pub fn gradient_consistency_error(kind: KernelKind, h: f64, dim: Dim, dx: f64) -> f64 {
    let support = support_radius(kind, h);
    let n = ((support / dx).ceil() as i32) + 1;
    let mut m = [[0.0f64; 3]; 3];
    for ix in -n..=n {
        for iy in -n..=n {
            let jz_range = match dim {
                Dim::D2 => 0..=0,
                Dim::D3 => -n..=n,
            };
            for iz in jz_range {
                if ix == 0 && iy == 0 && iz == 0 {
                    continue;
                }
                let rij = Vec3::new(ix as f64 * dx, iy as f64 * dx, iz as f64 * dx);
                if rij.norm() >= support {
                    continue;
                }
                let vol = match dim {
                    Dim::D2 => dx * dx,
                    Dim::D3 => dx * dx * dx,
                };
                let g = grad_w(kind, rij, h, dim); // ∇_i W
                                                   // A = Σ V (rj−ri) ⊗ ∇_i W  ≈ +I
                m[0][0] += vol * rij.x * g.x;
                m[0][1] += vol * rij.x * g.y;
                m[0][2] += vol * rij.x * g.z;
                m[1][0] += vol * rij.y * g.x;
                m[1][1] += vol * rij.y * g.y;
                m[1][2] += vol * rij.y * g.z;
                m[2][0] += vol * rij.z * g.x;
                m[2][1] += vol * rij.z * g.y;
                m[2][2] += vol * rij.z * g.z;
            }
        }
    }
    let dim_n = match dim {
        Dim::D2 => 2,
        Dim::D3 => 3,
    };
    let mut err: f64 = 0.0;
    for a in 0..dim_n {
        for b in 0..dim_n {
            let target = if a == b { 1.0 } else { 0.0 };
            err = err.max(fabs(m[a][b] - target));
        }
    }
    err
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn support_radii() {
        assert!((support_radius(KernelKind::CubicSpline, 0.1) - 0.2).abs() < 1e-15);
        assert!((support_radius(KernelKind::Poly6, 0.1) - 0.1).abs() < 1e-15);
    }

    #[test]
    fn cubic_positive_inside() {
        let h = 0.1;
        assert!(cubic_spline_w(0.0, h, Dim::D3) > 0.0);
        assert!(cubic_spline_w(0.15, h, Dim::D3) > 0.0);
        assert_eq!(cubic_spline_w(0.2, h, Dim::D3), 0.0);
    }
}
