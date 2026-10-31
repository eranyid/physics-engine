//! Two-body Kepler utilities: orbital elements ↔ state, Stumpff functions,
//! universal-variable propagation (Danby / Newton start).
//!
//! References (formulas only):
//! - Battin, R.H. *An Introduction to the Mathematics and Methods of Astrodynamics*.
//! - Danby, J.M.A. *Fundamentals of Celestial Mechanics*.
//! - Vallado, D.A. *Fundamentals of Astrodynamics and Applications*.

use pe_math::libm_ops;
use pe_math::Vec3;

use crate::constants::TWO_PI;

/// Classical Keplerian orbital elements (osculating) about a central μ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeplerElements {
    /// Semi-major axis [m] (negative for hyperbolic).
    pub a: f64,
    /// Eccentricity [dimensionless].
    pub e: f64,
    /// Inclination [rad].
    pub i: f64,
    /// Right ascension of ascending node Ω [rad].
    pub raan: f64,
    /// Argument of periapsis ω [rad].
    pub arg_p: f64,
    /// True anomaly ν [rad].
    pub true_anomaly: f64,
}

/// Kepler helper errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeplerError {
    /// Degenerate / parabolic edge not handled by this path.
    Degenerate,
    /// Newton iteration failed to converge.
    NoConvergence,
    /// Invalid input (e.g. μ ≤ 0).
    InvalidInput,
}

impl core::fmt::Display for KeplerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Degenerate => write!(f, "degenerate orbit"),
            Self::NoConvergence => write!(f, "Kepler solver did not converge"),
            Self::InvalidInput => write!(f, "invalid Kepler input"),
        }
    }
}

impl std::error::Error for KeplerError {}

/// Mean motion `n = √(μ / |a|³)` [rad/s] (elliptic uses a > 0).
#[inline]
pub fn mean_motion(mu: f64, a: f64) -> f64 {
    let aa = pe_math::libm_ops::fabs(a);
    libm_ops::sqrt(mu / (aa * aa * aa))
}

/// Elliptic orbital period `T = 2π √(a³/μ)` [s].
#[inline]
pub fn orbital_period(mu: f64, a: f64) -> f64 {
    TWO_PI * libm_ops::sqrt((a * a * a) / mu)
}

/// Stumpff `C(z)` = (1 − cos√z)/z  (z>0), (cosh√−z − 1)/(−z) (z<0), ½ (z=0).
///
/// Series form used near zero for stability (Battin).
pub fn stumpff_c(z: f64) -> f64 {
    if libm_ops::fabs(z) < 1e-8 {
        // 1/2 - z/24 + z²/720 - …
        0.5 - z / 24.0 + (z * z) / 720.0
    } else if z > 0.0 {
        let s = libm_ops::sqrt(z);
        (1.0 - libm_ops::cos(s)) / z
    } else {
        let s = libm_ops::sqrt(-z);
        // cosh(s) = (e^s + e^{-s})/2
        let es = libm_ops::exp(s);
        let cosh = 0.5 * (es + 1.0 / es);
        (cosh - 1.0) / (-z)
    }
}

/// Stumpff `S(z)` = (√z − sin√z)/z^{3/2} etc. Series near zero.
pub fn stumpff_s(z: f64) -> f64 {
    if libm_ops::fabs(z) < 1e-8 {
        // 1/6 - z/120 + z²/5040 - …
        1.0 / 6.0 - z / 120.0 + (z * z) / 5040.0
    } else if z > 0.0 {
        let s = libm_ops::sqrt(z);
        (s - libm_ops::sin(s)) / (z * s)
    } else {
        let s = libm_ops::sqrt(-z);
        let es = libm_ops::exp(s);
        let sinh = 0.5 * (es - 1.0 / es);
        (sinh - s) / ((-z) * s)
    }
}

/// Convert position/velocity to classical elements (two-body, μ [m³/s²]).
///
/// Frame: inertial; line of nodes from `ẑ × h`. Equatorial / circular edge cases
/// set Ω or ω to 0 by convention.
pub fn state_to_elements(mu: f64, r: Vec3, v: Vec3) -> Result<KeplerElements, KeplerError> {
    if mu <= 0.0 {
        return Err(KeplerError::InvalidInput);
    }
    let rn = r.norm();
    if rn == 0.0 {
        return Err(KeplerError::Degenerate);
    }
    let h = r.cross(v);
    let hn = h.norm();
    if hn == 0.0 {
        return Err(KeplerError::Degenerate);
    }
    let n_vec = Vec3::new(-h.y, h.x, 0.0); // ẑ × h
    let nn = n_vec.norm();
    let e_vec = (v.cross(h) / mu) - r / rn;
    let e = e_vec.norm();
    let energy = 0.5 * v.norm_squared() - mu / rn;
    let a = if libm_ops::fabs(energy) < 1e-30 {
        return Err(KeplerError::Degenerate); // parabola
    } else {
        -mu / (2.0 * energy)
    };
    let i = libm_ops::acos((h.z / hn).clamp(-1.0, 1.0));
    let raan = if nn < 1e-14 {
        0.0
    } else {
        let mut o = libm_ops::acos((n_vec.x / nn).clamp(-1.0, 1.0));
        if n_vec.y < 0.0 {
            o = TWO_PI - o;
        }
        o
    };
    let arg_p = if e < 1e-14 {
        0.0
    } else if nn < 1e-14 {
        // equatorial: use longitude of periapsis from x-axis
        let mut w = libm_ops::atan2(e_vec.y, e_vec.x);
        if w < 0.0 {
            w += TWO_PI;
        }
        w
    } else {
        let mut w = libm_ops::acos((n_vec.dot(e_vec) / (nn * e)).clamp(-1.0, 1.0));
        if e_vec.z < 0.0 {
            w = TWO_PI - w;
        }
        w
    };
    let true_anomaly = if e < 1e-14 {
        // circular: argument of latitude
        if nn < 1e-14 {
            let mut nu = libm_ops::atan2(r.y, r.x);
            if nu < 0.0 {
                nu += TWO_PI;
            }
            nu
        } else {
            let mut u = libm_ops::acos((n_vec.dot(r) / (nn * rn)).clamp(-1.0, 1.0));
            if r.z < 0.0 {
                u = TWO_PI - u;
            }
            u
        }
    } else {
        let mut nu = libm_ops::acos((e_vec.dot(r) / (e * rn)).clamp(-1.0, 1.0));
        if r.dot(v) < 0.0 {
            nu = TWO_PI - nu;
        }
        nu
    };
    Ok(KeplerElements {
        a,
        e,
        i,
        raan,
        arg_p,
        true_anomaly,
    })
}

/// Convert classical elements to position/velocity (μ [m³/s²]).
pub fn elements_to_state(mu: f64, el: KeplerElements) -> Result<(Vec3, Vec3), KeplerError> {
    if mu <= 0.0 || el.a == 0.0 {
        return Err(KeplerError::InvalidInput);
    }
    let (s_o, c_o) = (libm_ops::sin(el.raan), libm_ops::cos(el.raan));
    let (s_i, c_i) = (libm_ops::sin(el.i), libm_ops::cos(el.i));
    let (s_w, c_w) = (libm_ops::sin(el.arg_p), libm_ops::cos(el.arg_p));
    let (s_v, c_v) = (
        libm_ops::sin(el.true_anomaly),
        libm_ops::cos(el.true_anomaly),
    );

    let p = el.a * (1.0 - el.e * el.e);
    if p <= 0.0 {
        return Err(KeplerError::Degenerate);
    }
    let r_pqw = Vec3::new(
        p * c_v / (1.0 + el.e * c_v),
        p * s_v / (1.0 + el.e * c_v),
        0.0,
    );
    let v_pqw = Vec3::new(
        -libm_ops::sqrt(mu / p) * s_v,
        libm_ops::sqrt(mu / p) * (el.e + c_v),
        0.0,
    );
    // PQW → IJK
    let r = rotate_pqw(r_pqw, s_o, c_o, s_i, c_i, s_w, c_w);
    let v = rotate_pqw(v_pqw, s_o, c_o, s_i, c_i, s_w, c_w);
    Ok((r, v))
}

fn rotate_pqw(v: Vec3, s_o: f64, c_o: f64, s_i: f64, c_i: f64, s_w: f64, c_w: f64) -> Vec3 {
    let r00 = c_o * c_w - s_o * s_w * c_i;
    let r01 = -c_o * s_w - s_o * c_w * c_i;
    let r02 = s_o * s_i;
    let r10 = s_o * c_w + c_o * s_w * c_i;
    let r11 = -s_o * s_w + c_o * c_w * c_i;
    let r12 = -c_o * s_i;
    let r20 = s_w * s_i;
    let r21 = c_w * s_i;
    let r22 = c_i;
    Vec3::new(
        r00 * v.x + r01 * v.y + r02 * v.z,
        r10 * v.x + r11 * v.y + r12 * v.z,
        r20 * v.x + r21 * v.y + r22 * v.z,
    )
}

/// Universal-variable Kepler propagation by time `dt` [s].
///
/// Newton iteration on χ with Danby-style starter (Vallado / Battin). Returns
/// `(r(t+dt), v(t+dt))`.
pub fn propagate_universal(
    mu: f64,
    r0: Vec3,
    v0: Vec3,
    dt: f64,
) -> Result<(Vec3, Vec3), KeplerError> {
    if mu <= 0.0 {
        return Err(KeplerError::InvalidInput);
    }
    if dt == 0.0 {
        return Ok((r0, v0));
    }
    let r0n = r0.norm();
    let v0n2 = v0.norm_squared();
    let vr0 = r0.dot(v0) / r0n;
    let alpha = 2.0 / r0n - v0n2 / mu; // 1/a

    // Danby / Vallado starter for χ
    let chi = if alpha > 1e-12 {
        // elliptic
        libm_ops::sqrt(mu) * dt * alpha
    } else if alpha < -1e-12 {
        // hyperbolic (Battin / Vallado)
        let sign = if dt >= 0.0 { 1.0 } else { -1.0 };
        let num = -2.0 * mu * alpha * dt
            / (r0n * vr0 * alpha + sign * libm_ops::sqrt(-mu * alpha) * (1.0 - r0n * alpha));
        sign * libm_ops::log(num.max(1e-300)) / libm_ops::sqrt(-alpha)
    } else {
        // near-parabola: Barker's equation cubic starter (Vallado)
        let h = r0.cross(v0).norm();
        let p = h * h / mu;
        let s = 0.5
            * (core::f64::consts::PI / 2.0
                - libm_ops::atan(3.0 * libm_ops::sqrt(mu / (p * p * p)) * dt));
        let w = libm_ops::atan(libm_ops::cbrt(libm_ops::tan(s)));
        libm_ops::sqrt(p) * 2.0 * (1.0 / libm_ops::tan(2.0 * w))
    };
    let mut chi = chi;

    let sqrt_mu = libm_ops::sqrt(mu);
    let sigma0 = r0.dot(v0) / sqrt_mu; // (r·v)/√μ
    for _ in 0..100 {
        let z = alpha * chi * chi;
        let c = stumpff_c(z);
        let s = stumpff_s(z);
        // F(χ) = σ₀ χ² C + (1 − α r₀) χ³ S + r₀ χ − √μ Δt
        let f = sigma0 * chi * chi * c + (1.0 - alpha * r0n) * chi * chi * chi * s + r0n * chi
            - sqrt_mu * dt;
        // dF/dχ = σ₀ χ (1 − z S) + (1 − α r₀) χ² C + r₀   (= r)
        let fp = sigma0 * chi * (1.0 - z * s) + (1.0 - alpha * r0n) * chi * chi * c + r0n;
        if libm_ops::fabs(fp) < 1e-30 {
            return Err(KeplerError::NoConvergence);
        }
        let mut dchi = f / fp;
        // Damped step if the correction is huge relative to χ (multi-rev).
        if libm_ops::fabs(dchi) > libm_ops::fabs(chi) + 1.0 {
            dchi *= 0.5;
        }
        chi -= dchi;
        if libm_ops::fabs(dchi) < 1e-10 * (1.0 + libm_ops::fabs(chi)) {
            let z = alpha * chi * chi;
            let c = stumpff_c(z);
            let s = stumpff_s(z);
            let f_lag = 1.0 - (chi * chi / r0n) * c;
            let g_lag = dt - (chi * chi * chi / sqrt_mu) * s;
            let r_vec = r0 * f_lag + v0 * g_lag;
            let rn = r_vec.norm();
            if rn == 0.0 {
                return Err(KeplerError::Degenerate);
            }
            let fdot = (sqrt_mu / (rn * r0n)) * chi * (z * s - 1.0);
            let gdot = 1.0 - (chi * chi / rn) * c;
            let v_vec = r0 * fdot + v0 * gdot;
            return Ok((r_vec, v_vec));
        }
    }
    Err(KeplerError::NoConvergence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::EARTH_MU;
    use approx::assert_relative_eq;

    #[test]
    fn circular_leo_period() {
        let r = 7_000_000.0;
        let t = orbital_period(EARTH_MU, r);
        assert_relative_eq!(t, 5828.5, max_relative = 2e-4);
    }
}
