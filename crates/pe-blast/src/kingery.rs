//! Swisdak (1994) simplified Kingery–Bulmash hemispherical surface-burst fits.
//!
//! # Primary source (coefficients transcribed verbatim)
//! Swisdak, M. M., Jr. (1994). *Simplified Kingery Airblast Calculations*.
//! Indian Head Division, Naval Surface Warfare Center; presented at the
//! 26th DoD Explosives Safety Seminar. NTIS ADA526744 / public release.
//! Table 1 (metric): incident pressure, reflected pressure, arrival time,
//! positive duration, impulses, shock-front velocity.
//!
//! Functional form (Swisdak Table 1):
//! `Y = exp(A + B U + C U² + D U³ + E U⁴ + F U⁵ + G U⁶)`, `U = ln(Z)`,
//! with `Z` in m·kg^{-1/3}. Valid `Z` ranges are enforced by **clamping** and a
//! returned `in_range` flag — never silent extrapolation past the table.
//!
//! Hemispherical TNT **surface** burst engineering curves only. Not a free-air
//! spherical chart; not chemistry or charge design.

use pe_math::libm_ops;

/// Result of a Swisdak / Kingery–Bulmash evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KingeryResult {
    /// Evaluated quantity (see each function for SI units).
    pub value: f64,
    /// `true` if the input `Z` lay inside the published validity interval.
    pub in_range: bool,
    /// The `Z` actually used (may be clamped).
    pub z_used: f64,
}

#[derive(Clone, Copy)]
struct Poly {
    z_lo: f64,
    z_hi: f64,
    c: [f64; 7],
}

impl Poly {
    fn eval(&self, z: f64) -> f64 {
        let u = libm_ops::log(z.max(1e-30));
        let mut acc = self.c[0];
        let mut up = u;
        for k in 1..7 {
            acc += self.c[k] * up;
            up *= u;
        }
        libm_ops::exp(acc)
    }
}

fn eval_piecewise(z: f64, pieces: &[Poly]) -> KingeryResult {
    let global_lo = pieces.iter().map(|p| p.z_lo).fold(f64::INFINITY, f64::min);
    let global_hi = pieces
        .iter()
        .map(|p| p.z_hi)
        .fold(f64::NEG_INFINITY, f64::max);
    let in_range = z >= global_lo && z <= global_hi;
    let z_used = z.clamp(global_lo, global_hi);
    // Pick the piece whose interval contains z_used (inclusive at both ends;
    // prefer the lower piece at exact breakpoints by scanning forward).
    let mut chosen = &pieces[0];
    for p in pieces {
        if z_used >= p.z_lo && z_used <= p.z_hi {
            chosen = p;
            break;
        }
    }
    KingeryResult {
        value: chosen.eval(z_used),
        in_range,
        z_used,
    }
}

// --- Incident pressure P_i [kPa] — Swisdak Table 1 metric -------------------
// RANGE 0.2–2.9 / 2.9–23.8 / 23.8–198.5
const PI_PIECES: [Poly; 3] = [
    Poly {
        z_lo: 0.2,
        z_hi: 2.9,
        c: [7.2106, -2.1069, -0.3229, 0.1117, 0.0685, 0.0, 0.0],
    },
    Poly {
        z_lo: 2.9,
        z_hi: 23.8,
        c: [7.5938, -3.0523, 0.40977, 0.0261, -0.01267, 0.0, 0.0],
    },
    Poly {
        z_lo: 23.8,
        z_hi: 198.5,
        c: [6.0536, -1.4066, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
];

// Reflected pressure P_r [kPa]
const PR_PIECES: [Poly; 2] = [
    Poly {
        z_lo: 0.06,
        z_hi: 2.00,
        c: [9.006, -2.6893, -0.6295, 0.1011, 0.29255, 0.13505, 0.019736],
    },
    Poly {
        z_lo: 2.00,
        z_hi: 40.0,
        c: [8.8396, -1.733, -2.64, 2.293, -0.8232, 0.14247, -0.0099],
    },
];

// Time of arrival T_a [ms / kg^{1/3}]
const TOA_PIECES: [Poly; 2] = [
    Poly {
        z_lo: 0.06,
        z_hi: 1.50,
        c: [-0.7604, 1.8058, 0.1257, -0.0437, -0.0310, -0.00669, 0.0],
    },
    Poly {
        z_lo: 1.50,
        z_hi: 40.0,
        c: [-0.7137, 1.5732, 0.5561, -0.4213, 0.1054, -0.00929, 0.0],
    },
];

// Positive phase duration T [ms / kg^{1/3}]
const TD_PIECES: [Poly; 3] = [
    Poly {
        z_lo: 0.2,
        z_hi: 1.02,
        c: [0.5426, 3.2299, -1.5931, -5.9667, -4.0815, -0.9149, 0.0],
    },
    Poly {
        z_lo: 1.02,
        z_hi: 2.80,
        c: [0.5440, 2.7082, -9.7354, 14.3425, -9.7791, 2.8535, 0.0],
    },
    Poly {
        z_lo: 2.80,
        z_hi: 40.0,
        c: [-2.4608, 7.1639, -5.6215, 2.2711, -0.44994, 0.03486, 0.0],
    },
];

// Incident impulse I_i [kPa·ms / kg^{1/3}]
const II_PIECES: [Poly; 4] = [
    Poly {
        z_lo: 0.2,
        z_hi: 0.96,
        c: [5.522, 1.117, 0.6, -0.292, -0.087, 0.0, 0.0],
    },
    Poly {
        z_lo: 0.96,
        z_hi: 2.38,
        c: [5.465, -0.308, -1.464, 1.362, -0.432, 0.0, 0.0],
    },
    Poly {
        z_lo: 2.38,
        z_hi: 33.7,
        c: [5.2749, -0.4677, -0.2499, 0.0588, -0.00554, 0.0, 0.0],
    },
    Poly {
        z_lo: 33.7,
        z_hi: 158.7,
        c: [5.9825, -1.062, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
];

// Shock front velocity [km/s]
const SFV_PIECES: [Poly; 2] = [
    Poly {
        z_lo: 0.06,
        z_hi: 1.50,
        c: [0.1794, -0.956, -0.0866, 0.109, 0.0699, 0.01218, 0.0],
    },
    Poly {
        z_lo: 1.50,
        z_hi: 40.0,
        c: [0.2597, -1.326, 0.3767, 0.0396, -0.0351, 0.00432, 0.0],
    },
];

/// Incident peak overpressure [Pa] at scaled distance `z` [m·kg^{-1/3}].
///
/// Table stores kPa; converted to Pa. Validity roughly `0.2 ≤ Z ≤ 198.5`.
pub fn incident_overpressure(z: f64) -> KingeryResult {
    let mut r = eval_piecewise(z, &PI_PIECES);
    r.value *= 1e3; // kPa → Pa
    r
}

/// Normally reflected peak overpressure [Pa] (`0.06 ≤ Z ≤ 40`).
pub fn reflected_overpressure(z: f64) -> KingeryResult {
    let mut r = eval_piecewise(z, &PR_PIECES);
    r.value *= 1e3;
    r
}

/// Scaled arrival time [s · kg^{-1/3}] (`0.06 ≤ Z ≤ 40`).
/// Table stores ms/kg^{1/3}; converted to s/kg^{1/3}.
pub fn scaled_arrival_time(z: f64) -> KingeryResult {
    let mut r = eval_piecewise(z, &TOA_PIECES);
    r.value *= 1e-3;
    r
}

/// Arrival time [s] for charge `w_eq_kg` [kg] at range giving scaled `z`.
pub fn arrival_time(z: f64, w_eq_kg: f64) -> KingeryResult {
    let mut r = scaled_arrival_time(z);
    r.value *= libm_ops::cbrt(w_eq_kg.max(0.0));
    r
}

/// Scaled positive-phase duration [s · kg^{-1/3}] (`0.2 ≤ Z ≤ 40`).
pub fn scaled_positive_duration(z: f64) -> KingeryResult {
    let mut r = eval_piecewise(z, &TD_PIECES);
    r.value *= 1e-3;
    r
}

/// Positive-phase duration [s].
pub fn positive_duration(z: f64, w_eq_kg: f64) -> KingeryResult {
    let mut r = scaled_positive_duration(z);
    r.value *= libm_ops::cbrt(w_eq_kg.max(0.0));
    r
}

/// Scaled incident positive impulse [Pa·s · kg^{-1/3}].
/// Table: kPa·ms/kg^{1/3} → Pa·s/kg^{1/3} via ×1 (kPa·ms = Pa·s).
pub fn scaled_incident_impulse(z: f64) -> KingeryResult {
    eval_piecewise(z, &II_PIECES) // already Pa·s / kg^{1/3} numerically
}

/// Incident positive impulse [Pa·s].
pub fn incident_impulse(z: f64, w_eq_kg: f64) -> KingeryResult {
    let mut r = scaled_incident_impulse(z);
    r.value *= libm_ops::cbrt(w_eq_kg.max(0.0));
    r
}

/// Shock-front velocity [m/s] (`0.06 ≤ Z ≤ 40`). Table stores km/s.
pub fn shock_front_velocity(z: f64) -> KingeryResult {
    let mut r = eval_piecewise(z, &SFV_PIECES);
    r.value *= 1e3;
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn incident_at_z1_matches_table_poly() {
        // Direct evaluation of Swisdak coeffs for Z=1 in first Pi piece.
        let r = incident_overpressure(1.0);
        assert!(r.in_range);
        let u = 0.0_f64;
        let ln_pi_kpa = 7.2106 + -2.1069 * u;
        let expect_pa = libm_ops::exp(ln_pi_kpa) * 1e3;
        assert_relative_eq!(r.value, expect_pa, epsilon = 1e-6);
    }

    #[test]
    fn out_of_range_clamps_and_flags() {
        let r = incident_overpressure(0.01);
        assert!(!r.in_range);
        assert_relative_eq!(r.z_used, 0.2, epsilon = 0.0);
        let r2 = incident_overpressure(500.0);
        assert!(!r2.in_range);
        assert_relative_eq!(r2.z_used, 198.5, epsilon = 0.0);
    }
}
