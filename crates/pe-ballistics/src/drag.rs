//! Mach-dependent drag: generic tables, G1/G7 standard tables, analytic stand-in.
//!
//! # Interpolation
//! [`DragTable`] supports **linear** (`cd_linear`) and **monotone cubic**
//! Fritsch–Carlson PCHIP (`cd_monotone`) interpolation of `(Mach, C_D)` knots.
//! Standard G1/G7 lookups in this crate use **linear** interpolation by default
//! (documented; matches common open ballistics calculators).
//!
//! # G1 / G7
//! Published BRL/Aberdeen Cd–Mach knots transcribed from JBM public downloads
//! (free for public use). See [`crate::g1g7_tables`] and [`THIRD_PARTY.md`](../../../THIRD_PARTY.md).
//!
//! # Analytic model
//! A simple engineering `C_D(M)` remains available for demos. It is **not** a
//! G1/G7 standard drag function.

use pe_math::libm_ops;

use crate::g1g7_tables::{G1_CD, G1_MACH, G7_CD, G7_MACH};

/// Drag-model reference tag on an ammo profile.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DragModelRef {
    /// Published G1 standard-projectile Cd(Mach) table (JBM/BRL).
    ///
    /// Serde also accepts the legacy name `"G1StandIn"`.
    #[serde(alias = "G1StandIn")]
    G1,
    /// Published G7 standard-projectile Cd(Mach) table (JBM/BRL).
    ///
    /// Serde also accepts the legacy name `"G7StandIn"`.
    #[serde(alias = "G7StandIn")]
    G7,
    /// Custom Mach→C_D table.
    CustomTable,
    /// Built-in analytic `C_D(M)` (see [`analytic_cd`]).
    Analytic,
}

/// Tabulated `C_D(M)` knots (Mach ascending).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DragTable {
    /// Mach numbers [dimensionless], strictly ascending.
    pub mach: Vec<f64>,
    /// Drag coefficients [dimensionless].
    pub cd: Vec<f64>,
}

impl DragTable {
    /// Construct; returns `None` if lengths differ or Mach not ascending.
    pub fn new(mach: Vec<f64>, cd: Vec<f64>) -> Option<Self> {
        if mach.len() != cd.len() || mach.len() < 2 {
            return None;
        }
        for i in 1..mach.len() {
            if mach[i] <= mach[i - 1] {
                return None;
            }
        }
        Some(Self { mach, cd })
    }

    /// Linear interpolation of `C_D` at Mach `m` (clamped to table ends).
    pub fn cd_linear(&self, m: f64) -> f64 {
        let m = m.max(0.0);
        if m <= self.mach[0] {
            return self.cd[0];
        }
        let n = self.mach.len();
        if m >= self.mach[n - 1] {
            return self.cd[n - 1];
        }
        let mut i = 0;
        while i + 1 < n && self.mach[i + 1] < m {
            i += 1;
        }
        let m0 = self.mach[i];
        let m1 = self.mach[i + 1];
        let t = (m - m0) / (m1 - m0);
        self.cd[i] * (1.0 - t) + self.cd[i + 1] * t
    }

    /// Monotone cubic (Fritsch–Carlson PCHIP) interpolation of `C_D` at Mach `m`.
    ///
    /// Clamped to table ends. Guarantees no overshoot between knots when the
    /// tabulated sequence is locally monotone.
    pub fn cd_monotone(&self, m: f64) -> f64 {
        let m = m.max(0.0);
        let n = self.mach.len();
        if m <= self.mach[0] {
            return self.cd[0];
        }
        if m >= self.mach[n - 1] {
            return self.cd[n - 1];
        }
        let mut i = 0;
        while i + 1 < n && self.mach[i + 1] < m {
            i += 1;
        }
        // Endpoint slopes via one-sided differences; interior via Fritsch–Carlson.
        let slopes = pchip_slopes(&self.mach, &self.cd);
        let h = self.mach[i + 1] - self.mach[i];
        let t = (m - self.mach[i]) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        h00 * self.cd[i] + h10 * h * slopes[i] + h01 * self.cd[i + 1] + h11 * h * slopes[i + 1]
    }
}

/// Fritsch–Carlson monotone cubic slopes at knots.
fn pchip_slopes(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut d = vec![0.0; n];
    let mut delta = vec![0.0; n - 1];
    for i in 0..n - 1 {
        delta[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    }
    d[0] = delta[0];
    d[n - 1] = delta[n - 2];
    for i in 1..n - 1 {
        if delta[i - 1] * delta[i] <= 0.0 {
            d[i] = 0.0;
        } else {
            let w1 = 2.0 * (x[i + 1] - x[i]) + (x[i] - x[i - 1]);
            let w2 = (x[i + 1] - x[i]) + 2.0 * (x[i] - x[i - 1]);
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
        }
    }
    d
}

/// lb/in² → kg/m² for a *mass / diameter²* quantity (no π/4).
///
/// `1 lb/in² = 0.45359237 / 0.0254² ≈ 703.069580719 kg/m²`.
pub const LB_IN2_TO_KG_M2: f64 = 0.45359237 / (0.0254 * 0.0254);

/// Published G1/G7 BC [lb/in²] → SI BC [kg/m²] for use with `A_ref = m / BC_si`.
///
/// Imperial BC is `m / (d² i)` (mass over diameter-squared). This crate's SI BC
/// is `m / (A i)` with `A = π d²/4`, so
/// `BC_si = BC_imperial · (lb/in²→kg/m²) · (4/π)`.
///
/// Combined factor ≈ `894.665` kg/m² per (lb/in²). Matches the classical
/// `a = −½ ρ v² Cd_std (π/4) / BC_{m/d²}` form used by JBM / py_ballisticcalc
/// (`SDF = Cd · 2.08551e−4 / BC` in imperial FPS units).
pub const BC_LB_IN2_TO_KG_M2: f64 = LB_IN2_TO_KG_M2 * 4.0 / core::f64::consts::PI;

/// Convert a published G1/G7 BC in lb/in² to SI kg/m² (`m/(A i)`).
pub fn bc_imperial_to_si(bc_lb_per_in2: f64) -> f64 {
    bc_lb_per_in2 * BC_LB_IN2_TO_KG_M2
}

/// Convert SI BC [kg/m²] (`m/(A i)`) back to imperial G1/G7 BC [lb/in²].
pub fn bc_si_to_imperial(bc_kg_per_m2: f64) -> f64 {
    bc_kg_per_m2 / BC_LB_IN2_TO_KG_M2
}

/// Built-in G1 standard drag table (JBM/BRL transcription).
pub fn g1_table() -> DragTable {
    DragTable {
        mach: G1_MACH.to_vec(),
        cd: G1_CD.to_vec(),
    }
}

/// Built-in G7 standard drag table (JBM/BRL transcription).
pub fn g7_table() -> DragTable {
    DragTable {
        mach: G7_MACH.to_vec(),
        cd: G7_CD.to_vec(),
    }
}

/// Linear-interpolated standard-projectile `C_D` for G1 or G7.
pub fn standard_cd(model: DragModelRef, mach: f64) -> f64 {
    match model {
        DragModelRef::G1 => g1_table().cd_linear(mach),
        DragModelRef::G7 => g7_table().cd_linear(mach),
        DragModelRef::CustomTable | DragModelRef::Analytic => analytic_cd(mach),
    }
}

/// Analytic engineering `C_D(M)`:
/// - `M < 0.8`: constant `cd0`
/// - `0.8 ≤ M < 1.2`: smooth rise to `cd_peak`
/// - `M ≥ 1.2`: `cd_peak · (1.2/M)^α` decay
///
/// Default: `cd0=0.20`, `cd_peak=0.45`, `α=0.5` — **model**, not a published
/// projectile drag law.
pub fn analytic_cd(mach: f64) -> f64 {
    analytic_cd_params(mach, 0.20, 0.45, 0.5)
}

/// Parameterized analytic `C_D(M)`.
pub fn analytic_cd_params(mach: f64, cd0: f64, cd_peak: f64, alpha: f64) -> f64 {
    let m = mach.max(0.0);
    if m < 0.8 {
        cd0
    } else if m < 1.2 {
        let t = (m - 0.8) / 0.4;
        // Smoothstep
        let s = t * t * (3.0 - 2.0 * t);
        cd0 + (cd_peak - cd0) * s
    } else {
        cd_peak * libm_ops::pow(1.2 / m, alpha)
    }
}

/// Drag force magnitude [N]: `½ ρ V² A C_D`.
///
/// - `rho` [kg/m³], `speed` [m/s], `area` [m²], `cd` [dimensionless].
pub fn drag_force_magnitude(rho: f64, speed: f64, area: f64, cd: f64) -> f64 {
    0.5 * rho * speed * speed * area * cd
}

/// Convert a ballistic coefficient `BC` [kg/m²] (sectional-density style) to an
/// effective reference area scale. Engineering: `A_ref = m / BC` when BC is in
/// kg/m² (mass / area). With a G1/G7 **standard** Cd table this yields
/// `a_drag = −½ ρ v² Cd_std(M) / BC_si`, the SI form of the classical BC scaling.
pub fn area_from_bc_si(mass_kg: f64, bc_kg_per_m2: f64) -> f64 {
    if bc_kg_per_m2 <= 0.0 {
        return 0.0;
    }
    mass_kg / bc_kg_per_m2
}

/// Caliber area [m²] from diameter [m]: `π (d/2)²`.
pub fn caliber_area(diameter_m: f64) -> f64 {
    let r = 0.5 * diameter_m;
    core::f64::consts::PI * r * r
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn g1_g7_knot_counts_and_endpoints() {
        let g1 = g1_table();
        let g7 = g7_table();
        assert_eq!(g1.mach.len(), 79);
        assert_eq!(g7.mach.len(), 84);
        assert_relative_eq!(g1.cd_linear(0.0), 0.2629, epsilon = 0.0);
        assert_relative_eq!(g1.cd_linear(1.0), 0.4805, epsilon = 0.0);
        assert_relative_eq!(g1.cd_linear(5.0), 0.4988, epsilon = 0.0);
        assert_relative_eq!(g7.cd_linear(0.0), 0.1198, epsilon = 0.0);
        assert_relative_eq!(g7.cd_linear(1.05), 0.4043, epsilon = 0.0);
        assert_relative_eq!(g7.cd_linear(5.0), 0.1618, epsilon = 0.0);
    }

    #[test]
    fn linear_midpoint_and_monotone_no_overshoot() {
        let t = DragTable::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.0]).unwrap();
        assert_relative_eq!(t.cd_linear(0.5), 0.5, epsilon = 1e-15);
        // PCHIP on a tent should stay in [0,1].
        for k in 0..21 {
            let m = 0.1 * k as f64;
            let y = t.cd_monotone(m);
            assert!((-1e-12..=1.0 + 1e-12).contains(&y), "m={m} y={y}");
        }
    }

    #[test]
    fn bc_imperial_conversion_roundtrip() {
        let si = bc_imperial_to_si(0.5);
        assert_relative_eq!(si, 0.5 * BC_LB_IN2_TO_KG_M2, epsilon = 0.0);
        assert_relative_eq!(bc_si_to_imperial(si), 0.5, epsilon = 1e-12);
    }
}
