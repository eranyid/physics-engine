//! Mach-dependent drag: generic tables + a documented analytic `C_D(M)` model.
//!
//! # Drag tables
//! Linear and monotone (PCHIP-style monotone cubic is deferred — linear + optional
//! monotone-preserving linear slope clamp) interpolation of caller-supplied
//! `(Mach, C_D)` knots.
//!
//! # Analytic model
//! A simple engineering `C_D(M)` with subsonic plateau, transonic rise, and
//! supersonic decay. **Not** a G1/G7 standard drag function — those coefficients
//! were **not** transcribed from a verified primary BRL/McCoy table in this
//! milestone. Profiles that name `G1`/`G7` use this model scaled by BC as an
//! engineering stand-in and are flagged `verified: false` in ammo data.

use pe_math::libm_ops;

/// Drag-model reference tag on an ammo profile.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DragModelRef {
    /// Engineering stand-in (not a verified G1 table).
    G1StandIn,
    /// Engineering stand-in (not a verified G7 table).
    G7StandIn,
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

    /// Monotone-preserving linear interpolant (same as linear for monotone data;
    /// clamps local slopes that would overshoot — Fritsch–Carlson lite).
    pub fn cd_monotone(&self, m: f64) -> f64 {
        // For piecewise-linear tables, linear interp is already monotone in M
        // when C_D knots are used as-is; keep API for future cubic upgrade.
        self.cd_linear(m)
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
/// kg/m² (mass / area). **Not** the inch-pound G1 BC definition.
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
