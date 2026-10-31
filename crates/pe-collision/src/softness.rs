//! Soft-constraint coefficients (mass-independent Hertz / damping form).
//!
//! Clean-room reimplementation of the formulas published by Erin Catto
//! (Solver2D blog, 2024-02-05; Box2D v3 release notes) and the mass-independent
//! simplification attributed there to Ross Nordby (Bepu). No Box2D source was
//! copied — see `THIRD_PARTY.md`.
//!
//! References:
//! - <https://box2d.org/posts/2024/02/solver2d/>
//! - Erin Catto, GDC 2011, "Soft Constraints — Reinventing the Spring"

use core::f64::consts::PI;

use pe_math::libm_ops::sqrt;

/// Soft-constraint scales derived from Hertz, damping ratio, and sub-step `h`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Softness {
    /// Bias rate [1/s]: multiplies contact separation into the velocity target.
    pub bias_rate: f64,
    /// Non-dimensional mass coefficient (scales effective mass).
    pub mass_scale: f64,
    /// Non-dimensional impulse coefficient (damps accumulated impulse growth).
    pub impulse_scale: f64,
}

impl Softness {
    /// Hard constraint (no spring/damper): `Δλ = −m_eff · v_n`.
    pub const HARD: Self = Self {
        bias_rate: 0.0,
        mass_scale: 1.0,
        impulse_scale: 0.0,
    };

    /// Zero response (unused / disabled spring).
    pub const ZERO: Self = Self {
        bias_rate: 0.0,
        mass_scale: 0.0,
        impulse_scale: 0.0,
    };

    /// Build softness from Hertz [Hz], damping ratio ζ, and sub-step size `h` [s].
    ///
    /// ```text
    /// ω = 2π f
    /// a1 = 2ζ + ω h
    /// a2 = ω h · a1
    /// a3 = 1 / (1 + a2)
    /// bias_rate     = ω / a1
    /// mass_scale    = a2 · a3
    /// impulse_scale = a3
    /// ```
    pub fn from_hertz_zeta(hertz: f64, zeta: f64, h: f64) -> Self {
        if hertz <= 0.0 || h <= 0.0 {
            return Self::ZERO;
        }
        let omega = 2.0 * PI * hertz;
        let a1 = 2.0 * zeta + h * omega;
        let a2 = h * omega * a1;
        let a3 = 1.0 / (1.0 + a2);
        Self {
            bias_rate: omega / a1,
            mass_scale: a2 * a3,
            impulse_scale: a3,
        }
    }
}

/// Soft-step solver tuning (Box2D v3 defaults, lifted to 3D contacts).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftStepParams {
    /// Number of velocity/position sub-steps per full `dt` (default 4).
    pub substeps: usize,
    /// Contact spring frequency [Hz] before Nyquist-style capping (default 30).
    pub contact_hertz: f64,
    /// Contact damping ratio ζ (default 10 — heavily overdamped).
    pub contact_zeta: f64,
    /// Cap Hertz at `fraction * (1/h)` where `h = dt/substeps` (default 0.25).
    pub hertz_cap_fraction: f64,
    /// Restitution speed threshold [m/s] for the post-substep bounce pass (default 1).
    pub restitution_threshold: f64,
}

impl Default for SoftStepParams {
    fn default() -> Self {
        Self {
            substeps: 4,
            contact_hertz: 30.0,
            contact_zeta: 10.0,
            hertz_cap_fraction: 0.25,
            restitution_threshold: 1.0,
        }
    }
}

impl SoftStepParams {
    /// Softness for one sub-step of size `h`, with Hertz capped to
    /// `hertz_cap_fraction / h`.
    pub fn contact_softness(&self, h: f64) -> Softness {
        let inv_h = if h > 0.0 { 1.0 / h } else { 0.0 };
        let cap = self.hertz_cap_fraction * inv_h;
        let hertz = if self.contact_hertz < cap {
            self.contact_hertz
        } else {
            cap
        };
        Softness::from_hertz_zeta(hertz, self.contact_zeta, h)
    }
}

/// Contact solver mode. `Pgs` is the oracle-compatible default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SolverMode {
    /// Classic projected Gauss–Seidel / sequential impulses (stage-2 oracle).
    #[default]
    Pgs,
    /// Soft-step: sub-steps + soft constraints + relax + restitution pass.
    SoftStep,
}

impl SolverMode {
    /// Parse `"pgs"` / `"soft_step"` (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "pgs" | "sequential" | "si" => Some(Self::Pgs),
            "soft_step" | "softstep" | "soft" | "tgs_soft" => Some(Self::SoftStep),
            _ => None,
        }
    }

    /// Stable name for bindings / docs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pgs => "pgs",
            Self::SoftStep => "soft_step",
        }
    }
}

/// Magnitude helper used by sleep / wake checks (portable sqrt).
#[inline]
pub fn speed3(v: pe_math::Vec3) -> f64 {
    sqrt(v.norm_squared())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn softness_mass_plus_impulse_is_one() {
        let s = Softness::from_hertz_zeta(30.0, 10.0, 1.0 / 240.0);
        assert_relative_eq!(s.mass_scale + s.impulse_scale, 1.0, epsilon = 1e-12);
        assert!(s.bias_rate > 0.0);
    }

    #[test]
    fn hertz_is_capped_to_fraction_of_substep_rate() {
        let p = SoftStepParams {
            contact_hertz: 1.0e6,
            hertz_cap_fraction: 0.25,
            ..SoftStepParams::default()
        };
        let h = 1.0 / 240.0;
        let soft = p.contact_softness(h);
        let capped = Softness::from_hertz_zeta(0.25 * 240.0, p.contact_zeta, h);
        assert_eq!(soft, capped);
    }
}
