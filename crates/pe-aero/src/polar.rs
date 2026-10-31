//! Wing lift/drag polar with a linear stall break and pitch-moment derivatives.
//!
//! Body axes (right-handed, matching the engine's z-up world when level):
//! `+x` nose forward, `+y` right wing, `+z` up.
//! Angle of attack `α = atan2(−v_z, v_x)` from body velocity (positive nose-up).
//! Sideslip / β is ignored (planar longitudinal model); at `V → 0` we clamp α, β → 0.
//!
//! Compressibility is not modelled; there is no Prandtl–Glauert factor and no
//! special handling of Mach ≈ 1 singularities.

use pe_math::libm_ops;

/// Lift curve with stall + parabolic drag polar `C_D = C_D0 + C_L²/(π e AR)`.
///
/// `S` [m²] reference area, `c` [m] mean aerodynamic chord, `b` [m] span
/// (`AR = b²/S` if `aspect_ratio` is `None`). Coefficients dimensionless; angles [rad].
#[derive(Clone, Debug)]
pub struct WingPolar {
    /// Reference wing area [m²].
    pub s: f64,
    /// Mean aerodynamic chord [m].
    pub c: f64,
    /// Wing span [m].
    pub b: f64,
    /// Zero-lift drag coefficient [dimensionless].
    pub cd0: f64,
    /// Oswald efficiency factor [dimensionless].
    pub e: f64,
    /// Zero-α lift coefficient [dimensionless].
    pub cl0: f64,
    /// Lift-curve slope [1/rad] (thin-airfoil ≈ 2π).
    pub cl_alpha: f64,
    /// Stall angle of attack [rad].
    pub alpha_stall: f64,
    /// Post-stall `|dC_L/dα|` [1/rad].
    pub cl_fall: f64,
    /// Aspect ratio `b²/S` [dimensionless].
    pub aspect_ratio: f64,
    /// Zero-α pitching-moment coefficient [dimensionless].
    pub cm0: f64,
    /// Pitch stiffness `C_mα` [1/rad] (stable if < 0 with `C_Lα > 0`).
    pub cm_alpha: f64,
    /// Pitch damping `C_mq` [1/rad] (per nondimensional pitch rate).
    pub cm_q: f64,
}

/// Builder / validation error for [`WingPolar`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolarError;

impl core::fmt::Display for PolarError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid wing polar parameters")
    }
}

impl std::error::Error for PolarError {}

impl WingPolar {
    /// Construct a polar; `aspect_ratio = None` ⇒ `b²/S`.
    ///
    /// Angles in radians. Returns [`PolarError`] if `S`, `c`, `b`, `e`, or AR ≤ 0.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s: f64,
        c: f64,
        b: f64,
        cd0: f64,
        e: f64,
        cl0: f64,
        cl_alpha: f64,
        alpha_stall: f64,
        cl_fall: f64,
        aspect_ratio: Option<f64>,
        cm0: f64,
        cm_alpha: f64,
        cm_q: f64,
    ) -> Result<Self, PolarError> {
        if s <= 0.0 || c <= 0.0 || b <= 0.0 {
            return Err(PolarError);
        }
        let ar = aspect_ratio.unwrap_or(b * b / s);
        if e <= 0.0 || ar <= 0.0 {
            return Err(PolarError);
        }
        Ok(Self {
            s,
            c,
            b,
            cd0,
            e,
            cl0,
            cl_alpha,
            alpha_stall,
            cl_fall,
            aspect_ratio: ar,
            cm0,
            cm_alpha,
            cm_q,
        })
    }

    /// Light GA-like defaults matching the Python test helper.
    pub fn light_aircraft() -> Self {
        Self::new(
            16.0,
            1.6,
            10.0,
            0.025,
            0.8,
            0.2,
            5.7,
            14.0_f64.to_radians(),
            1.2,
            None,
            0.0,
            -0.8,
            -12.0,
        )
        .expect("valid defaults")
    }

    /// Elliptic-wing thin-airfoil lift slope `C_Lα = 2π / (1 + 2/AR)` [1/rad].
    pub fn elliptic_cl_alpha(aspect_ratio: f64) -> f64 {
        2.0 * core::f64::consts::PI / (1.0 + 2.0 / aspect_ratio)
    }

    /// Induced-drag factor for an elliptic wing: `C_Di = C_L² / (π AR)`.
    pub fn elliptic_induced_drag(cl: f64, aspect_ratio: f64) -> f64 {
        (cl * cl) / (core::f64::consts::PI * aspect_ratio)
    }

    /// Lift coefficient vs angle of attack `α` [rad], with a linear stall break.
    pub fn cl(&self, alpha: f64) -> f64 {
        let a_s = self.alpha_stall;
        let cl_lin = self.cl0 + self.cl_alpha * alpha;
        if libm_ops::fabs(alpha) <= a_s {
            return cl_lin;
        }
        let sign = if alpha > 0.0 { 1.0 } else { -1.0 };
        let cl_peak = self.cl0 + self.cl_alpha * sign * a_s;
        cl_peak - sign * self.cl_fall * (libm_ops::fabs(alpha) - a_s)
    }

    /// Parabolic drag polar including induced drag [dimensionless].
    pub fn cd(&self, cl: f64) -> f64 {
        self.cd0 + (cl * cl) / (core::f64::consts::PI * self.e * self.aspect_ratio)
    }

    /// Pitching moment coefficient; `q` [rad/s] body pitch rate, `V` [m/s] airspeed.
    ///
    /// Uses `V_safe = max(V, 1e-6)` so the damping term does not blow up as `V → 0`.
    pub fn cm(&self, alpha: f64, q: f64, v: f64) -> f64 {
        let v_safe = v.max(1e-6);
        self.cm0 + self.cm_alpha * alpha + self.cm_q * (q * self.c / (2.0 * v_safe))
    }

    /// Lift-to-drag ratio at `α` [rad]. Returns `+∞` if `C_D = 0`.
    pub fn l_over_d(&self, alpha: f64) -> f64 {
        let cl = self.cl(alpha);
        let cd = self.cd(cl);
        if cd == 0.0 {
            f64::INFINITY
        } else {
            cl / cd
        }
    }
}

/// Dynamic pressure `q̄ = ½ ρ V²` [Pa] (`ρ` [kg/m³], `V` [m/s]).
pub fn dynamic_pressure(rho: f64, v: f64) -> f64 {
    0.5 * rho * v * v
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    #[test]
    fn elliptic_wing_published_formulas() {
        let ar = 8.0;
        let cla = WingPolar::elliptic_cl_alpha(ar);
        assert_relative_eq!(
            cla,
            2.0 * core::f64::consts::PI / (1.0 + 2.0 / ar),
            max_relative = 1e-15
        );
        let cl = 1.0;
        assert_relative_eq!(
            WingPolar::elliptic_induced_drag(cl, ar),
            cl * cl / (core::f64::consts::PI * ar),
            max_relative = 1e-15
        );
    }

    #[test]
    fn drag_polar_and_induced() {
        let p = WingPolar::light_aircraft();
        let cl = 0.8;
        let cd_ind = cl * cl / (core::f64::consts::PI * p.e * p.aspect_ratio);
        assert_abs_diff_eq!(p.aspect_ratio, 100.0 / 16.0, epsilon = 1e-15);
        assert_abs_diff_eq!(p.cd(cl), p.cd0 + cd_ind, epsilon = 1e-15);
    }

    #[test]
    fn stall_break() {
        let p = WingPolar::light_aircraft();
        let a_s = p.alpha_stall;
        assert!(p.cl(a_s + 0.05) < p.cl(a_s));
        let a = a_s + 5.0_f64.to_radians();
        let cl_lin = p.cl0 + p.cl_alpha * a;
        assert!(p.cl(a) < cl_lin);
    }
}
