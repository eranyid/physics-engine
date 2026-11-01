//! North American Navion published data (Nelson / Teper / NASA).
//!
//! # Citations (public / textbook tables — numbers only, clean-room code)
//!
//! - Nelson, R. C. *Flight Stability and Automatic Control*, Appendix B,
//!   Table B.1 (general aviation airplane: Navion). Coefficients per radian;
//!   `W = 2750` lbf; `I_x,I_y,I_z = 1048, 3000, 3530` slug·ft²; CG 29.5% MAC.
//!   Nelson cites Teper (STI TR 176-1) and Heffley & Jewell (NASA CR-2144).
//! - Teper, G. L. *Aircraft Stability and Control Data*, STI TR 176-1, 1969.
//! - Heffley, R. K.; Jewell, W. F. *Aircraft Handling Qualities Data*,
//!   NASA CR-2144, December 1972 (public NASA report).
//! - NASA TN D-6643 (1972) Table I: `S = 17.112 m² (184 ft²)`,
//!   `c̄ = 1.74 m (5.7 ft)` — geometry cross-check (public domain).
//!
//! Trim condition used with these derivatives: sea level, `V = 176` ft/s,
//! `C_L = 0.41`, `C_D = 0.05` (Nelson Table B.1).
//!
//! Lateral set uses the standard reproduction of Nelson/Teper
//! (`C_nβ = +0.071`, `C_Yδr = 0.157`, `C_lδa = 0.1342`, …). All fields below
//! are marked `verified: true`.

use pe_math::{Vec3, STANDARD_GRAVITY};

use super::derivatives::{ControlInputs, StabilityDerivatives};
use super::forces::SixDofModel;
use crate::atmosphere::{DensityModel, RHO0};

/// Conversion: 1 foot = 0.3048 m (exact).
const FT: f64 = 0.3048;
/// Conversion: 1 slug·ft² → kg·m².
const SLUG_FT2: f64 = 1.355_817_948_331_400_4;
/// Conversion: 1 lbf → N.
const LBF: f64 = 4.448_221_615_260_5;

/// Bundled Navion data with provenance flag.
#[derive(Clone, Debug)]
pub struct NavionData {
    /// Six-DOF model in SI.
    pub model: SixDofModel,
    /// Reference true airspeed for the published derivative set [m/s].
    pub v_ref: f64,
    /// Published trim lift coefficient at `v_ref` (Nelson Table B.1).
    pub cl_trim: f64,
    /// Published trim drag coefficient at `v_ref` (Nelson Table B.1).
    pub cd_trim: f64,
    /// Approximate trim AoA [rad] consistent with `C_L = C_Lα α` and `C_L0 = 0`.
    pub alpha_trim: f64,
    /// `true` when numbers are taken from the cited public/textbook sources.
    pub verified: bool,
    /// Short citation string for docs / errors.
    pub citation: &'static str,
}

/// Nelson Appendix B / Teper Navion at sea level, SI.
///
/// Geometry: `S = 184 ft²`, `b = 33.4 ft`, `c̄ = 5.7 ft` (NASA TN D-6643 /
/// Nelson reference geometry). Mass from `W = 2750` lbf / `g₀`.
///
/// Absolute coefficients: Nelson tabulates flight-condition `C_L = 0.41` and
/// `C_D = 0.05` together with slopes. We take `C_L0 = 0` so
/// `α_trim = C_L / C_Lα`, and shift drag so `C_D(α_trim) = 0.05`:
/// `C_D0 = 0.05 − C_Dα α_trim`.
pub fn navion_nelson() -> NavionData {
    let s = 184.0 * FT * FT;
    let b = 33.4 * FT;
    let c = 5.7 * FT;
    let weight_n = 2750.0 * LBF;
    let mass = weight_n / STANDARD_GRAVITY;
    let ixx = 1048.0 * SLUG_FT2;
    let iyy = 3000.0 * SLUG_FT2;
    let izz = 3530.0 * SLUG_FT2;
    let v_ref = 176.0 * FT;

    let cl_trim = 0.41;
    let cd_trim = 0.05;
    let cl_alpha = 4.44;
    let cd_alpha = 0.33;
    let alpha_trim = cl_trim / cl_alpha;

    let deriv = StabilityDerivatives {
        cl0: 0.0,
        cl_alpha,
        // Nelson Table B.1: CL_α̇ = 0.0, Cm_α̇ = −4.36
        cl_alphadot: 0.0,
        cl_q: 3.8,
        cl_de: 0.355,
        cd0: cd_trim - cd_alpha * alpha_trim,
        cd_alpha,
        cm0: 0.0,
        cm_alpha: -0.683,
        cm_alphadot: -4.36,
        cm_q: -9.96,
        cm_de: -0.923,
        cy_beta: -0.564,
        cy_p: 0.0,
        cy_r: 0.0,
        cy_da: 0.0,
        cy_dr: 0.157,
        cl_beta: -0.074,
        cl_p: -0.410,
        cl_r: 0.107,
        cl_da: 0.1342,
        cl_dr: 0.0118,
        cn_beta: 0.071,
        cn_p: -0.0575,
        cn_r: -0.125,
        cn_da: -0.0035,
        cn_dr: -0.072,
    };

    let model = SixDofModel {
        mass,
        s,
        b,
        c,
        inertia_diag: (ixx, iyy, izz),
        deriv,
        thrust: 0.0,
        thrust_axis: Vec3::unit_x(),
        thrust_offset: Vec3::zero(),
        density: DensityModel::Constant(RHO0),
        wind: Vec3::zero(),
        controls: ControlInputs::default(),
        alpha_trim: 0.0, // absolute α; α_trim stored on NavionData
        verified: true,
        ground_height: None,
    };

    NavionData {
        model,
        v_ref,
        cl_trim,
        cd_trim,
        alpha_trim,
        verified: true,
        citation: "Nelson App. B Table B.1 / Teper STI TR 176-1 / NASA CR-2144; geometry NASA TN D-6643 Table I",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn navion_mass_from_weight() {
        let n = navion_nelson();
        let w_lbf = n.model.mass * STANDARD_GRAVITY / LBF;
        assert_relative_eq!(w_lbf, 2750.0, max_relative = 1e-9);
    }

    #[test]
    fn navion_geometry_si() {
        let n = navion_nelson();
        // 184 ft² · (0.3048 m/ft)² — NASA TN D-6643 lists 17.112 m² (rounded).
        assert_relative_eq!(n.model.s, 184.0 * FT * FT, max_relative = 1e-15);
        assert_relative_eq!(n.model.b, 33.4 * FT, max_relative = 1e-15);
        assert_relative_eq!(n.model.c, 5.7 * FT, max_relative = 1e-15);
        assert!(n.verified);
        assert_relative_eq!(
            n.model.deriv.cl(n.alpha_trim, 0.0, 0.0),
            n.cl_trim,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            n.model.deriv.cd(n.alpha_trim),
            n.cd_trim,
            max_relative = 1e-12
        );
    }
}
