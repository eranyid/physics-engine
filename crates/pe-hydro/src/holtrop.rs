//! Holtrop–Mennen (1982) calm-water resistance prediction.
//!
//! # Primary source (verified numerical example)
//!
//! Holtrop, J. & Mennen, G.G.J. (1982). An approximate power prediction method.
//! *International Shipbuilding Progress* **29**(335), 166–170 — Section 5
//! numerical example (hypothetical single-screw ship at 25 knots).
//!
//! Formulas transcribed clean-room from that paper. The 1984 re-analysis
//! (Holtrop, J., ISP **31**(363), 272–276) updates the form-factor and high-Fn
//! wave formulas; this module implements the **1982** decomposition that
//! matches the published Section 5 table. 1984 form-factor is available via
//! [`form_factor_1_plus_k_1984`] (labelled; not used by the 1982 oracle).
//!
//! # Total resistance
//!
//! ```text
//! R_T = R_F (1+k₁) + R_APP + R_W + R_B + R_TR + R_A
//! ```
//!
//! with ITTC-1957 `C_F`, Holtrop form factor, appendage, wave-making, bulb,
//! transom, and model–ship correlation terms.
//!
//! # Units
//!
//! SI throughout. Speed may be supplied in m/s; the 1982 example uses
//! 25 knots (= 25 × 1852/3600 m/s).

use pe_math::libm_ops;
use pe_math::STANDARD_GRAVITY;

use crate::error::{require_finite, require_non_negative, require_positive, HydroError};
use crate::resistance::{try_froude_number, try_ittc57_cf_from_speed, NU_SEAWATER_15C_NOMINAL};

/// Seawater density [kg/m³] used in Holtrop & Mennen (1982) §5 (15 °C).
pub const RHO_SEAWATER_HOLTROP_15C: f64 = 1025.0;

/// Knots → m/s (`1852/3600`).
pub const KNOTS_TO_MS: f64 = 1852.0 / 3600.0;

/// Stern shape coefficient guidelines (Holtrop & Mennen 1982 Table).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SternType {
    /// V-shaped sections (`C_stern = −10`).
    VShaped,
    /// Normal section shape (`C_stern = 0`).
    Normal,
    /// U-shaped sections with Hogner stern (`C_stern = +10`).
    UShapedHogner,
}

impl SternType {
    /// Published `C_stern` value.
    pub fn c_stern(self) -> f64 {
        match self {
            SternType::VShaped => -10.0,
            SternType::Normal => 0.0,
            SternType::UShapedHogner => 10.0,
        }
    }
}

/// Hull / appendage inputs for Holtrop–Mennen resistance (SI).
#[derive(Clone, Copy, Debug)]
pub struct HoltropInput {
    /// Waterline length `L` [m].
    pub l: f64,
    /// Moulded breadth `B` [m].
    pub b: f64,
    /// Mean moulded draught `T = (T_F + T_A)/2` [m] (used in form / wave).
    pub t: f64,
    /// Forward draught `T_F` [m].
    pub t_f: f64,
    /// Aft draught `T_A` [m].
    pub t_a: f64,
    /// Moulded displaced volume `∇` [m³].
    pub displacement: f64,
    /// Midship section coefficient `C_M`.
    pub c_m: f64,
    /// Waterplane area coefficient `C_WP`.
    pub c_wp: f64,
    /// Longitudinal centre of buoyancy as **percent of L** forward of midships
    /// (`lcb`; aft is negative). Holtrop 1982 §5 result table uses −0.75.
    pub lcb_pct: f64,
    /// Transverse bulb area at the still-water stem intersection `A_BT` [m²].
    pub a_bt: f64,
    /// Height of bulb-area centroid above keel `h_B` [m].
    pub h_b: f64,
    /// Immersed transom area at rest `A_T` [m²].
    pub a_t: f64,
    /// Appendage wetted area `S_APP` [m²].
    pub s_app: f64,
    /// Equivalent appendage form factor `(1+k₂)_eq`.
    pub one_plus_k2: f64,
    /// Stern shape coefficient `C_stern`.
    pub c_stern: f64,
    /// Ship speed [m/s].
    pub speed: f64,
    /// Fluid density [kg/m³].
    pub rho: f64,
    /// Kinematic viscosity [m²/s].
    pub nu: f64,
    /// Gravity [m/s²].
    pub g: f64,
    /// Optional half angle of entrance [degrees]. `None` → Holtrop regression.
    pub i_e_deg: Option<f64>,
    /// Optional hull wetted area [m²]. `None` → Holtrop approximation.
    pub s_hull: Option<f64>,
}

impl HoltropInput {
    /// Holtrop & Mennen (1982) §5 hypothetical single-screw ship at 25 knots.
    ///
    /// Inputs transcribed from the published table; `lcb_pct = −0.75` is the
    /// value relative to midships of `L` used in the result coefficients.
    pub fn holtrop_1982_example() -> Self {
        let speed = 25.0 * KNOTS_TO_MS;
        Self {
            l: 205.0,
            b: 32.0,
            t: 10.0,
            t_f: 10.0,
            t_a: 10.0,
            displacement: 37_500.0,
            c_m: 0.980,
            c_wp: 0.750,
            lcb_pct: -0.75,
            a_bt: 20.0,
            h_b: 4.0,
            a_t: 16.0,
            s_app: 50.0,
            one_plus_k2: 1.50,
            c_stern: 10.0,
            speed,
            rho: RHO_SEAWATER_HOLTROP_15C,
            nu: NU_SEAWATER_15C_NOMINAL,
            g: STANDARD_GRAVITY,
            i_e_deg: None,
            s_hull: None,
        }
    }

    /// Block coefficient `C_B = ∇ / (L B T)`.
    pub fn c_b(&self) -> f64 {
        self.displacement / (self.l * self.b * self.t)
    }

    /// Prismatic coefficient `C_P = C_B / C_M`.
    pub fn c_p(&self) -> f64 {
        self.c_b() / self.c_m
    }

    /// Validate finite / positive domains for Holtrop evaluation.
    pub fn try_validate(&self) -> Result<(), HydroError> {
        require_positive(self.l, "L")?;
        require_positive(self.b, "B")?;
        require_positive(self.t, "T")?;
        require_positive(self.t_f, "T_F")?;
        require_positive(self.t_a, "T_A")?;
        require_positive(self.displacement, "displacement")?;
        require_positive(self.c_m, "C_M")?;
        require_positive(self.c_wp, "C_WP")?;
        require_finite(self.lcb_pct, "lcb")?;
        require_non_negative(self.a_bt, "A_BT")?;
        require_non_negative(self.h_b, "h_B")?;
        require_non_negative(self.a_t, "A_T")?;
        require_non_negative(self.s_app, "S_APP")?;
        require_positive(self.one_plus_k2, "1+k2")?;
        require_finite(self.c_stern, "C_stern")?;
        require_non_negative(self.speed, "speed")?;
        require_positive(self.rho, "rho")?;
        require_positive(self.nu, "nu")?;
        require_positive(self.g, "g")?;
        if let Some(ie) = self.i_e_deg {
            require_finite(ie, "i_E")?;
        }
        if let Some(s) = self.s_hull {
            require_positive(s, "S")?;
        }
        let cp = self.c_p();
        if !(cp > 0.0 && cp < 1.0) {
            return Err(HydroError::InvalidParameter("C_P must be in (0,1)"));
        }
        Ok(())
    }
}

/// Component breakdown of Holtrop–Mennen calm-water resistance [N / —].
#[derive(Clone, Copy, Debug)]
pub struct HoltropResult {
    /// Froude number based on `L`.
    pub froude: f64,
    /// Reynolds number based on `L`.
    pub reynolds: f64,
    /// ITTC-1957 frictional coefficient `C_F`.
    pub c_f: f64,
    /// Hull wetted area `S` [m²].
    pub s: f64,
    /// Form factor `1+k₁`.
    pub one_plus_k1: f64,
    /// Length of run `L_R` [m].
    pub l_r: f64,
    /// Half angle of entrance [degrees].
    pub i_e_deg: f64,
    /// Frictional resistance `R_F` [N].
    pub r_f: f64,
    /// Appendage resistance `R_APP` [N].
    pub r_app: f64,
    /// Wave-making resistance `R_W` [N].
    pub r_w: f64,
    /// Bulb near-surface pressure resistance `R_B` [N].
    pub r_b: f64,
    /// Immersed-transom pressure resistance `R_TR` [N].
    pub r_tr: f64,
    /// Model–ship correlation resistance `R_A` [N].
    pub r_a: f64,
    /// Correlation allowance `C_A`.
    pub c_a: f64,
    /// Total resistance `R_T` [N].
    pub r_total: f64,
    /// Effective (towing) power `P_E = R_T · V` [W].
    pub p_e: f64,
}

/// Length of run `L_R` [m] (Holtrop 1982).
pub fn length_of_run(l: f64, c_p: f64, lcb_pct: f64) -> f64 {
    l * (1.0 - c_p + 0.06 * c_p * lcb_pct / (4.0 * c_p - 1.0))
}

/// Holtrop 1982 form factor `1+k₁`.
pub fn form_factor_1_plus_k1_1982(b: f64, t: f64, l: f64, c_p: f64, lcb_pct: f64, c_stern: f64) -> f64 {
    let l_r = length_of_run(l, c_p, lcb_pct);
    let t_over_l = t / l;
    let c12 = if t_over_l > 0.05 {
        libm_ops::pow(t_over_l, 0.2228446)
    } else if t_over_l > 0.02 {
        48.20 * libm_ops::pow(t_over_l - 0.02, 2.078) + 0.479948
    } else {
        0.479948
    };
    let c13 = 1.0 + 0.003 * c_stern;
    c13 * (0.93
        + c12
            * libm_ops::pow(b / l_r, 0.92497)
            * libm_ops::pow(0.95 - c_p, -0.521448)
            * libm_ops::pow(1.0 - c_p + 0.0225 * lcb_pct, 0.6906))
}

/// Holtrop 1984 form factor `1+k₁` (not used by the 1982 §5 oracle).
pub fn form_factor_1_plus_k_1984(
    b: f64,
    t: f64,
    l: f64,
    displacement: f64,
    c_p: f64,
    lcb_pct: f64,
    c_stern: f64,
) -> f64 {
    let l_r = length_of_run(l, c_p, lcb_pct);
    let c14 = 1.0 + 0.011 * c_stern;
    0.93 + 0.487118
        * c14
        * libm_ops::pow(b / l, 1.06806)
        * libm_ops::pow(t / l, 0.46106)
        * libm_ops::pow(l / l_r, 0.121563)
        * libm_ops::pow(l * l * l / displacement, 0.36486)
        * libm_ops::pow(1.0 - c_p, -0.604247)
}

/// Holtrop wetted-surface approximation [m²].
pub fn wetted_surface(
    l: f64,
    b: f64,
    t: f64,
    c_m: f64,
    c_b: f64,
    c_wp: f64,
    a_bt: f64,
) -> f64 {
    l * (2.0 * t + b)
        * libm_ops::sqrt(c_m)
        * (0.453 + 0.4425 * c_b - 0.2862 * c_m - 0.003467 * (b / t) + 0.3696 * c_wp)
        + 2.38 * a_bt / c_b
}

/// Half angle of entrance [degrees] (Holtrop 1982 regression).
pub fn half_angle_of_entrance(
    l: f64,
    b: f64,
    c_wp: f64,
    c_p: f64,
    lcb_pct: f64,
    l_r: f64,
    displacement: f64,
) -> f64 {
    let vol_term = 100.0 * displacement / (l * l * l);
    1.0 + 89.0
        * libm_ops::exp(
            -(libm_ops::pow(l / b, 0.80856)
                * libm_ops::pow(1.0 - c_wp, 0.30484)
                * libm_ops::pow(1.0 - c_p - 0.0225 * lcb_pct, 0.6367)
                * libm_ops::pow(l_r / b, 0.34574)
                * libm_ops::pow(vol_term, 0.16302)),
        )
}

fn c7_coeff(b_over_l: f64) -> f64 {
    if b_over_l < 0.11 {
        0.229577 * libm_ops::pow(b_over_l, 0.33333)
    } else if b_over_l <= 0.25 {
        b_over_l
    } else {
        0.5 - 0.0625 / b_over_l // 0.5 - 0.0625*(L/B) = 0.5 - 0.0625/b_over_l
    }
}

fn c16_coeff(c_p: f64) -> f64 {
    if c_p < 0.80 {
        8.07981 * c_p - 13.8673 * c_p * c_p + 6.984388 * c_p * c_p * c_p
    } else {
        1.73014 - 0.7067 * c_p
    }
}

fn c15_coeff(l: f64, displacement: f64) -> f64 {
    let ratio = (l * l * l) / displacement;
    if ratio < 512.0 {
        -1.69385
    } else if ratio <= 1727.0 {
        -1.69385 + (l / libm_ops::cbrt(displacement) - 8.0) / 2.36
    } else {
        0.0
    }
}

/// Evaluate Holtrop–Mennen 1982 calm-water resistance.
pub fn try_holtrop_mennen(input: &HoltropInput) -> Result<HoltropResult, HydroError> {
    input.try_validate()?;
    let c_b = input.c_b();
    let c_p = input.c_p();
    require_finite(c_b, "C_B")?;
    require_finite(c_p, "C_P")?;

    let froude = try_froude_number(input.speed, input.l, input.g)?;
    let reynolds = if input.speed > 0.0 {
        input.speed * input.l / input.nu
    } else {
        0.0
    };
    let c_f = try_ittc57_cf_from_speed(input.speed, input.l, input.nu)?;
    let s = match input.s_hull {
        Some(s) => s,
        None => wetted_surface(input.l, input.b, input.t, input.c_m, c_b, input.c_wp, input.a_bt),
    };
    require_finite(s, "S")?;

    let l_r = length_of_run(input.l, c_p, input.lcb_pct);
    let one_plus_k1 = form_factor_1_plus_k1_1982(
        input.b,
        input.t,
        input.l,
        c_p,
        input.lcb_pct,
        input.c_stern,
    );
    require_finite(one_plus_k1, "1+k1")?;

    let i_e = match input.i_e_deg {
        Some(v) => v,
        None => half_angle_of_entrance(
            input.l,
            input.b,
            input.c_wp,
            c_p,
            input.lcb_pct,
            l_r,
            input.displacement,
        ),
    };
    require_finite(i_e, "i_E")?;

    let dyn_p = 0.5 * input.rho * input.speed * input.speed;
    let r_f = dyn_p * s * c_f;
    let r_app = dyn_p * input.s_app * input.one_plus_k2 * c_f;

    // Wave resistance coefficients (1982).
    let b_over_l = input.b / input.l;
    let c7 = c7_coeff(b_over_l);
    let c1 = 2223105.0
        * libm_ops::pow(c7, 3.78613)
        * libm_ops::pow(input.t / input.b, 1.07961)
        * libm_ops::pow(90.0 - i_e, -1.37565);

    let c3 = if input.a_bt > 0.0 {
        0.56 * libm_ops::pow(input.a_bt, 1.5)
            / (input.b
                * input.t
                * (0.31 * libm_ops::sqrt(input.a_bt) + input.t_f - input.h_b))
    } else {
        0.0
    };
    let c2 = if input.a_bt > 0.0 {
        libm_ops::exp(-1.89 * libm_ops::sqrt(c3.max(0.0)))
    } else {
        1.0
    };
    let c5 = 1.0 - 0.8 * input.a_t / (input.b * input.t * input.c_m);

    let lambda = if input.l / input.b < 12.0 {
        1.446 * c_p - 0.03 * (input.l / input.b)
    } else {
        1.446 * c_p - 0.36
    };
    let m1 = 0.0140407 * (input.l / input.t)
        - 1.75254 * libm_ops::cbrt(input.displacement) / input.l
        - 4.79323 * b_over_l
        - c16_coeff(c_p);
    let c15 = c15_coeff(input.l, input.displacement);
    let m2 = if froude > 1e-12 {
        c15 * c_p * c_p * libm_ops::exp(-0.1 * libm_ops::pow(froude, -2.0))
    } else {
        0.0
    };
    let d = -0.9;
    let r_w = if froude > 1e-12 {
        c1 * c2
            * c5
            * input.displacement
            * input.rho
            * input.g
            * libm_ops::exp(
                m1 * libm_ops::pow(froude, d) + m2 * libm_ops::cos(lambda * libm_ops::pow(froude, -2.0)),
            )
    } else {
        0.0
    };

    // Bulb near-surface pressure resistance.
    let r_b = if input.a_bt > 0.0 {
        let pb_den = input.t_f - 1.5 * input.h_b;
        if pb_den.abs() < 1e-15 {
            0.0
        } else {
            let p_b = 0.56 * libm_ops::sqrt(input.a_bt) / pb_den;
            let fni_den = input.g
                * (input.t_f - input.h_b - 0.25 * libm_ops::sqrt(input.a_bt))
                + 0.15 * input.speed * input.speed;
            if fni_den <= 0.0 {
                0.0
            } else {
                let f_ni = input.speed / libm_ops::sqrt(fni_den);
                0.11 * libm_ops::exp(-3.0 / (p_b * p_b))
                    * libm_ops::pow(f_ni, 3.0)
                    * libm_ops::pow(input.a_bt, 1.5)
                    * input.rho
                    * input.g
                    / (1.0 + f_ni * f_ni)
            }
        }
    } else {
        0.0
    };

    // Transom resistance.
    let r_tr = if input.a_t > 0.0 {
        let fn_t = input.speed
            / libm_ops::sqrt(2.0 * input.g * input.a_t / (input.b + input.b * input.c_wp));
        let c6 = if fn_t < 5.0 {
            0.2 * (1.0 - 0.2 * fn_t)
        } else {
            0.0
        };
        dyn_p * input.a_t * c6
    } else {
        0.0
    };

    // Model–ship correlation.
    let c4 = if input.t_f / input.l <= 0.04 {
        input.t_f / input.l
    } else {
        0.04
    };
    let c_a = 0.006 * libm_ops::pow(input.l + 100.0, -0.16) - 0.00205
        + 0.003 * libm_ops::sqrt(input.l / 7.5) * libm_ops::pow(c_b, 4.0) * c2 * (0.04 - c4);
    let r_a = dyn_p * s * c_a;

    let r_total = r_f * one_plus_k1 + r_app + r_w + r_b + r_tr + r_a;
    let p_e = r_total * input.speed;

    for (name, v) in [
        ("R_F", r_f),
        ("R_APP", r_app),
        ("R_W", r_w),
        ("R_B", r_b),
        ("R_TR", r_tr),
        ("R_A", r_a),
        ("R_total", r_total),
        ("C_A", c_a),
    ] {
        require_finite(v, name)?;
    }

    Ok(HoltropResult {
        froude,
        reynolds,
        c_f,
        s,
        one_plus_k1,
        l_r,
        i_e_deg: i_e,
        r_f,
        r_app,
        r_w,
        r_b,
        r_tr,
        r_a,
        c_a,
        r_total,
        p_e,
    })
}

/// Infallible wrapper around [`try_holtrop_mennen`] (returns zeros on error).
pub fn holtrop_mennen(input: &HoltropInput) -> HoltropResult {
    try_holtrop_mennen(input).unwrap_or(HoltropResult {
        froude: 0.0,
        reynolds: 0.0,
        c_f: 0.0,
        s: 0.0,
        one_plus_k1: 1.0,
        l_r: 0.0,
        i_e_deg: 0.0,
        r_f: 0.0,
        r_app: 0.0,
        r_w: 0.0,
        r_b: 0.0,
        r_tr: 0.0,
        r_a: 0.0,
        c_a: 0.0,
        r_total: 0.0,
        p_e: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn holtrop_1982_section5_oracle() {
        // Holtrop & Mennen (1982) ISP 29(335) §5 numerical example — published
        // result table (forces in kN, power in kW).
        let r = try_holtrop_mennen(&HoltropInput::holtrop_1982_example()).unwrap();

        assert_relative_eq!(r.froude, 0.2868, epsilon = 5e-4);
        assert_relative_eq!(r.c_f, 0.001390, epsilon = 5e-6);
        assert_relative_eq!(r.s, 7381.45, epsilon = 1.0); // m²
        assert_relative_eq!(r.one_plus_k1, 1.156, epsilon = 2e-3);
        assert_relative_eq!(r.l_r, 81.385, epsilon = 0.05);
        assert_relative_eq!(r.i_e_deg, 12.08, epsilon = 0.15);

        assert_relative_eq!(r.r_f / 1e3, 869.63, epsilon = 2.0); // kN
        assert_relative_eq!(r.r_app / 1e3, 8.83, epsilon = 0.2);
        assert_relative_eq!(r.r_w / 1e3, 557.11, epsilon = 5.0);
        assert_relative_eq!(r.r_b / 1e3, 0.049, epsilon = 0.05);
        assert_relative_eq!(r.r_tr / 1e3, 0.0, epsilon = 0.05);
        assert_relative_eq!(r.c_a, 0.000352, epsilon = 5e-6);
        assert_relative_eq!(r.r_a / 1e3, 221.98, epsilon = 2.0);
        assert_relative_eq!(r.r_total / 1e3, 1793.26, epsilon = 8.0); // kN
        assert_relative_eq!(r.p_e / 1e3, 23063.0, epsilon = 120.0); // kW
    }

    #[test]
    fn try_holtrop_rejects_nan() {
        let mut inp = HoltropInput::holtrop_1982_example();
        inp.l = f64::NAN;
        assert!(matches!(
            try_holtrop_mennen(&inp),
            Err(HydroError::InvalidSize("L") | HydroError::NonFinite(_))
        ));
    }
}
