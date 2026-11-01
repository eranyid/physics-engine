//! Nondimensional stability / control derivatives (per radian).

/// Control-surface deflections [rad].
#[derive(Clone, Copy, Debug, Default)]
pub struct ControlInputs {
    /// Elevator deflection [rad] (positive trailing-edge down → nose-down `C_m`).
    pub delta_e: f64,
    /// Aileron deflection [rad] (positive right aileron trailing-edge down → left roll).
    pub delta_a: f64,
    /// Rudder deflection [rad] (positive trailing-edge left → positive yaw).
    pub delta_r: f64,
}

/// Linearised nondimensional stability and control derivatives (per radian).
///
/// Force/moment coefficients:
/// ```text
/// C_L = C_L0 + C_Lα α + C_Lq q̂ + C_Lδe δe
/// C_D = C_D0 + C_Dα α
/// C_Y = C_Yβ β + C_Yp p̂ + C_Yr r̂ + C_Yδa δa + C_Yδr δr
/// C_l = C_lβ β + C_lp p̂ + C_lr r̂ + C_lδa δa + C_lδr δr
/// C_m = C_m0 + C_mα α + C_mq q̂ + C_mδe δe
/// C_n = C_nβ β + C_np p̂ + C_nr r̂ + C_nδa δa + C_nδr δr
/// ```
/// with `p̂ = p b/(2V)`, `q̂ = q c̄/(2V)`, `r̂ = r b/(2V)`.
#[derive(Clone, Copy, Debug)]
pub struct StabilityDerivatives {
    /// Trim / zero-α lift coefficient.
    pub cl0: f64,
    /// `C_Lα` [1/rad].
    pub cl_alpha: f64,
    /// `C_Lq` [1/rad] (per nondimensional pitch rate).
    pub cl_q: f64,
    /// `C_Lδe` [1/rad].
    pub cl_de: f64,
    /// Zero-lift drag coefficient.
    pub cd0: f64,
    /// `C_Dα` [1/rad].
    pub cd_alpha: f64,
    /// Zero-α pitching-moment coefficient.
    pub cm0: f64,
    /// `C_mα` [1/rad].
    pub cm_alpha: f64,
    /// `C_mq` [1/rad].
    pub cm_q: f64,
    /// `C_mδe` [1/rad].
    pub cm_de: f64,
    /// `C_Yβ` [1/rad].
    pub cy_beta: f64,
    /// `C_Yp` [1/rad].
    pub cy_p: f64,
    /// `C_Yr` [1/rad].
    pub cy_r: f64,
    /// `C_Yδa` [1/rad].
    pub cy_da: f64,
    /// `C_Yδr` [1/rad].
    pub cy_dr: f64,
    /// `C_lβ` [1/rad].
    pub cl_beta: f64,
    /// `C_lp` [1/rad].
    pub cl_p: f64,
    /// `C_lr` [1/rad].
    pub cl_r: f64,
    /// `C_lδa` [1/rad].
    pub cl_da: f64,
    /// `C_lδr` [1/rad].
    pub cl_dr: f64,
    /// `C_nβ` [1/rad].
    pub cn_beta: f64,
    /// `C_np` [1/rad].
    pub cn_p: f64,
    /// `C_nr` [1/rad].
    pub cn_r: f64,
    /// `C_nδa` [1/rad].
    pub cn_da: f64,
    /// `C_nδr` [1/rad].
    pub cn_dr: f64,
}

impl StabilityDerivatives {
    /// Evaluate `C_L` at α [rad], nondim. pitch rate `q_hat`, elevator `de` [rad].
    pub fn cl(&self, alpha: f64, q_hat: f64, de: f64) -> f64 {
        self.cl0 + self.cl_alpha * alpha + self.cl_q * q_hat + self.cl_de * de
    }

    /// Evaluate `C_D` at α [rad].
    pub fn cd(&self, alpha: f64) -> f64 {
        self.cd0 + self.cd_alpha * alpha
    }

    /// Evaluate `C_m` at α [rad], `q_hat`, elevator `de` [rad].
    pub fn cm(&self, alpha: f64, q_hat: f64, de: f64) -> f64 {
        self.cm0 + self.cm_alpha * alpha + self.cm_q * q_hat + self.cm_de * de
    }

    /// Evaluate `C_Y` at β [rad], `p_hat`, `r_hat`, aileron/rudder [rad].
    pub fn cy(&self, beta: f64, p_hat: f64, r_hat: f64, da: f64, dr: f64) -> f64 {
        self.cy_beta * beta
            + self.cy_p * p_hat
            + self.cy_r * r_hat
            + self.cy_da * da
            + self.cy_dr * dr
    }

    /// Evaluate rolling-moment coeff. `C_l`.
    pub fn c_l_roll(&self, beta: f64, p_hat: f64, r_hat: f64, da: f64, dr: f64) -> f64 {
        self.cl_beta * beta
            + self.cl_p * p_hat
            + self.cl_r * r_hat
            + self.cl_da * da
            + self.cl_dr * dr
    }

    /// Evaluate yawing-moment coeff. `C_n`.
    pub fn cn(&self, beta: f64, p_hat: f64, r_hat: f64, da: f64, dr: f64) -> f64 {
        self.cn_beta * beta
            + self.cn_p * p_hat
            + self.cn_r * r_hat
            + self.cn_da * da
            + self.cn_dr * dr
    }
}
