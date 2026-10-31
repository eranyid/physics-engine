//! Von Mises J2 plasticity with linear isotropic hardening (1-D and 3-D).
//!
//! Clean-room textbook return mapping (Simo & Hughes / continuum mechanics
//! identities). No third-party simulation source translated.
//!
//! # 1-D uniaxial
//!
//! Trial stress `σ_tr = E (ε − ε_p)`. Yield `f = |σ_tr| − (σ_y + H α)`.
//! If `f > 0`, plastic increment `Δγ = f / (E + H)`, update
//! `ε_p ← ε_p + Δγ sgn(σ_tr)`, `α ← α + Δγ`, `σ ← σ_tr − E Δγ sgn(σ_tr)`.
//! Continuum tangent in plastic loading: `E_t = E H / (E + H)`.
//!
//! # 3-D
//!
//! Deviatoric trial `s_tr = dev(σ_tr)`, `σ_eq = √(3/2) ‖s_tr‖`.
//! `Δγ = f / (3μ + H)`, `s ← (1 − 3μ Δγ / σ_eq) s_tr`, hydrostatic unchanged.

use pe_math::libm_ops;

/// Linear isotropic hardening J2 material parameters (SI).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct J2Params {
    /// Young's modulus E [Pa] (1-D path) / used with ν for 3-D Lamé.
    pub youngs_modulus: f64,
    /// Poisson's ratio ν (3-D only; unused in pure 1-D).
    pub poisson_ratio: f64,
    /// Initial yield stress σ_y [Pa].
    pub yield_strength: f64,
    /// Linear isotropic hardening modulus H [Pa] (`σ_y_eff = σ_y + H α`).
    pub hardening_modulus: f64,
}

impl J2Params {
    /// Shear modulus μ = E / (2(1+ν)).
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Continuum elasto-plastic tangent in uniaxial plastic loading: E_t = E H/(E+H).
    pub fn uniaxial_tangent(&self) -> f64 {
        let e = self.youngs_modulus;
        let h = self.hardening_modulus;
        if (e + h).abs() < 1e-30 {
            return 0.0;
        }
        e * h / (e + h)
    }
}

/// 1-D uniaxial plastic state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UniaxialPlasticState {
    /// Plastic strain ε_p [dimensionless].
    pub plastic_strain: f64,
    /// Equivalent plastic strain α ≥ 0 [dimensionless].
    pub eq_plastic_strain: f64,
    /// Current Cauchy stress σ [Pa].
    pub stress: f64,
}

/// Result of one 1-D return-mapping step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniaxialReturn {
    /// Updated state.
    pub state: UniaxialPlasticState,
    /// `true` if the step was plastic (yield surface active).
    pub plastic: bool,
    /// Plastic multiplier increment Δγ ≥ 0.
    pub delta_gamma: f64,
}

/// 1-D von Mises / |σ| yield with linear isotropic hardening.
pub fn return_map_uniaxial(
    params: &J2Params,
    state: UniaxialPlasticState,
    strain: f64,
) -> UniaxialReturn {
    let e = params.youngs_modulus;
    let h = params.hardening_modulus;
    let sig_y = params.yield_strength;
    let eps_p = state.plastic_strain;
    let alpha = state.eq_plastic_strain;

    let sig_tr = e * (strain - eps_p);
    let f_tr = libm_ops::fabs(sig_tr) - (sig_y + h * alpha);
    if f_tr <= 0.0 {
        return UniaxialReturn {
            state: UniaxialPlasticState {
                plastic_strain: eps_p,
                eq_plastic_strain: alpha,
                stress: sig_tr,
            },
            plastic: false,
            delta_gamma: 0.0,
        };
    }
    let denom = e + h;
    let dg = if denom > 0.0 { f_tr / denom } else { 0.0 };
    let sgn = if sig_tr >= 0.0 { 1.0 } else { -1.0 };
    UniaxialReturn {
        state: UniaxialPlasticState {
            plastic_strain: eps_p + dg * sgn,
            eq_plastic_strain: alpha + dg,
            stress: sig_tr - e * dg * sgn,
        },
        plastic: true,
        delta_gamma: dg,
    }
}

/// Symmetric 3×3 stress/strain tensor stored row-major (Voigt-friendly accessors).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SymTensor3 {
    /// σ_xx / ε_xx.
    pub xx: f64,
    /// σ_yy / ε_yy.
    pub yy: f64,
    /// σ_zz / ε_zz.
    pub zz: f64,
    /// σ_yz / ε_yz (engineering shear unused — tensor component).
    pub yz: f64,
    /// σ_xz / ε_xz.
    pub xz: f64,
    /// σ_xy / ε_xy.
    pub xy: f64,
}

impl SymTensor3 {
    /// Zero tensor.
    pub fn zero() -> Self {
        Self::default()
    }

    /// Hydrostatic pressure p = −tr/3 (compression positive convention unused;
    /// returns mean stress tr/3).
    pub fn mean(&self) -> f64 {
        (self.xx + self.yy + self.zz) / 3.0
    }

    /// Deviator `s = σ − mean I`.
    pub fn deviator(&self) -> Self {
        let m = self.mean();
        Self {
            xx: self.xx - m,
            yy: self.yy - m,
            zz: self.zz - m,
            yz: self.yz,
            xz: self.xz,
            xy: self.xy,
        }
    }

    /// Frobenius norm ‖·‖_F.
    pub fn frobenius(&self) -> f64 {
        libm_ops::sqrt(
            self.xx * self.xx
                + self.yy * self.yy
                + self.zz * self.zz
                + 2.0 * (self.yz * self.yz + self.xz * self.xz + self.xy * self.xy),
        )
    }

    /// Von Mises equivalent stress √(3/2) ‖s‖ for a stress tensor (uses full σ;
    /// callers typically pass the deviator).
    pub fn von_mises_eq(&self) -> f64 {
        let s = self.deviator();
        libm_ops::sqrt(1.5) * s.frobenius()
    }

    /// Scale.
    pub fn scale(&self, a: f64) -> Self {
        Self {
            xx: self.xx * a,
            yy: self.yy * a,
            zz: self.zz * a,
            yz: self.yz * a,
            xz: self.xz * a,
            xy: self.xy * a,
        }
    }

    /// Add.
    pub fn add(&self, o: &Self) -> Self {
        Self {
            xx: self.xx + o.xx,
            yy: self.yy + o.yy,
            zz: self.zz + o.zz,
            yz: self.yz + o.yz,
            xz: self.xz + o.xz,
            xy: self.xy + o.xy,
        }
    }

    /// Isotropic Hooke: σ = λ tr(ε) I + 2μ ε from strain tensor.
    pub fn from_hooke_strain(strain: &Self, lambda: f64, mu: f64) -> Self {
        let tr = strain.xx + strain.yy + strain.zz;
        Self {
            xx: lambda * tr + 2.0 * mu * strain.xx,
            yy: lambda * tr + 2.0 * mu * strain.yy,
            zz: lambda * tr + 2.0 * mu * strain.zz,
            yz: 2.0 * mu * strain.yz,
            xz: 2.0 * mu * strain.xz,
            xy: 2.0 * mu * strain.xy,
        }
    }
}

/// 3-D J2 plastic state (strain-driven).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct J2PlasticState3 {
    /// Plastic strain tensor ε_p.
    pub plastic_strain: SymTensor3,
    /// Equivalent plastic strain α.
    pub eq_plastic_strain: f64,
    /// Current Cauchy stress.
    pub stress: SymTensor3,
}

/// Result of a 3-D J2 return map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct J2Return3 {
    /// Updated state.
    pub state: J2PlasticState3,
    /// Plastic step?
    pub plastic: bool,
    /// Δγ ≥ 0.
    pub delta_gamma: f64,
}

/// 3-D J2 radial return with linear isotropic hardening (strain-driven).
///
/// Elastic trial: `σ_tr = λ tr(ε−ε_p) I + 2μ (ε−ε_p)`.
pub fn return_map_j2_3d(
    params: &J2Params,
    state: J2PlasticState3,
    strain: SymTensor3,
) -> J2Return3 {
    let e = params.youngs_modulus;
    let nu = params.poisson_ratio;
    let mu = e / (2.0 * (1.0 + nu));
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let h = params.hardening_modulus;
    let sig_y = params.yield_strength;

    let eps_e = SymTensor3 {
        xx: strain.xx - state.plastic_strain.xx,
        yy: strain.yy - state.plastic_strain.yy,
        zz: strain.zz - state.plastic_strain.zz,
        yz: strain.yz - state.plastic_strain.yz,
        xz: strain.xz - state.plastic_strain.xz,
        xy: strain.xy - state.plastic_strain.xy,
    };
    let sig_tr = SymTensor3::from_hooke_strain(&eps_e, lambda, mu);
    let s_tr = sig_tr.deviator();
    let seq = libm_ops::sqrt(1.5) * s_tr.frobenius();
    let f_tr = seq - (sig_y + h * state.eq_plastic_strain);
    if f_tr <= 0.0 || seq < 1e-30 {
        return J2Return3 {
            state: J2PlasticState3 {
                plastic_strain: state.plastic_strain,
                eq_plastic_strain: state.eq_plastic_strain,
                stress: sig_tr,
            },
            plastic: false,
            delta_gamma: 0.0,
        };
    }
    let dg = f_tr / (3.0 * mu + h);
    // n = (3/2) s / σ_eq  ⇒  Δε_p = Δγ n,  s_new = s_tr − 2μ Δγ (3/2 s/σ_eq) / (3/2 wait)
    // Standard: s = (1 − 3μ Δγ / σ_eq) s_tr
    let factor = 1.0 - (3.0 * mu * dg) / seq;
    let s_new = s_tr.scale(factor);
    let mean = sig_tr.mean();
    let stress = with_hydrostatic(s_new, mean);
    // Δε_p = Δγ · (3/2) s_tr / σ_eq
    let n_scale = 1.5 * dg / seq;
    let deps_p = s_tr.scale(n_scale);
    J2Return3 {
        state: J2PlasticState3 {
            plastic_strain: state.plastic_strain.add(&deps_p),
            eq_plastic_strain: state.eq_plastic_strain + dg,
            stress,
        },
        plastic: true,
        delta_gamma: dg,
    }
}

fn with_hydrostatic(s: SymTensor3, mean: f64) -> SymTensor3 {
    SymTensor3 {
        xx: s.xx + mean,
        yy: s.yy + mean,
        zz: s.zz + mean,
        yz: s.yz,
        xz: s.xz,
        xy: s.xy,
    }
}

/// Drive a uniaxial tension path with prescribed total strain history.
///
/// Returns `(strains, stresses, permanent_strains_at_end)` where permanent
/// strain after unload to σ=0 satisfies `ε_perm ≈ ε − σ/E` on the plastic branch.
pub fn uniaxial_tension_curve(
    params: &J2Params,
    strains: &[f64],
) -> (Vec<f64>, Vec<f64>, UniaxialPlasticState) {
    let mut state = UniaxialPlasticState::default();
    let mut sigs = Vec::with_capacity(strains.len());
    for &eps in strains {
        let r = return_map_uniaxial(params, state, eps);
        state = r.state;
        sigs.push(state.stress);
    }
    (strains.to_vec(), sigs, state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn steelish() -> J2Params {
        J2Params {
            youngs_modulus: 2.0e11,
            poisson_ratio: 0.3,
            yield_strength: 2.5e8,
            hardening_modulus: 2.0e9, // Et ≈ 1.98e9
        }
    }

    #[test]
    fn yield_onset_and_hardening_slope() {
        let p = steelish();
        let e = p.youngs_modulus;
        let sig_y = p.yield_strength;
        let eps_y = sig_y / e;
        // Just below yield
        let r0 = return_map_uniaxial(&p, UniaxialPlasticState::default(), 0.99 * eps_y);
        assert!(!r0.plastic);
        assert_relative_eq!(r0.state.stress, 0.99 * sig_y, max_relative = 1e-12);
        // Into plastic regime
        let eps1 = eps_y + 1e-3;
        let r1 = return_map_uniaxial(&p, UniaxialPlasticState::default(), eps1);
        assert!(r1.plastic);
        let et = p.uniaxial_tangent();
        let sig_expect = sig_y + et * (eps1 - eps_y);
        let rel = (r1.state.stress - sig_expect).abs() / sig_expect;
        assert!(rel < 0.01, "hardening slope rel err {rel}");
    }

    #[test]
    fn unload_elastic_and_permanent_strain() {
        let p = steelish();
        let e = p.youngs_modulus;
        let eps_peak = p.yield_strength / e + 2e-3;
        let r = return_map_uniaxial(&p, UniaxialPlasticState::default(), eps_peak);
        let sig = r.state.stress;
        let eps_perm_analytic = eps_peak - sig / e;
        assert_relative_eq!(
            r.state.plastic_strain,
            eps_perm_analytic,
            max_relative = 1e-12
        );
        // Unload to same stress via elastic: ε_unload = ε_p + σ_target/E with σ_target=0
        let r_u = return_map_uniaxial(&p, r.state, r.state.plastic_strain);
        assert!(!r_u.plastic);
        assert!(r_u.state.stress.abs() / p.yield_strength < 1e-12);
    }

    #[test]
    fn j2_3d_hydrostatic_elastic_and_shear_returns_to_surface() {
        let p = steelish();
        // Pure dilatation never yields under J2.
        let hydro = SymTensor3 {
            xx: 2e-3,
            yy: 2e-3,
            zz: 2e-3,
            ..SymTensor3::zero()
        };
        let r_h = return_map_j2_3d(&p, J2PlasticState3::default(), hydro);
        assert!(!r_h.plastic);

        // Large shear strain → plastic; σ_eq must sit on σ_y + H α.
        let shear = SymTensor3 {
            xx: 5e-3,
            yy: -5e-3,
            zz: 0.0,
            ..SymTensor3::zero()
        };
        let r = return_map_j2_3d(&p, J2PlasticState3::default(), shear);
        assert!(r.plastic);
        let seq = r.state.stress.von_mises_eq();
        let sig_y_eff = p.yield_strength + p.hardening_modulus * r.state.eq_plastic_strain;
        let rel = (seq - sig_y_eff).abs() / sig_y_eff;
        assert!(
            rel < 0.01,
            "J2 surface rel err {rel}: seq={seq} sy={sig_y_eff}"
        );
    }
}
