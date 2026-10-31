//! First-order thrust models (fixed, propeller, jet) as [`ForceGenerator`]s.
//!
//! Forces [N] along a body-fixed axis. Density scaling uses ISA or a constant.
//! No advance-ratio maps or intake momentum drag.

use pe_core::forces::ForceGenerator;
use pe_core::world::{State, World};
use pe_math::libm_ops;
use pe_math::Vec3;

use crate::atmosphere::{DensityModel, RHO0};

fn normalize_axis(axis: Vec3) -> Vec3 {
    let n = axis.norm();
    if n < 1e-15 {
        Vec3::unit_x()
    } else {
        axis / n
    }
}

/// Constant thrust [N] along a body-fixed axis (default +x, nose).
#[derive(Clone, Debug)]
pub struct FixedThrust {
    /// Body index.
    pub body: usize,
    /// Thrust magnitude [N].
    pub thrust: f64,
    /// Unit axis in the body frame [dimensionless].
    pub axis: Vec3,
}

impl FixedThrust {
    /// Thrust [N] on `body` along `axis_body` (normalized).
    pub fn new(body: usize, thrust: f64, axis_body: Vec3) -> Self {
        Self {
            body,
            thrust,
            axis: normalize_axis(axis_body),
        }
    }
}

impl ForceGenerator for FixedThrust {
    fn apply(&self, _world: &World, s: &State, _t: f64, f: &mut [Vec3], _torque: &mut [Vec3]) {
        let i = self.body;
        f[i] += s.q[i].rotate(self.axis * self.thrust);
    }
}

/// Simple propeller: `T = min(T_static, η P / max(V, V_min)) · throttle · (ρ/ρ_sl)`.
///
/// `T_static` [N], shaft power `P` [W], efficiency `η` [dimensionless], throttle ∈ [0, 1].
#[derive(Clone, Debug)]
pub struct PropellerThrust {
    /// Body index.
    pub body: usize,
    /// Static thrust ceiling [N].
    pub t_static: f64,
    /// Shaft power [W].
    pub power: f64,
    /// Propulsive efficiency [dimensionless].
    pub eta: f64,
    /// Throttle ∈ [0, 1] [dimensionless].
    pub throttle: f64,
    /// Floor on airspeed in the power-limited branch [m/s].
    pub v_min: f64,
    /// Reference sea-level density [kg/m³].
    pub rho_sl: f64,
    /// Density model.
    pub density: DensityModel,
    /// Unit thrust axis in the body frame.
    pub axis: Vec3,
    /// Wind velocity [m/s] (world frame).
    pub wind: Vec3,
}

impl PropellerThrust {
    /// Build a propeller model. Units: `T_static` [N], `power` [W], `v_min` [m/s].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        body: usize,
        t_static: f64,
        power: f64,
        eta: f64,
        throttle: f64,
        v_min: f64,
        rho_sl: f64,
        density: DensityModel,
        axis_body: Vec3,
        wind: Vec3,
    ) -> Self {
        Self {
            body,
            t_static,
            power,
            eta,
            throttle,
            v_min,
            rho_sl,
            density,
            axis: normalize_axis(axis_body),
            wind,
        }
    }

    /// Thrust magnitude [N] at airspeed `V` [m/s] and density `ρ` [kg/m³].
    pub fn thrust_magnitude(&self, v: f64, rho: f64) -> f64 {
        let t_power = self.eta * self.power / v.max(self.v_min);
        let t = t_power.min(self.t_static) * self.throttle * (rho / self.rho_sl);
        t.max(0.0)
    }
}

impl ForceGenerator for PropellerThrust {
    fn apply(&self, _world: &World, s: &State, _t: f64, f: &mut [Vec3], _torque: &mut [Vec3]) {
        let i = self.body;
        let v = (s.v[i] - self.wind).norm();
        let rho = self.density.density_at(s.x[i].z);
        let t = self.thrust_magnitude(v, rho);
        f[i] += s.q[i].rotate(self.axis * t);
    }
}

/// Simple jet: `T = T_sl · throttle · (ρ/ρ_sl)^n` along a body axis.
///
/// `T_sl` [N], throttle ∈ [0, 1], exponent `n` [dimensionless].
#[derive(Clone, Debug)]
pub struct JetThrust {
    /// Body index.
    pub body: usize,
    /// Sea-level / reference thrust [N].
    pub t_sl: f64,
    /// Throttle ∈ [0, 1] [dimensionless].
    pub throttle: f64,
    /// Density-scaling exponent [dimensionless].
    pub n: f64,
    /// Reference density [kg/m³].
    pub rho_sl: f64,
    /// Density model.
    pub density: DensityModel,
    /// Unit thrust axis in the body frame.
    pub axis: Vec3,
}

impl JetThrust {
    /// Build a jet model. `T_sl` [N]; `n` is the density exponent.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        body: usize,
        t_sl: f64,
        throttle: f64,
        n: f64,
        rho_sl: f64,
        density: DensityModel,
        axis_body: Vec3,
    ) -> Self {
        Self {
            body,
            t_sl,
            throttle,
            n,
            rho_sl,
            density,
            axis: normalize_axis(axis_body),
        }
    }

    /// Thrust magnitude [N] at density `ρ` [kg/m³].
    pub fn thrust_magnitude(&self, rho: f64) -> f64 {
        (self.t_sl * self.throttle * libm_ops::pow(rho / self.rho_sl, self.n)).max(0.0)
    }
}

impl Default for JetThrust {
    fn default() -> Self {
        Self::new(0, 0.0, 1.0, 0.7, RHO0, DensityModel::Isa, Vec3::unit_x())
    }
}

impl ForceGenerator for JetThrust {
    fn apply(&self, _world: &World, s: &State, _t: f64, f: &mut [Vec3], _torque: &mut [Vec3]) {
        let i = self.body;
        let rho = self.density.density_at(s.x[i].z);
        let t = self.thrust_magnitude(rho);
        f[i] += s.q[i].rotate(self.axis * t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn jet_and_prop_magnitudes() {
        let jet = JetThrust::new(
            0,
            10_000.0,
            0.5,
            1.0,
            RHO0,
            DensityModel::Constant(RHO0),
            Vec3::unit_x(),
        );
        assert_abs_diff_eq!(jet.thrust_magnitude(RHO0), 5000.0, epsilon = 1e-12);
        assert_abs_diff_eq!(jet.thrust_magnitude(0.5 * RHO0), 2500.0, epsilon = 1e-12);

        let prop = PropellerThrust::new(
            0,
            2000.0,
            100_000.0,
            0.8,
            1.0,
            1.0,
            RHO0,
            DensityModel::Constant(RHO0),
            Vec3::unit_x(),
            Vec3::zero(),
        );
        assert_abs_diff_eq!(prop.thrust_magnitude(80.0, RHO0), 1000.0, epsilon = 1e-12);
        assert_abs_diff_eq!(prop.thrust_magnitude(1.0, RHO0), 2000.0, epsilon = 1e-12);
    }
}
