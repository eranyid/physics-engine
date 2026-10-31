//! Blade-element / momentum (BEM) theory for rotors and propellers.
//!
//! Glauert axial momentum theory combined with blade-element loads, Prandtl tip
//! loss, and induced-velocity fixed-point iteration with a convergence guard.
//! Sectional lift/drag come from [`WingPolar`] (existing stall + parabolic drag).
//!
//! # Hover / actuator disk
//!
//! Ideal momentum theory (Glauert):
//! - Induced velocity in hover: `v_h = √(T / (2 ρ A))`
//! - Ideal power: `P_i = T v_h`
//! - Figure of merit: `FM = T v_h / P_shaft` (≤ 1 for physical rotors)
//!
//! # Blade element
//!
//! At radius `r`, inflow angle `φ = atan2(V_∞ + v_i, Ω r)`, AoA `α = θ − φ`,
//! `dT = N_b (dL cos φ − dD sin φ) F`, tip loss
//! `F = (2/π) arccos(exp(−f))`, `f = (N_b/2)(R−r)/(r sin φ)` (Prandtl).
//!
//! # Coefficients
//!
//! Any blade twist / Clα beyond the polar defaults used in demos are **nominal
//! / unverified** engineering choices — labelled in tests and [`docs/ROTOR.md`].
//!
//! # References
//! - Glauert, H. (1935). Airplane propellers. In Durand, *Aerodynamic Theory*.
//! - Leishman, J. G. *Principles of Helicopter Aerodynamics* (momentum + BEMT).
//! - Prandtl tip-loss factor (as quoted in Leishman / Glauert).
//!
//! # Deferred
//! - Swirl / tangential induction `a'`; yawed inflow; unsteady dynamic stall;
//!   compressible tip corrections; wake contraction models.

use pe_core::forces::ForceGenerator;
use pe_core::world::{State, World};
use pe_math::libm_ops;
use pe_math::Vec3;

use crate::atmosphere::{DensityModel, RHO0};
use crate::polar::WingPolar;

/// Ideal hover induced velocity `v_h = √(T/(2 ρ A))` [m/s].
pub fn hover_induced_velocity(thrust: f64, rho: f64, area: f64) -> f64 {
    assert!(thrust >= 0.0 && rho > 0.0 && area > 0.0);
    libm_ops::sqrt(thrust / (2.0 * rho * area))
}

/// Ideal actuator-disk power `P_i = T v_h` [W] in hover.
pub fn ideal_hover_power(thrust: f64, rho: f64, area: f64) -> f64 {
    thrust * hover_induced_velocity(thrust, rho, area)
}

/// Figure of merit `FM = T v_h / P` [dimensionless] (hover). Returns 0 if `P≤0`.
pub fn figure_of_merit(thrust: f64, power: f64, rho: f64, area: f64) -> f64 {
    if power <= 0.0 || thrust <= 0.0 {
        return 0.0;
    }
    ideal_hover_power(thrust, rho, area) / power
}

/// Momentum-theory axial induced velocity [m/s] for climb/advance `V` [m/s].
///
/// Solves `v_i = T / (2 ρ A (V + v_i))` ⇒
/// `v_i = −V/2 + √((V/2)² + T/(2ρA))` (Glauert).
pub fn induced_velocity_momentum(thrust: f64, rho: f64, area: f64, v_axial: f64) -> f64 {
    let vh2 = thrust / (2.0 * rho * area);
    -0.5 * v_axial + libm_ops::sqrt((0.5 * v_axial) * (0.5 * v_axial) + vh2)
}

/// Prandtl tip-loss factor `F ∈ (0, 1]`.
///
/// `f = (N_b/2) (R − r) / (r sin φ)`, `F = (2/π) arccos(e^{−f})`.
pub fn prandtl_tip_loss(n_blades: usize, radius: f64, r: f64, phi: f64) -> f64 {
    let s = libm_ops::fabs(libm_ops::sin(phi)).max(1e-6);
    let f = (n_blades as f64 * 0.5) * (radius - r).max(0.0) / (r.max(1e-9) * s);
    let e = libm_ops::exp(-f.min(40.0));
    // arccos domain [−1,1]
    let arg = e.clamp(-1.0, 1.0);
    2.0 / core::f64::consts::PI * libm_ops::acos(arg)
}

/// One radial blade station.
#[derive(Clone, Debug)]
pub struct BladeStation {
    /// Radius [m].
    pub r: f64,
    /// Chord [m].
    pub chord: f64,
    /// Geometric pitch / twist [rad] (from rotation plane).
    pub twist: f64,
}

/// Rotor / propeller geometry for BEM.
#[derive(Clone, Debug)]
pub struct RotorGeometry {
    /// Number of blades.
    pub n_blades: usize,
    /// Tip radius R [m].
    pub radius: f64,
    /// Hub / root cut-out [m].
    pub hub: f64,
    /// Radial stations (ascending `r`).
    pub stations: Vec<BladeStation>,
}

impl RotorGeometry {
    /// Uniform chord + linear twist helicopter-style rotor (demo).
    ///
    /// Twist from `twist_root` to `twist_tip` [rad]. Chord constant [m].
    /// **Nominal** planform for demos — not a certified blade.
    pub fn uniform_chord(
        n_blades: usize,
        radius: f64,
        hub: f64,
        chord: f64,
        twist_root: f64,
        twist_tip: f64,
        n_stations: usize,
    ) -> Self {
        assert!(n_blades >= 1 && radius > hub && n_stations >= 2 && chord > 0.0);
        let mut stations = Vec::with_capacity(n_stations);
        for i in 0..n_stations {
            let t = i as f64 / (n_stations - 1) as f64;
            let r = hub + t * (radius - hub);
            let twist = twist_root + t * (twist_tip - twist_root);
            stations.push(BladeStation { r, chord, twist });
        }
        Self {
            n_blades,
            radius,
            hub,
            stations,
        }
    }

    /// Ideal-twist + Betz chord for uniform disk loading under inflow `v_i`.
    ///
    /// `θ(r) = atan2(v_i, Ω r) + α_design`, `c(r) = c_tip · (R/r)` (Glauert/Betz).
    /// Callers should keep `R/hub ≲ 3` so root chord stays moderate.
    #[allow(clippy::too_many_arguments)]
    pub fn ideal_hover_twist(
        n_blades: usize,
        radius: f64,
        hub: f64,
        chord_tip: f64,
        omega: f64,
        v_i: f64,
        alpha_design: f64,
        n_stations: usize,
    ) -> Self {
        assert!(n_blades >= 1 && radius > hub && n_stations >= 2 && chord_tip > 0.0 && omega > 0.0);
        let mut stations = Vec::with_capacity(n_stations);
        for i in 0..n_stations {
            let t = i as f64 / (n_stations - 1) as f64;
            let r = hub + t * (radius - hub);
            let phi = libm_ops::atan2(v_i, omega * r);
            let chord = chord_tip * (radius / r.max(1e-9));
            stations.push(BladeStation {
                r,
                chord,
                twist: phi + alpha_design,
            });
        }
        Self {
            n_blades,
            radius,
            hub,
            stations,
        }
    }

    /// Disk area `π R²` [m²].
    pub fn disk_area(&self) -> f64 {
        core::f64::consts::PI * self.radius * self.radius
    }
}

/// Result of one BEM evaluation.
#[derive(Clone, Debug)]
pub struct BemResult {
    /// Thrust [N] (along rotor axis, + in the thrust direction).
    pub thrust: f64,
    /// Shaft torque [N·m].
    pub torque: f64,
    /// Shaft power `Q · Ω` [W].
    pub power: f64,
    /// Annulus-area-weighted mean induced velocity [m/s].
    pub induced_velocity: f64,
    /// Thrust-weighted mean induced velocity `Σ v_i dT / T` [m/s].
    ///
    /// For `C_D = 0`, blade-element power satisfies `P = T · v_i^{tw}` to
    /// discretization accuracy (local identity `dP = v_i dT`).
    pub induced_velocity_thrust_wt: f64,
    /// Iterations used.
    pub iterations: usize,
    /// True if the induced-velocity residual met the tolerance.
    pub converged: bool,
    /// Final max relative `v_i` residual (last iteration).
    pub residual: f64,
}

/// Blade-element / momentum solver.
#[derive(Clone, Debug)]
pub struct BemRotor {
    /// Geometry.
    pub geom: RotorGeometry,
    /// Section polar (Cl, Cd vs α).
    pub polar: WingPolar,
    /// Max induced-velocity iterations.
    pub max_iter: usize,
    /// Relative residual tolerance on `v_i`.
    pub tol: f64,
    /// Under-relaxation for `v_i` updates ∈ (0, 1].
    pub relax: f64,
    /// If false, force Prandtl tip-loss factor `F = 1` (ideal-disk oracle).
    pub tip_loss: bool,
    /// If true, force sectional `C_D = 0` (ideal inviscid sections).
    pub zero_drag: bool,
}

impl BemRotor {
    /// Build a BEM rotor. Defaults: 80 iters, tol 1e-5, relax 0.35, tip loss on.
    pub fn new(geom: RotorGeometry, polar: WingPolar) -> Self {
        Self {
            geom,
            polar,
            max_iter: 80,
            tol: 1e-5,
            relax: 0.35,
            tip_loss: true,
            zero_drag: false,
        }
    }

    /// Ideal oracle mode: no tip loss, zero profile drag.
    pub fn ideal(mut self) -> Self {
        self.tip_loss = false;
        self.zero_drag = true;
        self.tol = 1e-8;
        self.max_iter = 200;
        self.relax = 0.5;
        self
    }

    /// Evaluate at tip speed `Ω` [rad/s], axial freestream `V_∞` [m/s]
    /// (positive into the disk from above for a propeller in advance; 0 = hover),
    /// density `ρ` [kg/m³].
    pub fn evaluate(&self, omega: f64, v_inf: f64, rho: f64) -> BemResult {
        assert!(omega > 0.0 && rho > 0.0);
        let n = self.geom.stations.len();
        // Mild positive seed (hover / climb). Keep fixed so the locked M11 hash
        // stays bit-stable across solver refactors; ideal oracles still converge.
        let v_seed = libm_ops::fabs(v_inf).max(1.0) * 0.05;
        let mut vi = vec![v_seed; n];
        let mut drs = vec![0.0; n];
        for i in 0..n {
            let r = self.geom.stations[i].r;
            drs[i] = if i == 0 {
                let r_next = self.geom.stations[1].r;
                0.5 * (r_next - r) + (r - self.geom.hub)
            } else if i + 1 == n {
                let r_prev = self.geom.stations[n - 2].r;
                (self.geom.radius - r) + 0.5 * (r - r_prev)
            } else {
                0.5 * (self.geom.stations[i + 1].r - self.geom.stations[i - 1].r)
            };
        }

        let mut thrust = 0.0;
        let mut torque = 0.0;
        let mut iters = 0;
        let mut converged = false;
        let mut residual: f64 = f64::INFINITY;
        let mut d_thrust = vec![0.0; n];

        for it in 0..self.max_iter {
            iters = it + 1;
            let mut t_sum = 0.0;
            let mut q_sum = 0.0;
            let mut vi_new = vec![0.0; n];
            let mut max_rel: f64 = 0.0;

            for (i, st) in self.geom.stations.iter().enumerate() {
                let r = st.r;
                let dr = drs[i];

                let v_ax = v_inf + vi[i];
                let v_t = omega * r;
                let phi = libm_ops::atan2(v_ax, v_t);
                let alpha = st.twist - phi;
                let w = libm_ops::hypot(v_ax, v_t);
                let cl = self.polar.cl(alpha);
                let cd = if self.zero_drag {
                    0.0
                } else {
                    self.polar.cd(cl)
                };
                let qbar = 0.5 * rho * w * w;
                let dl = qbar * st.chord * cl * dr;
                let dd = qbar * st.chord * cd * dr;
                let f_tip = if self.tip_loss {
                    prandtl_tip_loss(self.geom.n_blades, self.geom.radius, r, phi)
                } else {
                    1.0
                };
                let nb = self.geom.n_blades as f64;
                let dt = nb * (dl * libm_ops::cos(phi) - dd * libm_ops::sin(phi)) * f_tip;
                let dq = nb * (dl * libm_ops::sin(phi) + dd * libm_ops::cos(phi)) * r * f_tip;
                d_thrust[i] = dt;
                t_sum += dt;
                q_sum += dq;

                // Local momentum: dT = 4 π r dr ρ (V+vi) vi F  (Glauert; F in mass flux).
                // ⇒ vi from quadratic. Guard tip where F→0.
                let a_ann = 4.0 * core::f64::consts::PI * r * dr * f_tip.max(1e-3);
                let rhs = if a_ann > 1e-18 && dt > 0.0 {
                    dt / (rho * a_ann)
                } else {
                    0.0
                };
                let vi_m =
                    -0.5 * v_inf + libm_ops::sqrt((0.5 * v_inf) * (0.5 * v_inf) + rhs.max(0.0));
                let vi_u = self.relax * vi_m + (1.0 - self.relax) * vi[i];
                let denom = libm_ops::fabs(vi[i]).max(1e-6);
                max_rel = max_rel.max(libm_ops::fabs(vi_u - vi[i]) / denom);
                vi_new[i] = vi_u.max(0.0);
            }

            vi = vi_new;
            thrust = t_sum;
            torque = q_sum;
            residual = max_rel;
            if max_rel < self.tol {
                converged = true;
                break;
            }
        }

        // Annulus area weight ∝ r dr (disk-average induced velocity).
        let mut vi_mean = 0.0;
        let mut wsum = 0.0;
        let mut vi_tw = 0.0;
        let mut tsum_pos = 0.0;
        for (i, st) in self.geom.stations.iter().enumerate() {
            let w = st.r * drs[i];
            vi_mean += vi[i] * w;
            wsum += w;
            if d_thrust[i] > 0.0 {
                vi_tw += vi[i] * d_thrust[i];
                tsum_pos += d_thrust[i];
            }
        }
        if wsum > 0.0 {
            vi_mean /= wsum;
        }
        if tsum_pos > 0.0 {
            vi_tw /= tsum_pos;
        }

        BemResult {
            thrust,
            torque,
            power: torque * omega,
            induced_velocity: vi_mean,
            induced_velocity_thrust_wt: vi_tw,
            iterations: iters,
            converged,
            residual,
        }
    }
}

/// Force generator: BEM rotor thrust + torque on a body.
///
/// Thrust along body `axis`; torque about `axis` (reaction on the airframe).
#[derive(Clone, Debug)]
pub struct BemRotorForce {
    /// Body index.
    pub body: usize,
    /// BEM model.
    pub rotor: BemRotor,
    /// Rotor speed Ω [rad/s].
    pub omega: f64,
    /// Advance / climb speed along the axis in the body frame sense [m/s]
    /// (added to the relative wind projection).
    pub density: DensityModel,
    /// Thrust axis in the body frame (unit).
    pub axis: Vec3,
    /// Wind [m/s] world frame.
    pub wind: Vec3,
}

impl BemRotorForce {
    /// Build a BEM force generator. `omega` [rad/s] > 0.
    pub fn new(
        body: usize,
        rotor: BemRotor,
        omega: f64,
        density: DensityModel,
        axis_body: Vec3,
        wind: Vec3,
    ) -> Self {
        let n = axis_body.norm();
        let axis = if n < 1e-15 {
            Vec3::unit_z()
        } else {
            axis_body / n
        };
        Self {
            body,
            rotor,
            omega,
            density,
            axis,
            wind,
        }
    }
}

impl ForceGenerator for BemRotorForce {
    fn apply(&self, _world: &World, s: &State, _t: f64, f: &mut [Vec3], torque: &mut [Vec3]) {
        let i = self.body;
        let axis_w = s.q[i].rotate(self.axis);
        let v_rel = s.v[i] - self.wind;
        // Axial freestream opposing thrust axis (propeller: +axis = thrust,
        // advance V∞ = −v·axis when flying forward along +axis).
        let v_inf = -v_rel.dot(axis_w);
        let rho = self.density.density_at(s.x[i].z);
        let r = self.rotor.evaluate(
            self.omega,
            v_inf.max(-0.5 * self.omega * self.rotor.geom.radius),
            rho,
        );
        f[i] += axis_w * r.thrust;
        // Rotor torque reaction on the body (opposite to blade drag torque).
        torque[i] += axis_w * (-r.torque);
    }
}

/// Thin-airfoil polar for rotor demos: `Cl = 2π α`, `Cd = Cd0 + Cl²/(π e AR)`.
///
/// **Nominal** section — not an airfoil wind-tunnel polar.
pub fn nominal_rotor_polar(chord: f64, cd0: f64) -> WingPolar {
    // Fake a high-AR strip polar so induced drag is small at the section level
    // (BEM already accounts for induced via momentum). AR large ⇒ Cd ≈ Cd0 + small.
    WingPolar::new(
        chord * 1.0, // S unused for Cl(α)
        chord,
        100.0 * chord,
        cd0,
        0.9,
        0.0,
        2.0 * core::f64::consts::PI,
        15.0_f64.to_radians(),
        1.0,
        Some(100.0),
        0.0,
        0.0,
        0.0,
    )
    .expect("polar")
}

/// Inviscid thin-airfoil polar (`C_D0 = 0`); pair with [`BemRotor::ideal`] / `zero_drag`.
pub fn ideal_rotor_polar(chord: f64) -> WingPolar {
    nominal_rotor_polar(chord, 0.0)
}

/// Sea-level density convenience.
pub fn rho_sl() -> f64 {
    RHO0
}
