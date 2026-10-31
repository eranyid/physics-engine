//! Wisdom–Holman / WHFast-style symplectic map (clean-room).
//!
//! References (papers / textbooks only — **no** REBOUND GPL source):
//! - Wisdom & Holman (1991), *AJ* **102**, 1528 — mixed-variable symplectic map.
//! - Rein & Tamayo (2015), *MNRAS* **452**, 376 — WHFast: Jacobi transforms,
//!   Hamiltonian split `H = H_0 + H_Kepler + H_Interaction`, DKD map (arXiv:1506.01084).
//! - Duncan, Levison & Lee (1998) — democratic-heliocentric alternative.
//!
//! Kepler drifts use [`crate::kepler::propagate_universal`] (Stumpff / universal
//! variables already in-tree). Symplectic correctors are deferred.

use pe_math::libm_ops;
use pe_math::Vec3;

use crate::field::GravityField;
use crate::integrator::{Integrator, StepResult};
use crate::kepler::propagate_universal;
use crate::state::NBodyState;

/// Coordinate system for the Wisdom–Holman split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WhfastCoords {
    /// Jacobi coordinates (Rein & Tamayo 2015 / Wisdom & Holman 1991).
    #[default]
    Jacobi,
    /// Democratic heliocentric coordinates (Duncan et al. 1998).
    DemocraticHeliocentric,
}

/// WHFast-style second-order Wisdom–Holman integrator.
#[derive(Clone, Debug)]
pub struct Whfast {
    /// Gravitational constant [m³ kg⁻¹ s⁻²] (or `1` in geometric units).
    pub g_const: f64,
    /// Coordinate representation.
    pub coords: WhfastCoords,
    /// Human-readable status after the last step.
    pub last_status: &'static str,
    /// Cached cumulative interior masses `M_i = Σ_{j=0}^{i} m_j`.
    m_cum: Vec<f64>,
    /// Jacobi / helio scratch positions.
    rj: Vec<Vec3>,
    /// Jacobi / helio scratch velocities.
    vj: Vec<Vec3>,
    /// Cartesian acceleration scratch.
    a_cart: Vec<Vec3>,
}

impl Default for Whfast {
    fn default() -> Self {
        Self::new(1.0, WhfastCoords::Jacobi)
    }
}

/// Construct WHFast with gravitational constant `g_const` (Jacobi coords).
pub fn whfast(g_const: f64) -> Whfast {
    Whfast::new(g_const, WhfastCoords::Jacobi)
}

/// Construct WHFast with coordinate choice.
pub fn whfast_with(g_const: f64, coords: WhfastCoords) -> Whfast {
    Whfast::new(g_const, coords)
}

impl Whfast {
    /// New integrator.
    pub fn new(g_const: f64, coords: WhfastCoords) -> Self {
        Self {
            g_const,
            coords,
            last_status: "",
            m_cum: Vec::new(),
            rj: Vec::new(),
            vj: Vec::new(),
            a_cart: Vec::new(),
        }
    }

    fn ensure(&mut self, n: usize) {
        if self.rj.len() != n {
            self.rj = vec![Vec3::zero(); n];
            self.vj = vec![Vec3::zero(); n];
            self.a_cart = vec![Vec3::zero(); n];
            self.m_cum = vec![0.0; n];
        }
    }

    fn refresh_masses(&mut self, mass: &[f64]) {
        self.m_cum[0] = mass[0];
        for i in 1..mass.len() {
            self.m_cum[i] = self.m_cum[i - 1] + mass[i];
        }
    }

    /// Cartesian → Jacobi (Rein & Tamayo 2015 §3.1).
    fn cart_to_jacobi(&mut self, state: &NBodyState) {
        let n = state.n();
        let mut racc = state.x[0] * state.mass[0];
        let mut vacc = state.v[0] * state.mass[0];
        for i in 1..n {
            let m_im1 = self.m_cum[i - 1];
            self.rj[i] = state.x[i] - racc / m_im1;
            self.vj[i] = state.v[i] - vacc / m_im1;
            racc = racc * (1.0 + state.mass[i] / m_im1) + self.rj[i] * state.mass[i];
            vacc = vacc * (1.0 + state.mass[i] / m_im1) + self.vj[i] * state.mass[i];
        }
        self.rj[0] = racc / self.m_cum[n - 1];
        self.vj[0] = vacc / self.m_cum[n - 1];
    }

    /// Jacobi → Cartesian (inverse of the §3.1 forward map).
    fn jacobi_to_cart(&self, state: &mut NBodyState) {
        let n = state.n();
        let mut r = self.rj[0];
        let mut v = self.vj[0];
        for i in (1..n).rev() {
            let w = state.mass[i] / self.m_cum[i];
            r -= self.rj[i] * w;
            v -= self.vj[i] * w;
            state.set_position(i, self.rj[i] + r);
            state.v[i] = self.vj[i] + v;
        }
        state.set_position(0, r);
        state.v[0] = v;
    }

    /// Drift under `H_Kepler + H_0` for time `dt` (Jacobi).
    fn drift_jacobi(&mut self, state: &NBodyState, dt: f64) {
        let n = state.n();
        self.rj[0] += self.vj[0] * dt;
        for i in 1..n {
            let mu = self.g_const * self.m_cum[i];
            match propagate_universal(mu, self.rj[i], self.vj[i], dt) {
                Ok((r, v)) => {
                    self.rj[i] = r;
                    self.vj[i] = v;
                }
                Err(_) => {
                    self.rj[i] += self.vj[i] * dt;
                }
            }
        }
    }

    /// Interaction kick (Jacobi): `a_int = a_Nbody^{Jac} − a_Kepler`.
    fn kick_jacobi(&mut self, state: &mut NBodyState, dt: f64) {
        let n = state.n();
        self.jacobi_to_cart(state);
        for a in self.a_cart.iter_mut() {
            *a = Vec3::zero();
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let d = state.x[j] - state.x[i];
                let r2 = d.norm_squared();
                if r2 == 0.0 {
                    continue;
                }
                let inv_r3 = 1.0 / (r2 * libm_ops::sqrt(r2));
                let g = self.g_const * inv_r3;
                self.a_cart[i] += d * (g * state.mass[j]);
                self.a_cart[j] -= d * (g * state.mass[i]);
            }
        }
        // Accel → Jacobi (same linear map as velocity)
        let mut aj = vec![Vec3::zero(); n];
        let mut aacc = self.a_cart[0] * state.mass[0];
        for i in 1..n {
            let m_im1 = self.m_cum[i - 1];
            aj[i] = self.a_cart[i] - aacc / m_im1;
            aacc = aacc * (1.0 + state.mass[i] / m_im1) + aj[i] * state.mass[i];
        }
        aj[0] = aacc / self.m_cum[n - 1];

        for i in 1..n {
            let r2 = self.rj[i].norm_squared();
            if r2 > 0.0 {
                let inv_r3 = 1.0 / (r2 * libm_ops::sqrt(r2));
                let a_kep = self.rj[i] * (-self.g_const * self.m_cum[i] * inv_r3);
                self.vj[i] += (aj[i] - a_kep) * dt;
            }
        }
        self.vj[0] += aj[0] * dt;
    }

    // ——— Democratic heliocentric ———

    fn cart_to_helio(&mut self, state: &NBodyState) {
        let n = state.n();
        self.rj[0] = state.x[0];
        self.vj[0] = state.v[0];
        for i in 1..n {
            self.rj[i] = state.x[i] - state.x[0];
            self.vj[i] = state.v[i] - state.v[0];
        }
    }

    fn helio_to_cart(&self, state: &mut NBodyState) {
        let n = state.n();
        state.set_position(0, self.rj[0]);
        state.v[0] = self.vj[0];
        for i in 1..n {
            state.set_position(i, self.rj[0] + self.rj[i]);
            state.v[i] = self.vj[0] + self.vj[i];
        }
    }

    fn kick_helio(&mut self, state: &mut NBodyState, dt: f64) {
        let n = state.n();
        self.helio_to_cart(state);
        // Indirect acceleration of the star due to planets
        let mut a_indirect = Vec3::zero();
        for j in 1..n {
            let r2 = self.rj[j].norm_squared();
            if r2 == 0.0 {
                continue;
            }
            let inv_r3 = 1.0 / (r2 * libm_ops::sqrt(r2));
            a_indirect -= self.rj[j] * (self.g_const * state.mass[j] * inv_r3);
        }
        for i in 1..n {
            let mut a = a_indirect;
            for j in 1..n {
                if i == j {
                    continue;
                }
                let d = state.x[j] - state.x[i];
                let r2 = d.norm_squared();
                if r2 == 0.0 {
                    continue;
                }
                let inv_r3 = 1.0 / (r2 * libm_ops::sqrt(r2));
                a += d * (self.g_const * state.mass[j] * inv_r3);
            }
            self.vj[i] += a * dt;
        }
        // Conserve linear momentum: adjust star velocity
        let mut mtot = state.mass[0];
        let mut p_planets = Vec3::zero();
        for i in 1..n {
            mtot += state.mass[i];
            // inertial planet velocity = v0 + vh; after kick vh changed
            p_planets += (self.vj[0] + self.vj[i]) * state.mass[i];
        }
        let p_tot = self.vj[0] * state.mass[0] + p_planets;
        // Re-express with updated vh: P = m0 v0 + Σ mi (v0 + vh) = M v0 + Σ mi vh
        let mut sum_mhv = Vec3::zero();
        for i in 1..n {
            sum_mhv += self.vj[i] * state.mass[i];
        }
        let vcom = p_tot / mtot;
        self.vj[0] = vcom - sum_mhv / mtot;
    }

    fn step_jacobi(
        &mut self,
        _field: &dyn GravityField,
        state: &mut NBodyState,
        t: f64,
        dt: f64,
    ) -> StepResult {
        let n = state.n();
        if n < 2 {
            self.last_status = "WHFast Jacobi needs ≥2 bodies";
            return StepResult {
                t,
                dt_taken: 0.0,
                err_est: -1.0,
            };
        }
        self.ensure(n);
        self.refresh_masses(&state.mass);
        self.cart_to_jacobi(state);
        self.drift_jacobi(state, 0.5 * dt);
        self.kick_jacobi(state, dt);
        self.drift_jacobi(state, 0.5 * dt);
        self.jacobi_to_cart(state);
        self.last_status = "ok";
        StepResult {
            t: t + dt,
            dt_taken: dt,
            err_est: 0.0,
        }
    }

    fn step_helio(
        &mut self,
        _field: &dyn GravityField,
        state: &mut NBodyState,
        t: f64,
        dt: f64,
    ) -> StepResult {
        let n = state.n();
        if n < 2 {
            self.last_status = "WHFast helio needs ≥2 bodies";
            return StepResult {
                t,
                dt_taken: 0.0,
                err_est: -1.0,
            };
        }
        self.ensure(n);
        self.refresh_masses(&state.mass);
        self.cart_to_helio(state);

        // Track COM before planet Kepler so we can restore after relative drifts.
        let mut mtot = 0.0;
        let mut com0 = Vec3::zero();
        for i in 0..n {
            mtot += state.mass[i];
            com0 += state.x[i] * state.mass[i];
        }
        com0 /= mtot;
        let mut vcom = Vec3::zero();
        for i in 0..n {
            vcom += state.v[i] * state.mass[i];
        }
        vcom /= mtot;

        self.drift_helio_kepler_only(state, 0.5 * dt);
        com0 += vcom * (0.5 * dt);
        self.reanchor_star_to_com(state, com0, vcom);

        self.kick_helio(state, dt);
        // Refresh vcom after kick (momentum conserved → same vcom)
        self.helio_to_cart(state);
        vcom = {
            let mut p = Vec3::zero();
            for i in 0..n {
                p += state.v[i] * state.mass[i];
            }
            p / mtot
        };
        com0 = {
            let mut c = Vec3::zero();
            for i in 0..n {
                c += state.x[i] * state.mass[i];
            }
            c / mtot
        };
        self.cart_to_helio(state);

        self.drift_helio_kepler_only(state, 0.5 * dt);
        com0 += vcom * (0.5 * dt);
        self.reanchor_star_to_com(state, com0, vcom);
        self.helio_to_cart(state);
        self.last_status = "ok";
        StepResult {
            t: t + dt,
            dt_taken: dt,
            err_est: 0.0,
        }
    }

    fn drift_helio_kepler_only(&mut self, state: &NBodyState, dt: f64) {
        let n = state.n();
        for i in 1..n {
            let mu = self.g_const * (state.mass[0] + state.mass[i]);
            match propagate_universal(mu, self.rj[i], self.vj[i], dt) {
                Ok((r, v)) => {
                    self.rj[i] = r;
                    self.vj[i] = v;
                }
                Err(_) => {
                    self.rj[i] += self.vj[i] * dt;
                }
            }
        }
    }

    fn reanchor_star_to_com(&mut self, state: &NBodyState, com: Vec3, vcom: Vec3) {
        let n = state.n();
        let mut mtot = state.mass[0];
        let mut sum = Vec3::zero();
        let mut sumv = Vec3::zero();
        for i in 1..n {
            mtot += state.mass[i];
            sum += self.rj[i] * state.mass[i];
            sumv += self.vj[i] * state.mass[i];
        }
        self.rj[0] = com - sum / mtot;
        self.vj[0] = vcom - sumv / mtot;
    }
}

impl Integrator for Whfast {
    fn name(&self) -> &'static str {
        match self.coords {
            WhfastCoords::Jacobi => "whfast_jacobi",
            WhfastCoords::DemocraticHeliocentric => "whfast_helio",
        }
    }

    fn step(
        &mut self,
        field: &dyn GravityField,
        state: &mut NBodyState,
        t: f64,
        dt: f64,
    ) -> StepResult {
        // `field` is unused: WHFast evaluates Newtonian gravity internally with
        // `g_const` (the split requires the Kepler/interaction decomposition).
        let _ = field;
        match self.coords {
            WhfastCoords::Jacobi => self.step_jacobi(field, state, t, dt),
            WhfastCoords::DemocraticHeliocentric => self.step_helio(field, state, t, dt),
        }
    }
}
