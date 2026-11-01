//! Wisdom–Holman / WHFast-style symplectic map (clean-room).
//!
//! References (papers / textbooks only — **no** REBOUND GPL source):
//! - Wisdom & Holman (1991), *AJ* **102**, 1528 — mixed-variable symplectic map.
//! - Rein & Tamayo (2015), *MNRAS* **452**, 376 — WHFast: Jacobi transforms,
//!   Hamiltonian split `H = H_0 + H_Kepler + H_Interaction`, DKD map (arXiv:1506.01084).
//! - Wisdom, J. (2006). Symplectic correctors for canonical heliocentric n-body
//!   maps. *AJ* **131**, 2294 — corrector coefficients (`α=√(7/40)`, `β=1/(48α)`).
//! - Duncan, Levison & Lee (1998) — democratic-heliocentric alternative.
//!
//! Kepler drifts use [`crate::kepler::propagate_universal`] (Stumpff / universal
//! variables already in-tree). Symplectic correctors (orders 3/5/7/11) convert
//! between physical and mapping coordinates (Rein & Tamayo §2.4 / Wisdom 2006).

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

/// Symplectic corrector order (Wisdom 2006 compact coefficients).
///
/// Correctors are applied once to enter mapping coordinates and inverted for
/// physical outputs; they are **not** part of every DKD step. Error scaling with
/// a corrector of order `K` is `O(ε dt^K) + O(ε² dt²)` versus `O(ε dt²)` for
/// plain DKD (Rein & Tamayo 2015).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WhfastCorrector {
    /// No corrector (plain second-order DKD).
    #[default]
    None,
    /// Third-order (2-stage) corrector.
    Order3,
    /// Fifth-order (4-stage) corrector.
    Order5,
    /// Seventh-order (6-stage) corrector.
    Order7,
    /// Eleventh-order (10-stage) corrector — Rein & Tamayo default.
    Order11,
}

impl WhfastCorrector {
    /// Published corrector order `K` used in `O(ε dt^K)` (0 = disabled).
    pub fn order(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Order3 => 3,
            Self::Order5 => 5,
            Self::Order7 => 7,
            Self::Order11 => 11,
        }
    }
}

/// `α = √(7/40)` from Wisdom (2006).
fn corrector_alpha() -> f64 {
    libm_ops::sqrt(7.0 / 40.0)
}

/// `β = 1/(48 α)`.
fn corrector_beta() -> f64 {
    1.0 / (48.0 * corrector_alpha())
}

/// `(a, b)` arguments to successive `Z(a·dt, b·dt)` calls (application order).
///
/// Coefficients from Wisdom (2006); application order matches the Rein & Tamayo
/// (2015) WHFast description of the Wisdom corrector (Z-sequence for orders
/// 3/5/7/11). `A`=Kepler, `B`=interaction; `X(a,b)=A(a)B(b)A(−a)`;
/// `Z(a,b)=X(a,b)X(−a,−b)`.
fn corrector_z_args(order: WhfastCorrector) -> Vec<(f64, f64)> {
    let a1 = corrector_alpha();
    let a2 = 2.0 * a1;
    let a3 = 3.0 * a1;
    let a4 = 4.0 * a1;
    let a5 = 5.0 * a1;
    let beta = corrector_beta();
    match order {
        WhfastCorrector::None => Vec::new(),
        WhfastCorrector::Order3 => {
            // b31 = −β/2; forward: Z(a1, +β/2), Z(−a1, −β/2)
            let b31 = -0.5 * beta;
            vec![(a1, -b31), (-a1, b31)]
        }
        WhfastCorrector::Order5 => {
            let b51 = -(1.0 / 6.0) * beta;
            let b52 = (5.0 / 6.0) * beta;
            vec![(-a2, -b51), (-a1, -b52), (a1, b52), (a2, b51)]
        }
        WhfastCorrector::Order7 => {
            let b71 = (12361.0 / 246960.0) * beta;
            let b72 = -(22651.0 / 61740.0) * beta;
            let b73 = (53521.0 / 49392.0) * beta;
            vec![
                (-a3, -b71),
                (-a2, -b72),
                (-a1, -b73),
                (a1, b73),
                (a2, b72),
                (a3, b71),
            ]
        }
        WhfastCorrector::Order11 => {
            let b111 = (2_798_927.0 / 684_573_120.0) * beta;
            let b112 = -(329_447.0 / 6_985_440.0) * beta;
            let b113 = (895_249.0 / 3_622_080.0) * beta;
            let b114 = -(14_556_229.0 / 19_015_920.0) * beta;
            let b115 = (3_394_141.0 / 2_328_480.0) * beta;
            vec![
                (-a5, -b111),
                (-a4, -b112),
                (-a3, -b113),
                (-a2, -b114),
                (-a1, -b115),
                (a1, b115),
                (a2, b114),
                (a3, b113),
                (a4, b112),
                (a5, b111),
            ]
        }
    }
}

/// WHFast-style second-order Wisdom–Holman integrator.
#[derive(Clone, Debug)]
pub struct Whfast {
    /// Gravitational constant [m³ kg⁻¹ s⁻²] (or `1` in geometric units).
    pub g_const: f64,
    /// Coordinate representation.
    pub coords: WhfastCoords,
    /// Symplectic corrector (Jacobi recommended; ignored for helio unless set).
    pub corrector: WhfastCorrector,
    /// Human-readable status after the last step.
    pub last_status: &'static str,
    /// True after [`Self::to_mapping`] until [`Self::to_physical`].
    in_mapping: bool,
    /// Timestep associated with the active corrector transform.
    corrector_dt: f64,
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
    /// New integrator (no corrector).
    pub fn new(g_const: f64, coords: WhfastCoords) -> Self {
        Self {
            g_const,
            coords,
            corrector: WhfastCorrector::None,
            last_status: "",
            in_mapping: false,
            corrector_dt: 0.0,
            m_cum: Vec::new(),
            rj: Vec::new(),
            vj: Vec::new(),
            a_cart: Vec::new(),
        }
    }

    /// Builder: attach a symplectic corrector.
    pub fn with_corrector(mut self, c: WhfastCorrector) -> Self {
        self.corrector = c;
        self
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

    /// Wisdom `X(a,b) = A(a)∘B(b)∘A(−a)` (`∘` = apply right-first).
    ///
    /// Execution order: `A(−a)`, then `B(b)`, then `A(a)`.
    fn corrector_x(&mut self, state: &mut NBodyState, a: f64, b: f64) {
        if a != 0.0 {
            self.drift_jacobi(state, -a);
        }
        if b != 0.0 {
            self.kick_jacobi(state, b);
        }
        if a != 0.0 {
            self.drift_jacobi(state, a);
        }
    }

    /// Wisdom `Z(a,b) = X(a,b)∘X(−a,−b)` (apply `X(−a,−b)` first).
    fn corrector_z(&mut self, state: &mut NBodyState, a: f64, b: f64) {
        self.corrector_x(state, -a, -b);
        self.corrector_x(state, a, b);
    }

    fn apply_corrector_jacobi(&mut self, state: &mut NBodyState, h: f64, inverse: bool) {
        if self.corrector == WhfastCorrector::None || h == 0.0 {
            return;
        }
        let n = state.n();
        self.ensure(n);
        self.refresh_masses(&state.mass);
        self.cart_to_jacobi(state);
        let args = corrector_z_args(self.corrector);
        if inverse {
            // Inverse: reverse order and flip the interaction (b) sign (Wisdom 2006).
            for &(a, b) in args.iter().rev() {
                self.corrector_z(state, a * h, -b * h);
            }
        } else {
            for &(a, b) in &args {
                self.corrector_z(state, a * h, b * h);
            }
        }
        self.jacobi_to_cart(state);
    }

    /// Convert physical → mapping coordinates (call once before a corrected run).
    pub fn to_mapping(&mut self, state: &mut NBodyState, dt: f64) {
        if self.corrector == WhfastCorrector::None || self.coords != WhfastCoords::Jacobi {
            self.in_mapping = false;
            self.corrector_dt = dt;
            return;
        }
        if self.in_mapping {
            return;
        }
        self.apply_corrector_jacobi(state, dt, false);
        self.in_mapping = true;
        self.corrector_dt = dt;
    }

    /// Convert mapping → physical coordinates (call before reading energy / output).
    pub fn to_physical(&mut self, state: &mut NBodyState) {
        if !self.in_mapping || self.corrector == WhfastCorrector::None {
            self.in_mapping = false;
            return;
        }
        let dt = self.corrector_dt;
        self.apply_corrector_jacobi(state, dt, true);
        self.in_mapping = false;
    }

    /// Whether the integrator currently holds mapping (corrected) coordinates.
    pub fn in_mapping_coords(&self) -> bool {
        self.in_mapping
    }

    /// Integrate `n_steps` of size `dt`, applying correctors around the run.
    ///
    /// Applies [`Self::to_mapping`] once, steps, then [`Self::to_physical`].
    pub fn integrate_corrected(
        &mut self,
        field: &dyn GravityField,
        state: &mut NBodyState,
        t0: f64,
        dt: f64,
        n_steps: usize,
    ) -> f64 {
        self.to_mapping(state, dt);
        let mut t = t0;
        for _ in 0..n_steps {
            let res = self.step(field, state, t, dt);
            t = res.t;
            state.after_step();
        }
        self.to_physical(state);
        t
    }
}

impl Integrator for Whfast {
    fn name(&self) -> &'static str {
        match (self.coords, self.corrector) {
            (WhfastCoords::Jacobi, WhfastCorrector::None) => "whfast_jacobi",
            (WhfastCoords::Jacobi, WhfastCorrector::Order3) => "whfast_jacobi_c3",
            (WhfastCoords::Jacobi, WhfastCorrector::Order5) => "whfast_jacobi_c5",
            (WhfastCoords::Jacobi, WhfastCorrector::Order7) => "whfast_jacobi_c7",
            (WhfastCoords::Jacobi, WhfastCorrector::Order11) => "whfast_jacobi_c11",
            (WhfastCoords::DemocraticHeliocentric, _) => "whfast_helio",
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
