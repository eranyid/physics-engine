//! Steady wind and Dryden turbulence (seeded, deterministic).
//!
//! # References
//! - Dryden turbulence model: MIL-F-8785C / MIL-HDBK-1797; Hoblit, F. M.
//!   *Gust Loads on Aircraft*. Form filters on white noise for u_g, v_g, w_g.
//! - Discrete-time approximation of the Dryden shaping filters (transfer
//!   functions H_u, H_v, H_w) via exact exponential / bilinear-free one-step
//!   update with portable `libm` (no FMA).
//!
//! RNG: [`pe_math::Xoshiro256StarStar`] (same PRNG as spectral waves).

use pe_math::libm_ops;
use pe_math::{Vec3, Xoshiro256StarStar};

/// Constant / slowly varying wind in the world frame [m/s].
#[derive(Clone, Debug)]
pub struct SteadyWind {
    /// Wind velocity [m/s] (world frame; blowing *to* this direction).
    pub velocity: Vec3,
}

impl SteadyWind {
    /// Construct from components [m/s].
    pub fn new(vx: f64, vy: f64, vz: f64) -> Self {
        Self {
            velocity: Vec3::new(vx, vy, vz),
        }
    }

    /// Horizontal wind of speed `speed` [m/s] from heading `from_deg` [deg]
    /// meteorological (direction wind comes from, CW from north). World +x east,
    /// +y north, +z up.
    pub fn from_met(speed: f64, from_deg: f64) -> Self {
        let rad = from_deg * core::f64::consts::PI / 180.0;
        // Wind *to* = from + 180°: u (east) = −speed sin(from), v (north) = −speed cos(from)
        let vx = -speed * libm_ops::sin(rad);
        let vy = -speed * libm_ops::cos(rad);
        Self::new(vx, vy, 0.0)
    }
}

/// Dryden turbulence intensities / scale lengths (SI).
#[derive(Clone, Copy, Debug)]
pub struct DrydenParams {
    /// Longitudinal turbulence intensity σ_u [m/s].
    pub sigma_u: f64,
    /// Lateral intensity σ_v [m/s].
    pub sigma_v: f64,
    /// Vertical intensity σ_w [m/s].
    pub sigma_w: f64,
    /// Longitudinal scale L_u [m].
    pub l_u: f64,
    /// Lateral scale L_v [m].
    pub l_v: f64,
    /// Vertical scale L_w [m].
    pub l_w: f64,
}

impl DrydenParams {
    /// Light turbulence at low altitude (engineering defaults, not a flight-manual table).
    pub fn light_low_altitude() -> Self {
        Self {
            sigma_u: 1.5,
            sigma_v: 1.5,
            sigma_w: 1.0,
            l_u: 200.0,
            l_v: 200.0,
            l_w: 50.0,
        }
    }
}

/// First-order Dryden shaping state for one axis.
#[derive(Clone, Debug)]
struct DrydenAxis {
    /// Filter state.
    x: f64,
    /// Intensity σ [m/s].
    sigma: f64,
    /// Scale L [m].
    length: f64,
}

impl DrydenAxis {
    fn new(sigma: f64, length: f64) -> Self {
        Self {
            x: 0.0,
            sigma,
            length,
        }
    }

    /// Advance with true airspeed `v` [m/s], timestep `dt` [s], unit white noise `w`.
    ///
    /// Longitudinal Dryden: `H_u(s) = σ_u √(2 L_u / (π V)) / (1 + s L_u / V)`.
    /// Discrete: `x ← a x + b w`, output = x, with `a = exp(−V dt / L)`,
    /// `b = σ √(1 − a²)` so stationary variance = σ² (exact for this 1st-order
    /// surrogate of the Dryden u-filter; v/w use the same 1st-order form as a
    /// deterministic engineering approximation — full 2nd-order Dryden v/w is
    /// deferred).
    fn step(&mut self, v: f64, dt: f64, w: f64) -> f64 {
        let v_abs = v.max(1.0);
        let a = libm_ops::exp(-v_abs * dt / self.length.max(1e-6));
        let b = self.sigma * libm_ops::sqrt((1.0 - a * a).max(0.0));
        self.x = a * self.x + b * w;
        self.x
    }
}

/// Dryden turbulence generator (body/world gust velocities).
#[derive(Clone, Debug)]
pub struct DrydenTurbulence {
    /// Parameters.
    pub params: DrydenParams,
    rng: Xoshiro256StarStar,
    u: DrydenAxis,
    v: DrydenAxis,
    w: DrydenAxis,
    /// Last gust [m/s] in wind axes (u forward, v right, w down-positive in
    /// aircraft body aero convention — here we return world-aligned gust with
    /// +z up, so w_gust world = −w_dryden).
    last: Vec3,
}

impl DrydenTurbulence {
    /// Seeded Dryden turbulence.
    pub fn new(params: DrydenParams, seed: u64) -> Self {
        Self {
            u: DrydenAxis::new(params.sigma_u, params.l_u),
            v: DrydenAxis::new(params.sigma_v, params.l_v),
            w: DrydenAxis::new(params.sigma_w, params.l_w),
            params,
            rng: Xoshiro256StarStar::seed(seed),
            last: Vec3::zero(),
        }
    }

    /// Advance filters; `airspeed` [m/s], `dt` [s]. Returns gust velocity in
    /// a north-east-up frame with components `(u_g, v_g, w_g_up)` [m/s].
    pub fn step(&mut self, airspeed: f64, dt: f64) -> Vec3 {
        let wu = self.rng.normal();
        let wv = self.rng.normal();
        let ww = self.rng.normal();
        let ug = self.u.step(airspeed, dt, wu);
        let vg = self.v.step(airspeed, dt, wv);
        let wg_down = self.w.step(airspeed, dt, ww);
        self.last = Vec3::new(ug, vg, -wg_down);
        self.last
    }

    /// Last gust sample [m/s].
    pub fn last_gust(&self) -> Vec3 {
        self.last
    }
}

/// Combine steady wind + optional Dryden gust into a total wind vector [m/s].
#[derive(Clone, Debug)]
pub struct WindField {
    /// Steady component.
    pub steady: SteadyWind,
    /// Optional turbulence (None → steady only).
    pub dryden: Option<DrydenTurbulence>,
}

impl WindField {
    /// Steady only.
    pub fn steady(v: Vec3) -> Self {
        Self {
            steady: SteadyWind { velocity: v },
            dryden: None,
        }
    }

    /// Steady + Dryden.
    pub fn with_dryden(steady: SteadyWind, dryden: DrydenTurbulence) -> Self {
        Self {
            steady,
            dryden: Some(dryden),
        }
    }

    /// Total wind at this step (advances Dryden when present).
    pub fn step(&mut self, airspeed: f64, dt: f64) -> Vec3 {
        let gust = match &mut self.dryden {
            Some(d) => d.step(airspeed, dt),
            None => Vec3::zero(),
        };
        self.steady.velocity + gust
    }

    /// Current wind without advancing (steady + last gust).
    pub fn current(&self) -> Vec3 {
        let gust = self
            .dryden
            .as_ref()
            .map(|d| d.last_gust())
            .unwrap_or(Vec3::zero());
        self.steady.velocity + gust
    }
}
