//! Blast loading on rigid bodies and pressure–impulse damage helpers.
//!
//! Applies Friedlander pressure–time histories over exposed faces as impulses
//! on a [`pe_core::World`] body, or as continuous forces through
//! [`pe_core::ExternalLoads`]. Peak / arrival / duration can use Kinney
//! (legacy hash path) or Swisdak Kingery–Bulmash with Rankine–Hugoniot
//! reflected / dynamic-pressure contributions. Optional AABB ray shadowing
//! attenuates load when an occluder blocks the charge→face segment.
//!
//! # Model limits (honest)
//! - Engineering similarity / empirical charts, **not** 3-D CFD.
//! - Swisdak curves are hemispherical **surface-burst** fits with valid-`Z`
//!   clamp; free-air Kinney is a separate closed form.
//! - Incidence uses `max(0, cos θ)` with a reflected/incident blend — no
//!   clearing, no Mach stem, no multiple reflections.
//! - Dynamic-pressure drag uses a caller `C_d` on the RH post-shock `q`
//!   (engineering; not a validated aero database).
//! - AABB shadowing is conservative and ignores diffraction.
//! - Positive duration / TOA from KB when selected; legacy path uses an
//!   engineering `t_d(Z)` and sound-speed proxy TOA.
//!
//! # References
//! - Friedlander waveform (see [`crate::friedlander`]).
//! - Kinney & Graham (1985); Swisdak (1994) Kingery–Bulmash.
//! - Rankine–Hugoniot normal reflection / dynamic pressure
//!   ([`crate::rankine_hugoniot`]).
//! - Generic pressure–impulse (P–I) damage curves: Biggs (1964) — **caller
//!   supplies** curve parameters (not weapon-specific).

use pe_core::{ExternalLoads, World};
use pe_math::libm_ops;
use pe_math::Vec3;

use crate::error::{require_finite, require_non_negative, require_positive, BlastError};
use crate::friedlander;
use crate::kingery;
use crate::kinney::{peak_overpressure, P0_DEFAULT};
use crate::occlusion::{is_shadowed, OccluderAabb};
use crate::rankine_hugoniot::{self, GAMMA_AIR};
use crate::scaling::scaled_distance;

/// Peak / arrival / duration source for face loading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverpressureModel {
    /// Kinney–Graham free-air peak; engineering TOA / `t_d` (legacy hash path).
    Kinney,
    /// Swisdak simplified Kingery–Bulmash surface-burst charts.
    KingeryBulmash,
}

/// How face-normal incidence mixes incident vs reflected overpressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflectionMix {
    /// Incident only (legacy).
    IncidentOnly,
    /// `p = p_i + (p_r − p_i)·cos²θ` for facing faces (engineering wall blend).
    IncidentReflected,
}

/// Parameters for enhanced rigid-body blast loading (M20).
#[derive(Clone, Copy, Debug)]
pub struct BlastLoadParams {
    /// Peak / TOA / duration model.
    pub overpressure: OverpressureModel,
    /// Incident / reflected mix.
    pub reflection: ReflectionMix,
    /// Dynamic-pressure drag coefficient `C_d` (multiplies RH `q`·area·incidence).
    /// Use `0` to disable. **Unverified** aero database — caller-tuned.
    pub drag_cd: f64,
    /// Multiplier applied when the face is shadowed (`0` = full shadow).
    pub shadow_factor: f64,
    /// Ideal-gas γ for Rankine–Hugoniot helpers.
    pub gamma: f64,
}

impl Default for BlastLoadParams {
    fn default() -> Self {
        Self {
            overpressure: OverpressureModel::KingeryBulmash,
            reflection: ReflectionMix::IncidentReflected,
            drag_cd: 1.0,
            shadow_factor: 0.0,
            gamma: GAMMA_AIR,
        }
    }
}

impl BlastLoadParams {
    /// Legacy Kinney-only parameters matching the pre-M20 pyramid hash path.
    pub fn legacy_kinney() -> Self {
        Self {
            overpressure: OverpressureModel::Kinney,
            reflection: ReflectionMix::IncidentOnly,
            drag_cd: 0.0,
            shadow_factor: 1.0, // ignore occlusion
            gamma: GAMMA_AIR,
        }
    }

    /// Fallible validation: finite `drag_cd`/`shadow_factor`/`gamma`, `gamma > 1`,
    /// and `shadow_factor ≥ 0`.
    pub fn try_validate(self) -> Result<Self, BlastError> {
        require_finite(self.drag_cd, "drag_cd")?;
        require_non_negative(self.shadow_factor, "shadow_factor")?;
        require_finite(self.gamma, "gamma")?;
        if self.gamma <= 1.0 {
            return Err(BlastError::InvalidParameter("gamma must be > 1"));
        }
        Ok(self)
    }
}

/// Blast source for engineering loads (TNT-equivalent mass only — no chemistry).
#[derive(Clone, Copy, Debug)]
pub struct BlastSource {
    /// Detonation / burst origin [m].
    pub origin: Vec3,
    /// TNT-equivalent mass [kg] (`W_eq`).
    pub w_eq_kg: f64,
    /// Burst time [s] (world clock).
    pub t0: f64,
    /// Ambient pressure [Pa].
    pub p0: f64,
    /// Positive-phase duration scale [s] at `Z = 1` (Kinney / engineering path).
    pub td_at_z1: f64,
    /// Friedlander decay `b` [dimensionless].
    pub decay_b: f64,
}

impl BlastSource {
    /// Default engineering source at the origin.
    ///
    /// Panics if `w_eq_kg` is non-finite or not strictly positive. Prefer
    /// [`try_new`](Self::try_new) at crate / FFI boundaries.
    pub fn new(origin: Vec3, w_eq_kg: f64) -> Self {
        Self::try_new(origin, w_eq_kg).expect("blast source: w_eq_kg must be positive")
    }

    /// Fallible constructor: finite origin, `w_eq_kg > 0`, default ambient /
    /// duration / decay.
    pub fn try_new(origin: Vec3, w_eq_kg: f64) -> Result<Self, BlastError> {
        if !origin.is_finite() {
            return Err(BlastError::NonFinite("origin"));
        }
        require_positive(w_eq_kg, "w_eq_kg")?;
        Ok(Self {
            origin,
            w_eq_kg,
            t0: 0.0,
            p0: P0_DEFAULT,
            td_at_z1: 0.01,
            decay_b: 1.0,
        })
    }

    /// Validate all scalar fields after manual construction / mutation.
    pub fn try_validate(self) -> Result<Self, BlastError> {
        if !self.origin.is_finite() {
            return Err(BlastError::NonFinite("origin"));
        }
        require_positive(self.w_eq_kg, "w_eq_kg")?;
        require_finite(self.t0, "t0")?;
        require_positive(self.p0, "p0")?;
        require_positive(self.td_at_z1, "td_at_z1")?;
        require_non_negative(self.decay_b, "decay_b")?;
        Ok(self)
    }

    /// Scaled distance [m·kg^{-1/3}] to world point.
    pub fn z_at(&self, point: Vec3) -> f64 {
        let r = (point - self.origin).norm();
        scaled_distance(r, self.w_eq_kg, 1.0).unwrap_or(f64::INFINITY)
    }

    /// Approximate arrival time [s] after `t0` using sound-speed proxy
    /// `t_a ≈ R / a_eff` with `a_eff` from Kinney peak (engineering, not KB TOA).
    /// Prefer Swisdak arrival when available via [`crate::kingery::arrival_time`].
    pub fn arrival_time_approx(&self, point: Vec3) -> f64 {
        let r = (point - self.origin).norm();
        let z = self.z_at(point);
        let ps = peak_overpressure(z, self.p0);
        let a0 = 340.0; // m/s, engineering ambient
        let xi = ps / self.p0;
        let ms = libm_ops::sqrt((1.0 + 6.0 * xi / 7.0).max(1.0)); // γ=1.4 approx
        let us = (ms * a0).max(a0);
        r / us
    }

    /// Positive duration [s] at point: `t_d = t_d(Z=1) · Z^{1/2}` (engineering).
    pub fn positive_duration(&self, point: Vec3) -> f64 {
        let z = self.z_at(point).max(1e-6);
        self.td_at_z1 * libm_ops::sqrt(z)
    }

    /// Arrival time [s] after `t0` for the selected model.
    pub fn arrival_time(&self, point: Vec3, model: OverpressureModel) -> f64 {
        match model {
            OverpressureModel::Kinney => self.arrival_time_approx(point),
            OverpressureModel::KingeryBulmash => {
                let z = self.z_at(point);
                kingery::arrival_time(z, self.w_eq_kg).value
            }
        }
    }

    /// Positive duration [s] for the selected model.
    pub fn duration(&self, point: Vec3, model: OverpressureModel) -> f64 {
        match model {
            OverpressureModel::Kinney => self.positive_duration(point),
            OverpressureModel::KingeryBulmash => {
                let z = self.z_at(point);
                kingery::positive_duration(z, self.w_eq_kg).value
            }
        }
    }
}

/// Exposed face for blast loading.
#[derive(Clone, Copy, Debug)]
pub struct ExposedFace {
    /// Body index in the [`World`].
    pub body: usize,
    /// Face center [m] (world).
    pub center: Vec3,
    /// Outward unit normal (world).
    pub normal: Vec3,
    /// Exposed area [m²].
    pub area: f64,
}

impl ExposedFace {
    /// Fallible face constructor: finite pose vectors and `area > 0`.
    pub fn try_new(body: usize, center: Vec3, normal: Vec3, area: f64) -> Result<Self, BlastError> {
        if !center.is_finite() {
            return Err(BlastError::NonFinite("face.center"));
        }
        if !normal.is_finite() {
            return Err(BlastError::NonFinite("face.normal"));
        }
        require_positive(area, "face.area")?;
        Ok(Self {
            body,
            center,
            normal,
            area,
        })
    }
}

/// Angle-of-incidence factor: `max(0, cos θ)` with `θ` between blast direction
/// and inward face normal (facing the charge).
pub fn incidence_factor(blast_dir_from_charge: Vec3, face_outward: Vec3) -> f64 {
    let d = blast_dir_from_charge.normalize();
    let n = face_outward.normalize();
    (-d.dot(n)).max(0.0)
}

/// Standoff attenuation in addition to Kinney `p_s(Z)` — identity (1.0) by
/// default; kept as a hook for caller-supplied factors.
pub fn standoff_factor(_z: f64) -> f64 {
    1.0
}

/// Hopkinson–Cranz peak overpressure [Pa] for the selected model at `z`.
pub fn peak_incident_pa(z: f64, p0: f64, model: OverpressureModel) -> f64 {
    match model {
        OverpressureModel::Kinney => peak_overpressure(z, p0) * standoff_factor(z),
        OverpressureModel::KingeryBulmash => kingery::incident_overpressure(z).value,
    }
}

/// Peak face pressure [Pa] (incident / reflected blend) before Friedlander and
/// before dynamic-pressure drag. Does not include shadowing.
pub fn peak_face_overpressure(
    src: &BlastSource,
    face: &ExposedFace,
    params: &BlastLoadParams,
) -> f64 {
    let z = src.z_at(face.center);
    let p_i = peak_incident_pa(z, src.p0, params.overpressure);
    let dir = face.center - src.origin;
    let mu = incidence_factor(dir, face.normal);
    if mu <= 0.0 {
        return 0.0;
    }
    let p_peak = match params.reflection {
        ReflectionMix::IncidentOnly => p_i,
        ReflectionMix::IncidentReflected => {
            let p_r = match params.overpressure {
                OverpressureModel::Kinney => {
                    rankine_hugoniot::reflected_overpressure(p_i, src.p0, params.gamma)
                }
                OverpressureModel::KingeryBulmash => kingery::reflected_overpressure(z).value,
            };
            // Engineering blend: face-on → reflected, grazing → incident.
            p_i + (p_r - p_i) * (mu * mu)
        }
    };
    p_peak * mu
}

/// Dynamic-pressure drag contribution [Pa] (× `C_d` later) at the face standoff.
pub fn dynamic_pressure_pa(src: &BlastSource, face: &ExposedFace, params: &BlastLoadParams) -> f64 {
    if params.drag_cd <= 0.0 {
        return 0.0;
    }
    let z = src.z_at(face.center);
    let p_i = peak_incident_pa(z, src.p0, params.overpressure);
    let xi = p_i / src.p0.max(1e-30);
    rankine_hugoniot::dynamic_pressure(xi, src.p0, params.gamma)
}

/// Instantaneous overpressure [Pa] on a face at world time `t` [s] (legacy
/// Kinney path — used by the locked blast-pyramid hash).
pub fn face_pressure(src: &BlastSource, face: &ExposedFace, t: f64) -> f64 {
    face_pressure_params(src, face, t, &BlastLoadParams::legacy_kinney(), &[])
}

/// Instantaneous effective face pressure [Pa] including reflection blend,
/// Friedlander envelope, dynamic-pressure drag, and optional shadowing.
pub fn face_pressure_params(
    src: &BlastSource,
    face: &ExposedFace,
    t: f64,
    params: &BlastLoadParams,
    occluders: &[OccluderAabb],
) -> f64 {
    let ta = src.t0 + src.arrival_time(face.center, params.overpressure);
    let td = src.duration(face.center, params.overpressure);
    let tau = t - ta;
    let p_peak = peak_face_overpressure(src, face, params);
    if p_peak <= 0.0 {
        return 0.0;
    }
    let dir = face.center - src.origin;
    let mu = incidence_factor(dir, face.normal);
    let q = dynamic_pressure_pa(src, face, params);
    let p_wave = friedlander::pressure(tau, p_peak / mu.max(1e-30), td, src.decay_b) * mu;
    // Drag uses the same Friedlander envelope shape on q (engineering).
    let p_drag = if params.drag_cd > 0.0 && mu > 0.0 {
        friedlander::pressure(tau, q, td, src.decay_b) * params.drag_cd * mu
    } else {
        0.0
    };
    let mut p = p_wave + p_drag;
    if params.shadow_factor < 1.0 - 1e-15
        && is_shadowed(src.origin, face.center, face.body, occluders)
    {
        p *= params.shadow_factor;
    }
    p
}

/// Apply Friedlander impulse over `dt` [s] to a world body (legacy path).
///
/// Impulse `J = p A dt` along the inward normal (blast pushes the face inward).
/// Kept bit-stable for [`crate::blast_pyramid_hash`].
pub fn apply_blast_impulse(
    world: &mut World,
    src: &BlastSource,
    face: &ExposedFace,
    t: f64,
    dt: f64,
) {
    let p = face_pressure(src, face, t);
    if p <= 0.0 || face.area <= 0.0 {
        return;
    }
    let n_in = -face.normal.normalize();
    let j = n_in * (p * face.area * dt);
    world.apply_impulse(face.body, j, face.center);
}

/// Apply enhanced blast impulse (KB / RH / shadowing) via `World::apply_impulse`.
pub fn apply_blast_impulse_params(
    world: &mut World,
    src: &BlastSource,
    face: &ExposedFace,
    t: f64,
    dt: f64,
    params: &BlastLoadParams,
    occluders: &[OccluderAabb],
) {
    let p = face_pressure_params(src, face, t, params, occluders);
    if p <= 0.0 || face.area <= 0.0 {
        return;
    }
    let n_in = -face.normal.normalize();
    let j = n_in * (p * face.area * dt);
    world.apply_impulse(face.body, j, face.center);
}

/// Write continuous blast forces for one face into [`ExternalLoads`].
///
/// Force `F = p A n_inward`, torque `(r − COM) × F` with `r = face.center`.
/// Caller should [`ExternalLoads::clear`] once per step before accumulating
/// all faces, then `World::step` so the integrator samples the loads.
pub fn accumulate_blast_external(
    loads: &ExternalLoads,
    world: &World,
    src: &BlastSource,
    face: &ExposedFace,
    t: f64,
    params: &BlastLoadParams,
    occluders: &[OccluderAabb],
) {
    let p = face_pressure_params(src, face, t, params, occluders);
    if p <= 0.0 || face.area <= 0.0 || face.body >= world.n {
        return;
    }
    let n_in = -face.normal.normalize();
    let force = n_in * (p * face.area);
    let com = world.state.x[face.body];
    let torque = (face.center - com).cross(force);
    loads.add(face.body, force, torque);
}

/// Accumulate all faces into `loads` (does **not** clear — caller clears).
pub fn accumulate_blast_external_all(
    loads: &ExternalLoads,
    world: &World,
    src: &BlastSource,
    faces: &[ExposedFace],
    t: f64,
    params: &BlastLoadParams,
    occluders: &[OccluderAabb],
) {
    // Ascending body, then face index order is caller's responsibility; we
    // walk the given slice in order (deterministic if faces are sorted).
    for face in faces {
        accumulate_blast_external(loads, world, src, face, t, params, occluders);
    }
}

/// Generic pressure–impulse damage criterion (caller-supplied curve).
///
/// A point `(P, I)` fails (damage) if it lies above the iso-damage curve
/// approximated as `(P − P0)(I − I0) = C` (hyperbolic Biggs-style).
#[derive(Clone, Copy, Debug)]
pub struct PressureImpulseCurve {
    /// Pressure asymptote [Pa].
    pub p0: f64,
    /// Impulse asymptote [Pa·s].
    pub i0: f64,
    /// Hyperbola constant [Pa²·s].
    pub c: f64,
}

impl PressureImpulseCurve {
    /// `true` if `(p_peak [Pa], impulse [Pa·s])` exceeds the damage curve.
    pub fn is_damaged(&self, p_peak: f64, impulse: f64) -> bool {
        if p_peak <= self.p0 || impulse <= self.i0 {
            return false;
        }
        (p_peak - self.p0) * (impulse - self.i0) >= self.c
    }
}

/// Integrate positive impulse [Pa·s] on a face from arrival through `t_d`
/// (legacy Kinney path).
pub fn face_positive_impulse(src: &BlastSource, face: &ExposedFace) -> f64 {
    face_positive_impulse_params(src, face, &BlastLoadParams::legacy_kinney())
}

/// Analytic positive specific impulse [Pa·s] for enhanced params (no shadowing;
/// shadowing is time-dependent under motion — use stepwise integration).
pub fn face_positive_impulse_params(
    src: &BlastSource,
    face: &ExposedFace,
    params: &BlastLoadParams,
) -> f64 {
    let z = src.z_at(face.center);
    let td = src.duration(face.center, params.overpressure);
    let p_peak = peak_face_overpressure(src, face, params);
    let dir = face.center - src.origin;
    let mu = incidence_factor(dir, face.normal);
    if mu <= 0.0 || p_peak <= 0.0 {
        return 0.0;
    }
    let p_s = p_peak / mu;
    let i_wave = friedlander::positive_impulse(p_s, td, src.decay_b) * mu;
    let q = dynamic_pressure_pa(src, face, params);
    let i_drag = if params.drag_cd > 0.0 {
        friedlander::positive_impulse(q, td, src.decay_b) * params.drag_cd * mu
    } else {
        0.0
    };
    let _ = z;
    i_wave + i_drag
}

/// Rigid-wall normal reflection coefficient `p_r / p_s` from Rankine–Hugoniot.
pub fn rigid_wall_reflection_coefficient(ps_over_p0: f64, gamma: f64) -> f64 {
    rankine_hugoniot::from_overpressure_ratio(ps_over_p0, gamma).reflected_overpressure_ratio
}
