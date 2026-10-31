//! Blast loading on rigid bodies and pressure–impulse damage helpers.
//!
//! Applies a Friedlander pressure–time history over an exposed face as an
//! impulse on a [`pe_core::World`] body, with arrival time from shock radius,
//! a simple standoff and angle-of-incidence factor.
//!
//! # References
//! - Friedlander waveform (see [`crate::friedlander`]).
//! - Kinney & Graham (1985) for engineering load factors.
//! - Generic pressure–impulse (P–I) damage curves: Biggs, J. M. (1964).
//!   *Introduction to Structural Dynamics*; UFC structural-response charts —
//!   **caller supplies** curve parameters (not weapon-specific).

use pe_core::World;
use pe_math::libm_ops;
use pe_math::Vec3;

use crate::friedlander;
use crate::kinney::{peak_overpressure, P0_DEFAULT};
use crate::scaling::scaled_distance;

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
    /// Positive-phase duration scale [s] at `Z = 1` (engineering; caller-tuned).
    pub td_at_z1: f64,
    /// Friedlander decay `b` [dimensionless].
    pub decay_b: f64,
}

impl BlastSource {
    /// Default engineering source at the origin.
    pub fn new(origin: Vec3, w_eq_kg: f64) -> Self {
        Self {
            origin,
            w_eq_kg,
            t0: 0.0,
            p0: P0_DEFAULT,
            td_at_z1: 0.01,
            decay_b: 1.0,
        }
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
        // Effective shock speed ~ a0 * Ms with Ms from weak RH; floor at a0.
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

/// Angle-of-incidence factor: `max(0, cos θ)` with `θ` between blast direction
/// and inward face normal (facing the charge).
pub fn incidence_factor(blast_dir_from_charge: Vec3, face_outward: Vec3) -> f64 {
    // Blast arrives opposite to outward normal for a face looking at the charge.
    let d = blast_dir_from_charge.normalize();
    let n = face_outward.normalize();
    (-d.dot(n)).max(0.0)
}

/// Standoff attenuation in addition to Kinney `p_s(Z)` — identity (1.0) by
/// default; kept as a hook for caller-supplied factors.
pub fn standoff_factor(_z: f64) -> f64 {
    1.0
}

/// Instantaneous overpressure [Pa] on a face at world time `t` [s].
pub fn face_pressure(src: &BlastSource, face: &ExposedFace, t: f64) -> f64 {
    let ta = src.t0 + src.arrival_time_approx(face.center);
    let td = src.positive_duration(face.center);
    let tau = t - ta;
    let z = src.z_at(face.center);
    let ps = peak_overpressure(z, src.p0) * standoff_factor(z);
    let dir = face.center - src.origin;
    let inc = incidence_factor(dir, face.normal);
    friedlander::pressure(tau, ps, td, src.decay_b) * inc
}

/// Apply Friedlander impulse over `dt` [s] to a world body (force ≈ p·A·n_inward).
///
/// Impulse `J = p A dt` along the inward normal (blast pushes the face inward).
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

/// Integrate positive impulse [Pa·s] on a face from arrival through `t_d`.
pub fn face_positive_impulse(src: &BlastSource, face: &ExposedFace) -> f64 {
    let z = src.z_at(face.center);
    let ps = peak_overpressure(z, src.p0) * standoff_factor(z);
    let td = src.positive_duration(face.center);
    let dir = face.center - src.origin;
    let inc = incidence_factor(dir, face.normal);
    friedlander::positive_impulse(ps, td, src.decay_b) * inc
}
