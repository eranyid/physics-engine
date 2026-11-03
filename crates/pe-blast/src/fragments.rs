//! Gurney fragment velocity and Mott fragment-mass distribution.
//!
//! Fragments can be spawned into a [`pe_ballistics::ProjectileSystem`] as
//! separately simulated ballistic bodies (G1 drag by default).
//!
//! # References
//! - Gurney, R. W. (1943). The initial velocities of fragments from bombs,
//!   shells, and grenades. BRL Report 405 — energy methods with caller-supplied
//!   or table `√(2E)` [m/s]. **No explosive chemistry or recipes.**
//! - Mott, N. F. (1947). Fragmentation of shell cases. *Proc. R. Soc. A* —
//!   exponential mass distribution sampling.
//! - Cooper, P. W. *Explosives Engineering* — commonly cited Gurney constants
//!   (see [`GurneyConstant`] provenance).
//!
//! RNG: [`pe_math::Xoshiro256StarStar`] (shared with pe-aero / pe-hydro).

use pe_ballistics::{
    AmmoProfile, BallisticCoefficient, DragModelRef, FieldProvenance, MuzzlePose, ProjectileSystem,
    ShotId, AMMO_PROFILE_VERSION,
};
use pe_math::libm_ops;
use pe_math::{Vec3, Xoshiro256StarStar};

use crate::error::{require_finite, require_positive, BlastError};

/// Charge-to-metal mass ratio geometries for Gurney.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GurneyGeometry {
    /// Flat sandwich / open-faced slab: `V = √(2E) · √(C/M / (1 + C/M))`.
    Flat,
    /// Sheet / open sandwich (alias of [`Flat`] for API clarity).
    Sheet,
    /// Cylinder: `V = √(2E) · √(C/M / (1 + ½ C/M))`.
    Cylindrical,
    /// Sphere: `V = √(2E) · √(C/M / (1 + ⅗ C/M))`.
    Spherical,
}

impl GurneyGeometry {
    fn cm_factor(self, cm: f64) -> f64 {
        match self {
            GurneyGeometry::Flat | GurneyGeometry::Sheet => cm / (1.0 + cm),
            GurneyGeometry::Cylindrical => cm / (1.0 + 0.5 * cm),
            GurneyGeometry::Spherical => cm / (1.0 + 0.6 * cm),
        }
    }
}

/// Named Gurney `√(2E)` constant [m/s] with provenance.
#[derive(Clone, Copy, Debug)]
pub struct GurneyConstant {
    /// Display name (e.g. `"TNT"`).
    pub name: &'static str,
    /// `√(2E)` [m/s].
    pub sqrt_2e_m_s: f64,
    /// `true` only when the exact number was taken from a fetched primary table.
    pub verified: bool,
    /// Short provenance note.
    pub source_note: &'static str,
}

/// Built-in Gurney constants (fixed order).
///
/// Values labelled `verified: false` are **unverified** widely-circulated
/// engineering numbers (Cooper / handbook summaries) — not locked to a
/// primary BRL table cell in this tree. Callers may still use them for
/// demos; production should pass a sourced `√(2E)`.
pub fn gurney_constants() -> &'static [GurneyConstant] {
    &GURNEY_TABLE
}

const GURNEY_TABLE: [GurneyConstant; 3] = [
    GurneyConstant {
        name: "TNT",
        sqrt_2e_m_s: 2440.0,
        verified: false,
        source_note:
            "unverified: widely cited Cooper/handbook TNT √(2E)≈2.44 km/s — not a fetched BRL cell",
    },
    GurneyConstant {
        name: "CompB",
        sqrt_2e_m_s: 2700.0,
        verified: false,
        source_note: "unverified: widely cited Comp B √(2E)≈2.70 km/s — not a fetched primary cell",
    },
    GurneyConstant {
        name: "HMX",
        sqrt_2e_m_s: 2970.0,
        verified: false,
        source_note: "unverified: handbook-order HMX √(2E)≈2.97 km/s — not a fetched primary cell",
    },
];

/// Look up a built-in Gurney constant by name (case-sensitive).
pub fn gurney_constant_by_name(name: &str) -> Option<&'static GurneyConstant> {
    GURNEY_TABLE.iter().find(|c| c.name == name)
}

/// Gurney fragment velocity [m/s].
///
/// - `gurney_sqrt_2e` = `√(2E)` [m/s] — **caller-supplied** Gurney energy
///   parameter (not derived from compositions here).
/// - `charge_to_mass` = `C/M` [dimensionless] (> 0).
pub fn gurney_velocity(
    gurney_sqrt_2e: f64,
    charge_to_mass: f64,
    geometry: GurneyGeometry,
) -> Option<f64> {
    try_gurney_velocity(gurney_sqrt_2e, charge_to_mass, geometry).ok()
}

/// Fallible Gurney velocity with typed [`BlastError`].
pub fn try_gurney_velocity(
    gurney_sqrt_2e: f64,
    charge_to_mass: f64,
    geometry: GurneyGeometry,
) -> Result<f64, BlastError> {
    require_positive(gurney_sqrt_2e, "gurney_sqrt_2e")?;
    require_positive(charge_to_mass, "charge_to_mass")?;
    let factor = geometry.cm_factor(charge_to_mass);
    let v = gurney_sqrt_2e * libm_ops::sqrt(factor);
    require_finite(v, "gurney_velocity")?;
    Ok(v)
}

/// Ideal Gurney metal kinetic energy budget [J]: `½ M V²` with
/// `V = gurney_velocity(...)` and casing mass `M` [kg].
pub fn gurney_metal_ke(casing_mass_kg: f64, velocity_m_s: f64) -> f64 {
    0.5 * casing_mass_kg * velocity_m_s * velocity_m_s
}

/// Mott fragment mass distribution: `N(>m) ∝ exp(−√(m/μ))` style sampling.
///
/// Samples a fragment mass [kg] given characteristic mass `mu_kg` [kg] using
/// the inverse-CDF of an exponential-of-sqrt law:
/// `m = μ · [−ln(1 − U)]²` with `U ~ Uniform(0,1)`.
///
/// Reference: Mott (1947); engineering form used in fragmenting-case models.
pub fn mott_sample_mass(rng: &mut Xoshiro256StarStar, mu_kg: f64) -> f64 {
    let u = rng.next_f64().clamp(1e-12, 1.0 - 1e-12);
    let x = -libm_ops::log(1.0 - u);
    mu_kg * x * x
}

/// Closed-form mean of the Mott sampling law `m = μ X²`, `X ~ Exp(1)`:
/// `E[m] = μ E[X²] = 2 μ`.
pub fn mott_mean_mass(mu_kg: f64) -> f64 {
    2.0 * mu_kg
}

/// Sample `n` fragment masses [kg] (deterministic given seed).
pub fn mott_sample_many(seed: u64, mu_kg: f64, n: usize) -> Vec<f64> {
    let mut rng = Xoshiro256StarStar::seed(seed);
    (0..n).map(|_| mott_sample_mass(&mut rng, mu_kg)).collect()
}

/// One fragment ready to spawn as a ballistic body.
#[derive(Clone, Copy, Debug)]
pub struct FragmentSpec {
    /// Mass [kg].
    pub mass_kg: f64,
    /// Launch position [m].
    pub position: Vec3,
    /// Launch velocity [m/s].
    pub velocity: Vec3,
}

/// Build fragment specs from Gurney speed + Mott masses with deterministic
/// spherical directions (Fibonacci lattice on the sphere, then RNG jitter
/// disabled — pure lattice for bit-stability).
///
/// Directions are unit vectors; speed is identical Gurney `V` for every
/// fragment (ideal Gurney). Masses from Mott; if `normalize_mass` is set, masses
/// are rescaled so `Σ m = casing_mass_kg` (momentum/energy budget bookkeeping).
pub fn build_fragment_specs(
    origin: Vec3,
    gurney_v: f64,
    masses_kg: &[f64],
    casing_mass_kg: Option<f64>,
) -> Vec<FragmentSpec> {
    let n = masses_kg.len();
    if n == 0 || gurney_v <= 0.0 {
        return Vec::new();
    }
    let mut masses: Vec<f64> = masses_kg.to_vec();
    if let Some(m_tot) = casing_mass_kg {
        let s: f64 = masses.iter().sum();
        if s > 0.0 && m_tot > 0.0 {
            let scale = m_tot / s;
            for m in &mut masses {
                *m *= scale;
            }
        }
    }
    let dirs = fibonacci_sphere(n);
    masses
        .into_iter()
        .zip(dirs)
        .map(|(mass_kg, d)| FragmentSpec {
            mass_kg,
            position: origin,
            velocity: d * gurney_v,
        })
        .collect()
}

/// Deterministic Fibonacci sphere directions (unit), index order 0..n-1.
pub fn fibonacci_sphere(n: usize) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    let golden = core::f64::consts::PI * (3.0 - libm_ops::sqrt(5.0));
    for i in 0..n {
        let y = 1.0 - (2.0 * (i as f64) + 1.0) / (n as f64);
        let r = libm_ops::sqrt((1.0 - y * y).max(0.0));
        let theta = golden * (i as f64);
        let x = libm_ops::cos(theta) * r;
        let z = libm_ops::sin(theta) * r;
        out.push(Vec3::new(x, y, z).normalize());
    }
    out
}

/// Total fragment kinetic energy [J].
pub fn fragments_kinetic_energy(specs: &[FragmentSpec]) -> f64 {
    specs
        .iter()
        .map(|f| 0.5 * f.mass_kg * f.velocity.norm_squared())
        .sum()
}

/// Total fragment linear momentum [kg·m/s].
pub fn fragments_momentum(specs: &[FragmentSpec]) -> Vec3 {
    let mut p = Vec3::zero();
    for f in specs {
        p += f.velocity * f.mass_kg;
    }
    p
}

/// Default fragment ammo profile (analytic drag, zero dispersion) for mass `m`
/// and muzzle speed equal to `|v|`.
pub fn fragment_ammo_profile(
    name: &str,
    mass_kg: f64,
    speed_m_s: f64,
    caliber_m: f64,
) -> AmmoProfile {
    AmmoProfile {
        version: AMMO_PROFILE_VERSION,
        name: name.to_string(),
        caliber_m,
        mass_kg,
        muzzle_velocity_m_s: speed_m_s,
        bc: BallisticCoefficient {
            value_si: (mass_kg / (core::f64::consts::PI * (0.5 * caliber_m).powi(2))).max(1.0),
            drag_model: DragModelRef::Analytic,
            table: None,
            provenance: FieldProvenance {
                note: "blast fragment stand-in; unverified BC".into(),
                verified: false,
            },
        },
        length_m: caliber_m * 2.0,
        twist_m_per_rev: 0.0,
        form_factor: 1.0,
        barrel_length_ref_m: None,
        velocity_std_m_s: 0.0,
        dispersion_std_rad: 0.0,
        provenance: FieldProvenance {
            note: "Gurney/Mott blast fragment".into(),
            verified: false,
        },
    }
}

/// Spawn fragment specs into a [`ProjectileSystem`] (ascending `ShotId`).
///
/// Each fragment gets its own ammo profile (mass / speed). Spin effects should
/// be disabled by the caller for debris (`sys.enable_spin_effects = false`).
pub fn spawn_fragments_ballistic(
    sys: &mut ProjectileSystem,
    specs: &[FragmentSpec],
    base_shot_id: u64,
    caliber_m: f64,
) {
    for (i, spec) in specs.iter().enumerate() {
        let speed = spec.velocity.norm();
        let dir = if speed > 1e-15 {
            spec.velocity * (1.0 / speed)
        } else {
            Vec3::unit_x()
        };
        let profile = fragment_ammo_profile(
            &format!("frag_{}", base_shot_id + i as u64),
            spec.mass_kg,
            speed,
            caliber_m,
        );
        let pidx = sys.add_profile(profile);
        sys.spawn(
            ShotId(base_shot_id + i as u64),
            pidx,
            MuzzlePose {
                position: spec.position,
                direction: dir,
                barrel_length_m: None,
            },
        );
    }
}

/// End-to-end: Mott sample → Gurney V → normalize mass → spawn ballistic.
pub fn detonate_cased_charge(
    sys: &mut ProjectileSystem,
    origin: Vec3,
    gurney_sqrt_2e: f64,
    charge_to_mass: f64,
    geometry: GurneyGeometry,
    casing_mass_kg: f64,
    mott_mu_kg: f64,
    n_fragments: usize,
    seed: u64,
    base_shot_id: u64,
    caliber_m: f64,
) -> Option<(f64, Vec<FragmentSpec>)> {
    let v = gurney_velocity(gurney_sqrt_2e, charge_to_mass, geometry)?;
    let masses = mott_sample_many(seed, mott_mu_kg, n_fragments);
    let specs = build_fragment_specs(origin, v, &masses, Some(casing_mass_kg));
    spawn_fragments_ballistic(sys, &specs, base_shot_id, caliber_m);
    Some((v, specs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn flat_equals_sqrt_half_when_cm_one() {
        let v = gurney_velocity(1000.0, 1.0, GurneyGeometry::Flat).unwrap();
        assert_relative_eq!(v, 1000.0 * libm_ops::sqrt(0.5), epsilon = 1e-12);
    }

    #[test]
    fn sheet_matches_flat() {
        let a = gurney_velocity(2000.0, 0.4, GurneyGeometry::Flat).unwrap();
        let b = gurney_velocity(2000.0, 0.4, GurneyGeometry::Sheet).unwrap();
        assert_relative_eq!(a, b, epsilon = 0.0);
    }

    #[test]
    fn mott_seed_deterministic() {
        let a = mott_sample_many(7, 0.01, 16);
        let b = mott_sample_many(7, 0.01, 16);
        assert_eq!(a, b);
    }

    #[test]
    fn mott_mean_closed_form() {
        assert_relative_eq!(mott_mean_mass(0.05), 0.1, epsilon = 0.0);
    }
}
