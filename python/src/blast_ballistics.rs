//! PyO3 bindings for `pe-blast` and `pe-ballistics`.

use pe_ballistics::ammo::{bundled_profiles, AmmoProfile};
use pe_ballistics::oracles::{vacuum_max_height, vacuum_range};
use pe_ballistics::projectile::{MuzzlePose, ProjectileSystem, ShotId};
use pe_ballistics::{ballistics_volley_hash, g0};
use pe_blast::{
    blast_pyramid_hash, friedlander_pressure, peak_overpressure, positive_impulse, scaled_distance,
    xi0_gamma_14_spherical,
};
use pe_math::Vec3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Canonical blast-pyramid determinism hash.
#[pyfunction]
fn canonical_blast_hash() -> u64 {
    blast_pyramid_hash()
}

/// Canonical 100-bullet volley determinism hash.
#[pyfunction]
fn canonical_ballistics_hash() -> u64 {
    ballistics_volley_hash()
}

/// Hopkinson–Cranz scaled distance `Z = R / W_eq^{1/3}` [m·kg^{-1/3}].
#[pyfunction]
fn blast_scaled_distance(range_m: f64, charge_mass_kg: f64, tnt_equivalence: f64) -> PyResult<f64> {
    scaled_distance(range_m, charge_mass_kg, tnt_equivalence)
        .ok_or_else(|| PyValueError::new_err("non-physical inputs"))
}

/// Kinney–Graham peak overpressure [Pa] at scaled distance `z`.
#[pyfunction]
#[pyo3(signature = (z, p0=101_325.0))]
fn blast_peak_overpressure(z: f64, p0: f64) -> f64 {
    peak_overpressure(z, p0)
}

/// Friedlander pressure [Pa] at time `t` [s] after arrival.
#[pyfunction]
fn blast_friedlander(t: f64, p_s: f64, t_d: f64, b: f64) -> f64 {
    friedlander_pressure(t, p_s, t_d, b)
}

/// Friedlander positive impulse [Pa·s].
#[pyfunction]
fn blast_friedlander_impulse(p_s: f64, t_d: f64, b: f64) -> f64 {
    positive_impulse(p_s, t_d, b)
}

/// Sedov–Taylor ξ₀ for γ=1.4 spherical.
#[pyfunction]
fn blast_sedov_xi0() -> f64 {
    xi0_gamma_14_spherical()
}

/// Vacuum projectile range [m].
#[pyfunction]
#[pyo3(signature = (v0, theta_rad, g=None))]
fn ballistics_vacuum_range(v0: f64, theta_rad: f64, g: Option<f64>) -> f64 {
    vacuum_range(v0, theta_rad, g.unwrap_or_else(g0))
}

/// Vacuum max height [m].
#[pyfunction]
#[pyo3(signature = (v0, theta_rad, g=None))]
fn ballistics_vacuum_max_height(v0: f64, theta_rad: f64, g: Option<f64>) -> f64 {
    vacuum_max_height(v0, theta_rad, g.unwrap_or_else(g0))
}

/// List bundled ammo profile names.
#[pyfunction]
fn ammo_bundled_names() -> Vec<String> {
    bundled_profiles().into_iter().map(|p| p.name).collect()
}

/// Muzzle energy [J] for a bundled profile by name.
#[pyfunction]
fn ammo_muzzle_energy(name: &str) -> PyResult<f64> {
    bundled_profiles()
        .into_iter()
        .find(|p| p.name == name)
        .map(|p| p.muzzle_energy_j())
        .ok_or_else(|| PyValueError::new_err(format!("unknown ammo profile {name:?}")))
}

/// Ammo profile as a JSON string (bundled by name).
#[pyfunction]
fn ammo_profile_json(name: &str) -> PyResult<String> {
    bundled_profiles()
        .into_iter()
        .find(|p| p.name == name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown ammo profile {name:?}")))?
        .to_json()
        .map_err(PyValueError::new_err)
}

/// Demo: fire `n` identical zero-dispersion shots along +x; return final x positions.
#[pyfunction]
#[pyo3(signature = (profile_name, n=10, dt=1.0/120.0, steps=60))]
fn ballistics_demo_volley(
    profile_name: &str,
    n: usize,
    dt: f64,
    steps: usize,
) -> PyResult<Vec<f64>> {
    let mut profile = bundled_profiles()
        .into_iter()
        .find(|p| p.name == profile_name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown ammo {profile_name:?}")))?;
    profile.velocity_std_m_s = 0.0;
    profile.dispersion_std_rad = 0.0;
    let mut sys = ProjectileSystem::new(0.0);
    let idx = sys.add_profile(profile);
    for i in 0..n {
        sys.spawn(
            ShotId(i as u64 + 1),
            idx,
            MuzzlePose {
                position: Vec3::new(0.0, i as f64 * 0.1, 1.0),
                direction: Vec3::unit_x(),
                barrel_length_m: None,
            },
        );
    }
    for _ in 0..steps {
        sys.step(dt, &[], &[]);
    }
    Ok(sys.projectiles.iter().map(|p| p.x.x).collect())
}

/// Parse ammo JSON and return muzzle velocity [m/s].
#[pyfunction]
fn ammo_from_json_muzzle_velocity(json: &str) -> PyResult<f64> {
    let p = AmmoProfile::from_json(json).map_err(PyValueError::new_err)?;
    Ok(p.muzzle_velocity_m_s)
}

/// Register blast + ballistics bindings.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(canonical_blast_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_ballistics_hash, m)?)?;
    m.add_function(wrap_pyfunction!(blast_scaled_distance, m)?)?;
    m.add_function(wrap_pyfunction!(blast_peak_overpressure, m)?)?;
    m.add_function(wrap_pyfunction!(blast_friedlander, m)?)?;
    m.add_function(wrap_pyfunction!(blast_friedlander_impulse, m)?)?;
    m.add_function(wrap_pyfunction!(blast_sedov_xi0, m)?)?;
    m.add_function(wrap_pyfunction!(ballistics_vacuum_range, m)?)?;
    m.add_function(wrap_pyfunction!(ballistics_vacuum_max_height, m)?)?;
    m.add_function(wrap_pyfunction!(ammo_bundled_names, m)?)?;
    m.add_function(wrap_pyfunction!(ammo_muzzle_energy, m)?)?;
    m.add_function(wrap_pyfunction!(ammo_profile_json, m)?)?;
    m.add_function(wrap_pyfunction!(ballistics_demo_volley, m)?)?;
    m.add_function(wrap_pyfunction!(ammo_from_json_muzzle_velocity, m)?)?;
    Ok(())
}
