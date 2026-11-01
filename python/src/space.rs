//! PyO3 bindings for `pe-space` (M7).

use pe_math::Vec3;
use pe_space::constants::{EARTH_J2, EARTH_MU, EARTH_RADIUS};
use pe_space::diagnostics::{angular_momentum, kinetic_energy, total_energy};
use pe_space::field::{CentralMu, GravityField, NBodyPointMass};
use pe_space::dop853_hash;
use pe_space::figure_eight_hash;
use pe_space::ias15_hash;
use pe_space::integrator::{AdaptiveStep, Integrator};
use pe_space::integrators::{
    Dop853, Ias15, Leapfrog, Rk4, Whfast, WhfastCoords, WhfastCorrector, Yoshida4,
};
use pe_space::kepler::{
    elements_to_state, orbital_period, propagate_universal, state_to_elements, KeplerElements,
};
use pe_space::lambert::lambert_izzo;
use pe_space::perturbations::{accel_j2, j2_nodal_rate, J2Model};
use pe_space::scenarios::{figure_eight_state, FIGURE_EIGHT_PERIOD_RETURN};
use pe_space::state::NBodyState;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

fn vec3_from_seq(obj: &Bound<'_, PyAny>) -> PyResult<Vec3> {
    let arr: Vec<f64> = obj.extract()?;
    if arr.len() != 3 {
        return Err(PyValueError::new_err("expected length-3 sequence [x,y,z]"));
    }
    Ok(Vec3::new(arr[0], arr[1], arr[2]))
}

fn state_from_lists(
    masses: Vec<f64>,
    positions: Vec<Vec<f64>>,
    velocities: Vec<Vec<f64>>,
) -> PyResult<NBodyState> {
    if masses.len() != positions.len() || masses.len() != velocities.len() {
        return Err(PyValueError::new_err(
            "masses, positions, velocities must have the same length",
        ));
    }
    let mut x = Vec::with_capacity(positions.len());
    let mut v = Vec::with_capacity(velocities.len());
    for (p, vel) in positions.iter().zip(velocities.iter()) {
        if p.len() != 3 || vel.len() != 3 {
            return Err(PyValueError::new_err(
                "each position/velocity must be length 3",
            ));
        }
        x.push(Vec3::new(p[0], p[1], p[2]));
        v.push(Vec3::new(vel[0], vel[1], vel[2]));
    }
    Ok(NBodyState::new(masses, x, v))
}

/// Canonical figure-eight N-body hash (cross-OS determinism).
#[pyfunction]
fn canonical_space_hash() -> u64 {
    figure_eight_hash()
}

/// Canonical IAS15 eccentric-Kepler hash (cross-OS determinism).
#[pyfunction]
fn canonical_ias15_hash() -> u64 {
    ias15_hash()
}

/// Canonical DOP853 eccentric-Kepler hash (cross-OS determinism).
#[pyfunction]
fn canonical_dop853_hash() -> u64 {
    dop853_hash()
}

/// Orbital period `T = 2π √(a³/μ)` [s].
#[pyfunction]
fn kepler_period(mu: f64, a: f64) -> f64 {
    orbital_period(mu, a)
}

/// Convert Keplerian elements to position/velocity (SI). Returns `(r[3], v[3])`.
#[pyfunction]
#[pyo3(signature = (mu, a, e, i, raan, arg_p, true_anomaly))]
fn kepler_elements_to_state(
    mu: f64,
    a: f64,
    e: f64,
    i: f64,
    raan: f64,
    arg_p: f64,
    true_anomaly: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let el = KeplerElements {
        a,
        e,
        i,
        raan,
        arg_p,
        true_anomaly,
    };
    let (r, v) = elements_to_state(mu, el).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((r.to_array().to_vec(), v.to_array().to_vec()))
}

/// Convert position/velocity to Keplerian elements (SI).
#[pyfunction]
fn kepler_state_to_elements(
    mu: f64,
    r: &Bound<'_, PyAny>,
    v: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    let r = vec3_from_seq(r)?;
    let v = vec3_from_seq(v)?;
    let el = state_to_elements(mu, r, v).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((el.a, el.e, el.i, el.raan, el.arg_p, el.true_anomaly))
}

/// Universal-variable Kepler propagation by `dt` [s]. Returns `(r[3], v[3])`.
#[pyfunction]
fn kepler_propagate(
    mu: f64,
    r: &Bound<'_, PyAny>,
    v: &Bound<'_, PyAny>,
    dt: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let r = vec3_from_seq(r)?;
    let v = vec3_from_seq(v)?;
    let (rf, vf) =
        propagate_universal(mu, r, v, dt).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((rf.to_array().to_vec(), vf.to_array().to_vec()))
}

/// Lambert solver (0-rev). Returns `(v1[3], v2[3])`.
#[pyfunction]
#[pyo3(signature = (mu, r1, r2, tof, short_path=true))]
fn lambert(
    mu: f64,
    r1: &Bound<'_, PyAny>,
    r2: &Bound<'_, PyAny>,
    tof: f64,
    short_path: bool,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let r1 = vec3_from_seq(r1)?;
    let r2 = vec3_from_seq(r2)?;
    let sol = lambert_izzo(mu, r1, r2, tof, short_path, 0)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((sol.v1.to_array().to_vec(), sol.v2.to_array().to_vec()))
}

/// Analytic J₂ nodal regression rate [rad/s].
#[pyfunction]
#[pyo3(signature = (a, e, i, mu=None, radius=None, j2=None))]
fn j2_nodal_regression_rate(
    a: f64,
    e: f64,
    i: f64,
    mu: Option<f64>,
    radius: Option<f64>,
    j2: Option<f64>,
) -> f64 {
    let model = J2Model {
        mu: mu.unwrap_or(EARTH_MU),
        radius: radius.unwrap_or(EARTH_RADIUS),
        j2: j2.unwrap_or(EARTH_J2),
    };
    j2_nodal_rate(&model, a, e, i)
}

/// J₂ acceleration at position `r` [m] (Earth defaults).
#[pyfunction]
fn j2_acceleration(r: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let r = vec3_from_seq(r)?;
    Ok(accel_j2(&J2Model::earth(), r).to_array().to_vec())
}

/// Integrate an N-body / central-μ system; return final positions as a flat
/// length-`3N` list (row-major).
#[pyfunction]
#[pyo3(signature = (masses, positions, velocities, t_end, dt, integrator="leapfrog", mu=None, geometric=false, g_const=None))]
#[allow(clippy::too_many_arguments)]
fn nbody_integrate(
    py: Python<'_>,
    masses: Vec<f64>,
    positions: Vec<Vec<f64>>,
    velocities: Vec<Vec<f64>>,
    t_end: f64,
    dt: f64,
    integrator: &str,
    mu: Option<f64>,
    geometric: bool,
    g_const: Option<f64>,
) -> PyResult<Vec<f64>> {
    let mut state = state_from_lists(masses, positions, velocities)?;
    if !matches!(
        integrator,
        "leapfrog"
            | "yoshida4"
            | "rk4"
            | "ias15"
            | "dop853"
            | "whfast"
            | "whfast_helio"
            | "whfast_c5"
            | "whfast_c11"
    ) {
        return Err(PyValueError::new_err(format!(
            "unknown integrator {integrator:?}; expected leapfrog|yoshida4|rk4|ias15|dop853|whfast|whfast_helio|whfast_c5|whfast_c11"
        )));
    }
    let integ_name = integrator.to_string();
    let use_mu = mu;
    let use_geom = geometric;
    let g = g_const.unwrap_or(if geometric { 1.0 } else { pe_math::G });

    py.allow_threads(|| {
        let central = CentralMu::new(use_mu.unwrap_or(EARTH_MU));
        let nb = NBodyPointMass::geometric();
        let si = NBodyPointMass::si();
        let field: &dyn GravityField = if use_mu.is_some() {
            &central
        } else if use_geom {
            &nb
        } else {
            &si
        };
        match integ_name.as_str() {
            "leapfrog" => {
                let mut integ = Leapfrog::default();
                integ.integrate_to(field, &mut state, 0.0, t_end, dt);
            }
            "yoshida4" => {
                let mut integ = Yoshida4::default();
                integ.integrate_to(field, &mut state, 0.0, t_end, dt);
            }
            "ias15" => {
                state.enable_compensated_positions();
                let mut integ = Ias15::new(AdaptiveStep {
                    dt,
                    ..AdaptiveStep::default()
                });
                integ.integrate_to(field, &mut state, 0.0, t_end, dt);
            }
            "dop853" => {
                let mut integ = Dop853::new(AdaptiveStep {
                    dt,
                    ..AdaptiveStep::default()
                });
                integ.integrate_to(field, &mut state, 0.0, t_end, dt);
            }
            "whfast" => {
                let mut integ = Whfast::new(g, WhfastCoords::Jacobi);
                integ.integrate_to(field, &mut state, 0.0, t_end, dt);
            }
            "whfast_c5" => {
                let mut integ =
                    Whfast::new(g, WhfastCoords::Jacobi).with_corrector(WhfastCorrector::Order5);
                let n = (t_end / dt).round() as usize;
                integ.integrate_corrected(field, &mut state, 0.0, dt, n);
            }
            "whfast_c11" => {
                let mut integ =
                    Whfast::new(g, WhfastCoords::Jacobi).with_corrector(WhfastCorrector::Order11);
                let n = (t_end / dt).round() as usize;
                integ.integrate_corrected(field, &mut state, 0.0, dt, n);
            }
            "whfast_helio" => {
                let mut integ = Whfast::new(g, WhfastCoords::DemocraticHeliocentric);
                integ.integrate_to(field, &mut state, 0.0, t_end, dt);
            }
            _ => {
                let mut integ = Rk4;
                integ.integrate_to(field, &mut state, 0.0, t_end, dt);
            }
        }
    });

    let n = state.n();
    let mut flat = Vec::with_capacity(n * 3);
    for i in 0..n {
        flat.extend_from_slice(&state.x[i].to_array());
    }
    Ok(flat)
}

/// Figure-eight demo: integrate one period, return max |Δx|/|Δv|.
#[pyfunction]
#[pyo3(signature = (steps=4000, integrator="leapfrog"))]
fn figure_eight_period_error(steps: usize, integrator: &str) -> PyResult<f64> {
    let mut state = figure_eight_state();
    let x0 = state.x.clone();
    let v0 = state.v.clone();
    let field = NBodyPointMass::geometric();
    let t_end = if integrator == "ias15" {
        FIGURE_EIGHT_PERIOD_RETURN
    } else {
        pe_space::FIGURE_EIGHT_PERIOD
    };
    match integrator {
        "leapfrog" => {
            let mut integ = Leapfrog::default();
            let dt = t_end / steps as f64;
            let mut t = 0.0;
            for _ in 0..steps {
                let res = integ.step(&field, &mut state, t, dt);
                t = res.t;
            }
        }
        "ias15" => {
            state.enable_compensated_positions();
            let mut integ = Ias15::new(AdaptiveStep {
                dt: t_end / steps.max(1) as f64,
                dt_min: 1e-10,
                dt_max: t_end / 20.0,
                ..AdaptiveStep::default()
            });
            integ.integrate_to(&field, &mut state, 0.0, t_end, t_end / steps.max(1) as f64);
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown integrator {other:?}; expected leapfrog|ias15"
            )));
        }
    }
    let mut max_dx = 0.0_f64;
    for i in 0..3 {
        let dx = (state.x[i] - x0[i]).norm();
        let dv = (state.v[i] - v0[i]).norm();
        if dx > max_dx {
            max_dx = dx;
        }
        if dv > max_dx {
            max_dx = dv;
        }
    }
    let _ = (
        kinetic_energy(&state),
        angular_momentum(&state, Vec3::zero()),
        total_energy(&field, &state),
    );
    Ok(max_dx)
}

/// Register space bindings on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(canonical_space_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_ias15_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_dop853_hash, m)?)?;
    m.add_function(wrap_pyfunction!(kepler_period, m)?)?;
    m.add_function(wrap_pyfunction!(kepler_elements_to_state, m)?)?;
    m.add_function(wrap_pyfunction!(kepler_state_to_elements, m)?)?;
    m.add_function(wrap_pyfunction!(kepler_propagate, m)?)?;
    m.add_function(wrap_pyfunction!(lambert, m)?)?;
    m.add_function(wrap_pyfunction!(j2_nodal_regression_rate, m)?)?;
    m.add_function(wrap_pyfunction!(j2_acceleration, m)?)?;
    m.add_function(wrap_pyfunction!(nbody_integrate, m)?)?;
    m.add_function(wrap_pyfunction!(figure_eight_period_error, m)?)?;
    m.add("EARTH_MU", EARTH_MU)?;
    m.add("EARTH_RADIUS", EARTH_RADIUS)?;
    m.add("EARTH_J2", EARTH_J2)?;
    Ok(())
}
