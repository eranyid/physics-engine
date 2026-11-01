//! PyO3 bindings for 6-DOF flight dynamics (`pe-aero::fdm`).

use pe_aero::atmosphere::RHO0;
use pe_aero::scenarios::{
    canonical_dryden_gust_hash as dryden_hash_rs, canonical_fdm_hash as fdm_hash_rs,
};
use pe_aero::{
    coordinated_turn_load_factor, coordinated_turn_rate, linearise_longitudinal, navion_nelson,
    phugoid_approximation, trim_steady_level, TrimGuess, TrimStatus,
};
use pe_math::STANDARD_GRAVITY;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

/// Canonical 6-DOF FDM packed-state hash (cross-OS lock).
#[pyfunction]
fn canonical_fdm_hash() -> u64 {
    fdm_hash_rs()
}

/// Canonical Dryden gust-sequence hash (cross-OS lock).
#[pyfunction]
fn canonical_dryden_gust_hash() -> u64 {
    dryden_hash_rs()
}

/// Lanchester / Etkin phugoid approximation `(omega_n, zeta)`.
#[pyfunction]
#[pyo3(signature = (v, cl, cd, g=None))]
fn fdm_phugoid_approximation(v: f64, cl: f64, cd: f64, g: Option<f64>) -> (f64, f64) {
    phugoid_approximation(v, cl, cd, g.unwrap_or(STANDARD_GRAVITY))
}

/// Coordinated-turn load factor `n = 1/cos(phi)`.
#[pyfunction]
fn fdm_coordinated_turn_n(phi_rad: f64) -> f64 {
    coordinated_turn_load_factor(phi_rad)
}

/// Coordinated-turn rate `g tan(phi)/V` [rad/s].
#[pyfunction]
#[pyo3(signature = (phi_rad, v, g=None))]
fn fdm_coordinated_turn_rate(phi_rad: f64, v: f64, g: Option<f64>) -> f64 {
    coordinated_turn_rate(phi_rad, v, g.unwrap_or(STANDARD_GRAVITY))
}

/// Navion (Nelson) trim + longitudinal eigenmodes as a dict of measured values.
#[pyfunction]
fn fdm_navion_longitudinal_modes(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let nav = navion_nelson();
    if !nav.verified {
        return Err(PyValueError::new_err("Navion data marked unverified"));
    }
    let guess = TrimGuess {
        v: nav.v_ref,
        alpha: nav.alpha_trim,
        delta_e: -nav.model.deriv.cm_alpha * nav.alpha_trim / nav.model.deriv.cm_de,
        thrust: 0.5 * RHO0 * nav.v_ref * nav.v_ref * nav.model.s * nav.cd_trim,
        g: STANDARD_GRAVITY,
        rho: Some(RHO0),
    };
    let trim = trim_steady_level(&nav.model, guess);
    if trim.status != TrimStatus::Converged {
        return Err(PyValueError::new_err(format!(
            "trim failed, residual={}",
            trim.residual_norm
        )));
    }
    let (w_approx, z_approx) = phugoid_approximation(trim.v, trim.cl, trim.cd, STANDARD_GRAVITY);
    let ss = linearise_longitudinal(&nav.model, &trim, RHO0, STANDARD_GRAVITY);
    let modes = ss
        .modes
        .ok_or_else(|| PyValueError::new_err("mode classification failed"))?;
    let d = PyDict::new(py);
    d.set_item("verified", true)?;
    d.set_item("citation", nav.citation)?;
    d.set_item("v", trim.v)?;
    d.set_item("alpha", trim.alpha)?;
    d.set_item("delta_e", trim.delta_e)?;
    d.set_item("thrust", trim.thrust)?;
    d.set_item("CL", trim.cl)?;
    d.set_item("CD", trim.cd)?;
    d.set_item("trim_residual", trim.residual_norm)?;
    d.set_item("phugoid_omega_n", modes.phugoid.omega_n())?;
    d.set_item("phugoid_zeta", modes.phugoid.zeta())?;
    d.set_item("short_period_omega_n", modes.short_period.omega_n())?;
    d.set_item("short_period_zeta", modes.short_period.zeta())?;
    d.set_item("phugoid_omega_approx", w_approx)?;
    d.set_item("phugoid_zeta_approx", z_approx)?;
    Ok(d.into())
}

/// Register FDM helpers on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(canonical_fdm_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_dryden_gust_hash, m)?)?;
    m.add_function(wrap_pyfunction!(fdm_phugoid_approximation, m)?)?;
    m.add_function(wrap_pyfunction!(fdm_coordinated_turn_n, m)?)?;
    m.add_function(wrap_pyfunction!(fdm_coordinated_turn_rate, m)?)?;
    m.add_function(wrap_pyfunction!(fdm_navion_longitudinal_modes, m)?)?;
    Ok(())
}
