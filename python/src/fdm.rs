//! PyO3 bindings for 6-DOF flight dynamics (`pe-aero::fdm`).

use pe_aero::atmosphere::RHO0;
use pe_aero::ground_effect::{mccormick_lift_factor, wieselsberger_sigma};
use pe_aero::scenarios::{
    canonical_dryden_gust_hash as dryden_hash_rs, canonical_fdm_gust_hash as fdm_gust_hash_rs,
    canonical_fdm_hash as fdm_hash_rs,
};
use pe_aero::{
    coordinated_turn_load_factor, coordinated_turn_rate, linearise_lateral, linearise_longitudinal,
    linearise_longitudinal_ex, navion_nelson, phugoid_approximation, short_period_approximation,
    short_period_approximation_with_alphadot, trim_steady_level, TrimGuess, TrimStatus,
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

/// Canonical FDM + held Dryden gust hash (cross-OS lock).
#[pyfunction]
fn canonical_fdm_gust_hash() -> u64 {
    fdm_gust_hash_rs()
}

/// Lanchester / Etkin phugoid approximation `(omega_n, zeta)` (sanity only).
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

/// Wieselsberger induced-drag factor `σ(h/b)`.
#[pyfunction]
fn fdm_wieselsberger_sigma(h_over_b: f64) -> f64 {
    wieselsberger_sigma(h_over_b)
}

/// McCormick ground-effect lift factor `K_L = 2 − σ(h/b)`.
#[pyfunction]
fn fdm_mccormick_lift_factor(h_over_b: f64) -> f64 {
    mccormick_lift_factor(h_over_b)
}

fn navion_trim() -> PyResult<(pe_aero::NavionData, pe_aero::TrimResult)> {
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
    Ok((nav, trim))
}

/// Navion (Nelson) trim + longitudinal eigenmodes as a dict of measured values.
#[pyfunction]
fn fdm_navion_longitudinal_modes(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let (nav, trim) = navion_trim()?;
    let (w_approx, z_approx) = phugoid_approximation(trim.v, trim.cl, trim.cd, STANDARD_GRAVITY);
    let ss = linearise_longitudinal(&nav.model, &trim, RHO0, STANDARD_GRAVITY);
    let modes = ss
        .modes
        .ok_or_else(|| PyValueError::new_err("mode classification failed"))?;
    let ss0 = linearise_longitudinal_ex(&nav.model, &trim, RHO0, STANDARD_GRAVITY, false);
    let modes0 = ss0
        .modes
        .ok_or_else(|| PyValueError::new_err("no-α̇ mode classification failed"))?;
    let (w_sp0, z_sp0) = short_period_approximation(&nav.model, trim.v, RHO0, trim.alpha);
    let (w_sp1, z_sp1) =
        short_period_approximation_with_alphadot(&nav.model, trim.v, RHO0, trim.alpha);
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
    d.set_item("short_period_zeta_no_alphadot", modes0.short_period.zeta())?;
    d.set_item("phugoid_omega_approx", w_approx)?;
    d.set_item("phugoid_zeta_approx", z_approx)?;
    d.set_item("short_period_omega_approx", w_sp0)?;
    d.set_item("short_period_zeta_approx", z_sp0)?;
    d.set_item("short_period_omega_approx_alphadot", w_sp1)?;
    d.set_item("short_period_zeta_approx_alphadot", z_sp1)?;
    Ok(d.into())
}

/// Navion lateral-directional eigenmodes (roll / spiral / Dutch roll).
#[pyfunction]
fn fdm_navion_lateral_modes(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let (nav, trim) = navion_trim()?;
    let ss = linearise_lateral(&nav.model, &trim, RHO0, STANDARD_GRAVITY);
    let modes = ss
        .modes
        .ok_or_else(|| PyValueError::new_err("lateral mode classification failed"))?;
    let d = PyDict::new(py);
    d.set_item("verified", true)?;
    d.set_item("citation", nav.citation)?;
    d.set_item("roll", modes.roll.re)?;
    d.set_item("spiral", modes.spiral.re)?;
    d.set_item("dutch_roll_omega_n", modes.dutch_roll.omega_n())?;
    d.set_item("dutch_roll_zeta", modes.dutch_roll.zeta())?;
    d.set_item("dutch_roll_re", modes.dutch_roll.re)?;
    d.set_item("dutch_roll_im", modes.dutch_roll.im)?;
    Ok(d.into())
}

/// Register FDM helpers on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(canonical_fdm_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_dryden_gust_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_fdm_gust_hash, m)?)?;
    m.add_function(wrap_pyfunction!(fdm_phugoid_approximation, m)?)?;
    m.add_function(wrap_pyfunction!(fdm_coordinated_turn_n, m)?)?;
    m.add_function(wrap_pyfunction!(fdm_coordinated_turn_rate, m)?)?;
    m.add_function(wrap_pyfunction!(fdm_wieselsberger_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(fdm_mccormick_lift_factor, m)?)?;
    m.add_function(wrap_pyfunction!(fdm_navion_longitudinal_modes, m)?)?;
    m.add_function(wrap_pyfunction!(fdm_navion_lateral_modes, m)?)?;
    Ok(())
}
