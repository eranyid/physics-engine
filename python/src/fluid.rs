//! PyO3 bindings for `pe-fluid` (NumPy array exports).

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pe_fluid::diagnostics::{kinetic_energy, mean_density_error, total_mass};
use pe_fluid::oracles::{hydrostatic_pressure, ritter_front, ritter_height};
use pe_fluid::particles::{fill_box, fill_box_boundary, mass_from_spacing, Particles};
use pe_fluid::pbf::{PbfParams, PbfWorld};
use pe_fluid::scenarios::{dfsph_tank_hash, fluid_tank_hash};
use pe_fluid::{Dim, KernelKind};
use pe_math::{Vec3, STANDARD_GRAVITY};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Canonical PBF fluid-tank determinism hash.
#[pyfunction]
fn canonical_fluid_hash() -> u64 {
    fluid_tank_hash()
}

/// Canonical DFSPH fluid-tank determinism hash.
#[pyfunction]
fn canonical_dfsph_hash() -> u64 {
    dfsph_tank_hash()
}

/// Ritter dam-break front `x_f = 2 √(g h0) t` [m].
#[pyfunction]
fn fluid_ritter_front(g: f64, h0: f64, t: f64) -> f64 {
    ritter_front(g, h0, t)
}

/// Ritter height [m] at `x` [m], time `t` [s].
#[pyfunction]
fn fluid_ritter_height(g: f64, h0: f64, x: f64, t: f64) -> f64 {
    ritter_height(g, h0, x, t)
}

/// Hydrostatic `p = ρ g h` [Pa].
#[pyfunction]
fn fluid_hydrostatic_pressure(rho: f64, g: f64, depth: f64) -> f64 {
    hydrostatic_pressure(rho, g, depth)
}

fn parse_dim(s: &str) -> PyResult<Dim> {
    match s {
        "2" | "2d" | "D2" => Ok(Dim::D2),
        "3" | "3d" | "D3" => Ok(Dim::D3),
        _ => Err(PyValueError::new_err("dim must be '2' or '3'")),
    }
}

/// Small DFSPH tank demo; returns `(positions [N,3], densities [N], mean_density_error)`.
#[pyfunction]
#[pyo3(signature = (n_side=4, steps=40, dt=1.0/120.0, dim="3"))]
#[allow(clippy::type_complexity)]
fn fluid_dfsph_tank_demo<'py>(
    py: Python<'py>,
    n_side: usize,
    steps: usize,
    dt: f64,
    dim: &str,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    use pe_fluid::particles::{
        fill_box, fill_box_boundary, mass_calibrated_to_rest, recomputed_akinci_psi, Particles,
    };
    use pe_fluid::{DfsphParams, DfsphWorld};
    let dim = parse_dim(dim)?;
    let dx = 0.05;
    let h = 2.0 * dx;
    let rho0 = 1000.0;
    let mass = mass_calibrated_to_rest(rho0, dx, h, dim, KernelKind::CubicSpline);
    let mut parts = Particles::new(dim, rho0, mass, h);
    let nz = match dim {
        Dim::D2 => 1,
        Dim::D3 => n_side,
    };
    fill_box(
        &mut parts,
        Vec3::new(0.1, 0.1, 0.1),
        n_side,
        n_side,
        nz,
        dx,
        Vec3::zero(),
    );
    let zmax = match dim {
        Dim::D2 => 0.0,
        Dim::D3 => 0.1 + n_side as f64 * dx + 0.05,
    };
    fill_box_boundary(
        &mut parts,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(
            0.1 + n_side as f64 * dx + 0.1,
            0.1 + n_side as f64 * dx + 0.1,
            zmax.max(0.45),
        ),
        dx,
        1,
    );
    recomputed_akinci_psi(&mut parts, KernelKind::CubicSpline);
    let gravity = match dim {
        Dim::D2 => Vec3::new(0.0, -STANDARD_GRAVITY, 0.0),
        Dim::D3 => Vec3::new(0.0, 0.0, -STANDARD_GRAVITY),
    };
    let params = DfsphParams {
        gravity,
        particle_diameter: dx,
        adaptive_dt: false,
        max_density_iters: 4,
        max_divergence_iters: 4,
        density_error_tol: 0.01,
        divergence_error_tol: 0.01,
        ..DfsphParams::default()
    };
    let mut world = DfsphWorld::new(parts, params);
    py.allow_threads(|| {
        for _ in 0..steps {
            world.step(dt);
        }
    });
    let n = world.particles.fluid_count();
    let mut pos = Vec::with_capacity(n * 3);
    let mut dens = Vec::with_capacity(n);
    for i in 0..world.particles.len() {
        if world.particles.kind[i] == pe_fluid::ParticleKind::Fluid {
            pos.extend_from_slice(&world.particles.x[i].to_array());
            dens.push(world.particles.rho[i]);
        }
    }
    let err = mean_density_error(&world.particles);
    let pos_arr = PyArray1::from_vec(py, pos)
        .reshape([n, 3])
        .map_err(|e| PyValueError::new_err(format!("reshape: {e}")))?;
    let dens_arr = PyArray1::from_vec(py, dens);
    Ok((pos_arr, dens_arr, err))
}

/// Small PBF tank demo; returns `(positions [N,3], densities [N], mean_density_error)`.
#[pyfunction]
#[pyo3(signature = (n_side=4, steps=40, dt=1.0/120.0, dim="3"))]
#[allow(clippy::type_complexity)]
fn fluid_pbf_tank_demo<'py>(
    py: Python<'py>,
    n_side: usize,
    steps: usize,
    dt: f64,
    dim: &str,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    let dim = parse_dim(dim)?;
    let dx = 0.05;
    let h = 2.2 * dx;
    let rho0 = 1000.0;
    let mass = mass_from_spacing(rho0, dx, dim);
    let mut parts = Particles::new(dim, rho0, mass, h);
    let nz = match dim {
        Dim::D2 => 1,
        Dim::D3 => n_side,
    };
    fill_box(
        &mut parts,
        Vec3::new(0.1, 0.1, 0.1),
        n_side,
        n_side,
        nz,
        dx,
        Vec3::zero(),
    );
    let zmax = match dim {
        Dim::D2 => 0.0,
        Dim::D3 => 0.1 + n_side as f64 * dx + 0.05,
    };
    fill_box_boundary(
        &mut parts,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(
            0.1 + n_side as f64 * dx + 0.1,
            0.1 + n_side as f64 * dx + 0.1,
            zmax.max(0.45),
        ),
        dx,
        1,
    );
    let gravity = match dim {
        Dim::D2 => Vec3::new(0.0, -STANDARD_GRAVITY, 0.0),
        Dim::D3 => Vec3::new(0.0, 0.0, -STANDARD_GRAVITY),
    };
    let params = PbfParams {
        gravity,
        iterations: 2,
        art_pressure_k: 0.0,
        ..PbfParams::default()
    };
    let mut world = PbfWorld::new(parts, params);
    py.allow_threads(|| {
        for _ in 0..steps {
            world.step(dt);
        }
    });
    let n = world.particles.fluid_count();
    let mut pos = Vec::with_capacity(n * 3);
    let mut dens = Vec::with_capacity(n);
    for i in 0..world.particles.len() {
        if world.particles.kind[i] == pe_fluid::ParticleKind::Fluid {
            pos.extend_from_slice(&world.particles.x[i].to_array());
            dens.push(world.particles.rho[i]);
        }
    }
    let err = mean_density_error(&world.particles);
    let pos_arr = PyArray1::from_vec(py, pos)
        .reshape([n, 3])
        .map_err(|e| PyValueError::new_err(format!("reshape: {e}")))?;
    let dens_arr = PyArray1::from_vec(py, dens);
    Ok((pos_arr, dens_arr, err))
}

/// Kernel `W(0, h)` sanity for poly6 in 3-D (used by pytest).
#[pyfunction]
fn fluid_poly6_w0(h: f64) -> f64 {
    pe_fluid::w(KernelKind::Poly6, 0.0, h, Dim::D3)
}

/// Total fluid mass of the canonical hash world after 0 steps.
#[pyfunction]
fn fluid_canonical_mass() -> f64 {
    let w = pe_fluid::fluid_hash_world();
    total_mass(&w.particles)
}

/// Kinetic energy after canonical hash steps.
#[pyfunction]
fn fluid_canonical_ke_after_hash_steps() -> f64 {
    let mut w = pe_fluid::fluid_hash_world();
    for _ in 0..pe_fluid::FLUID_HASH_STEPS {
        w.step(pe_fluid::FLUID_HASH_DT);
    }
    kinetic_energy(&w.particles)
}

/// Register fluid bindings.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(canonical_fluid_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_dfsph_hash, m)?)?;
    m.add_function(wrap_pyfunction!(fluid_ritter_front, m)?)?;
    m.add_function(wrap_pyfunction!(fluid_ritter_height, m)?)?;
    m.add_function(wrap_pyfunction!(fluid_hydrostatic_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(fluid_pbf_tank_demo, m)?)?;
    m.add_function(wrap_pyfunction!(fluid_dfsph_tank_demo, m)?)?;
    m.add_function(wrap_pyfunction!(fluid_poly6_w0, m)?)?;
    m.add_function(wrap_pyfunction!(fluid_canonical_mass, m)?)?;
    m.add_function(wrap_pyfunction!(fluid_canonical_ke_after_hash_steps, m)?)?;
    Ok(())
}
