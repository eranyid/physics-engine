//! PyO3 bindings for `pe-soft` (XPBD soft bodies + materials).

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pe_math::Vec3;
use pe_soft::builders::{make_cloth, make_rope, make_soft_cube};
use pe_soft::collision::{StaticColliders, StaticPlane};
use pe_soft::material::{
    builtin_materials, lame_lambda, lame_mu, material_by_name, stretch_compliance,
};
use pe_soft::particle::Particle;
use pe_soft::plasticity::{return_map_uniaxial, J2Params, UniaxialPlasticState};
use pe_soft::scenarios::{materials_canonical_hash, soft_canonical_hash};
use pe_soft::solver::{SoftParams, SoftWorld};
use pe_soft::thermal::{free_expansion_delta_l, rod_fourier_mode1};
use pe_soft::{kinetic_energy, total_energy};
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

/// Canonical soft-body determinism hash.
#[pyfunction]
fn canonical_soft_hash() -> u64 {
    soft_canonical_hash()
}

/// Canonical materials-track determinism hash (plasticity + thermal + fracture).
#[pyfunction]
fn canonical_materials_hash() -> u64 {
    materials_canonical_hash()
}

/// List built-in material names (nominal + handbook; see `verified` flag).
#[pyfunction]
fn soft_material_names() -> Vec<String> {
    builtin_materials()
        .into_iter()
        .map(|m| m.name.to_string())
        .collect()
}

/// Lamé (λ, μ) from E, ν. Returns `None` fields as Python `None` via Option.
#[pyfunction]
fn soft_lame_parameters(e: f64, nu: f64) -> (Option<f64>, Option<f64>) {
    (lame_lambda(e, nu), lame_mu(e, nu))
}

/// Stretch compliance α = 1/(E A / L₀).
#[pyfunction]
fn soft_stretch_compliance(e: f64, rest_length: f64, area: f64) -> Option<f64> {
    stretch_compliance(e, rest_length, area)
}

/// Soft continuum material as a Python dict (nominal table lookup).
#[pyfunction]
fn soft_material_dict(name: &str) -> PyResult<std::collections::HashMap<String, String>> {
    // Return stringly-typed map for simplicity + verified flag.
    let m = material_by_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown material {name:?}")))?;
    let mut out = std::collections::HashMap::new();
    out.insert("name".into(), m.name.to_string());
    out.insert("density".into(), format!("{}", m.density));
    out.insert("youngs_modulus".into(), format!("{}", m.youngs_modulus));
    out.insert("poisson_ratio".into(), format!("{}", m.poisson_ratio));
    out.insert("friction".into(), format!("{}", m.friction));
    out.insert("restitution".into(), format!("{}", m.restitution));
    out.insert("yield_strength".into(), format!("{}", m.yield_strength));
    out.insert(
        "hardening_modulus".into(),
        format!("{}", m.hardening_modulus),
    );
    out.insert(
        "thermal_conductivity".into(),
        format!("{}", m.thermal_conductivity),
    );
    out.insert("specific_heat".into(), format!("{}", m.specific_heat));
    out.insert(
        "thermal_expansion".into(),
        format!("{}", m.thermal_expansion),
    );
    out.insert("fracture_stress".into(), format!("{}", m.fracture_stress));
    out.insert("fracture_strain".into(), format!("{}", m.fracture_strain));
    out.insert("verified".into(), format!("{}", m.verified));
    out.insert("source_note".into(), m.source_note.to_string());
    Ok(out)
}

/// 1-D J2 return map: returns `(stress, plastic_strain, eq_plastic_strain, plastic_flag)`.
#[pyfunction]
fn soft_j2_uniaxial_step(
    e: f64,
    nu: f64,
    yield_strength: f64,
    hardening: f64,
    strain: f64,
    plastic_strain: f64,
    eq_plastic_strain: f64,
) -> (f64, f64, f64, bool) {
    let p = J2Params {
        youngs_modulus: e,
        poisson_ratio: nu,
        yield_strength,
        hardening_modulus: hardening,
    };
    let state = UniaxialPlasticState {
        plastic_strain,
        eq_plastic_strain,
        stress: 0.0,
    };
    let r = return_map_uniaxial(&p, state, strain);
    (
        r.state.stress,
        r.state.plastic_strain,
        r.state.eq_plastic_strain,
        r.plastic,
    )
}

/// Free thermal expansion `ΔL = α L₀ ΔT`.
#[pyfunction]
fn soft_free_expansion_delta_l(l0: f64, alpha: f64, d_t: f64) -> f64 {
    free_expansion_delta_l(l0, alpha, d_t)
}

/// Analytic 1-D rod Fourier mode-1 temperature.
#[pyfunction]
fn soft_rod_fourier_mode1(x: f64, t: f64, length: f64, t0: f64, diffusivity: f64) -> f64 {
    rod_fourier_mode1(x, t, length, t0, diffusivity)
}

/// XPBD soft world exposed to Python.
#[pyclass(name = "SoftWorld")]
struct PySoftWorld {
    inner: SoftWorld,
}

#[pymethods]
impl PySoftWorld {
    /// Empty soft world.
    #[new]
    fn new() -> Self {
        Self {
            inner: SoftWorld::new(),
        }
    }

    /// Build a pinned-end rope from `a` to `b`.
    #[staticmethod]
    #[pyo3(signature = (a, b, segments=10, mass=1.0, compliance=1e-6, pin_ends=true))]
    fn rope(
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        segments: usize,
        mass: f64,
        compliance: f64,
        pin_ends: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: make_rope(
                vec3_from_seq(a)?,
                vec3_from_seq(b)?,
                segments,
                mass,
                compliance,
                pin_ends,
            ),
        })
    }

    /// Build a cloth grid (optionally pin four corners).
    #[staticmethod]
    #[pyo3(signature = (origin, nx=8, ny=8, dx=0.1, dy=0.1, mass=0.5, stretch_compliance=1e-6, bend_compliance=1e-4, pin_corners=true))]
    #[allow(clippy::too_many_arguments)]
    fn cloth(
        origin: &Bound<'_, PyAny>,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        mass: f64,
        stretch_compliance: f64,
        bend_compliance: f64,
        pin_corners: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: make_cloth(
                vec3_from_seq(origin)?,
                nx,
                ny,
                dx,
                dy,
                mass,
                stretch_compliance,
                bend_compliance,
                pin_corners,
            ),
        })
    }

    /// Soft cube / jelly from a built-in material name.
    #[staticmethod]
    #[pyo3(signature = (origin, size=0.3, divisions=2, material="jelly", neo_hookean=true))]
    fn soft_cube(
        origin: &Bound<'_, PyAny>,
        size: f64,
        divisions: usize,
        material: &str,
        neo_hookean: bool,
    ) -> PyResult<Self> {
        let mat = material_by_name(material)
            .ok_or_else(|| PyValueError::new_err(format!("unknown material {material:?}")))?;
        Ok(Self {
            inner: make_soft_cube(vec3_from_seq(origin)?, size, divisions, &mat, neo_hookean),
        })
    }

    /// Particle count.
    #[getter]
    fn n(&self) -> usize {
        self.inner.len()
    }

    /// Simulation time [s].
    #[getter]
    fn t(&self) -> f64 {
        self.inner.t
    }

    /// Positions as `(n, 3)` NumPy array.
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.len();
        let flat = self.inner.positions_flat();
        let arr = PyArray1::from_vec(py, flat);
        arr.reshape([n, 3]).expect("reshape x")
    }

    /// Velocities as `(n, 3)` NumPy array.
    fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.len();
        let flat = self.inner.velocities_flat();
        let arr = PyArray1::from_vec(py, flat);
        arr.reshape([n, 3]).expect("reshape v")
    }

    /// Packed state `(x‖v)` length `6n`.
    fn pack<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.pack())
    }

    /// Kinetic energy [J].
    fn kinetic_energy(&self) -> f64 {
        kinetic_energy(&self.inner)
    }

    /// Total energy estimate [J].
    fn total_energy(&self) -> f64 {
        total_energy(&self.inner)
    }

    /// Configure substeps / damping / gravity.
    #[pyo3(signature = (substeps=None, iterations=None, damping=None, gravity=None))]
    fn set_params(
        &mut self,
        substeps: Option<usize>,
        iterations: Option<usize>,
        damping: Option<f64>,
        gravity: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if let Some(s) = substeps {
            self.inner.params.substeps = s;
        }
        if let Some(i) = iterations {
            self.inner.params.iterations = i;
        }
        if let Some(d) = damping {
            self.inner.params.damping = d;
        }
        if let Some(g) = gravity {
            self.inner.params.gravity = vec3_from_seq(g)?;
        }
        let _ = SoftParams::default();
        Ok(())
    }

    /// Add a static ground plane `n·x = offset`.
    #[pyo3(signature = (normal, offset=0.0, friction=0.4, restitution=0.0))]
    fn add_ground_plane(
        &mut self,
        normal: &Bound<'_, PyAny>,
        offset: f64,
        friction: f64,
        restitution: f64,
    ) -> PyResult<()> {
        self.inner.colliders.planes.push(StaticPlane::new(
            vec3_from_seq(normal)?,
            offset,
            friction,
            restitution,
        ));
        Ok(())
    }

    /// Clear static colliders.
    fn clear_colliders(&mut self) {
        self.inner.colliders = StaticColliders::default();
    }

    /// Add a dynamic particle; returns index.
    fn add_particle(&mut self, position: &Bound<'_, PyAny>, mass: f64) -> PyResult<usize> {
        Ok(self
            .inner
            .add_particle(Particle::dynamic(vec3_from_seq(position)?, mass)))
    }

    /// Pin particle `i`.
    fn pin(&mut self, i: usize) -> PyResult<()> {
        if i >= self.inner.len() {
            return Err(PyValueError::new_err("particle index out of range"));
        }
        self.inner.pin(i);
        Ok(())
    }

    /// One outer step; **releases the GIL**.
    fn step(&mut self, py: Python<'_>, dt: f64) {
        py.allow_threads(|| {
            self.inner.step(dt);
        });
    }

    /// Run to `t_end` with fixed `dt`; **releases the GIL**.
    fn run(&mut self, py: Python<'_>, t_end: f64, dt: f64) {
        py.allow_threads(|| {
            self.inner.run_for(t_end, dt);
        });
    }
}

/// Register soft-body bindings.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySoftWorld>()?;
    m.add_function(wrap_pyfunction!(canonical_soft_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_materials_hash, m)?)?;
    m.add_function(wrap_pyfunction!(soft_material_names, m)?)?;
    m.add_function(wrap_pyfunction!(soft_material_dict, m)?)?;
    m.add_function(wrap_pyfunction!(soft_lame_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(soft_stretch_compliance, m)?)?;
    m.add_function(wrap_pyfunction!(soft_j2_uniaxial_step, m)?)?;
    m.add_function(wrap_pyfunction!(soft_free_expansion_delta_l, m)?)?;
    m.add_function(wrap_pyfunction!(soft_rod_fourier_mode1, m)?)?;
    Ok(())
}
