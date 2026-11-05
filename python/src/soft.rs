//! PyO3 bindings for `pe-soft` (XPBD soft bodies + materials).

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pe_math::Vec3;
use pe_soft::builders::{make_cloth, make_rope, make_soft_cube};
use pe_soft::cohesive_fracture::{
    analytic_gc, cohesive_canonical_hash, damage_from_delta_max, make_cohesive_bar,
    quasistatic_cohesive_open, traction_virgin_envelope, BilinearCzmParams, CohesiveSoftWorld,
};
use pe_soft::collision::{StaticColliders, StaticPlane};
use pe_soft::finite_strain_plastic::{
    analytic_log_permanent_strain, fs_plastic_canonical_hash, make_fs_plastic_bar,
    quasistatic_log_stretch_unload, FsPlasticSoftWorld,
};
use pe_soft::material::{
    builtin_materials, lame_lambda, lame_mu, material_by_name, stretch_compliance,
};
use pe_soft::particle::Particle;
use pe_soft::plastic_coupling::{
    analytic_permanent_strain, make_plastic_bar, plastic_canonical_hash,
    quasistatic_stretch_unload, PlasticSoftWorld,
};
use pe_soft::plasticity::{return_map_uniaxial, J2Params, UniaxialPlasticState};
use pe_soft::scenarios::{materials_canonical_hash, soft_canonical_hash, thermal_canonical_hash};
use pe_soft::solver::{SoftParams, SoftWorld};
use pe_soft::spectral_hencky_plastic::{
    analytic_isochoric_uniaxial_seq, be_plastic_canonical_hash, make_be_plastic_bar,
    BePlasticSoftWorld,
};
use pe_soft::thermal::{
    free_expansion_delta_l, init_rod_sine_mode, robin_lumped_exp, rod_fourier_mode1,
    semi_infinite_erfc, ThermalMesh, ThermalProps, ThermalScheme,
};
use pe_soft::{
    kinetic_energy, make_momentum_exchange_scene, make_pendulum_bob, make_rope_on_free_body,
    make_soft_on_box_on_ground, soft_coupling_canonical_hash, total_energy, SoftRigidWorld,
};
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

/// Semi-infinite solid erfc solution `T_i + (T_s−T_i) erfc(x/(2√(κt)))`.
#[pyfunction]
fn soft_semi_infinite_erfc(x: f64, t: f64, t_i: f64, t_s: f64, diffusivity: f64) -> f64 {
    semi_infinite_erfc(x, t, t_i, t_s, diffusivity)
}

/// Lumped Robin cooling `T_∞ + (T₀−T_∞) exp(−t/τ)`.
#[pyfunction]
fn soft_robin_lumped_exp(t: f64, t0: f64, t_inf: f64, tau: f64) -> f64 {
    robin_lumped_exp(t, t0, t_inf, tau)
}

/// Canonical M29 implicit-thermal determinism hash.
#[pyfunction]
fn canonical_thermal_hash() -> u64 {
    thermal_canonical_hash()
}

/// Canonical M30 plastic XPBD coupling determinism hash.
#[pyfunction]
fn canonical_plastic_hash() -> u64 {
    plastic_canonical_hash()
}

/// Canonical M31 finite-strain plastic determinism hash.
#[pyfunction]
fn canonical_fs_plastic_hash() -> u64 {
    fs_plastic_canonical_hash()
}

/// Canonical M32 spectral Hencky / Kirchhoff (`b_e`) plastic determinism hash.
#[pyfunction]
fn canonical_be_plastic_hash() -> u64 {
    be_plastic_canonical_hash()
}

/// Analytic elastic-trial von Mises for isochoric uniaxial stretch: `3 μ |ln λ|`.
#[pyfunction]
fn soft_analytic_isochoric_uniaxial_seq(e: f64, nu: f64, lambda_stretch: f64) -> f64 {
    let mu = e / (2.0 * (1.0 + nu));
    analytic_isochoric_uniaxial_seq(mu, lambda_stretch)
}

/// Analytic permanent strain `ε_p` after virgin stretch to `eps_peak`.
#[pyfunction]
fn soft_analytic_permanent_strain(
    e: f64,
    nu: f64,
    yield_strength: f64,
    hardening: f64,
    eps_peak: f64,
) -> f64 {
    let p = J2Params {
        youngs_modulus: e,
        poisson_ratio: nu,
        yield_strength,
        hardening_modulus: hardening,
    };
    analytic_permanent_strain(&p, eps_peak)
}

/// Analytic Hencky permanent strain after virgin log stretch to `eps_log`.
#[pyfunction]
fn soft_analytic_log_permanent_strain(
    e: f64,
    nu: f64,
    yield_strength: f64,
    hardening: f64,
    eps_log: f64,
) -> f64 {
    let p = J2Params {
        youngs_modulus: e,
        poisson_ratio: nu,
        yield_strength,
        hardening_modulus: hardening,
    };
    analytic_log_permanent_strain(&p, eps_log)
}

/// Quasistatic stretch-hold-unload; returns `(final_length, eps_p, analytic_eps_p)`.
#[pyfunction]
#[pyo3(signature = (e, nu, yield_strength, hardening, l0, area, eps_target, hold_steps=80, settle_steps=800, dt=1.0/240.0))]
#[allow(clippy::too_many_arguments)]
fn soft_quasistatic_stretch_unload(
    e: f64,
    nu: f64,
    yield_strength: f64,
    hardening: f64,
    l0: f64,
    area: f64,
    eps_target: f64,
    hold_steps: usize,
    settle_steps: usize,
    dt: f64,
) -> (f64, f64, f64) {
    let p = J2Params {
        youngs_modulus: e,
        poisson_ratio: nu,
        yield_strength,
        hardening_modulus: hardening,
    };
    quasistatic_stretch_unload(p, l0, area, eps_target, hold_steps, settle_steps, dt)
}

/// Quasistatic log-strain stretch-hold-unload by stretch ratio `lambda`.
///
/// Returns `(final_length, eps_p, analytic_eps_p)`.
#[pyfunction]
#[pyo3(signature = (e, nu, yield_strength, hardening, l0, area, lambda, hold_steps=80, settle_steps=800, dt=1.0/240.0))]
#[allow(clippy::too_many_arguments)]
fn soft_quasistatic_log_stretch_unload(
    e: f64,
    nu: f64,
    yield_strength: f64,
    hardening: f64,
    l0: f64,
    area: f64,
    lambda: f64,
    hold_steps: usize,
    settle_steps: usize,
    dt: f64,
) -> (f64, f64, f64) {
    let p = J2Params {
        youngs_modulus: e,
        poisson_ratio: nu,
        yield_strength,
        hardening_modulus: hardening,
    };
    quasistatic_log_stretch_unload(p, l0, area, lambda, hold_steps, settle_steps, dt)
}

/// Implicit / explicit 1-D thermal rod exposed to Python (NumPy temperatures).
#[pyclass(name = "ThermalRod")]
struct PyThermalRod {
    mesh: ThermalMesh,
    fixed: Vec<bool>,
    length: f64,
}

#[pymethods]
impl PyThermalRod {
    /// Uniform rod with `n` nodes on `[0, length]`, cross-section `area`.
    #[new]
    #[pyo3(signature = (n, length, area, density, specific_heat, conductivity, expansion, t_ref=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: usize,
        length: f64,
        area: f64,
        density: f64,
        specific_heat: f64,
        conductivity: f64,
        expansion: f64,
        t_ref: f64,
    ) -> PyResult<Self> {
        if n < 2 || length <= 0.0 || area <= 0.0 {
            return Err(PyValueError::new_err(
                "need n>=2, length>0, area>0 for ThermalRod",
            ));
        }
        let props = ThermalProps {
            density,
            specific_heat,
            conductivity,
            expansion,
        };
        let mesh = ThermalMesh::rod_1d(n, length, area, props, t_ref);
        let fixed = vec![false; n];
        Ok(Self {
            mesh,
            fixed,
            length,
        })
    }

    /// Number of nodes.
    #[getter]
    fn n(&self) -> usize {
        self.mesh.len()
    }

    /// Explicit CFL-safe dt suggestion.
    fn max_stable_dt(&self) -> f64 {
        self.mesh.max_stable_dt()
    }

    /// Diffusivity κ = k/(ρ c).
    fn diffusivity(&self) -> f64 {
        self.mesh.props.diffusivity()
    }

    /// Total thermal energy Σ C_i T_i.
    fn total_energy(&self) -> f64 {
        self.mesh.total_energy()
    }

    /// Copy temperatures into a NumPy length-`n` array.
    fn temperatures<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.mesh.temperature)
    }

    /// Set Dirichlet flags (length `n` bool array / sequence).
    fn set_fixed(&mut self, fixed: Vec<bool>) -> PyResult<()> {
        if fixed.len() != self.mesh.len() {
            return Err(PyValueError::new_err("fixed length must equal n"));
        }
        self.fixed = fixed;
        Ok(())
    }

    /// Set temperatures from a length-`n` sequence.
    fn set_temperatures(&mut self, t: Vec<f64>) -> PyResult<()> {
        if t.len() != self.mesh.len() {
            return Err(PyValueError::new_err("temperature length must equal n"));
        }
        self.mesh.temperature = t;
        Ok(())
    }

    /// Set Robin `h*A` and `T_inf` at node `i`.
    fn set_robin(&mut self, i: usize, h_a: f64, t_inf: f64) -> PyResult<()> {
        if i >= self.mesh.len() {
            return Err(PyValueError::new_err("node index out of range"));
        }
        self.mesh.robin_ha[i] = h_a;
        self.mesh.robin_tinf[i] = t_inf;
        Ok(())
    }

    /// Set Neumann nodal power [W] into node `i`.
    fn set_neumann_power(&mut self, i: usize, power: f64) -> PyResult<()> {
        if i >= self.mesh.len() {
            return Err(PyValueError::new_err("node index out of range"));
        }
        self.mesh.neumann_power[i] = power;
        Ok(())
    }

    /// Initialize first Fourier sine mode (Dirichlet ends ≈ 0).
    fn init_sine_mode(&mut self, t0: f64) {
        init_rod_sine_mode(&mut self.mesh, self.length, t0);
    }

    /// Advance one step. `scheme`: `"explicit"`, `"be"` / `"backward_euler"`, or
    /// `"cn"` / `"crank_nicolson"`.
    fn step(&mut self, py: Python<'_>, dt: f64, scheme: &str) -> PyResult<bool> {
        let sch = match scheme {
            "explicit" => ThermalScheme::Explicit,
            "be" | "backward_euler" => ThermalScheme::BackwardEuler,
            "cn" | "crank_nicolson" => ThermalScheme::CrankNicolson,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown thermal scheme {other:?}; use explicit|be|cn"
                )));
            }
        };
        let fixed = self.fixed.clone();
        Ok(py.allow_threads(|| self.mesh.step_scheme(dt, sch, &fixed)))
    }
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

/// Canonical soft↔rigid coupling determinism hash (M28).
#[pyfunction]
fn canonical_soft_coupling_hash() -> u64 {
    soft_coupling_canonical_hash()
}

/// Two-way soft XPBD ↔ dynamic rigid coupling world.
#[pyclass(name = "SoftRigidWorld")]
struct PySoftRigidWorld {
    inner: SoftRigidWorld,
}

#[pymethods]
impl PySoftRigidWorld {
    /// Rope of mass `m_rope` attached to a free rigid sphere of mass `m_body`.
    #[staticmethod]
    #[pyo3(signature = (m_rope=1.0, m_body=3.0, segments=6, compliance=1e-6))]
    fn rope_on_free_body(m_rope: f64, m_body: f64, segments: usize, compliance: f64) -> Self {
        Self {
            inner: make_rope_on_free_body(m_rope, m_body, segments, compliance),
        }
    }

    /// Soft jelly cube resting on a dynamic box on a ground plane.
    #[staticmethod]
    fn soft_on_box_on_ground() -> Self {
        Self {
            inner: make_soft_on_box_on_ground(),
        }
    }

    /// Heavy rigid bob on a stiff rope (top pinned). Returns world; length is 1 m.
    #[staticmethod]
    #[pyo3(signature = (m_bob=5.0, m_rope=0.05, length=1.0, segments=8, compliance=1e-8))]
    fn pendulum_bob(
        m_bob: f64,
        m_rope: f64,
        length: f64,
        segments: usize,
        compliance: f64,
    ) -> Self {
        let (world, _l) = make_pendulum_bob(m_bob, m_rope, length, segments, compliance);
        Self { inner: world }
    }

    /// Isolated soft-particle cluster vs free box (no gravity) for momentum checks.
    #[staticmethod]
    fn momentum_exchange() -> Self {
        Self {
            inner: make_momentum_exchange_scene(),
        }
    }

    /// Soft particle count.
    #[getter]
    fn n_soft(&self) -> usize {
        self.inner.soft.len()
    }

    /// Rigid body count.
    #[getter]
    fn n_rigid(&self) -> usize {
        self.inner.rigid.n
    }

    /// Simulation time [s] (soft clock).
    #[getter]
    fn t(&self) -> f64 {
        self.inner.soft.t
    }

    /// Soft positions `(n, 3)`.
    fn soft_x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.soft.len();
        let flat = self.inner.soft.positions_flat();
        let arr = PyArray1::from_vec(py, flat);
        arr.reshape([n, 3]).expect("reshape soft_x")
    }

    /// Rigid COM positions `(n_rigid, 3)`.
    fn rigid_x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.rigid.n;
        let mut flat = Vec::with_capacity(n * 3);
        for i in 0..n {
            flat.extend_from_slice(&self.inner.rigid.state.x[i].to_array());
        }
        let arr = PyArray1::from_vec(py, flat);
        arr.reshape([n, 3]).expect("reshape rigid_x")
    }

    /// Combined centre of mass `[x,y,z]`.
    fn combined_com(&self) -> [f64; 3] {
        self.inner.combined_com().to_array()
    }

    /// Total linear momentum `[px,py,pz]` [kg m/s].
    fn total_linear_momentum(&self) -> [f64; 3] {
        self.inner.total_linear_momentum().to_array()
    }

    /// Total mechanical energy estimate [J].
    fn total_energy(&self) -> f64 {
        self.inner.total_energy()
    }

    /// Last outer-step ground support force `[Fx,Fy,Fz]` [N].
    fn ground_force(&self) -> [f64; 3] {
        self.inner.diagnostics.ground_force.to_array()
    }

    /// Configure soft substeps / damping / gravity (also drives rigid gravity).
    #[pyo3(signature = (substeps=None, iterations=None, damping=None, gravity=None))]
    fn set_params(
        &mut self,
        substeps: Option<usize>,
        iterations: Option<usize>,
        damping: Option<f64>,
        gravity: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if let Some(s) = substeps {
            self.inner.soft.params.substeps = s;
        }
        if let Some(i) = iterations {
            self.inner.soft.params.iterations = i;
        }
        if let Some(d) = damping {
            self.inner.soft.params.damping = d;
        }
        if let Some(g) = gravity {
            self.inner.soft.params.gravity = vec3_from_seq(g)?;
        }
        Ok(())
    }

    /// One outer coupled step; **releases the GIL**.
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

/// Soft XPBD bar with J2 plastic rest coupling (M30).
#[pyclass(name = "PlasticBar")]
struct PyPlasticBar {
    inner: PlasticSoftWorld,
}

#[pymethods]
impl PyPlasticBar {
    /// Single pinned–free plastic distance bar.
    #[new]
    #[pyo3(signature = (e, nu, yield_strength, hardening, l0=1.0, area=0.01, mass=1.0, substeps=32))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        e: f64,
        nu: f64,
        yield_strength: f64,
        hardening: f64,
        l0: f64,
        area: f64,
        mass: f64,
        substeps: usize,
    ) -> PyResult<Self> {
        if l0 <= 0.0 || area <= 0.0 || mass <= 0.0 || e <= 0.0 {
            return Err(PyValueError::new_err(
                "need l0>0, area>0, mass>0, E>0 for PlasticBar",
            ));
        }
        let p = J2Params {
            youngs_modulus: e,
            poisson_ratio: nu,
            yield_strength,
            hardening_modulus: hardening,
        };
        Ok(Self {
            inner: make_plastic_bar(p, l0, area, mass, substeps),
        })
    }

    /// Simulation time [s].
    #[getter]
    fn t(&self) -> f64 {
        self.inner.t()
    }

    /// Number of particles (2 for a single bar).
    #[getter]
    fn n(&self) -> usize {
        self.inner.soft.len()
    }

    /// Current free-end length [m].
    fn length(&self) -> f64 {
        let p = &self.inner.soft.particles;
        (p[1].x - p[0].x).norm()
    }

    /// Current XPBD rest length [m].
    fn rest_length(&self) -> f64 {
        self.inner.soft.constraints.distances[0].rest_length
    }

    /// Plastic strain / eq plastic / stress `(ε_p, α, σ)`.
    fn plastic_state(&self) -> (f64, f64, f64) {
        let s = &self.inner.bars[0].state;
        (s.plastic_strain, s.eq_plastic_strain, s.stress)
    }

    /// Set free-end position along x (pinned end stays at origin).
    fn set_free_end_x(&mut self, x: f64) {
        self.inner.soft.particles[1].x.x = x;
        self.inner.soft.particles[1].v = pe_math::Vec3::zero();
    }

    /// Temporarily pin (`inv_mass=0`) or unpin the free end.
    fn set_free_end_pinned(&mut self, pinned: bool) {
        if pinned {
            self.inner.soft.particles[1].inv_mass = 0.0;
            self.inner.soft.particles[1].v = pe_math::Vec3::zero();
        } else {
            self.inner.soft.particles[1].inv_mass = 1.0;
        }
    }

    /// Set damping [1/s].
    fn set_damping(&mut self, damping: f64) {
        self.inner.soft.params.damping = damping;
    }

    /// One XPBD + plastic rest-update step; **releases the GIL**.
    fn step(&mut self, py: Python<'_>, dt: f64) {
        py.allow_threads(|| {
            self.inner.step(dt);
        });
    }

    /// Particle positions as NumPy `(n, 3)`.
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.soft.len();
        let flat = self.inner.soft.positions_flat();
        let arr = PyArray1::from_vec(py, flat);
        arr.reshape([n, 3]).expect("reshape plastic_bar_x")
    }
}

/// Soft XPBD bar with finite-strain (Hencky) J2 plastic rest coupling (M31).
#[pyclass(name = "FsPlasticBar")]
struct PyFsPlasticBar {
    inner: FsPlasticSoftWorld,
}

#[pymethods]
impl PyFsPlasticBar {
    /// Single pinned–free finite-strain plastic distance bar.
    #[new]
    #[pyo3(signature = (e, nu, yield_strength, hardening, l0=1.0, area=0.01, mass=1.0, substeps=32))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        e: f64,
        nu: f64,
        yield_strength: f64,
        hardening: f64,
        l0: f64,
        area: f64,
        mass: f64,
        substeps: usize,
    ) -> PyResult<Self> {
        if l0 <= 0.0 || area <= 0.0 || mass <= 0.0 || e <= 0.0 {
            return Err(PyValueError::new_err(
                "need l0>0, area>0, mass>0, E>0 for FsPlasticBar",
            ));
        }
        let p = J2Params {
            youngs_modulus: e,
            poisson_ratio: nu,
            yield_strength,
            hardening_modulus: hardening,
        };
        Ok(Self {
            inner: make_fs_plastic_bar(p, l0, area, mass, substeps),
        })
    }

    /// Simulation time [s].
    #[getter]
    fn t(&self) -> f64 {
        self.inner.t()
    }

    /// Number of particles (2 for a single bar).
    #[getter]
    fn n(&self) -> usize {
        self.inner.soft.len()
    }

    /// Current free-end length [m].
    fn length(&self) -> f64 {
        let p = &self.inner.soft.particles;
        (p[1].x - p[0].x).norm()
    }

    /// Current XPBD rest length [m] (`L₀ exp(ε_p)`).
    fn rest_length(&self) -> f64 {
        self.inner.soft.constraints.distances[0].rest_length
    }

    /// Plastic Hencky strain / eq plastic / stress `(ε_p, α, σ)`.
    fn plastic_state(&self) -> (f64, f64, f64) {
        let s = &self.inner.bars[0].state;
        (s.plastic_strain, s.eq_plastic_strain, s.stress)
    }

    /// Set free-end position along x (pinned end stays at origin).
    fn set_free_end_x(&mut self, x: f64) {
        self.inner.soft.particles[1].x.x = x;
        self.inner.soft.particles[1].v = pe_math::Vec3::zero();
    }

    /// Temporarily pin (`inv_mass=0`) or unpin the free end.
    fn set_free_end_pinned(&mut self, pinned: bool) {
        if pinned {
            self.inner.soft.particles[1].inv_mass = 0.0;
            self.inner.soft.particles[1].v = pe_math::Vec3::zero();
        } else {
            self.inner.soft.particles[1].inv_mass = 1.0;
        }
    }

    /// Set damping [1/s].
    fn set_damping(&mut self, damping: f64) {
        self.inner.soft.params.damping = damping;
    }

    /// One XPBD + finite-strain plastic rest-update step; **releases the GIL**.
    fn step(&mut self, py: Python<'_>, dt: f64) {
        py.allow_threads(|| {
            self.inner.step(dt);
        });
    }

    /// Particle positions as NumPy `(n, 3)`.
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.soft.len();
        let flat = self.inner.soft.positions_flat();
        let arr = PyArray1::from_vec(py, flat);
        arr.reshape([n, 3]).expect("reshape fs_plastic_bar_x")
    }
}

/// Canonical M33 cohesive-zone fracture determinism hash.
#[pyfunction]
fn canonical_cohesive_hash() -> u64 {
    cohesive_canonical_hash()
}

/// Analytic bilinear CZM fracture energy `G_c = ½ σ_c δ_f` [J/m²].
#[pyfunction]
fn soft_analytic_gc(sigma_c: f64, delta_c: f64, delta_f: f64) -> PyResult<f64> {
    let p = BilinearCzmParams {
        sigma_c,
        delta_c,
        delta_f,
    };
    if !p.is_valid() {
        return Err(PyValueError::new_err(
            "need sigma_c>0 and 0<delta_c<delta_f for soft_analytic_gc",
        ));
    }
    Ok(analytic_gc(&p))
}

/// Virgin-envelope traction at opening `delta` [Pa].
#[pyfunction]
fn soft_czm_virgin_traction(sigma_c: f64, delta_c: f64, delta_f: f64, delta: f64) -> PyResult<f64> {
    let p = BilinearCzmParams {
        sigma_c,
        delta_c,
        delta_f,
    };
    if !p.is_valid() {
        return Err(PyValueError::new_err(
            "need sigma_c>0 and 0<delta_c<delta_f for soft_czm_virgin_traction",
        ));
    }
    Ok(traction_virgin_envelope(&p, delta))
}

/// Irreversible bilinear CZM damage from historical max opening.
#[pyfunction]
fn soft_czm_damage(sigma_c: f64, delta_c: f64, delta_f: f64, delta_max: f64) -> PyResult<f64> {
    let p = BilinearCzmParams {
        sigma_c,
        delta_c,
        delta_f,
    };
    if !p.is_valid() {
        return Err(PyValueError::new_err(
            "need sigma_c>0 and 0<delta_c<delta_f for soft_czm_damage",
        ));
    }
    Ok(damage_from_delta_max(&p, delta_max))
}

/// Quasistatic cohesive open: `(opening, damage, traction, failed)`.
#[pyfunction]
#[pyo3(signature = (sigma_c, delta_c, delta_f, l0, area, target_opening, settle_steps=80, dt=1.0/240.0))]
#[allow(clippy::too_many_arguments)]
fn soft_quasistatic_cohesive_open(
    sigma_c: f64,
    delta_c: f64,
    delta_f: f64,
    l0: f64,
    area: f64,
    target_opening: f64,
    settle_steps: usize,
    dt: f64,
) -> PyResult<(f64, f64, f64, bool)> {
    let p = BilinearCzmParams {
        sigma_c,
        delta_c,
        delta_f,
    };
    if !p.is_valid() || l0 <= 0.0 || area <= 0.0 {
        return Err(PyValueError::new_err(
            "need valid CZM params, l0>0, area>0 for soft_quasistatic_cohesive_open",
        ));
    }
    Ok(quasistatic_cohesive_open(
        p,
        l0,
        area,
        target_opening,
        settle_steps,
        dt,
    ))
}

/// Soft XPBD bar with spectral Hencky world (M32; bars share M31 Hencky rest).
#[pyclass(name = "BePlasticBar")]
struct PyBePlasticBar {
    inner: BePlasticSoftWorld,
}

#[pymethods]
impl PyBePlasticBar {
    /// Single pinned–free Hencky plastic distance bar in a spectral-`b_e` world.
    #[new]
    #[pyo3(signature = (e, nu, yield_strength, hardening, l0=1.0, area=0.01, mass=1.0, substeps=32))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        e: f64,
        nu: f64,
        yield_strength: f64,
        hardening: f64,
        l0: f64,
        area: f64,
        mass: f64,
        substeps: usize,
    ) -> PyResult<Self> {
        if l0 <= 0.0 || area <= 0.0 || mass <= 0.0 || e <= 0.0 {
            return Err(PyValueError::new_err(
                "need l0>0, area>0, mass>0, E>0 for BePlasticBar",
            ));
        }
        let p = J2Params {
            youngs_modulus: e,
            poisson_ratio: nu,
            yield_strength,
            hardening_modulus: hardening,
        };
        Ok(Self {
            inner: make_be_plastic_bar(p, l0, area, mass, substeps),
        })
    }

    /// Simulation time [s].
    #[getter]
    fn t(&self) -> f64 {
        self.inner.t()
    }

    /// Number of particles (2 for a single bar).
    #[getter]
    fn n(&self) -> usize {
        self.inner.soft.len()
    }

    /// Current free-end length [m].
    fn length(&self) -> f64 {
        let p = &self.inner.soft.particles;
        (p[1].x - p[0].x).norm()
    }

    /// Current XPBD rest length [m] (`L₀ exp(ε_p)`).
    fn rest_length(&self) -> f64 {
        self.inner.soft.constraints.distances[0].rest_length
    }

    /// Plastic Hencky strain / eq plastic / stress `(ε_p, α, σ)`.
    fn plastic_state(&self) -> (f64, f64, f64) {
        let s = &self.inner.bars[0].state;
        (s.plastic_strain, s.eq_plastic_strain, s.stress)
    }

    /// Set free-end position along x (pinned end stays at origin).
    fn set_free_end_x(&mut self, x: f64) {
        self.inner.soft.particles[1].x.x = x;
        self.inner.soft.particles[1].v = pe_math::Vec3::zero();
    }

    /// Temporarily pin (`inv_mass=0`) or unpin the free end.
    fn set_free_end_pinned(&mut self, pinned: bool) {
        if pinned {
            self.inner.soft.particles[1].inv_mass = 0.0;
            self.inner.soft.particles[1].v = pe_math::Vec3::zero();
        } else {
            self.inner.soft.particles[1].inv_mass = 1.0;
        }
    }

    /// Set damping [1/s].
    fn set_damping(&mut self, damping: f64) {
        self.inner.soft.params.damping = damping;
    }

    /// One XPBD + spectral plastic rest-update step; **releases the GIL**.
    fn step(&mut self, py: Python<'_>, dt: f64) {
        py.allow_threads(|| {
            self.inner.step(dt);
        });
    }

    /// Particle positions as NumPy `(n, 3)`.
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.soft.len();
        let flat = self.inner.soft.positions_flat();
        let arr = PyArray1::from_vec(py, flat);
        arr.reshape([n, 3]).expect("reshape be_plastic_bar_x")
    }
}

/// Soft XPBD bar with bilinear cohesive-zone bond (M33).
#[pyclass(name = "CohesiveBar")]
struct PyCohesiveBar {
    inner: CohesiveSoftWorld,
}

#[pymethods]
impl PyCohesiveBar {
    /// Single pinned–free cohesive distance bar.
    #[new]
    #[pyo3(signature = (sigma_c, delta_c, delta_f, l0=1.0, area=0.01, mass=1.0, substeps=32))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        sigma_c: f64,
        delta_c: f64,
        delta_f: f64,
        l0: f64,
        area: f64,
        mass: f64,
        substeps: usize,
    ) -> PyResult<Self> {
        let p = BilinearCzmParams {
            sigma_c,
            delta_c,
            delta_f,
        };
        if !p.is_valid() || l0 <= 0.0 || area <= 0.0 || mass <= 0.0 {
            return Err(PyValueError::new_err(
                "need valid CZM (sigma_c>0, 0<delta_c<delta_f), l0>0, area>0, mass>0",
            ));
        }
        Ok(Self {
            inner: make_cohesive_bar(p, l0, area, mass, substeps),
        })
    }

    /// Simulation time [s].
    #[getter]
    fn t(&self) -> f64 {
        self.inner.time()
    }

    /// Number of particles (2 for a single bar).
    #[getter]
    fn n(&self) -> usize {
        self.inner.soft.len()
    }

    /// Current free-end length [m].
    fn length(&self) -> f64 {
        let p = &self.inner.soft.particles;
        (p[1].x - p[0].x).norm()
    }

    /// Cold rest length [m].
    fn rest_length(&self) -> f64 {
        self.inner.bonds[0].cold_rest
    }

    /// Cohesive state `(opening, damage, traction, failed)`.
    fn cohesive_state(&self) -> (f64, f64, f64, bool) {
        let b = &self.inner.bonds[0];
        (b.opening, b.damage, b.traction, b.failed)
    }

    /// Crack-path bond indices in failure order.
    fn crack_path(&self) -> Vec<usize> {
        self.inner.crack_path.iter().map(|e| e.bond_index).collect()
    }

    /// Set free-end position along x (pinned end stays at origin).
    fn set_free_end_x(&mut self, x: f64) {
        self.inner.soft.particles[1].x.x = x;
        self.inner.soft.particles[1].v = pe_math::Vec3::zero();
    }

    /// Temporarily pin (`inv_mass=0`) or unpin the free end.
    fn set_free_end_pinned(&mut self, pinned: bool) {
        if pinned {
            self.inner.soft.particles[1].inv_mass = 0.0;
            self.inner.soft.particles[1].v = pe_math::Vec3::zero();
        } else {
            self.inner.soft.particles[1].inv_mass = 1.0;
        }
    }

    /// Set damping [1/s].
    fn set_damping(&mut self, damping: f64) {
        self.inner.soft.params.damping = damping;
    }

    /// One XPBD + cohesive update step; **releases the GIL**.
    fn step(&mut self, py: Python<'_>, dt: f64) {
        py.allow_threads(|| {
            self.inner.step(dt);
        });
    }

    /// Particle positions as NumPy `(n, 3)`.
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.soft.len();
        let flat = self.inner.soft.positions_flat();
        let arr = PyArray1::from_vec(py, flat);
        arr.reshape([n, 3]).expect("reshape cohesive_bar_x")
    }
}

/// Register soft-body bindings.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySoftWorld>()?;
    m.add_class::<PySoftRigidWorld>()?;
    m.add_class::<PyThermalRod>()?;
    m.add_class::<PyPlasticBar>()?;
    m.add_class::<PyFsPlasticBar>()?;
    m.add_class::<PyBePlasticBar>()?;
    m.add_class::<PyCohesiveBar>()?;
    m.add_function(wrap_pyfunction!(canonical_soft_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_materials_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_soft_coupling_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_thermal_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_plastic_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_fs_plastic_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_be_plastic_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_cohesive_hash, m)?)?;
    m.add_function(wrap_pyfunction!(soft_material_names, m)?)?;
    m.add_function(wrap_pyfunction!(soft_material_dict, m)?)?;
    m.add_function(wrap_pyfunction!(soft_lame_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(soft_stretch_compliance, m)?)?;
    m.add_function(wrap_pyfunction!(soft_j2_uniaxial_step, m)?)?;
    m.add_function(wrap_pyfunction!(soft_free_expansion_delta_l, m)?)?;
    m.add_function(wrap_pyfunction!(soft_rod_fourier_mode1, m)?)?;
    m.add_function(wrap_pyfunction!(soft_semi_infinite_erfc, m)?)?;
    m.add_function(wrap_pyfunction!(soft_robin_lumped_exp, m)?)?;
    m.add_function(wrap_pyfunction!(soft_analytic_permanent_strain, m)?)?;
    m.add_function(wrap_pyfunction!(soft_analytic_log_permanent_strain, m)?)?;
    m.add_function(wrap_pyfunction!(soft_analytic_isochoric_uniaxial_seq, m)?)?;
    m.add_function(wrap_pyfunction!(soft_quasistatic_stretch_unload, m)?)?;
    m.add_function(wrap_pyfunction!(soft_quasistatic_log_stretch_unload, m)?)?;
    m.add_function(wrap_pyfunction!(soft_analytic_gc, m)?)?;
    m.add_function(wrap_pyfunction!(soft_czm_virgin_traction, m)?)?;
    m.add_function(wrap_pyfunction!(soft_czm_damage, m)?)?;
    m.add_function(wrap_pyfunction!(soft_quasistatic_cohesive_open, m)?)?;
    Ok(())
}
