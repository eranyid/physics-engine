//! PyO3 bindings for stage-3 aero / hydro (M4).

use numpy::PyArray1;
use pe_aero::atmosphere::{self, DensityModel, RHO0};
use pe_aero::{
    lanchester_phugoid_period, level_flight_trim_earth, terminal_velocity_vertical, AircraftAero,
    FixedThrust, JetThrust, PropellerThrust, WingPolar,
};
use pe_core::diagnostics::state_hash64;
use pe_core::forces::{ForceGenerator, UniformGravity};
use pe_core::integrators::rk4;
use pe_core::{inertia_solid_box, RigidBody, World};
use pe_hydro::hull::{
    BoxHull, ConvexMeshHull, CylinderHull, Hull, SphereHull, RHO_FRESHWATER, RHO_SEAWATER,
};
use pe_hydro::{
    angle_of_vanishing_stability, box_added_mass_heave, box_added_mass_surge,
    box_righting_arm_curve, check_imo_is_code, heave_period_earth, metacentric_height,
    righting_arm, righting_moment, roll_period_earth, Buoyancy, HullDrag, WaveDrag,
};
use pe_math::{Quat, Vec3, STANDARD_GRAVITY};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

fn vec3_from_seq(obj: &Bound<'_, PyAny>) -> PyResult<Vec3> {
    let arr: Vec<f64> = obj.extract()?;
    if arr.len() != 3 {
        return Err(PyValueError::new_err("expected length-3 sequence [x,y,z]"));
    }
    Ok(Vec3::new(arr[0], arr[1], arr[2]))
}

fn density_from_opt(atmosphere: Option<&Bound<'_, PyAny>>) -> PyResult<DensityModel> {
    match atmosphere {
        None => Ok(DensityModel::Isa),
        Some(obj) => {
            if let Ok(rho) = obj.extract::<f64>() {
                Ok(DensityModel::Constant(rho))
            } else {
                // Callables are not supported across the FFI; pass a float density.
                Err(PyValueError::new_err(
                    "atmosphere must be a float density [kg/m^3] or omitted for ISA",
                ))
            }
        }
    }
}

/// Wing polar (SI). `S` [m²], `c` [m], `b` [m]; angles [rad].
#[pyclass(name = "WingPolar", module = "physics_engine_rs")]
#[derive(Clone)]
pub struct PyWingPolar {
    pub(crate) inner: WingPolar,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyWingPolar {
    #[new]
    #[pyo3(signature = (
        S, c, b, CD0=0.02, e=0.8, CL0=0.0, CL_alpha=None, alpha_stall=None,
        CL_fall=1.0, aspect_ratio=None, Cm0=0.0, Cm_alpha=-0.5, Cm_q=-8.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        S: f64,
        c: f64,
        b: f64,
        CD0: f64,
        e: f64,
        CL0: f64,
        CL_alpha: Option<f64>,
        alpha_stall: Option<f64>,
        CL_fall: f64,
        aspect_ratio: Option<f64>,
        Cm0: f64,
        Cm_alpha: f64,
        Cm_q: f64,
    ) -> PyResult<Self> {
        let cl_alpha = CL_alpha.unwrap_or(2.0 * core::f64::consts::PI);
        let a_stall = alpha_stall.unwrap_or(15.0_f64.to_radians());
        let inner = WingPolar::new(
            S,
            c,
            b,
            CD0,
            e,
            CL0,
            cl_alpha,
            a_stall,
            CL_fall,
            aspect_ratio,
            Cm0,
            Cm_alpha,
            Cm_q,
        )
        .map_err(|_| PyValueError::new_err("invalid WingPolar parameters"))?;
        Ok(Self { inner })
    }

    #[allow(non_snake_case)]
    fn CL(&self, alpha: f64) -> f64 {
        self.inner.cl(alpha)
    }

    #[allow(non_snake_case)]
    fn CD(&self, cl: f64) -> f64 {
        self.inner.cd(cl)
    }

    #[allow(non_snake_case)]
    fn Cm(&self, alpha: f64, q: f64, v: f64) -> f64 {
        self.inner.cm(alpha, q, v)
    }

    fn L_over_D(&self, alpha: f64) -> f64 {
        self.inner.l_over_d(alpha)
    }

    #[getter]
    fn aspect_ratio(&self) -> f64 {
        self.inner.aspect_ratio
    }

    #[getter]
    fn S(&self) -> f64 {
        self.inner.s
    }

    #[getter]
    fn c(&self) -> f64 {
        self.inner.c
    }

    #[getter]
    fn b(&self) -> f64 {
        self.inner.b
    }

    #[getter]
    fn CD0(&self) -> f64 {
        self.inner.cd0
    }

    #[getter]
    fn e(&self) -> f64 {
        self.inner.e
    }

    #[getter]
    fn CL0(&self) -> f64 {
        self.inner.cl0
    }

    #[getter]
    fn CL_alpha(&self) -> f64 {
        self.inner.cl_alpha
    }

    #[getter]
    fn alpha_stall(&self) -> f64 {
        self.inner.alpha_stall
    }

    #[getter]
    fn Cm_alpha(&self) -> f64 {
        self.inner.cm_alpha
    }
}

/// Aircraft aero force generator.
#[pyclass(name = "AircraftAero", module = "physics_engine_rs")]
#[derive(Clone)]
pub struct PyAircraftAero {
    pub(crate) inner: AircraftAero,
}

#[pymethods]
impl PyAircraftAero {
    #[new]
    #[pyo3(signature = (body, polar, atmosphere=None, wind=None))]
    fn new(
        body: usize,
        polar: &PyWingPolar,
        atmosphere: Option<&Bound<'_, PyAny>>,
        wind: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let density = density_from_opt(atmosphere)?;
        let wind = match wind {
            Some(w) => vec3_from_seq(w)?,
            None => Vec3::zero(),
        };
        Ok(Self {
            inner: AircraftAero::new(body, polar.inner.clone(), density, wind),
        })
    }
}

/// Fixed thrust [N] along a body axis.
#[pyclass(name = "FixedThrust", module = "physics_engine_rs")]
#[derive(Clone)]
pub struct PyFixedThrust {
    pub(crate) inner: FixedThrust,
}

#[pymethods]
impl PyFixedThrust {
    #[new]
    #[pyo3(signature = (body, thrust, axis_body=None))]
    fn new(body: usize, thrust: f64, axis_body: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let axis = match axis_body {
            Some(a) => vec3_from_seq(a)?,
            None => Vec3::unit_x(),
        };
        Ok(Self {
            inner: FixedThrust::new(body, thrust, axis),
        })
    }
}

/// Propeller thrust model.
#[pyclass(name = "PropellerThrust", module = "physics_engine_rs")]
#[derive(Clone)]
pub struct PyPropellerThrust {
    pub(crate) inner: PropellerThrust,
}

#[pymethods]
impl PyPropellerThrust {
    #[new]
    #[pyo3(signature = (
        body, T_static, power, eta=0.8, throttle=1.0, V_min=1.0, rho_sl=None,
        atmosphere=None, axis_body=None, wind=None
    ))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn new(
        body: usize,
        T_static: f64,
        power: f64,
        eta: f64,
        throttle: f64,
        V_min: f64,
        rho_sl: Option<f64>,
        atmosphere: Option<&Bound<'_, PyAny>>,
        axis_body: Option<&Bound<'_, PyAny>>,
        wind: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let axis = match axis_body {
            Some(a) => vec3_from_seq(a)?,
            None => Vec3::unit_x(),
        };
        let wind = match wind {
            Some(w) => vec3_from_seq(w)?,
            None => Vec3::zero(),
        };
        Ok(Self {
            inner: PropellerThrust::new(
                body,
                T_static,
                power,
                eta,
                throttle,
                V_min,
                rho_sl.unwrap_or(RHO0),
                density_from_opt(atmosphere)?,
                axis,
                wind,
            ),
        })
    }

    fn thrust_magnitude(&self, v: f64, rho: f64) -> f64 {
        self.inner.thrust_magnitude(v, rho)
    }
}

/// Jet thrust model.
#[pyclass(name = "JetThrust", module = "physics_engine_rs")]
#[derive(Clone)]
pub struct PyJetThrust {
    pub(crate) inner: JetThrust,
}

#[pymethods]
impl PyJetThrust {
    #[new]
    #[pyo3(signature = (
        body, T_sl, throttle=1.0, n=0.7, rho_sl=None, atmosphere=None, axis_body=None
    ))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn new(
        body: usize,
        T_sl: f64,
        throttle: f64,
        n: f64,
        rho_sl: Option<f64>,
        atmosphere: Option<&Bound<'_, PyAny>>,
        axis_body: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let axis = match axis_body {
            Some(a) => vec3_from_seq(a)?,
            None => Vec3::unit_x(),
        };
        Ok(Self {
            inner: JetThrust::new(
                body,
                T_sl,
                throttle,
                n,
                rho_sl.unwrap_or(RHO0),
                density_from_opt(atmosphere)?,
                axis,
            ),
        })
    }

    fn thrust_magnitude(&self, rho: f64) -> f64 {
        self.inner.thrust_magnitude(rho)
    }
}

/// Rectangular barge hull.
#[pyclass(name = "BoxHull", module = "physics_engine_rs")]
#[derive(Clone)]
pub struct PyBoxHull {
    pub(crate) inner: BoxHull,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyBoxHull {
    #[new]
    fn new(length: f64, beam: f64, depth: f64) -> Self {
        Self {
            inner: BoxHull::new(length, beam, depth),
        }
    }

    fn upright_draft_for_mass(&self, mass: f64, rho: f64) -> f64 {
        self.inner.upright_draft_for_mass(mass, rho)
    }

    fn upright_KB(&self, draft: f64) -> f64 {
        self.inner.upright_kb(draft)
    }

    fn upright_BM_roll(&self, draft: f64) -> f64 {
        self.inner.upright_bm_roll(draft)
    }

    #[pyo3(signature = (draft, kg, roll=true))]
    fn upright_GM(&self, draft: f64, kg: f64, roll: bool) -> f64 {
        self.inner.upright_gm(draft, kg, roll)
    }

    #[getter]
    fn length(&self) -> f64 {
        self.inner.length
    }

    #[getter]
    fn beam(&self) -> f64 {
        self.inner.beam
    }

    #[getter]
    fn depth(&self) -> f64 {
        self.inner.depth
    }
}

/// Vertical cylinder hull.
#[pyclass(name = "CylinderHull", module = "physics_engine_rs")]
#[derive(Clone)]
pub struct PyCylinderHull {
    pub(crate) inner: CylinderHull,
}

#[pymethods]
impl PyCylinderHull {
    #[new]
    fn new(radius: f64, height: f64) -> Self {
        Self {
            inner: CylinderHull::new(radius, height),
        }
    }

    fn upright_draft_for_mass(&self, mass: f64, rho: f64) -> f64 {
        self.inner.upright_draft_for_mass(mass, rho)
    }
}

/// Sphere hull.
#[pyclass(name = "SphereHull", module = "physics_engine_rs")]
#[derive(Clone)]
pub struct PySphereHull {
    pub(crate) inner: SphereHull,
}

#[pymethods]
impl PySphereHull {
    #[new]
    fn new(radius: f64) -> Self {
        Self {
            inner: SphereHull::new(radius),
        }
    }

    fn volume(&self) -> f64 {
        self.inner.volume()
    }

    fn upright_draft_for_mass(&self, mass: f64, rho: f64) -> f64 {
        self.inner.upright_draft_for_mass(mass, rho)
    }
}

/// Convex mesh hull.
#[pyclass(name = "ConvexMeshHull", module = "physics_engine_rs")]
#[derive(Clone)]
pub struct PyConvexMeshHull {
    pub(crate) inner: ConvexMeshHull,
}

#[pymethods]
impl PyConvexMeshHull {
    #[new]
    fn new(vertices: &Bound<'_, PyAny>, faces: &Bound<'_, PyAny>) -> PyResult<Self> {
        let verts_raw: Vec<Vec<f64>> = vertices.extract()?;
        let mut verts = Vec::with_capacity(verts_raw.len());
        for v in verts_raw {
            if v.len() != 3 {
                return Err(PyValueError::new_err("vertices must be Nx3"));
            }
            verts.push(Vec3::new(v[0], v[1], v[2]));
        }
        let faces: Vec<Vec<usize>> = faces.extract()?;
        Ok(Self {
            inner: ConvexMeshHull::new(verts, faces),
        })
    }
}

fn hull_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Hull> {
    if let Ok(h) = obj.extract::<PyRef<PyBoxHull>>() {
        return Ok(Hull::Box(h.inner));
    }
    if let Ok(h) = obj.extract::<PyRef<PyCylinderHull>>() {
        return Ok(Hull::Cylinder(h.inner));
    }
    if let Ok(h) = obj.extract::<PyRef<PySphereHull>>() {
        return Ok(Hull::Sphere(h.inner));
    }
    if let Ok(h) = obj.extract::<PyRef<PyConvexMeshHull>>() {
        return Ok(Hull::Mesh(h.inner.clone()));
    }
    Err(PyValueError::new_err(
        "hull must be BoxHull|CylinderHull|SphereHull|ConvexMeshHull",
    ))
}

/// Buoyancy force generator.
#[pyclass(name = "Buoyancy", module = "physics_engine_rs")]
#[derive(Clone)]
pub struct PyBuoyancy {
    pub(crate) inner: Buoyancy,
}

#[pymethods]
impl PyBuoyancy {
    #[new]
    #[pyo3(signature = (body, hull, rho=None, g=None, water_level=0.0, com_body=None))]
    fn new(
        body: usize,
        hull: &Bound<'_, PyAny>,
        rho: Option<f64>,
        g: Option<f64>,
        water_level: f64,
        com_body: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let com = match com_body {
            Some(c) => vec3_from_seq(c)?,
            None => Vec3::zero(),
        };
        Ok(Self {
            inner: Buoyancy::new(
                body,
                hull_from_py(hull)?,
                rho.unwrap_or(RHO_FRESHWATER),
                g.unwrap_or(STANDARD_GRAVITY),
                water_level,
                com,
            ),
        })
    }
}

/// Hull drag force generator.
#[pyclass(name = "HullDrag", module = "physics_engine_rs")]
#[derive(Clone)]
pub struct PyHullDrag {
    pub(crate) inner: HullDrag,
}

#[pymethods]
impl PyHullDrag {
    #[new]
    #[pyo3(signature = (
        body, S, L, C_f=0.004, C_w=0.0, rho=None, g=None, wind=None, include_heave=false
    ))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn new(
        body: usize,
        S: f64,
        L: f64,
        C_f: f64,
        C_w: f64,
        rho: Option<f64>,
        g: Option<f64>,
        wind: Option<&Bound<'_, PyAny>>,
        include_heave: bool,
    ) -> PyResult<Self> {
        let wind = match wind {
            Some(w) => vec3_from_seq(w)?,
            None => Vec3::zero(),
        };
        Ok(Self {
            inner: HullDrag::new(
                body,
                S,
                L,
                C_f,
                WaveDrag::Constant(C_w),
                rho.unwrap_or(RHO_FRESHWATER),
                g.unwrap_or(STANDARD_GRAVITY),
                wind,
                include_heave,
            ),
        })
    }
}

/// Try to extract a stage-3 force generator; returns `None` if not recognised.
pub fn try_extract_generator(item: &Bound<'_, PyAny>) -> PyResult<Option<Box<dyn ForceGenerator>>> {
    if let Ok(g) = item.extract::<PyRef<PyAircraftAero>>() {
        return Ok(Some(Box::new(g.inner.clone())));
    }
    if let Ok(g) = item.extract::<PyRef<PyFixedThrust>>() {
        return Ok(Some(Box::new(g.inner.clone())));
    }
    if let Ok(g) = item.extract::<PyRef<PyPropellerThrust>>() {
        return Ok(Some(Box::new(g.inner.clone())));
    }
    if let Ok(g) = item.extract::<PyRef<PyJetThrust>>() {
        return Ok(Some(Box::new(g.inner.clone())));
    }
    if let Ok(g) = item.extract::<PyRef<PyBuoyancy>>() {
        return Ok(Some(Box::new(g.inner.clone())));
    }
    if let Ok(g) = item.extract::<PyRef<PyHullDrag>>() {
        return Ok(Some(Box::new(g.inner.clone())));
    }
    Ok(None)
}

// ---- Free functions ---------------------------------------------------------

#[pyfunction]
#[pyo3(name = "isa_temperature")]
fn isa_temperature(h: f64) -> PyResult<f64> {
    atmosphere::temperature(h).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(name = "isa_pressure")]
fn isa_pressure(h: f64) -> PyResult<f64> {
    atmosphere::pressure(h).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(name = "isa_density")]
fn isa_density(h: f64) -> PyResult<f64> {
    atmosphere::density(h).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(name = "isa_speed_of_sound")]
fn isa_speed_of_sound(h: f64) -> PyResult<f64> {
    atmosphere::speed_of_sound(h).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(name = "isa_state")]
fn isa_state<'py>(py: Python<'py>, h: f64) -> PyResult<Bound<'py, PyDict>> {
    let s = atmosphere::state(h).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("temperature", s.temperature)?;
    d.set_item("pressure", s.pressure)?;
    d.set_item("density", s.density)?;
    d.set_item("speed_of_sound", s.speed_of_sound)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(name = "level_flight_trim")]
fn level_flight_trim_py<'py>(
    py: Python<'py>,
    mass: f64,
    polar: &PyWingPolar,
    rho: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let t = level_flight_trim_earth(mass, &polar.inner, rho);
    let d = PyDict::new(py);
    d.set_item("V", t.v)?;
    d.set_item("alpha", t.alpha)?;
    d.set_item("thrust", t.thrust)?;
    d.set_item("CL", t.cl)?;
    d.set_item("CD", t.cd)?;
    d.set_item("L_over_D", t.l_over_d)?;
    d.set_item("q_bar", t.q_bar)?;
    d.set_item("q_min_stall", t.q_min_stall)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(name = "terminal_velocity_vertical", signature = (mass, polar, rho, alpha=None))]
fn terminal_velocity_vertical_py(
    mass: f64,
    polar: &PyWingPolar,
    rho: f64,
    alpha: Option<f64>,
) -> f64 {
    let a = alpha.unwrap_or(-core::f64::consts::FRAC_PI_2);
    terminal_velocity_vertical(mass, &polar.inner, rho, a, STANDARD_GRAVITY)
}

#[pyfunction]
#[pyo3(name = "lanchester_phugoid_period", signature = (v, g=None))]
fn lanchester_phugoid_period_py(v: f64, g: Option<f64>) -> f64 {
    lanchester_phugoid_period(v, g.unwrap_or(STANDARD_GRAVITY))
}

#[pyfunction]
#[pyo3(name = "metacentric_height")]
fn metacentric_height_py(kb: f64, bm: f64, kg: f64) -> f64 {
    metacentric_height(kb, bm, kg)
}

#[pyfunction]
#[pyo3(name = "righting_arm")]
fn righting_arm_py(gm: f64, phi: f64) -> f64 {
    righting_arm(gm, phi)
}

#[pyfunction]
#[pyo3(name = "righting_moment", signature = (rho, volume, gz, g=None))]
fn righting_moment_py(rho: f64, volume: f64, gz: f64, g: Option<f64>) -> f64 {
    righting_moment(rho, volume, gz, g.unwrap_or(STANDARD_GRAVITY))
}

#[pyfunction]
#[pyo3(name = "heave_period")]
fn heave_period_py(mass: f64, rho: f64, a_wp: f64) -> f64 {
    heave_period_earth(mass, rho, a_wp)
}

#[pyfunction]
#[pyo3(name = "roll_period")]
fn roll_period_py(k: f64, gm: f64) -> f64 {
    roll_period_earth(k, gm)
}

#[pyfunction]
#[pyo3(name = "box_righting_arm_curve")]
fn box_righting_arm_curve_py<'py>(
    py: Python<'py>,
    hull: &PyBoxHull,
    draft: f64,
    kg: f64,
    phis: Vec<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let gz = box_righting_arm_curve(&hull.inner, draft, kg, &phis);
    PyArray1::from_vec(py, gz)
}

#[pyfunction]
#[pyo3(name = "angle_of_vanishing_stability", signature = (hull, draft, kg, phi_max=None, n=120))]
fn angle_of_vanishing_stability_py(
    hull: &PyBoxHull,
    draft: f64,
    kg: f64,
    phi_max: Option<f64>,
    n: usize,
) -> f64 {
    angle_of_vanishing_stability(
        &hull.inner,
        draft,
        kg,
        phi_max.unwrap_or(85.0_f64.to_radians()),
        n,
    )
}

#[pyfunction]
#[pyo3(name = "box_added_mass_heave")]
fn box_added_mass_heave_py(rho: f64, a_wp: f64, draft: f64) -> f64 {
    box_added_mass_heave(rho, a_wp, draft)
}

#[pyfunction]
#[pyo3(name = "box_added_mass_surge", signature = (rho, volume, Cx=0.5))]
#[allow(non_snake_case)]
fn box_added_mass_surge_py(rho: f64, volume: f64, Cx: f64) -> f64 {
    box_added_mass_surge(rho, volume, Cx)
}

#[pyfunction]
#[pyo3(name = "check_imo_is_code")]
fn check_imo_is_code_py<'py>(
    py: Python<'py>,
    phis: Vec<f64>,
    gz: Vec<f64>,
    gm0: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if phis.len() != gz.len() {
        return Err(PyValueError::new_err("phis and gz must have equal length"));
    }
    let r = check_imo_is_code(&phis, &gz, gm0);
    let d = PyDict::new(py);
    d.set_item("passed", r.passed)?;
    d.set_item("area_0_30", r.area_0_30)?;
    d.set_item("area_0_40", r.area_0_40)?;
    d.set_item("area_30_40", r.area_30_40)?;
    d.set_item("gz_at_or_above_30", r.gz_at_or_above_30)?;
    d.set_item("angle_of_max_gz", r.angle_of_max_gz)?;
    d.set_item("gm0", r.gm0)?;
    let fails = PyList::new(py, r.failures)?;
    d.set_item("failures", fails)?;
    Ok(d)
}

/// Canonical stage-3 aero+hydro hash (same scenario as `pe-hydro` binary).
#[pyfunction]
fn canonical_aero_hydro_hash() -> u64 {
    let polar = WingPolar::new(
        16.0,
        1.6,
        10.0,
        0.025,
        0.8,
        0.2,
        5.7,
        14.0_f64.to_radians(),
        1.2,
        None,
        0.0,
        0.0,
        0.0,
    )
    .expect("polar");
    let mass_ac = 1000.0;
    let q_att = Quat::from_axis_angle(Vec3::unit_y(), -0.05);
    let aircraft = RigidBody::new(mass_ac)
        .with_inertia(inertia_solid_box(mass_ac, 8.0, 10.0, 2.0))
        .with_position(Vec3::new(0.0, 0.0, 500.0))
        .with_velocity(Vec3::new(40.0, 0.0, 0.0))
        .with_orientation(q_att);

    let hull = BoxHull::new(12.0, 4.0, 2.0);
    let mass_ship = 24_000.0;
    let draft = hull.upright_draft_for_mass(mass_ship, RHO_FRESHWATER);
    let z_c = -draft + 0.5 * hull.depth;
    let ship = RigidBody::new(mass_ship)
        .with_inertia(inertia_solid_box(mass_ship, 12.0, 4.0, 2.0))
        .with_position(Vec3::new(30.0, 0.0, z_c))
        .with_orientation(Quat::from_axis_angle(Vec3::unit_x(), 0.02));

    let mut w = World::new(
        vec![aircraft, ship],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(AircraftAero::new(
                0,
                polar,
                DensityModel::Constant(RHO0),
                Vec3::zero(),
            )),
            Box::new(FixedThrust::new(0, 800.0, Vec3::unit_x())),
            Box::new(Buoyancy::freshwater(1, Hull::Box(hull), 0.0)),
        ],
    );
    w.run_for(1.0, 1e-2, rk4);
    state_hash64(&w.state)
}

/// Register stage-3 classes and free functions on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWingPolar>()?;
    m.add_class::<PyAircraftAero>()?;
    m.add_class::<PyFixedThrust>()?;
    m.add_class::<PyPropellerThrust>()?;
    m.add_class::<PyJetThrust>()?;
    m.add_class::<PyBoxHull>()?;
    m.add_class::<PyCylinderHull>()?;
    m.add_class::<PySphereHull>()?;
    m.add_class::<PyConvexMeshHull>()?;
    m.add_class::<PyBuoyancy>()?;
    m.add_class::<PyHullDrag>()?;

    m.add_function(wrap_pyfunction!(isa_temperature, m)?)?;
    m.add_function(wrap_pyfunction!(isa_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(isa_density, m)?)?;
    m.add_function(wrap_pyfunction!(isa_speed_of_sound, m)?)?;
    m.add_function(wrap_pyfunction!(isa_state, m)?)?;
    m.add_function(wrap_pyfunction!(level_flight_trim_py, m)?)?;
    m.add_function(wrap_pyfunction!(terminal_velocity_vertical_py, m)?)?;
    m.add_function(wrap_pyfunction!(lanchester_phugoid_period_py, m)?)?;
    m.add_function(wrap_pyfunction!(metacentric_height_py, m)?)?;
    m.add_function(wrap_pyfunction!(righting_arm_py, m)?)?;
    m.add_function(wrap_pyfunction!(righting_moment_py, m)?)?;
    m.add_function(wrap_pyfunction!(heave_period_py, m)?)?;
    m.add_function(wrap_pyfunction!(roll_period_py, m)?)?;
    m.add_function(wrap_pyfunction!(box_righting_arm_curve_py, m)?)?;
    m.add_function(wrap_pyfunction!(angle_of_vanishing_stability_py, m)?)?;
    m.add_function(wrap_pyfunction!(box_added_mass_heave_py, m)?)?;
    m.add_function(wrap_pyfunction!(box_added_mass_surge_py, m)?)?;
    m.add_function(wrap_pyfunction!(check_imo_is_code_py, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_aero_hydro_hash, m)?)?;

    m.add("RHO0", RHO0)?;
    m.add("RHO_FRESHWATER", RHO_FRESHWATER)?;
    m.add("RHO_SEAWATER", RHO_SEAWATER)?;
    m.add("ISA_T0", atmosphere::T0)?;
    m.add("ISA_P0", atmosphere::P0)?;
    Ok(())
}
