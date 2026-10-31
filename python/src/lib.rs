//! PyO3 bindings for stages 1–3 (`physics_engine_rs`).
//!
//! - Built as **abi3** (CPython 3.10+)
//! - GIL released during `step` / `run` (including contact steps)
//! - State export packs into a contiguous `Vec<f64>` then moves into NumPy
//!   (`PyArray::from_vec`) — one allocation, no second copy

mod aero_hydro;
mod blast_ballistics;
mod soft;
mod fluid;
mod space;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pe_collision::{
    BoxShape, BroadPhaseKind, Capsule, CcdParams, ContactWorld, ConvexHull, Joint, JointBreak,
    JointLimit, JointMotor, JointSoftParams, Material, Plane, RoundedShape, Shape, SoftStepParams,
    SolverMode, Sphere, StaticCollider, WORLD_BODY,
};
use pe_core::diagnostics::{
    angular_momentum as ang_mom, kinetic_energy, linear_momentum, potential_energy, state_hash64,
    total_energy,
};
use pe_core::forces::{
    ConstantTorque, ForceGenerator, LinearDrag, NewtonianGravity, QuadraticDrag, Spring,
    UniformGravity,
};
use pe_core::integrators::IntegratorKind;
use pe_core::{RigidBody, World};
use pe_math::{Quat, Vec3, G, STANDARD_GRAVITY};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

fn vec3_from_seq(obj: &Bound<'_, PyAny>) -> PyResult<Vec3> {
    let arr: Vec<f64> = obj.extract()?;
    if arr.len() != 3 {
        return Err(PyValueError::new_err("expected length-3 sequence [x,y,z]"));
    }
    Ok(Vec3::new(arr[0], arr[1], arr[2]))
}

fn quat_from_seq(obj: &Bound<'_, PyAny>) -> PyResult<Quat> {
    let arr: Vec<f64> = obj.extract()?;
    if arr.len() != 4 {
        return Err(PyValueError::new_err(
            "expected length-4 quaternion [w,x,y,z]",
        ));
    }
    Ok(Quat::new(arr[0], arr[1], arr[2], arr[3]).normalize())
}

fn inertia_from_obj(obj: &Bound<'_, PyAny>) -> PyResult<[[f64; 3]; 3]> {
    if let Ok(diag) = obj.extract::<Vec<f64>>() {
        if diag.len() == 3 {
            return Ok([
                [diag[0], 0.0, 0.0],
                [0.0, diag[1], 0.0],
                [0.0, 0.0, diag[2]],
            ]);
        }
        if diag.len() == 9 {
            return Ok([
                [diag[0], diag[1], diag[2]],
                [diag[3], diag[4], diag[5]],
                [diag[6], diag[7], diag[8]],
            ]);
        }
    }
    let rows: Vec<Vec<f64>> = obj.extract()?;
    if rows.len() != 3 || rows.iter().any(|r| r.len() != 3) {
        return Err(PyValueError::new_err(
            "inertia_body: diag[3], flat[9], or 3x3 nested",
        ));
    }
    Ok([
        [rows[0][0], rows[0][1], rows[0][2]],
        [rows[1][0], rows[1][1], rows[1][2]],
        [rows[2][0], rows[2][1], rows[2][2]],
    ])
}

fn parse_integrator(name: &str) -> PyResult<IntegratorKind> {
    IntegratorKind::parse(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown integrator {name:?}; expected explicit_euler|symplectic_euler|velocity_verlet|rk4"
        ))
    })
}

/// Initial rigid-body description (SI).
#[pyclass(name = "RigidBody", module = "physics_engine_rs")]
#[derive(Clone)]
struct PyRigidBody {
    inner: RigidBody,
}

#[pymethods]
impl PyRigidBody {
    #[new]
    #[pyo3(signature = (mass, position=None, velocity=None, inertia_body=None, orientation=None, angular_velocity=None, name=None))]
    fn new(
        mass: f64,
        position: Option<&Bound<'_, PyAny>>,
        velocity: Option<&Bound<'_, PyAny>>,
        inertia_body: Option<&Bound<'_, PyAny>>,
        orientation: Option<&Bound<'_, PyAny>>,
        angular_velocity: Option<&Bound<'_, PyAny>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        if mass <= 0.0 {
            return Err(PyValueError::new_err("mass must be positive"));
        }
        let mut b = RigidBody::new(mass);
        if let Some(p) = position {
            b.position = vec3_from_seq(p)?;
        }
        if let Some(v) = velocity {
            b.velocity = vec3_from_seq(v)?;
        }
        if let Some(i) = inertia_body {
            b = b.with_inertia(inertia_from_obj(i)?);
        }
        if let Some(q) = orientation {
            b.orientation = quat_from_seq(q)?;
        }
        if let Some(w) = angular_velocity {
            b.angular_velocity = vec3_from_seq(w)?;
        }
        if let Some(n) = name {
            b.name = n;
        }
        Ok(Self { inner: b })
    }

    #[getter]
    fn mass(&self) -> f64 {
        self.inner.mass
    }
}

/// `F = m g` with `g` [m/s²] (default `(0,0,-g0)`).
#[pyclass(name = "UniformGravity", module = "physics_engine_rs")]
#[derive(Clone)]
struct PyUniformGravity {
    inner: UniformGravity,
}

#[pymethods]
impl PyUniformGravity {
    #[new]
    #[pyo3(signature = (g=None))]
    fn new(g: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let g = match g {
            Some(obj) => vec3_from_seq(obj)?,
            None => Vec3::new(0.0, 0.0, -STANDARD_GRAVITY),
        };
        Ok(Self {
            inner: UniformGravity::new(g),
        })
    }
}

/// Pairwise Newtonian gravity.
#[pyclass(name = "NewtonianGravity", module = "physics_engine_rs")]
#[derive(Clone)]
struct PyNewtonianGravity {
    inner: NewtonianGravity,
}

#[pymethods]
impl PyNewtonianGravity {
    #[new]
    #[pyo3(signature = (g_const=None, softening=0.0))]
    fn new(g_const: Option<f64>, softening: f64) -> Self {
        Self {
            inner: NewtonianGravity::with_params(g_const.unwrap_or(G), softening),
        }
    }
}

/// Stokes drag `F = -b (v - wind)`, `b` [kg/s].
#[pyclass(name = "LinearDrag", module = "physics_engine_rs")]
#[derive(Clone)]
struct PyLinearDrag {
    inner: LinearDrag,
}

#[pymethods]
impl PyLinearDrag {
    #[new]
    #[pyo3(signature = (b, wind=None))]
    fn new(b: f64, wind: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let wind = match wind {
            Some(w) => vec3_from_seq(w)?,
            None => Vec3::zero(),
        };
        Ok(Self {
            inner: LinearDrag::new(b, wind),
        })
    }
}

/// Newton drag `F = -k |u| u`, `k` [kg/m].
#[pyclass(name = "QuadraticDrag", module = "physics_engine_rs")]
#[derive(Clone)]
struct PyQuadraticDrag {
    inner: QuadraticDrag,
}

#[pymethods]
impl PyQuadraticDrag {
    #[new]
    #[pyo3(signature = (k, wind=None))]
    fn new(k: f64, wind: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let wind = match wind {
            Some(w) => vec3_from_seq(w)?,
            None => Vec3::zero(),
        };
        Ok(Self {
            inner: QuadraticDrag::new(k, wind),
        })
    }

    #[staticmethod]
    fn coefficient(rho: f64, cd: f64, area: f64) -> f64 {
        QuadraticDrag::coefficient(rho, cd, area)
    }
}

/// Hookean spring to a fixed anchor, stiffness `k` [N/m].
#[pyclass(name = "Spring", module = "physics_engine_rs")]
#[derive(Clone)]
struct PySpring {
    inner: Spring,
}

#[pymethods]
impl PySpring {
    #[new]
    fn new(body: usize, anchor: &Bound<'_, PyAny>, k: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Spring::new(body, vec3_from_seq(anchor)?, k),
        })
    }
}

/// Constant world-frame torque [N m].
#[pyclass(name = "ConstantTorque", module = "physics_engine_rs")]
#[derive(Clone)]
struct PyConstantTorque {
    inner: ConstantTorque,
}

#[pymethods]
impl PyConstantTorque {
    #[new]
    fn new(body: usize, torque: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: ConstantTorque::new(body, vec3_from_seq(torque)?),
        })
    }
}

fn extract_generators(gens: &Bound<'_, PyAny>) -> PyResult<Vec<Box<dyn ForceGenerator>>> {
    let list = gens.downcast::<PyList>()?;
    let mut out: Vec<Box<dyn ForceGenerator>> = Vec::with_capacity(list.len());
    for item in list.iter() {
        if let Ok(g) = item.extract::<PyRef<PyUniformGravity>>() {
            out.push(Box::new(g.inner.clone()));
        } else if let Ok(g) = item.extract::<PyRef<PyNewtonianGravity>>() {
            out.push(Box::new(g.inner.clone()));
        } else if let Ok(g) = item.extract::<PyRef<PyLinearDrag>>() {
            out.push(Box::new(g.inner.clone()));
        } else if let Ok(g) = item.extract::<PyRef<PyQuadraticDrag>>() {
            out.push(Box::new(g.inner.clone()));
        } else if let Ok(g) = item.extract::<PyRef<PySpring>>() {
            out.push(Box::new(g.inner.clone()));
        } else if let Ok(g) = item.extract::<PyRef<PyConstantTorque>>() {
            out.push(Box::new(g.inner.clone()));
        } else if let Some(g) = aero_hydro::try_extract_generator(&item)? {
            out.push(g);
        } else {
            return Err(PyTypeError::new_err(
                "generators must be UniformGravity|NewtonianGravity|LinearDrag|QuadraticDrag|Spring|ConstantTorque|AircraftAero|FixedThrust|PropellerThrust|JetThrust|Buoyancy|HullDrag",
            ));
        }
    }
    Ok(out)
}

/// Simulation world (SoA). Stage-1 API mirror of the Python oracle.
#[pyclass(name = "World", module = "physics_engine_rs")]
struct PyWorld {
    inner: World,
}

#[pymethods]
impl PyWorld {
    #[new]
    #[pyo3(signature = (bodies, generators=None))]
    fn new(bodies: &Bound<'_, PyAny>, generators: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let list = bodies.downcast::<PyList>()?;
        let mut rbodies = Vec::with_capacity(list.len());
        for item in list.iter() {
            let b: PyRef<PyRigidBody> = item.extract()?;
            rbodies.push(b.inner.clone());
        }
        let gens = match generators {
            Some(g) => extract_generators(g)?,
            None => Vec::new(),
        };
        Ok(Self {
            inner: World::new(rbodies, gens),
        })
    }

    #[getter]
    fn t(&self) -> f64 {
        self.inner.t
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    /// Positions as `(N, 3)` float64 array [m]. Packs then moves into NumPy (no second copy).
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.n;
        let mut flat = Vec::with_capacity(n * 3);
        for p in &self.inner.state.x {
            flat.extend_from_slice(&p.to_array());
        }
        let arr = PyArray1::from_vec(py, flat);
        arr.reshape([n, 3]).expect("reshape x")
    }

    /// Quaternions as `(N, 4)` float64 `[w,x,y,z]`.
    #[getter]
    fn q<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.n;
        let mut flat = Vec::with_capacity(n * 4);
        for q in &self.inner.state.q {
            flat.extend_from_slice(&q.to_array());
        }
        let arr = PyArray1::from_vec(py, flat);
        arr.reshape([n, 4]).expect("reshape q")
    }

    /// Velocities as `(N, 3)` float64 [m/s].
    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.n;
        let mut flat = Vec::with_capacity(n * 3);
        for v in &self.inner.state.v {
            flat.extend_from_slice(&v.to_array());
        }
        let arr = PyArray1::from_vec(py, flat);
        arr.reshape([n, 3]).expect("reshape v")
    }

    /// World-frame angular momenta as `(N, 3)` float64 [kg m²/s].
    #[getter]
    #[allow(non_snake_case)]
    fn L<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.n;
        let mut flat = Vec::with_capacity(n * 3);
        for l in &self.inner.state.l {
            flat.extend_from_slice(&l.to_array());
        }
        let arr = PyArray1::from_vec(py, flat);
        arr.reshape([n, 3]).expect("reshape L")
    }

    /// Packed state `[x…, q…, v…, L…]` moved into a 1-D NumPy array.
    fn pack<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.state.pack())
    }

    /// FNV-1a hash of packed IEEE bits (cross-OS determinism).
    fn state_hash(&self) -> u64 {
        state_hash64(&self.inner.state)
    }

    /// One step; **releases the GIL** for the duration of the integrator.
    fn step(&mut self, py: Python<'_>, dt: f64, integrator: &str) -> PyResult<()> {
        let kind = parse_integrator(integrator)?;
        let f = kind.as_fn();
        py.allow_threads(|| {
            self.inner.step(dt, f);
        });
        Ok(())
    }

    /// Fixed-`dt` run to `t_end` [s]; **releases the GIL**.
    #[pyo3(signature = (t_end, dt, integrator))]
    fn run(&mut self, py: Python<'_>, t_end: f64, dt: f64, integrator: &str) -> PyResult<()> {
        let kind = parse_integrator(integrator)?;
        let f = kind.as_fn();
        py.allow_threads(|| {
            self.inner.run_for(t_end, dt, f);
        });
        Ok(())
    }

    /// Instantaneous forces/torques at current state: returns `(F, T)` arrays.
    #[pyo3(signature = (t=None))]
    #[allow(clippy::type_complexity)]
    fn forces<'py>(
        &self,
        py: Python<'py>,
        t: Option<f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let t = t.unwrap_or(self.inner.t);
        let (f, tau) = self.inner.forces(&self.inner.state, t);
        let n = self.inner.n;
        let mut ff = Vec::with_capacity(n * 3);
        let mut tt = Vec::with_capacity(n * 3);
        for i in 0..n {
            ff.extend_from_slice(&f[i].to_array());
            tt.extend_from_slice(&tau[i].to_array());
        }
        let fa = PyArray1::from_vec(py, ff).reshape([n, 3])?;
        let ta = PyArray1::from_vec(py, tt).reshape([n, 3])?;
        Ok((fa, ta))
    }

    fn kinetic_energy(&self) -> f64 {
        kinetic_energy(&self.inner, &self.inner.state)
    }

    fn potential_energy(&self) -> f64 {
        potential_energy(&self.inner, &self.inner.state)
    }

    fn total_energy(&self) -> f64 {
        total_energy(&self.inner, &self.inner.state)
    }

    fn linear_momentum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let p = linear_momentum(&self.inner, &self.inner.state);
        PyArray1::from_vec(py, p.to_array().to_vec())
    }

    #[pyo3(signature = (origin=None))]
    fn angular_momentum<'py>(
        &self,
        py: Python<'py>,
        origin: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let o = match origin {
            Some(obj) => vec3_from_seq(obj)?,
            None => Vec3::zero(),
        };
        let l = ang_mom(&self.inner, &self.inner.state, o);
        Ok(PyArray1::from_vec(py, l.to_array().to_vec()))
    }

    /// Masses as length-N array [kg] (moved into NumPy).
    #[getter]
    fn mass<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.mass.clone())
    }
}

#[pyfunction]
fn inertia_solid_sphere(mass: f64, radius: f64) -> [[f64; 3]; 3] {
    pe_core::inertia_solid_sphere(mass, radius)
}

#[pyfunction]
fn inertia_solid_box(mass: f64, a: f64, b: f64, c: f64) -> [[f64; 3]; 3] {
    pe_core::inertia_solid_box(mass, a, b, c)
}

#[pyfunction]
fn inertia_solid_cylinder(mass: f64, r: f64, h: f64) -> [[f64; 3]; 3] {
    pe_core::inertia_solid_cylinder(mass, r, h)
}

/// Canonical cross-OS hash scenario (same as Rust `cross_os_canonical_hash_fixture`).
#[pyfunction]
fn canonical_determinism_hash() -> u64 {
    let mut w = World::new(
        vec![
            RigidBody::new(1.0)
                .with_position(Vec3::new(0.0, 0.0, 10.0))
                .with_velocity(Vec3::new(1.0, 0.0, 0.0)),
            RigidBody::new(2.0)
                .with_position(Vec3::new(1.0, 0.0, 0.0))
                .with_velocity(Vec3::new(0.0, 0.5, 0.0))
                .with_inertia_diag(0.4, 0.5, 0.6)
                .with_angular_velocity(Vec3::new(0.1, 0.0, 0.2)),
        ],
        vec![
            Box::new(UniformGravity::earth()),
            Box::new(LinearDrag::new(0.01, Vec3::zero())),
            Box::new(Spring::new(0, Vec3::new(0.0, 0.0, 10.0), 5.0)),
            Box::new(ConstantTorque::new(1, Vec3::new(0.0, 0.0, 0.05))),
        ],
    );
    w.run_for(2.0, 1e-3, IntegratorKind::Rk4.as_fn());
    state_hash64(&w.state)
}

/// Canonical stage-2 collision hash (10-box pyramid, 600 steps).
#[pyfunction]
fn canonical_collision_hash() -> u64 {
    pe_collision::canonical_collision_hash()
}

/// Canonical M5a soft-step collision hash (same pyramid, soft-step solver).
#[pyfunction]
fn canonical_soft_step_collision_hash() -> u64 {
    pe_collision::canonical_soft_step_collision_hash()
}

/// Canonical M5b joints hash (double pendulum, soft-step).
#[pyfunction]
fn canonical_joints_hash() -> u64 {
    pe_collision::canonical_joints_hash()
}

/// Canonical M6 convex hash (hull + capsule + rounded vs static box).
#[pyfunction]
fn canonical_convex_hash() -> u64 {
    pe_collision::canonical_convex_hash()
}

// --- Stage 2 shapes / contact ------------------------------------------------

/// Sphere shape [m].
#[pyclass(name = "Sphere", module = "physics_engine_rs")]
#[derive(Clone)]
struct PySphere {
    inner: Sphere,
}

#[pymethods]
impl PySphere {
    #[new]
    fn new(radius: f64) -> Self {
        Self {
            inner: Sphere::new(radius),
        }
    }

    #[getter]
    fn radius(&self) -> f64 {
        self.inner.radius
    }

    fn volume(&self) -> f64 {
        self.inner.volume()
    }

    fn inertia(&self, mass: f64) -> [[f64; 3]; 3] {
        self.inner.inertia(mass)
    }
}

/// Oriented box half-extents `(hx, hy, hz)` [m].
#[pyclass(name = "Box", module = "physics_engine_rs")]
#[derive(Clone)]
struct PyBox {
    inner: BoxShape,
}

#[pymethods]
impl PyBox {
    #[new]
    fn new(half_extents: &Bound<'_, PyAny>) -> PyResult<Self> {
        let h = vec3_from_seq(half_extents)?;
        Ok(Self {
            inner: BoxShape::new(h.x, h.y, h.z),
        })
    }

    #[getter]
    fn half_extents(&self) -> [f64; 3] {
        self.inner.half_extents.to_array()
    }

    fn volume(&self) -> f64 {
        self.inner.volume()
    }

    fn inertia(&self, mass: f64) -> [[f64; 3]; 3] {
        self.inner.inertia(mass)
    }
}

/// Capsule: radius + half-height along body z [m].
#[pyclass(name = "Capsule", module = "physics_engine_rs")]
#[derive(Clone)]
struct PyCapsule {
    inner: Capsule,
}

#[pymethods]
impl PyCapsule {
    #[new]
    fn new(radius: f64, half_height: f64) -> Self {
        Self {
            inner: Capsule::new(radius, half_height),
        }
    }

    fn volume(&self) -> f64 {
        self.inner.volume()
    }

    fn inertia(&self, mass: f64) -> [[f64; 3]; 3] {
        self.inner.inertia(mass)
    }
}

/// Static half-space plane.
#[pyclass(name = "Plane", module = "physics_engine_rs")]
#[derive(Clone)]
struct PyPlane {
    inner: Plane,
}

#[pymethods]
impl PyPlane {
    #[new]
    #[pyo3(signature = (normal, offset=0.0))]
    fn new(normal: &Bound<'_, PyAny>, offset: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Plane::new(vec3_from_seq(normal)?, offset),
        })
    }
}

/// Contact material (`e`, `μ`).
#[pyclass(name = "Material", module = "physics_engine_rs")]
#[derive(Clone)]
struct PyMaterial {
    inner: Material,
}

#[pymethods]
impl PyMaterial {
    #[new]
    #[pyo3(signature = (restitution=0.0, friction=0.5))]
    fn new(restitution: f64, friction: f64) -> Self {
        Self {
            inner: Material::new(restitution, friction),
        }
    }

    #[getter]
    fn restitution(&self) -> f64 {
        self.inner.restitution
    }

    #[getter]
    fn friction(&self) -> f64 {
        self.inner.friction
    }
}

/// Convex polyhedron from points (body frame), optional Minkowski radius [m].
#[pyclass(name = "ConvexHull", module = "physics_engine_rs")]
#[derive(Clone)]
struct PyConvexHull {
    inner: ConvexHull,
}

#[pymethods]
impl PyConvexHull {
    #[new]
    #[pyo3(signature = (points, radius=0.0))]
    fn new(points: &Bound<'_, PyAny>, radius: f64) -> PyResult<Self> {
        let raw: Vec<Vec<f64>> = points.extract()?;
        let mut pts = Vec::with_capacity(raw.len());
        for p in raw {
            if p.len() != 3 {
                return Err(PyValueError::new_err("each point must be [x,y,z]"));
            }
            pts.push(Vec3::new(p[0], p[1], p[2]));
        }
        let h = ConvexHull::from_points_with_radius(&pts, radius)
            .map_err(|e| PyValueError::new_err(format!("convex hull failed: {e:?}")))?;
        Ok(Self { inner: h })
    }

    #[staticmethod]
    #[pyo3(signature = (hx, hy, hz, radius=0.0))]
    fn from_box(hx: f64, hy: f64, hz: f64, radius: f64) -> PyResult<Self> {
        let h = ConvexHull::from_box(hx, hy, hz, radius)
            .map_err(|e| PyValueError::new_err(format!("convex hull failed: {e:?}")))?;
        Ok(Self { inner: h })
    }

    fn volume(&self) -> f64 {
        self.inner.volume()
    }

    fn validate(&self) -> PyResult<()> {
        self.inner
            .validate()
            .map_err(|e| PyValueError::new_err(format!("invalid hull: {e:?}")))
    }

    #[getter]
    fn n_vertices(&self) -> usize {
        self.inner.vertices.len()
    }

    #[getter]
    fn n_faces(&self) -> usize {
        self.inner.faces.len()
    }

    #[getter]
    fn radius(&self) -> f64 {
        self.inner.radius
    }
}

/// Minkowski sum of an inner shape with a sphere of `radius` [m].
#[pyclass(name = "Rounded", module = "physics_engine_rs")]
#[derive(Clone)]
struct PyRounded {
    inner: RoundedShape,
}

#[pymethods]
impl PyRounded {
    #[new]
    fn new(inner: &Bound<'_, PyAny>, radius: f64) -> PyResult<Self> {
        if radius < 0.0 {
            return Err(PyValueError::new_err("radius must be >= 0"));
        }
        Ok(Self {
            inner: RoundedShape::new(shape_from_py(inner)?, radius),
        })
    }

    #[getter]
    fn radius(&self) -> f64 {
        self.inner.radius
    }
}

fn shape_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Shape> {
    if let Ok(s) = obj.extract::<PyRef<PySphere>>() {
        return Ok(Shape::Sphere(s.inner));
    }
    if let Ok(s) = obj.extract::<PyRef<PyBox>>() {
        return Ok(Shape::Box(s.inner));
    }
    if let Ok(s) = obj.extract::<PyRef<PyCapsule>>() {
        return Ok(Shape::Capsule(s.inner));
    }
    if let Ok(s) = obj.extract::<PyRef<PyPlane>>() {
        return Ok(Shape::Plane(s.inner));
    }
    if let Ok(s) = obj.extract::<PyRef<PyConvexHull>>() {
        return Ok(Shape::Convex(s.inner.clone()));
    }
    if let Ok(s) = obj.extract::<PyRef<PyRounded>>() {
        return Ok(Shape::Rounded(s.inner.clone()));
    }
    Err(PyTypeError::new_err(
        "shape must be Sphere|Box|Capsule|Plane|ConvexHull|Rounded",
    ))
}

/// Infinite-mass static collider.
#[pyclass(name = "StaticCollider", module = "physics_engine_rs")]
#[derive(Clone)]
struct PyStaticCollider {
    inner: StaticCollider,
}

#[pymethods]
impl PyStaticCollider {
    #[new]
    #[pyo3(signature = (shape, position=None, orientation=None, material=None))]
    fn new(
        shape: &Bound<'_, PyAny>,
        position: Option<&Bound<'_, PyAny>>,
        orientation: Option<&Bound<'_, PyAny>>,
        material: Option<PyRef<PyMaterial>>,
    ) -> PyResult<Self> {
        let mut c = StaticCollider::new(shape_from_py(shape)?);
        if let Some(p) = position {
            c.position = vec3_from_seq(p)?;
        }
        if let Some(q) = orientation {
            c.orientation = quat_from_seq(q)?;
        }
        if let Some(m) = material {
            c.material = m.inner;
        }
        Ok(Self { inner: c })
    }
}

/// Contact world: semi-implicit Euler + sequential impulses (stage 2).
#[pyclass(name = "ContactWorld", module = "physics_engine_rs")]
struct PyContactWorld {
    inner: ContactWorld,
}

#[pymethods]
impl PyContactWorld {
    #[new]
    #[pyo3(signature = (world, shapes, statics=None, materials=None, iterations=20, restitution_threshold=0.2, slop=1e-4, beta=0.2, warm_start=true, solver="pgs", allow_sleep=false, broad_phase="sap"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        world: &Bound<'_, PyAny>,
        shapes: &Bound<'_, PyAny>,
        statics: Option<&Bound<'_, PyAny>>,
        materials: Option<&Bound<'_, PyAny>>,
        iterations: usize,
        restitution_threshold: f64,
        slop: f64,
        beta: f64,
        warm_start: bool,
        solver: &str,
        allow_sleep: bool,
        broad_phase: &str,
    ) -> PyResult<Self> {
        // Accept either a PyWorld or rebuild from bodies+gens is not supported —
        // require PyWorld and *move* its inner state by cloning via pack/rebuild is heavy.
        // Instead: take ownership by extracting bodies is not available; we clone World
        // by re-reading state. Simplest path: require the caller pass a freshly built
        // World and we take it via a dedicated constructor that consumes generators.
        //
        // Practical approach matching oracle tests: accept PyWorld and deep-copy SoA +
        // rebuild with the same generators by requiring World was just constructed.
        let mut py_world: PyRefMut<PyWorld> = world.extract()?;
        // Move the world out by replacing with an empty placeholder.
        let taken = std::mem::replace(&mut py_world.inner, World::new(Vec::new(), Vec::new()));

        let shape_list = shapes.downcast::<PyList>()?;
        let mut shape_vec = Vec::with_capacity(shape_list.len());
        for item in shape_list.iter() {
            shape_vec.push(shape_from_py(&item)?);
        }

        let mut static_vec = Vec::new();
        if let Some(st) = statics {
            let list = st.downcast::<PyList>()?;
            for item in list.iter() {
                let s: PyRef<PyStaticCollider> = item.extract()?;
                static_vec.push(s.inner.clone());
            }
        }

        let mats = if let Some(m) = materials {
            let list = m.downcast::<PyList>()?;
            let mut out = Vec::with_capacity(list.len());
            for item in list.iter() {
                let mat: PyRef<PyMaterial> = item.extract()?;
                out.push(mat.inner);
            }
            Some(out)
        } else {
            None
        };

        let mut cw = ContactWorld::new(taken, shape_vec, static_vec, mats);
        cw.iterations = iterations;
        cw.restitution_threshold = restitution_threshold;
        cw.slop = slop;
        cw.beta = beta;
        cw.warm_start = warm_start;
        cw.solver_mode = SolverMode::parse(solver).ok_or_else(|| {
            PyValueError::new_err(format!("unknown solver {solver:?}; expected pgs|soft_step"))
        })?;
        cw.set_allow_sleep(allow_sleep);
        cw.broad_phase = match broad_phase.trim().to_ascii_lowercase().as_str() {
            "sap" | "sweep_and_prune" => BroadPhaseKind::Sap,
            "brute" | "brute_force" => BroadPhaseKind::Brute,
            "tree" | "dynamic_tree" | "aabb_tree" => BroadPhaseKind::DynamicTree,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown broad_phase {other:?}; expected sap|brute|tree"
                )));
            }
        };
        if cw.solver_mode == SolverMode::SoftStep {
            cw.soft_params = SoftStepParams::default();
        }
        Ok(Self { inner: cw })
    }

    #[getter]
    fn t(&self) -> f64 {
        self.inner.t()
    }

    #[getter]
    fn slop(&self) -> f64 {
        self.inner.slop
    }

    #[getter]
    fn solver(&self) -> &'static str {
        self.inner.solver_mode.as_str()
    }

    #[setter]
    fn set_solver(&mut self, name: &str) -> PyResult<()> {
        self.inner.solver_mode = SolverMode::parse(name).ok_or_else(|| {
            PyValueError::new_err(format!("unknown solver {name:?}; expected pgs|soft_step"))
        })?;
        Ok(())
    }

    #[getter]
    fn allow_sleep(&self) -> bool {
        self.inner.sleep.enabled
    }

    #[setter]
    fn set_allow_sleep(&mut self, allow: bool) {
        self.inner.set_allow_sleep(allow);
    }

    /// Per-body awake flags (length N). Sleeping bodies are `False`.
    #[getter]
    fn awake<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.inner.body_awake().to_vec())
    }

    fn wake_body(&mut self, i: usize) -> PyResult<()> {
        if i >= self.inner.world.n {
            return Err(PyValueError::new_err("body index out of range"));
        }
        self.inner.wake_body(i);
        Ok(())
    }

    /// Mark body `i` as a CCD bullet (TOI vs statics).
    fn set_bullet(&mut self, i: usize, bullet: bool) -> PyResult<()> {
        if i >= self.inner.world.n {
            return Err(PyValueError::new_err("body index out of range"));
        }
        self.inner.set_bullet(i, bullet);
        Ok(())
    }

    /// Enable/disable speculative contacts and bullet TOI.
    #[pyo3(signature = (speculative=true, toi=true, speculative_distance=0.02))]
    fn set_ccd(&mut self, speculative: bool, toi: bool, speculative_distance: f64) {
        self.inner.ccd = CcdParams {
            speculative,
            speculative_distance,
            toi,
            ..CcdParams::default()
        };
        // Preserve enable flags even though Default has them false.
        self.inner.ccd.speculative = speculative;
        self.inner.ccd.toi = toi;
        self.inner.ccd.speculative_distance = speculative_distance;
    }

    /// Add a distance joint. `body_b=None` ⇒ world. Returns joint id.
    #[pyo3(signature = (body_a, body_b, local_a, local_b, length, hertz=60.0, zeta=0.7, break_force=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn add_distance_joint(
        &mut self,
        body_a: usize,
        body_b: Option<usize>,
        local_a: [f64; 3],
        local_b: [f64; 3],
        length: f64,
        hertz: f64,
        zeta: f64,
        break_force: f64,
    ) -> usize {
        let b = body_b.unwrap_or(WORLD_BODY);
        let mut j = Joint::distance(
            body_a,
            b,
            Vec3::from_slice(&local_a),
            Vec3::from_slice(&local_b),
            length,
        );
        j.soft = JointSoftParams { hertz, zeta };
        if break_force > 0.0 {
            j.break_threshold = JointBreak {
                force: break_force,
                torque: 0.0,
            };
        }
        self.inner.add_joint(j)
    }

    /// Add a revolute joint with optional limits / motor.
    #[pyo3(signature = (body_a, body_b, local_a, local_b, axis_a, axis_b, ref_a, ref_b, hertz=60.0, zeta=0.7, limit=None, motor_speed=None, motor_max_torque=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn add_revolute_joint(
        &mut self,
        body_a: usize,
        body_b: Option<usize>,
        local_a: [f64; 3],
        local_b: [f64; 3],
        axis_a: [f64; 3],
        axis_b: [f64; 3],
        ref_a: [f64; 3],
        ref_b: [f64; 3],
        hertz: f64,
        zeta: f64,
        limit: Option<(f64, f64)>,
        motor_speed: Option<f64>,
        motor_max_torque: f64,
    ) -> usize {
        let b = body_b.unwrap_or(WORLD_BODY);
        let mut j = Joint::revolute(
            body_a,
            b,
            Vec3::from_slice(&local_a),
            Vec3::from_slice(&local_b),
            Vec3::from_slice(&axis_a),
            Vec3::from_slice(&axis_b),
            Vec3::from_slice(&ref_a),
            Vec3::from_slice(&ref_b),
        );
        j.soft = JointSoftParams { hertz, zeta };
        if let Some((lo, hi)) = limit {
            j.limit = JointLimit {
                enabled: true,
                lower: lo,
                upper: hi,
            };
        }
        if let Some(speed) = motor_speed {
            j.motor = JointMotor {
                enabled: true,
                target_speed: speed,
                max_force: motor_max_torque,
            };
        }
        self.inner.add_joint(j)
    }

    /// Joint broken flag.
    fn joint_broken(&self, id: usize) -> bool {
        self.inner.joint_broken(id)
    }

    /// Joint reaction force [N] as length-3 array.
    fn joint_reaction_force<'py>(&self, py: Python<'py>, id: usize) -> Bound<'py, PyArray1<f64>> {
        let f = self.inner.joint_reaction_force(id);
        PyArray1::from_vec(py, f.to_array().to_vec())
    }

    /// Underlying dynamic world state accessors (positions).
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.world.n;
        let mut flat = Vec::with_capacity(n * 3);
        for p in &self.inner.world.state.x {
            flat.extend_from_slice(&p.to_array());
        }
        PyArray1::from_vec(py, flat)
            .reshape([n, 3])
            .expect("reshape x")
    }

    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.world.n;
        let mut flat = Vec::with_capacity(n * 3);
        for v in &self.inner.world.state.v {
            flat.extend_from_slice(&v.to_array());
        }
        PyArray1::from_vec(py, flat)
            .reshape([n, 3])
            .expect("reshape v")
    }

    #[getter]
    #[allow(non_snake_case)]
    fn L<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.world.n;
        let mut flat = Vec::with_capacity(n * 3);
        for l in &self.inner.world.state.l {
            flat.extend_from_slice(&l.to_array());
        }
        PyArray1::from_vec(py, flat)
            .reshape([n, 3])
            .expect("reshape L")
    }

    #[getter]
    fn q<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.world.n;
        let mut flat = Vec::with_capacity(n * 4);
        for q in &self.inner.world.state.q {
            flat.extend_from_slice(&q.to_array());
        }
        PyArray1::from_vec(py, flat)
            .reshape([n, 4])
            .expect("reshape q")
    }

    fn pack<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.world.state.pack())
    }

    fn state_hash(&self) -> u64 {
        state_hash64(&self.inner.world.state)
    }

    fn kinetic_energy(&self) -> f64 {
        kinetic_energy(&self.inner.world, &self.inner.world.state)
    }

    fn linear_momentum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let p = linear_momentum(&self.inner.world, &self.inner.world.state);
        PyArray1::from_vec(py, p.to_array().to_vec())
    }

    /// One contact step; **releases the GIL**.
    fn step(&mut self, py: Python<'_>, dt: f64) {
        py.allow_threads(|| {
            self.inner.step(dt);
        });
    }

    /// Fixed-`dt` run to `t_end` [s]; **releases the GIL**.
    fn run(&mut self, py: Python<'_>, t_end: f64, dt: f64) {
        py.allow_threads(|| {
            self.inner.run_for(t_end, dt);
        });
    }
}

#[pymodule]
fn physics_engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRigidBody>()?;
    m.add_class::<PyWorld>()?;
    m.add_class::<PyUniformGravity>()?;
    m.add_class::<PyNewtonianGravity>()?;
    m.add_class::<PyLinearDrag>()?;
    m.add_class::<PyQuadraticDrag>()?;
    m.add_class::<PySpring>()?;
    m.add_class::<PyConstantTorque>()?;
    m.add_class::<PySphere>()?;
    m.add_class::<PyBox>()?;
    m.add_class::<PyCapsule>()?;
    m.add_class::<PyPlane>()?;
    m.add_class::<PyConvexHull>()?;
    m.add_class::<PyRounded>()?;
    m.add_class::<PyMaterial>()?;
    m.add_class::<PyStaticCollider>()?;
    m.add_class::<PyContactWorld>()?;
    m.add_function(wrap_pyfunction!(inertia_solid_sphere, m)?)?;
    m.add_function(wrap_pyfunction!(inertia_solid_box, m)?)?;
    m.add_function(wrap_pyfunction!(inertia_solid_cylinder, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_determinism_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_collision_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_soft_step_collision_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_joints_hash, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_convex_hash, m)?)?;
    aero_hydro::register(m)?;
    space::register(m)?;
    blast_ballistics::register(m)?;
    soft::register(m)?;
    fluid::register(m)?;
    m.add("G", G)?;
    m.add("STANDARD_GRAVITY", STANDARD_GRAVITY)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
