//! Heat conduction on particle / edge / tet meshes and thermal expansion coupling.
//!
//! # Conduction
//!
//! Lumped-capacity graph FEM / discrete Laplacian on an undirected edge set:
//! `C_i dT_i/dt = Σ_j κ_ij (T_j − T_i) + Q_i + h A_i (T_∞ − T_i)` with
//! `C_i = ρ c V_i` and edge conductance `κ_ij = k A_ij / L_ij` (1-D rod) or
//! assembled from linear tetrahedron conductivity matrices.
//!
//! Time integrators:
//! - **Explicit Euler** (CFL-limited)
//! - **Backward Euler** (θ = 1) — unconditionally stable
//! - **Crank–Nicolson** (θ = ½) — 2nd-order in time
//!
//! Implicit systems use a deterministic sparse solve: Thomas (tridiagonal 1-D)
//! or conjugate gradient (fixed iteration order, no `HashMap`, no `mul_add`).
//!
//! # Boundaries
//!
//! - Dirichlet: hold `T` (`fixed[i] = true`)
//! - Neumann: nodal power `neumann_power[i]` [W] into the body
//! - Insulated: default (no flux)
//! - Robin / convection: `robin_ha[i] = h A`, `robin_tinf[i]`
//!
//! # Analytic 1-D (Dirichlet ends)
//!
//! `T(x,t) = T₀ sin(π x / L) exp(−κ (π/L)² t)`, diffusivity `κ = k/(ρ c)`.
//!
//! Semi-infinite step: `T(x,t) = T_s + (T_i − T_s) erfc(x / (2 √(κ t)))`.
//!
//! # Expansion
//!
//! Free thermal expansion: `ΔL = α L₀ ΔT` (linear), applied as a rest-length
//! update on distance constraints; tet/volume rests scale isotropically.

pub mod sparse;

use pe_math::libm_ops;
use pe_math::Vec3;

use crate::constraint::{tetra_volume, DistanceConstraint, TetraFemConstraint, VolumeConstraint};

use self::sparse::{cg_solve, thomas_solve, CsrMatrix};

/// Thermal material properties (SI).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThermalProps {
    /// Density ρ [kg/m³].
    pub density: f64,
    /// Specific heat c_p [J/(kg·K)].
    pub specific_heat: f64,
    /// Thermal conductivity k [W/(m·K)].
    pub conductivity: f64,
    /// Linear thermal expansion coefficient α [1/K].
    pub expansion: f64,
}

impl ThermalProps {
    /// Thermal diffusivity κ = k / (ρ c) [m²/s].
    pub fn diffusivity(&self) -> f64 {
        let den = self.density * self.specific_heat;
        if den <= 0.0 {
            return 0.0;
        }
        self.conductivity / den
    }
}

/// Undirected conduction edge (sorted endpoints for determinism).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThermalEdge {
    /// Lower particle index.
    pub i: usize,
    /// Higher particle index.
    pub j: usize,
}

impl ThermalEdge {
    /// Construct with sorted endpoints.
    pub fn new(a: usize, b: usize) -> Self {
        if a <= b {
            Self { i: a, j: b }
        } else {
            Self { i: b, j: a }
        }
    }
}

/// Implicit / explicit time scheme for conduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThermalScheme {
    /// Forward Euler (CFL-limited).
    Explicit,
    /// Backward Euler (θ = 1).
    BackwardEuler,
    /// Crank–Nicolson (θ = ½).
    CrankNicolson,
}

impl ThermalScheme {
    /// θ-method weight (`None` for explicit).
    pub fn theta(self) -> Option<f64> {
        match self {
            Self::Explicit => None,
            Self::BackwardEuler => Some(1.0),
            Self::CrankNicolson => Some(0.5),
        }
    }
}

/// Lumped thermal mesh on particles.
#[derive(Clone, Debug)]
pub struct ThermalMesh {
    /// Temperature at each particle [K] (or °C offset — differences only matter
    /// for conduction; absolute used for expansion relative to `t_ref`).
    pub temperature: Vec<f64>,
    /// Lumped thermal capacity C_i = ρ c V_i [J/K].
    pub capacity: Vec<f64>,
    /// Conduction edges (fixed order — no HashMap).
    pub edges: Vec<ThermalEdge>,
    /// Conductance κ_ij [W/K] parallel to `edges`.
    pub conductance: Vec<f64>,
    /// Reference temperature for expansion [K].
    pub t_ref: f64,
    /// Material props (α, …).
    pub props: ThermalProps,
    /// Neumann nodal power into the body [W] (length = n; default 0).
    pub neumann_power: Vec<f64>,
    /// Robin `h A` [W/K] (length = n; default 0).
    pub robin_ha: Vec<f64>,
    /// Robin ambient temperature [K] (length = n).
    pub robin_tinf: Vec<f64>,
}

impl ThermalMesh {
    fn empty_bc(n: usize, t_ref: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (vec![0.0; n], vec![0.0; n], vec![t_ref; n])
    }

    /// Build a uniform 1-D rod of `n` nodes on `[0, L]` with equal spacing,
    /// cross-section `area`, and lumped capacities `ρ c A L / n`.
    pub fn rod_1d(n: usize, length: f64, area: f64, props: ThermalProps, t_ref: f64) -> Self {
        assert!(n >= 2 && length > 0.0 && area > 0.0);
        let dx = length / (n - 1) as f64;
        let c_node = props.density * props.specific_heat * area * length / n as f64;
        let g = props.conductivity * area / dx; // W/K
        let mut edges = Vec::with_capacity(n - 1);
        let mut conductance = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            edges.push(ThermalEdge::new(i, i + 1));
            conductance.push(g);
        }
        let (neumann_power, robin_ha, robin_tinf) = Self::empty_bc(n, t_ref);
        Self {
            temperature: vec![t_ref; n],
            capacity: vec![c_node; n],
            edges,
            conductance,
            t_ref,
            props,
            neumann_power,
            robin_ha,
            robin_tinf,
        }
    }

    /// Assemble a lumped thermal mesh from linear tetrahedra.
    ///
    /// Capacity: equal share of `ρ c |V|` on each of the four corners.
    /// Conductance: edge values `κ_ij = −K_ij` from the element conductivity
    /// matrix `K_ij = k |V| (∇N_i · ∇N_j)`.
    pub fn from_tets(
        positions: &[Vec3],
        tets: &[[usize; 4]],
        props: ThermalProps,
        t_ref: f64,
    ) -> Self {
        let n = positions.len();
        let mut capacity = vec![0.0; n];
        // Accumulate unique edges with summed conductance via sorted COO merge.
        let mut edge_i = Vec::new();
        let mut edge_j = Vec::new();
        let mut edge_g = Vec::new();
        let k = props.conductivity;
        let rho_c = props.density * props.specific_heat;
        for tet in tets {
            let x0 = positions[tet[0]];
            let x1 = positions[tet[1]];
            let x2 = positions[tet[2]];
            let x3 = positions[tet[3]];
            let vol = tetra_volume(x0, x1, x2, x3).abs();
            if vol < 1e-30 {
                continue;
            }
            let grads = match tet_shape_grads(x0, x1, x2, x3) {
                Some(g) => g,
                None => continue,
            };
            let c_corner = rho_c * vol * 0.25;
            for &idx in tet.iter() {
                capacity[idx] += c_corner;
            }
            // Element conductivity K_ab = k V (∇Na · ∇Nb); off-diag → edge κ.
            for a in 0..4 {
                for b in (a + 1)..4 {
                    let kij = k * vol * grads[a].dot(grads[b]);
                    let g = -kij; // conductance ≥ 0 when K_ab ≤ 0
                    if g <= 0.0 {
                        // Still assemble absolute coupling if gradient product > 0
                        // (rare geometry); use |K| so energy dissipates.
                        let g_abs = libm_ops::fabs(kij);
                        if g_abs > 0.0 {
                            let e = ThermalEdge::new(tet[a], tet[b]);
                            edge_i.push(e.i);
                            edge_j.push(e.j);
                            edge_g.push(g_abs);
                        }
                    } else {
                        let e = ThermalEdge::new(tet[a], tet[b]);
                        edge_i.push(e.i);
                        edge_j.push(e.j);
                        edge_g.push(g);
                    }
                }
            }
        }
        // Merge duplicate edges.
        let mut triples: Vec<(usize, usize, f64)> = edge_i
            .iter()
            .zip(edge_j.iter())
            .zip(edge_g.iter())
            .map(|((&i, &j), &g)| (i, j, g))
            .collect();
        triples.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut edges = Vec::new();
        let mut conductance = Vec::new();
        let mut i = 0usize;
        while i < triples.len() {
            let (ei, ej, mut g) = triples[i];
            i += 1;
            while i < triples.len() && triples[i].0 == ei && triples[i].1 == ej {
                g += triples[i].2;
                i += 1;
            }
            edges.push(ThermalEdge { i: ei, j: ej });
            conductance.push(g);
        }
        let (neumann_power, robin_ha, robin_tinf) = Self::empty_bc(n, t_ref);
        Self {
            temperature: vec![t_ref; n],
            capacity,
            edges,
            conductance,
            t_ref,
            props,
            neumann_power,
            robin_ha,
            robin_tinf,
        }
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.temperature.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.temperature.is_empty()
    }

    /// Total thermal energy Σ C_i T_i [J] (relative to 0 K).
    pub fn total_energy(&self) -> f64 {
        self.temperature
            .iter()
            .zip(self.capacity.iter())
            .map(|(t, c)| t * c)
            .sum()
    }

    /// True when edges form a single path `0—1—…—n-1` (enables Thomas).
    pub fn is_path_tridiagonal(&self) -> bool {
        let n = self.len();
        if n < 2 || self.edges.len() != n - 1 {
            return false;
        }
        for (k, e) in self.edges.iter().enumerate() {
            if e.i != k || e.j != k + 1 {
                return false;
            }
        }
        true
    }

    /// Explicit Euler conduction step. `fixed[i]=true` ⇒ Dirichlet (hold T).
    ///
    /// Returns `false` if a non-finite temperature appears.
    pub fn step_explicit(&mut self, dt: f64, fixed: &[bool]) -> bool {
        let n = self.len();
        if dt <= 0.0 || n == 0 {
            return true;
        }
        let mut d_e = vec![0.0; n]; // heat flow into node [W]
        for (e, &g) in self.edges.iter().zip(self.conductance.iter()) {
            let ti = self.temperature[e.i];
            let tj = self.temperature[e.j];
            let q = g * (tj - ti);
            d_e[e.i] += q;
            d_e[e.j] -= q;
        }
        for i in 0..n {
            if fixed.get(i).copied().unwrap_or(false) {
                continue;
            }
            let c = self.capacity[i];
            if c <= 0.0 {
                continue;
            }
            let robin = self.robin_ha[i] * (self.robin_tinf[i] - self.temperature[i]);
            d_e[i] += self.neumann_power[i] + robin;
            self.temperature[i] += dt * d_e[i] / c;
            if !self.temperature[i].is_finite() {
                return false;
            }
        }
        true
    }

    /// Suggested stable explicit dt: `dt < min_i C_i / (Σ_j κ_ij + h A_i)`.
    pub fn max_stable_dt(&self) -> f64 {
        let n = self.len();
        let mut deg = vec![0.0; n];
        for (e, &g) in self.edges.iter().zip(self.conductance.iter()) {
            deg[e.i] += g;
            deg[e.j] += g;
        }
        for i in 0..n {
            deg[i] += self.robin_ha[i];
        }
        let mut dt = f64::INFINITY;
        for i in 0..n {
            if deg[i] > 0.0 && self.capacity[i] > 0.0 {
                dt = dt.min(self.capacity[i] / deg[i]);
            }
        }
        if !dt.is_finite() {
            1.0
        } else {
            0.4 * dt
        }
    }

    /// θ-method implicit step. `theta = 1` → BE, `theta = 0.5` → CN.
    ///
    /// Uses Thomas when the mesh is a 1-D path; otherwise CG with a fixed
    /// iteration count of `n` for bit-stable deterministic solves.
    pub fn step_theta(&mut self, dt: f64, theta: f64, fixed: &[bool]) -> bool {
        self.step_theta_with(dt, theta, fixed, true)
    }

    /// Implicit step with optional fixed-iteration CG (hash / determinism).
    pub fn step_theta_with(
        &mut self,
        dt: f64,
        theta: f64,
        fixed: &[bool],
        fixed_cg_iters: bool,
    ) -> bool {
        let n = self.len();
        if dt <= 0.0 || n == 0 {
            return true;
        }
        let theta = theta.clamp(0.0, 1.0);
        if self.is_path_tridiagonal() {
            return self.step_theta_thomas(dt, theta, fixed);
        }
        self.step_theta_cg(dt, theta, fixed, fixed_cg_iters)
    }

    /// Backward Euler step.
    pub fn step_backward_euler(&mut self, dt: f64, fixed: &[bool]) -> bool {
        self.step_theta(dt, 1.0, fixed)
    }

    /// Crank–Nicolson step.
    pub fn step_crank_nicolson(&mut self, dt: f64, fixed: &[bool]) -> bool {
        self.step_theta(dt, 0.5, fixed)
    }

    /// Dispatch by [`ThermalScheme`].
    pub fn step_scheme(&mut self, dt: f64, scheme: ThermalScheme, fixed: &[bool]) -> bool {
        match scheme {
            ThermalScheme::Explicit => self.step_explicit(dt, fixed),
            ThermalScheme::BackwardEuler => self.step_backward_euler(dt, fixed),
            ThermalScheme::CrankNicolson => self.step_crank_nicolson(dt, fixed),
        }
    }

    fn apply_laplacian(&self, t: &[f64], out: &mut [f64]) {
        let n = self.len();
        for i in 0..n {
            out[i] = 0.0;
        }
        for (e, &g) in self.edges.iter().zip(self.conductance.iter()) {
            let ti = t[e.i];
            let tj = t[e.j];
            let q = g * (tj - ti);
            out[e.i] += q;
            out[e.j] -= q;
        }
        // Laplacian form L T with (L T)_i = Σ κ (T_i − T_j) = −(heat into i from edges)
        // Convert out from "heat into" to L T:
        for i in 0..n {
            out[i] = -out[i] + self.robin_ha[i] * t[i];
        }
    }

    fn step_theta_thomas(&mut self, dt: f64, theta: f64, fixed: &[bool]) -> bool {
        let n = self.len();
        let mut l_lower = vec![0.0; n];
        let mut l_diag = vec![0.0; n];
        let mut l_upper = vec![0.0; n];
        for (e, &g) in self.edges.iter().zip(self.conductance.iter()) {
            l_diag[e.i] += g;
            l_diag[e.j] += g;
            l_upper[e.i] -= g;
            l_lower[e.j] -= g;
        }
        for i in 0..n {
            l_diag[i] += self.robin_ha[i];
        }
        let t = &self.temperature;
        let mut lower = vec![0.0; n];
        let mut diag = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            let c = self.capacity[i];
            let lt_i = l_diag[i] * t[i]
                + (if i > 0 { l_lower[i] * t[i - 1] } else { 0.0 })
                + (if i + 1 < n {
                    l_upper[i] * t[i + 1]
                } else {
                    0.0
                });
            rhs[i] = c * t[i] - (1.0 - theta) * dt * lt_i
                + dt * (self.neumann_power[i] + self.robin_ha[i] * self.robin_tinf[i]);
            lower[i] = theta * dt * l_lower[i];
            upper[i] = theta * dt * l_upper[i];
            diag[i] = c + theta * dt * l_diag[i];
        }
        for i in 0..n {
            if fixed.get(i).copied().unwrap_or(false) {
                lower[i] = 0.0;
                upper[i] = 0.0;
                diag[i] = 1.0;
                rhs[i] = t[i];
            }
        }
        for i in 0..n {
            if fixed.get(i).copied().unwrap_or(false) {
                continue;
            }
            if i > 0 && fixed.get(i - 1).copied().unwrap_or(false) {
                rhs[i] -= lower[i] * t[i - 1];
                lower[i] = 0.0;
            }
            if i + 1 < n && fixed.get(i + 1).copied().unwrap_or(false) {
                rhs[i] -= upper[i] * t[i + 1];
                upper[i] = 0.0;
            }
        }
        if !thomas_solve(&lower, &diag, &upper, &mut rhs) {
            return false;
        }
        for i in 0..n {
            if fixed.get(i).copied().unwrap_or(false) {
                continue;
            }
            self.temperature[i] = rhs[i];
            if !self.temperature[i].is_finite() {
                return false;
            }
        }
        true
    }

    fn step_theta_cg(&mut self, dt: f64, theta: f64, fixed: &[bool], fixed_cg_iters: bool) -> bool {
        let n = self.len();
        let t = self.temperature.clone();
        let mut lt = vec![0.0; n];
        self.apply_laplacian(&t, &mut lt);

        // COO for A = C + θ dt L
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(self.capacity[i] + theta * dt * self.robin_ha[i]);
        }
        for (e, &g) in self.edges.iter().zip(self.conductance.iter()) {
            // L_ii += g, L_jj += g, L_ij = L_ji = -g
            // Add θ dt L contributions
            let s = theta * dt * g;
            vals[e.i] += s;
            vals[e.j] += s;
            rows.push(e.i);
            cols.push(e.j);
            vals.push(-s);
            rows.push(e.j);
            cols.push(e.i);
            vals.push(-s);
        }

        let mut rhs = vec![0.0; n];
        for i in 0..n {
            rhs[i] = self.capacity[i] * t[i] - (1.0 - theta) * dt * lt[i]
                + dt * (self.neumann_power[i] + self.robin_ha[i] * self.robin_tinf[i]);
        }

        // Dirichlet: replace row with identity; move known couplings out of free rows.
        // First, collect L contributions from fixed nodes into rhs for free nodes.
        // Simpler approach: after building A, zero row i / set A_ii=1 / rhs=T_i for fixed,
        // and for free rows zero columns j that are fixed (moving A_ij T_j to rhs).
        let a_tmp = CsrMatrix::from_coo(n, &rows, &cols, &vals);
        // Rebuild COO with Dirichlet handling.
        let mut rows2 = Vec::new();
        let mut cols2 = Vec::new();
        let mut vals2 = Vec::new();
        for i in 0..n {
            if fixed.get(i).copied().unwrap_or(false) {
                rows2.push(i);
                cols2.push(i);
                vals2.push(1.0);
                rhs[i] = t[i];
                continue;
            }
            for k in a_tmp.row_ptr[i]..a_tmp.row_ptr[i + 1] {
                let j = a_tmp.col_idx[k];
                let v = a_tmp.vals[k];
                if fixed.get(j).copied().unwrap_or(false) {
                    rhs[i] -= v * t[j];
                } else {
                    rows2.push(i);
                    cols2.push(j);
                    vals2.push(v);
                }
            }
        }
        let a = CsrMatrix::from_coo(n, &rows2, &cols2, &vals2);
        let mut x = t.clone();
        let ok = if fixed_cg_iters {
            cg_solve(&a, &rhs, &mut x, 1e-15, n, Some(n.max(1)))
        } else {
            cg_solve(&a, &rhs, &mut x, 1e-12, (n * 4).max(32), None)
        };
        if !ok {
            return false;
        }
        for i in 0..n {
            if fixed.get(i).copied().unwrap_or(false) {
                continue;
            }
            self.temperature[i] = x[i];
            if !self.temperature[i].is_finite() {
                return false;
            }
        }
        true
    }
}

/// Shape-function gradients for a linear tet (∇N₀…∇N₃).
fn tet_shape_grads(x0: Vec3, x1: Vec3, x2: Vec3, x3: Vec3) -> Option<[Vec3; 4]> {
    let col0 = x1 - x0;
    let col1 = x2 - x0;
    let col2 = x3 - x0;
    let dm = [
        [col0.x, col1.x, col2.x],
        [col0.y, col1.y, col2.y],
        [col0.z, col1.z, col2.z],
    ];
    let inv = crate::constraint::invert3(&dm)?;
    // Columns of Dm^{-T} = rows of Dm^{-1} are ∇N1, ∇N2, ∇N3.
    let g1 = Vec3::new(inv[0][0], inv[0][1], inv[0][2]);
    let g2 = Vec3::new(inv[1][0], inv[1][1], inv[1][2]);
    let g3 = Vec3::new(inv[2][0], inv[2][1], inv[2][2]);
    let g0 = -(g1 + g2 + g3);
    Some([g0, g1, g2, g3])
}

/// Analytic Fourier fundamental mode for a rod with T(0)=T(L)=0:
/// `T(x,t) = T0 sin(π x/L) exp(−κ (π/L)² t)`.
pub fn rod_fourier_mode1(x: f64, t: f64, length: f64, t0: f64, diffusivity: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let k = pi / length;
    t0 * libm_ops::sin(k * x) * libm_ops::exp(-diffusivity * k * k * t)
}

/// Initialize rod temperatures to the first Fourier sine mode (ends ≈ 0).
pub fn init_rod_sine_mode(mesh: &mut ThermalMesh, length: f64, t0: f64) {
    let n = mesh.len();
    let dx = length / (n - 1) as f64;
    for i in 0..n {
        let x = i as f64 * dx;
        mesh.temperature[i] = t0 * libm_ops::sin(core::f64::consts::PI * x / length);
    }
}

/// Semi-infinite solid, surface step to `t_s` at t=0⁺ (Carslaw & Jaeger / Incropera):
/// `T(x,t) = T_i + (T_s − T_i) erfc( x / (2 √(κ t)) )`
/// (equivalently `T_s + (T_i − T_s) erf(η)`).
pub fn semi_infinite_erfc(x: f64, t: f64, t_i: f64, t_s: f64, diffusivity: f64) -> f64 {
    if t <= 0.0 || diffusivity <= 0.0 {
        return if x <= 0.0 { t_s } else { t_i };
    }
    let eta = x / (2.0 * libm_ops::sqrt(diffusivity * t));
    t_i + (t_s - t_i) * libm_ops::erfc(eta)
}

/// Lumped Robin (Bi → 0) cooling: `T(t) = T_∞ + (T₀ − T_∞) exp(−t/τ)`,
/// `τ = ρ c V / (h A) = C / (h A)`.
pub fn robin_lumped_exp(t: f64, t0: f64, t_inf: f64, tau: f64) -> f64 {
    if tau <= 0.0 {
        return t_inf;
    }
    t_inf + (t0 - t_inf) * libm_ops::exp(-t / tau)
}

/// Free thermal expansion of a rest length: `L(T) = L₀ (1 + α (T − T_ref))`.
#[inline]
pub fn expanded_rest_length(l0: f64, alpha: f64, t: f64, t_ref: f64) -> f64 {
    l0 * (1.0 + alpha * (t - t_ref))
}

/// `ΔL = α L₀ ΔT` (oracle identity).
#[inline]
pub fn free_expansion_delta_l(l0: f64, alpha: f64, d_t: f64) -> f64 {
    alpha * l0 * d_t
}

/// Clamped-bar thermal stress (small-strain): `σ = −E α ΔT` (compression when heated).
#[inline]
pub fn clamped_thermal_stress(e: f64, alpha: f64, d_t: f64) -> f64 {
    -e * alpha * d_t
}

/// Apply thermal expansion to distance-constraint rest lengths using the
/// average endpoint temperature. Updates rest lengths in place; returns the
/// mean `|ΔL|` applied.
pub fn apply_thermal_expansion_distances(
    constraints: &mut [DistanceConstraint],
    mesh: &ThermalMesh,
) -> f64 {
    let alpha = mesh.props.expansion;
    let t_ref = mesh.t_ref;
    let mut sum = 0.0;
    let mut n = 0usize;
    for c in constraints.iter_mut() {
        if c.broken {
            continue;
        }
        let ti = mesh.temperature.get(c.i).copied().unwrap_or(t_ref);
        let tj = mesh.temperature.get(c.j).copied().unwrap_or(t_ref);
        let t_avg = 0.5 * (ti + tj);
        let l_new = expanded_rest_length(c.rest_length, alpha, t_avg, t_ref);
        sum += libm_ops::fabs(l_new - c.rest_length);
        c.rest_length = l_new;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// Expand rest lengths from a cold rest-length snapshot (preferred coupling API).
pub fn apply_thermal_expansion_from_cold(
    constraints: &mut [DistanceConstraint],
    cold_rest: &[f64],
    mesh: &ThermalMesh,
) {
    let alpha = mesh.props.expansion;
    let t_ref = mesh.t_ref;
    for (k, c) in constraints.iter_mut().enumerate() {
        if c.broken {
            continue;
        }
        let l0 = cold_rest.get(k).copied().unwrap_or(c.rest_length);
        let ti = mesh.temperature.get(c.i).copied().unwrap_or(t_ref);
        let tj = mesh.temperature.get(c.j).copied().unwrap_or(t_ref);
        let t_avg = 0.5 * (ti + tj);
        c.rest_length = expanded_rest_length(l0, alpha, t_avg, t_ref);
    }
}

/// Isotropic volume expansion `V(T) = V₀ (1 + α ΔT)³` from cold snapshot.
pub fn apply_thermal_expansion_volumes_from_cold(
    volumes: &mut [VolumeConstraint],
    cold_vol: &[f64],
    mesh: &ThermalMesh,
) {
    let alpha = mesh.props.expansion;
    let t_ref = mesh.t_ref;
    for (k, c) in volumes.iter_mut().enumerate() {
        let v0 = cold_vol.get(k).copied().unwrap_or(c.rest_volume);
        let mut t_avg = 0.0;
        for &idx in &c.indices {
            t_avg += mesh.temperature.get(idx).copied().unwrap_or(t_ref);
        }
        t_avg *= 0.25;
        let scale = 1.0 + alpha * (t_avg - t_ref);
        c.rest_volume = v0 * scale * scale * scale;
    }
}

/// Isotropic tet rest scaling: `Dm_inv ← Dm_inv / s`, `V₀ ← V₀ s³`, `s = 1+αΔT`.
pub fn apply_thermal_expansion_tets_from_cold(
    tets: &mut [TetraFemConstraint],
    cold_dm_inv: &[[[f64; 3]; 3]],
    cold_vol: &[f64],
    mesh: &ThermalMesh,
) {
    let alpha = mesh.props.expansion;
    let t_ref = mesh.t_ref;
    for (k, c) in tets.iter_mut().enumerate() {
        if c.broken {
            continue;
        }
        let mut t_avg = 0.0;
        for &idx in &c.indices {
            t_avg += mesh.temperature.get(idx).copied().unwrap_or(t_ref);
        }
        t_avg *= 0.25;
        let s = 1.0 + alpha * (t_avg - t_ref);
        if s.abs() < 1e-15 {
            continue;
        }
        let inv_s = 1.0 / s;
        let dm0 = cold_dm_inv.get(k).copied().unwrap_or(c.dm_inv);
        for r in 0..3 {
            for col in 0..3 {
                c.dm_inv[r][col] = dm0[r][col] * inv_s;
            }
        }
        let v0 = cold_vol.get(k).copied().unwrap_or(c.rest_volume);
        c.rest_volume = v0 * s * s * s;
    }
}

/// Steady-state analytic linear profile between Dirichlet ends T(0)=Ta, T(L)=Tb:
/// `T(x) = Ta + (Tb−Ta) x/L`.
#[inline]
pub fn steady_linear_profile(x: f64, length: f64, t_a: f64, t_b: f64) -> f64 {
    t_a + (t_b - t_a) * (x / length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn copperish() -> ThermalProps {
        ThermalProps {
            density: 8960.0,
            specific_heat: 385.0,
            conductivity: 400.0,
            expansion: 16.5e-6,
        }
    }

    #[test]
    fn free_expansion_oracle() {
        let l0 = 1.0;
        let alpha = 1e-5;
        let dt = 100.0;
        assert_relative_eq!(free_expansion_delta_l(l0, alpha, dt), 1e-3, epsilon = 1e-15);
    }

    #[test]
    fn insulated_energy_conserved_explicit() {
        let props = copperish();
        let mut mesh = ThermalMesh::rod_1d(17, 1.0, 0.01, props, 300.0);
        for i in 0..mesh.len() {
            mesh.temperature[i] =
                300.0 + 10.0 * libm_ops::sin(core::f64::consts::PI * i as f64 / 16.0);
        }
        let e0 = mesh.total_energy();
        let dt = mesh.max_stable_dt();
        let fixed = vec![false; mesh.len()];
        for _ in 0..200 {
            assert!(mesh.step_explicit(dt, &fixed));
        }
        let e1 = mesh.total_energy();
        let rel = (e1 - e0).abs() / e0.abs();
        assert!(rel < 1e-12, "insulated energy drift {rel}");
    }

    #[test]
    fn insulated_energy_conserved_backward_euler() {
        let props = copperish();
        let mut mesh = ThermalMesh::rod_1d(17, 1.0, 0.01, props, 300.0);
        for i in 0..mesh.len() {
            mesh.temperature[i] =
                300.0 + 10.0 * libm_ops::sin(core::f64::consts::PI * i as f64 / 16.0);
        }
        let e0 = mesh.total_energy();
        let dt = 10.0 * mesh.max_stable_dt();
        let fixed = vec![false; mesh.len()];
        for _ in 0..50 {
            assert!(mesh.step_backward_euler(dt, &fixed));
        }
        let rel = (mesh.total_energy() - e0).abs() / e0.abs();
        assert!(rel < 1e-11, "BE insulated energy drift {rel}");
    }

    #[test]
    fn erfc_surface_limit() {
        let t = semi_infinite_erfc(0.0, 1.0, 0.0, 100.0, 1e-4);
        assert_relative_eq!(t, 100.0, epsilon = 1e-12);
    }
}
