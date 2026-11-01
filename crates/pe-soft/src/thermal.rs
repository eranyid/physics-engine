//! Heat conduction on particle / edge meshes and thermal expansion coupling.
//!
//! # Conduction
//!
//! Lumped-capacity graph FEM / discrete Laplacian on an undirected edge set:
//! `C_i dT_i/dt = Σ_j κ_ij (T_j − T_i)` with `C_i = ρ c V_i` (or equal lumped
//! masses) and edge conductance `κ_ij = k A_ij / L_ij` (1-D rod: `A/L` per
//! segment). Explicit Euler with CFL-safe `dt`. Insulated boundaries: no flux
//! edges → total thermal energy `Σ C_i T_i` conserved.
//!
//! # Analytic 1-D rod (Dirichlet ends)
//!
//! For `T(0,t)=T(L,t)=0`, initial `T(x,0)=T₀ sin(π x / L)`:
//! `T(x,t) = T₀ sin(π x / L) exp(−κ (π/L)² t)` with diffusivity `κ = k/(ρ c)`.
//!
//! # Expansion
//!
//! Free thermal expansion: `ΔL = α L₀ ΔT` (linear), applied as a rest-length
//! update on distance constraints / uniaxial bars.

use pe_math::libm_ops;

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
}

impl ThermalMesh {
    /// Build a uniform 1-D rod of `n` nodes on `[0, L]` with equal spacing,
    /// cross-section `area`, and lumped capacities `ρ c A Δx` (end nodes half
    /// or full — we use equal `ρ c A L / n` for simplicity / Fourier tests with
    /// Dirichlet overwrite ends).
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
        Self {
            temperature: vec![t_ref; n],
            capacity: vec![c_node; n],
            edges,
            conductance,
            t_ref,
            props,
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
            let q = g * (tj - ti); // from j into i
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
            self.temperature[i] += dt * d_e[i] / c;
            if !self.temperature[i].is_finite() {
                return false;
            }
        }
        true
    }

    /// Suggested stable explicit dt: `dt < min_i C_i / Σ_j κ_ij` (with safety).
    pub fn max_stable_dt(&self) -> f64 {
        let n = self.len();
        let mut deg = vec![0.0; n];
        for (e, &g) in self.edges.iter().zip(self.conductance.iter()) {
            deg[e.i] += g;
            deg[e.j] += g;
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

/// Apply thermal expansion to distance-constraint rest lengths using the
/// average endpoint temperature. Updates rest lengths in place; returns the
/// mean `|ΔL|` applied.
pub fn apply_thermal_expansion_distances(
    constraints: &mut [crate::constraint::DistanceConstraint],
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
        // Interpret current rest_length as cold L0 only on first call — callers
        // should pass the cold rest length each time OR we expand from a stored
        // cold length. Here we treat `rest_length` as cold L₀ at construction and
        // overwrite with expanded length from a parallel cold buffer when provided.
        // For the simple API: assume `rest_length` currently holds cold L₀ and
        // the caller resets from a cold snapshot each step.
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
    constraints: &mut [crate::constraint::DistanceConstraint],
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
    fn insulated_energy_conserved() {
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
}
