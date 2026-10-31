//! Divergence-free SPH (DFSPH) — Bender & Koschier (2015 / TVCG 2017).
//!
//! Clean-room from the published algorithm: constant-density solver +
//! divergence-free solver sharing precomputed factors `α_i`, warm-started
//! stiffness, adaptive CFL timestep. Cubic spline kernel throughout (same
//! kernel for `W` and `∇W` as required by the paper).
//!
//! # Algorithm (one step)
//! 1. Non-pressure forces (gravity, XSPH viscosity, optional surface tension).
//! 2. Adapt `Δt` by CFL: `Δt ≤ 0.4 d / ‖v‖_max` (and force limit).
//! 3. Predict `v* ← v + Δt a_nonpressure`.
//! 4. Constant-density solver: correct `v*` so predicted `ρ* → ρ₀`.
//! 5. Advect `x ← x + Δt v*`; rebuild neighbours; recompute `ρ`, `α`.
//! 6. Divergence-free solver: correct `v*` so `Dρ/Dt → 0`.
//! 7. `v ← v*`.
//!
//! # References
//! - Bender, J. & Koschier, D. (2015). Divergence-Free SPH for Incompressible
//!   and Viscous Fluids. *SCA* / *IEEE TVCG* **23**(3), 2017.
//! - Akinci et al. (2012) boundary volumes `ψ` (boundary samples only).

use pe_math::libm_ops::sqrt;
use pe_math::Vec3;

use crate::cfl::{suggested_dt, CflParams};
use crate::grid::UniformGrid;
use crate::kernels::{grad_w, support_radius, w, KernelKind};
use crate::particles::{ParticleKind, Particles};
use crate::surface_tension::{surface_tension_accel, SurfaceTensionParams};
use crate::viscosity::Xsph;

/// DFSPH solver parameters.
#[derive(Clone, Debug)]
pub struct DfsphParams {
    /// Density / gradient kernel (paper: cubic spline).
    pub kernel: KernelKind,
    /// Max iterations for the constant-density solver.
    pub max_density_iters: usize,
    /// Max iterations for the divergence-free solver.
    pub max_divergence_iters: usize,
    /// Average relative density-error tolerance (e.g. `0.001` = 0.1 %).
    pub density_error_tol: f64,
    /// Average `|Dρ/Dt| / ρ₀` tolerance for divergence solver.
    pub divergence_error_tol: f64,
    /// Gravity [m/s²].
    pub gravity: Vec3,
    /// XSPH viscosity (Schechter–Bridson style; paper default for water).
    pub xsph: Xsph,
    /// Optional surface tension (disabled when `sigma = 0`).
    pub surface_tension: SurfaceTensionParams,
    /// Particle diameter `d` [m] for CFL (`Δt ≤ λ d / v_max`). Default = `h/2`.
    pub particle_diameter: f64,
    /// CFL velocity factor (paper uses 0.4 on diameter).
    pub cfl_lambda: f64,
    /// Enable adaptive timestep inside [`DfsphWorld::step_adaptive`].
    pub adaptive_dt: bool,
    /// Warm-start pressure stiffness across steps (paper §3.2–3.3).
    pub warm_start: bool,
}

impl Default for DfsphParams {
    fn default() -> Self {
        Self {
            kernel: KernelKind::CubicSpline,
            max_density_iters: 50,
            max_divergence_iters: 50,
            density_error_tol: 0.001,
            divergence_error_tol: 0.001,
            gravity: Vec3::new(0.0, 0.0, -9.80665),
            xsph: Xsph { epsilon: 0.05 },
            surface_tension: SurfaceTensionParams::disabled(),
            particle_diameter: 0.0, // filled from h in world ctor if 0
            cfl_lambda: 0.4,
            adaptive_dt: true,
            warm_start: true,
        }
    }
}

/// DFSPH world.
#[derive(Clone, Debug)]
pub struct DfsphWorld {
    /// Particles.
    pub particles: Particles,
    /// Neighbour grid.
    pub grid: UniformGrid,
    /// Parameters.
    pub params: DfsphParams,
    /// Simulation time [s].
    pub t: f64,
    /// Last adaptive / applied timestep [s].
    pub last_dt: f64,
    /// Density factors α_i [m⁵·s²/kg] (paper).
    alpha: Vec<f64>,
    /// Predicted velocities v*.
    v_star: Vec<Vec3>,
    /// Stiffness κ for constant-density solver.
    kappa: Vec<f64>,
    /// Stiffness κᵛ for divergence-free solver.
    kappa_v: Vec<f64>,
    /// Warm-start accumulators.
    kappa_warm: Vec<f64>,
    kappa_v_warm: Vec<f64>,
    /// Non-pressure accelerations.
    a_adv: Vec<Vec3>,
    /// Per-particle neighbour lists (deterministic grid order).
    neigh_lists: Vec<Vec<usize>>,
    /// Scratch for XSPH / ST.
    scratch_dv: Vec<Vec3>,
    /// Colour-field normals for surface tension.
    normals: Vec<Vec3>,
}

impl DfsphWorld {
    /// Create from particles + params.
    pub fn new(mut particles: Particles, mut params: DfsphParams) -> Self {
        if params.particle_diameter <= 0.0 {
            // Cubic support = 2h ⇒ typical lattice spacing ≈ h.
            params.particle_diameter = particles.h;
        }
        let support = support_radius(params.kernel, particles.h);
        let n = particles.len();
        let grid = UniformGrid::new(support, particles.dim);
        // Ensure D2 particles stay planar.
        if particles.dim == crate::Dim::D2 {
            for i in 0..n {
                particles.x[i].z = 0.0;
                particles.v[i].z = 0.0;
            }
        }
        let mut world = Self {
            particles,
            grid,
            params,
            t: 0.0,
            last_dt: 0.0,
            alpha: vec![0.0; n],
            v_star: vec![Vec3::zero(); n],
            kappa: vec![0.0; n],
            kappa_v: vec![0.0; n],
            kappa_warm: vec![0.0; n],
            kappa_v_warm: vec![0.0; n],
            a_adv: vec![Vec3::zero(); n],
            neigh_lists: vec![Vec::new(); n],
            scratch_dv: vec![Vec3::zero(); n],
            normals: vec![Vec3::zero(); n],
        };
        world.rebuild_neighbours();
        world.compute_density();
        world.compute_alpha();
        world
    }

    fn ensure_buffers(&mut self) {
        let n = self.particles.len();
        self.alpha.resize(n, 0.0);
        self.v_star.resize(n, Vec3::zero());
        self.kappa.resize(n, 0.0);
        self.kappa_v.resize(n, 0.0);
        self.kappa_warm.resize(n, 0.0);
        self.kappa_v_warm.resize(n, 0.0);
        self.a_adv.resize(n, Vec3::zero());
        self.scratch_dv.resize(n, Vec3::zero());
        self.normals.resize(n, Vec3::zero());
        if self.neigh_lists.len() != n {
            self.neigh_lists.resize(n, Vec::new());
        }
    }

    /// Rebuild neighbour lists from current positions (sorted grid).
    pub fn rebuild_neighbours(&mut self) {
        self.ensure_buffers();
        self.grid.rebuild(&self.particles.x);
        let n = self.particles.len();
        for i in 0..n {
            self.neigh_lists[i].clear();
            if self.particles.kind[i] == ParticleKind::Fluid {
                self.grid.neighbours(i, &mut self.neigh_lists[i]);
            }
        }
    }

    /// Density summation with Akinci boundary contribution.
    pub fn compute_density(&mut self) {
        let n = self.particles.len();
        let h = self.particles.h;
        let dim = self.particles.dim;
        let kern = self.params.kernel;
        let mass = self.particles.mass;
        let rho0 = self.particles.rest_density;
        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                self.particles.rho[i] = rho0;
                continue;
            }
            let xi = self.particles.x[i];
            let mut rho = mass * w(kern, 0.0, h, dim);
            for &j in &self.neigh_lists[i] {
                if j == i {
                    continue;
                }
                let r = (self.particles.x[j] - xi).norm();
                let wij = w(kern, r, h, dim);
                match self.particles.kind[j] {
                    ParticleKind::Fluid => rho += mass * wij,
                    ParticleKind::Boundary => rho += rho0 * self.particles.psi[j] * wij,
                }
            }
            self.particles.rho[i] = rho.max(1e-12);
        }
    }

    /// Precompute α_i = ρ_i / (|Σ m∇W|² + Σ |m∇W|²).
    ///
    /// The paper clamps the denominator at `1e−6`, which is appropriate for
    /// their scaled particle masses. In SI (water, `m ∼ ρ₀ dx^d`) we clamp
    /// relative to a characteristic `|m ∇W|²` scale so free-surface particles
    /// with sparse neighbourhoods do not explode.
    pub fn compute_alpha(&mut self) {
        let n = self.particles.len();
        let h = self.particles.h;
        let dim = self.particles.dim;
        let kern = self.params.kernel;
        let mass = self.particles.mass;
        let rho0 = self.particles.rest_density;
        // Characteristic |m ∇W| ~ m / h^{d+1}; clamp floor ~ (ε m / h^{d+1})².
        let h_safe = h.max(1e-12);
        let char_grad = match dim {
            crate::Dim::D2 => mass / (h_safe * h_safe * h_safe),
            crate::Dim::D3 => mass / (h_safe * h_safe * h_safe * h_safe),
        };
        let denom_floor = (1e-6 * char_grad * char_grad).max(1e-12);
        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                self.alpha[i] = 0.0;
                continue;
            }
            let xi = self.particles.x[i];
            let mut sum_grad = Vec3::zero();
            let mut sum_grad_sq = 0.0;
            let mut n_nbr = 0usize;
            for &j in &self.neigh_lists[i] {
                if j == i {
                    continue;
                }
                let rij = self.particles.x[j] - xi;
                let gw = grad_w(kern, rij, h, dim);
                let mj = match self.particles.kind[j] {
                    ParticleKind::Fluid => mass,
                    ParticleKind::Boundary => rho0 * self.particles.psi[j],
                };
                let g = gw * mj;
                sum_grad += g;
                sum_grad_sq += g.norm_squared();
                n_nbr += 1;
            }
            if n_nbr < 2 {
                // Isolated / pair: skip pressure (α=0).
                self.alpha[i] = 0.0;
                continue;
            }
            let denom = (sum_grad.norm_squared() + sum_grad_sq).max(denom_floor);
            self.alpha[i] = self.particles.rho[i] / denom;
        }
    }

    fn neighbour_mass(&self, j: usize) -> f64 {
        match self.particles.kind[j] {
            ParticleKind::Fluid => self.particles.mass,
            ParticleKind::Boundary => self.particles.rest_density * self.particles.psi[j],
        }
    }

    /// Density rate `Dρ_i/Dt = Σ_j m_j (v_i − v_j) · ∇_i W` using `v_star`.
    fn density_derivative(&self, i: usize) -> f64 {
        let h = self.particles.h;
        let dim = self.particles.dim;
        let kern = self.params.kernel;
        let xi = self.particles.x[i];
        let vi = self.v_star[i];
        let mut drho = 0.0;
        for &j in &self.neigh_lists[i] {
            if j == i {
                continue;
            }
            let rij = self.particles.x[j] - xi;
            let gw = grad_w(kern, rij, h, dim);
            let mj = self.neighbour_mass(j);
            let vj = match self.particles.kind[j] {
                ParticleKind::Fluid => self.v_star[j],
                ParticleKind::Boundary => Vec3::zero(),
            };
            drho += mj * (vi - vj).dot(gw);
        }
        drho
    }

    fn compute_nonpressure_accel(&mut self) {
        let n = self.particles.len();
        let g = self.params.gravity;
        for i in 0..n {
            if self.particles.kind[i] == ParticleKind::Fluid {
                self.a_adv[i] = g;
            } else {
                self.a_adv[i] = Vec3::zero();
            }
        }
        // Surface tension accelerations (optional).
        if self.params.surface_tension.sigma != 0.0 {
            surface_tension_accel(
                &self.particles,
                &self.neigh_lists,
                self.params.kernel,
                &self.params.surface_tension,
                &mut self.normals,
                &mut self.scratch_dv,
            );
            for i in 0..n {
                if self.particles.kind[i] == ParticleKind::Fluid {
                    self.a_adv[i] += self.scratch_dv[i];
                }
            }
        }
    }

    /// Apply XSPH velocity filter on `v_star` (fluid–fluid).
    fn apply_xsph_on_vstar(&mut self) {
        let eps = self.params.xsph.epsilon;
        if eps == 0.0 {
            return;
        }
        let n = self.particles.len();
        let h = self.particles.h;
        let dim = self.particles.dim;
        let kern = self.params.kernel;
        let mass = self.particles.mass;
        for d in &mut self.scratch_dv {
            *d = Vec3::zero();
        }
        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                continue;
            }
            let xi = self.particles.x[i];
            let vi = self.v_star[i];
            let rho_i = self.particles.rho[i];
            let mut dv = Vec3::zero();
            for &j in &self.neigh_lists[i] {
                if j == i || self.particles.kind[j] != ParticleKind::Fluid {
                    continue;
                }
                let r = (self.particles.x[j] - xi).norm();
                let wij = w(kern, r, h, dim);
                let rho_bar = 0.5 * (rho_i + self.particles.rho[j]);
                dv += (self.v_star[j] - vi) * (mass / rho_bar.max(1e-12) * wij);
            }
            self.scratch_dv[i] = dv * eps;
        }
        for i in 0..n {
            if self.particles.kind[i] == ParticleKind::Fluid {
                self.v_star[i] += self.scratch_dv[i];
            }
        }
    }

    /// Apply pressure velocity correction: `v ← v − Δt Σ m_j (κ_i/ρ_i + κ_j/ρ_j) ∇W`.
    fn apply_pressure_velocity_update(&mut self, dt: f64, use_divergence: bool) {
        let n = self.particles.len();
        let h = self.particles.h;
        let dim = self.particles.dim;
        let kern = self.params.kernel;
        let kappa = if use_divergence {
            &self.kappa_v
        } else {
            &self.kappa
        };
        for d in &mut self.scratch_dv {
            *d = Vec3::zero();
        }
        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                continue;
            }
            let xi = self.particles.x[i];
            let rho_i = self.particles.rho[i].max(1e-12);
            let ki = kappa[i];
            let mut dv = Vec3::zero();
            for &j in &self.neigh_lists[i] {
                if j == i {
                    continue;
                }
                let rij = self.particles.x[j] - xi;
                let gw = grad_w(kern, rij, h, dim);
                let mj = self.neighbour_mass(j);
                let kj = match self.particles.kind[j] {
                    ParticleKind::Fluid => kappa[j],
                    // Static boundary: κ_j = 0 (paper note on static neighbours).
                    ParticleKind::Boundary => 0.0,
                };
                let rho_j = match self.particles.kind[j] {
                    ParticleKind::Fluid => self.particles.rho[j].max(1e-12),
                    ParticleKind::Boundary => self.particles.rest_density,
                };
                dv += gw * (mj * (ki / rho_i + kj / rho_j));
            }
            self.scratch_dv[i] = dv * (-dt);
        }
        for i in 0..n {
            if self.particles.kind[i] == ParticleKind::Fluid {
                self.v_star[i] += self.scratch_dv[i];
                if self.particles.dim == crate::Dim::D2 {
                    self.v_star[i].z = 0.0;
                }
            }
        }
    }

    /// Constant-density solver (Algorithm 3). Returns iteration count.
    pub fn correct_density_error(&mut self, dt: f64) -> usize {
        let n = self.particles.len();
        let rho0 = self.particles.rest_density;
        let dt2 = dt * dt;
        let mut iters = 0usize;

        // Warm start: one pressure update with previous-step κ sum.
        if self.params.warm_start {
            for i in 0..n {
                self.kappa[i] = self.kappa_warm[i];
            }
            self.apply_pressure_velocity_update(dt, false);
        }
        self.kappa_warm.fill(0.0);

        loop {
            // Predict ρ* = ρ + Δt Dρ/Dt(v*).
            // Compression-only (ρ* > ρ₀)⁺: required for free-surface stability.
            // Under-dense SPH surface particles must not attract (tensile blow-up).
            // Bender & Koschier Algorithm 3 writes the two-sided form; production
            // DFSPH codes (and this crate) clamp to compression.
            let mut avg_err = 0.0;
            let mut count = 0usize;
            for i in 0..n {
                if self.particles.kind[i] != ParticleKind::Fluid {
                    self.kappa[i] = 0.0;
                    continue;
                }
                let rho_star = self.particles.rho[i] + dt * self.density_derivative(i);
                let err = (rho_star - rho0).max(0.0);
                avg_err += err / rho0;
                count += 1;
                // κ_i = (ρ* − ρ₀)⁺ / Δt² · α_i
                self.kappa[i] = err / dt2 * self.alpha[i];
                if self.params.warm_start {
                    self.kappa_warm[i] += self.kappa[i];
                }
            }
            if count > 0 {
                avg_err /= count as f64;
            }
            self.apply_pressure_velocity_update(dt, false);
            iters += 1;
            if (avg_err <= self.params.density_error_tol && iters >= 2)
                || iters >= self.params.max_density_iters
            {
                break;
            }
        }
        iters
    }

    /// Divergence-free solver (Algorithm 2). Returns iteration count.
    pub fn correct_divergence_error(&mut self, dt: f64) -> usize {
        let n = self.particles.len();
        let rho0 = self.particles.rest_density;
        let mut iters = 0usize;

        if self.params.warm_start {
            for i in 0..n {
                self.kappa_v[i] = self.kappa_v_warm[i];
            }
            self.apply_pressure_velocity_update(dt, true);
        }
        self.kappa_v_warm.fill(0.0);

        loop {
            // Compression-only: correct Dρ/Dt > 0 (approaching neighbours).
            let mut avg = 0.0;
            let mut count = 0usize;
            for i in 0..n {
                if self.particles.kind[i] != ParticleKind::Fluid {
                    self.kappa_v[i] = 0.0;
                    continue;
                }
                let drho = self.density_derivative(i).max(0.0);
                avg += drho / rho0;
                count += 1;
                // κᵛ_i = (1/Δt) (Dρ/Dt)^+ α_i
                self.kappa_v[i] = (1.0 / dt) * drho * self.alpha[i];
                if self.params.warm_start {
                    self.kappa_v_warm[i] += self.kappa_v[i];
                }
            }
            if count > 0 {
                avg /= count as f64;
            }
            self.apply_pressure_velocity_update(dt, true);
            iters += 1;
            if (avg <= self.params.divergence_error_tol && iters >= 1)
                || iters >= self.params.max_divergence_iters
            {
                break;
            }
        }
        iters
    }

    /// Suggest a CFL timestep from current state (paper: `0.4 d / v_max`).
    pub fn suggest_dt(&self) -> f64 {
        let mut vmax: f64 = 0.0;
        let mut amax: f64 = 0.0;
        for i in 0..self.particles.len() {
            if self.particles.kind[i] != ParticleKind::Fluid {
                continue;
            }
            vmax = vmax.max(self.particles.v[i].norm_squared());
            amax = amax.max(self.a_adv[i].norm_squared());
        }
        vmax = sqrt(vmax);
        amax = sqrt(amax);
        let d = self.params.particle_diameter.max(1e-12);
        let dt_v = self.params.cfl_lambda * d / vmax.max(1e-6);
        // Also respect acceleration CFL with sound-speed stand-in unused (incompressible).
        let dt_f = suggested_dt(
            self.particles.h,
            0.0,
            vmax,
            amax.max(1e-6),
            CflParams {
                lambda_v: self.params.cfl_lambda,
                lambda_f: 0.25,
            },
        );
        // Velocity CFL dominates; force CFL is a secondary clamp.
        dt_v.min(dt_f).clamp(1e-6, 1e-2)
    }

    /// Fixed-`dt` DFSPH step (ignores adaptive flag).
    pub fn step(&mut self, dt: f64) {
        self.ensure_buffers();
        let n = self.particles.len();
        let dt = dt.max(1e-12);
        self.last_dt = dt;

        self.compute_nonpressure_accel();

        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                self.v_star[i] = self.particles.v[i];
                continue;
            }
            self.v_star[i] = self.particles.v[i] + self.a_adv[i] * dt;
            if self.particles.dim == crate::Dim::D2 {
                self.v_star[i].z = 0.0;
            }
        }
        self.apply_xsph_on_vstar();

        self.correct_density_error(dt);

        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                continue;
            }
            self.particles.x[i] += self.v_star[i] * dt;
            if self.particles.dim == crate::Dim::D2 {
                self.particles.x[i].z = 0.0;
            }
        }

        self.rebuild_neighbours();
        self.compute_density();
        self.compute_alpha();

        self.correct_divergence_error(dt);

        for i in 0..n {
            if self.particles.kind[i] == ParticleKind::Fluid {
                self.particles.v[i] = self.v_star[i];
                // Store κ as a pressure diagnostic: p ≈ κ (paper EOS γ=1 form unused).
                self.particles.p[i] = self.kappa[i];
                if self.particles.dim == crate::Dim::D2 {
                    self.particles.v[i].z = 0.0;
                }
            }
        }
        self.t += dt;
    }

    /// One adaptive step; returns the `dt` used.
    pub fn step_adaptive(&mut self) -> f64 {
        self.ensure_buffers();
        self.compute_nonpressure_accel();
        let dt = if self.params.adaptive_dt {
            self.suggest_dt()
        } else {
            self.last_dt.max(1e-4)
        };
        // Re-enter fixed step (recomputes a_adv — cheap).
        self.step(dt);
        dt
    }

    /// Mean `|ρ/ρ₀ − 1|`.
    pub fn mean_density_error(&self) -> f64 {
        crate::diagnostics::mean_density_error(&self.particles)
    }

    /// Max fluid speed.
    pub fn v_max(&self) -> f64 {
        let mut m: f64 = 0.0;
        for i in 0..self.particles.len() {
            if self.particles.kind[i] == ParticleKind::Fluid {
                m = m.max(self.particles.v[i].norm_squared());
            }
        }
        sqrt(m)
    }
}
