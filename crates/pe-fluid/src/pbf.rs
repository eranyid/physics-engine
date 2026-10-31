//! Position Based Fluids (Macklin & Müller 2013).
//!
//! Density constraint `C_i = ρ_i/ρ₀ − 1`, λ-multipliers, artificial pressure
//! tensile-instability correction, optional vorticity confinement.
//!
//! # References
//! - Macklin, M. & Müller, M. (2013). Position Based Fluids. *ACM TOG* **32**.
//! - Müller et al. (2003) poly6 / spiky kernels (as used in the PBF paper).

use pe_math::libm_ops::{fabs, pow, sqrt};
use pe_math::Vec3;

use crate::grid::UniformGrid;
use crate::kernels::{grad_w, support_radius, w, KernelKind};
use crate::particles::{ParticleKind, Particles};

/// PBF solver parameters.
#[derive(Clone, Debug)]
pub struct PbfParams {
    /// Density kernel (poly6).
    pub density_kernel: KernelKind,
    /// Gradient kernel (spiky).
    pub grad_kernel: KernelKind,
    /// Constraint solver iterations.
    pub iterations: usize,
    /// Relaxation ε in λ denominator (CFM).
    pub epsilon_lambda: f64,
    /// Artificial pressure strength k.
    pub art_pressure_k: f64,
    /// Artificial pressure power n.
    pub art_pressure_n: f64,
    /// Δq / h for artificial pressure reference (paper uses 0.1–0.3).
    pub art_pressure_dq_over_h: f64,
    /// XSPH-like viscosity ε (paper §5).
    pub viscosity_eps: f64,
    /// Vorticity confinement ε (0 = off).
    pub vorticity_eps: f64,
    /// Gravity [m/s²].
    pub gravity: Vec3,
}

impl Default for PbfParams {
    fn default() -> Self {
        Self {
            density_kernel: KernelKind::Poly6,
            grad_kernel: KernelKind::Spiky,
            iterations: 5,
            // Macklin & Müller (2013) use ε ≈ 0.01 as CFM relaxation in λ.
            epsilon_lambda: 0.01,
            // Mild default; tank settles often use 0. Tensile correction is optional.
            art_pressure_k: 0.01,
            art_pressure_n: 4.0,
            art_pressure_dq_over_h: 0.25,
            viscosity_eps: 0.01,
            vorticity_eps: 0.0,
            gravity: Vec3::new(0.0, 0.0, -9.80665),
        }
    }
}

/// Position Based Fluids world.
#[derive(Clone, Debug)]
pub struct PbfWorld {
    /// Particles.
    pub particles: Particles,
    /// Neighbour grid.
    pub grid: UniformGrid,
    /// Parameters.
    pub params: PbfParams,
    /// Simulation time [s].
    pub t: f64,
    /// Predicted positions.
    pred: Vec<Vec3>,
    /// λ multipliers.
    lambda: Vec<f64>,
    /// Δp corrections.
    dp: Vec<Vec3>,
    /// Scratch neighbours.
    neigh: Vec<usize>,
    /// Densities at predicted positions.
    rho_pred: Vec<f64>,
}

impl PbfWorld {
    /// Create from particles + params.
    pub fn new(particles: Particles, params: PbfParams) -> Self {
        let support = support_radius(params.density_kernel, particles.h)
            .max(support_radius(params.grad_kernel, particles.h));
        let n = particles.len();
        let grid = UniformGrid::new(support, particles.dim);
        Self {
            particles,
            grid,
            params,
            t: 0.0,
            pred: vec![Vec3::zero(); n],
            lambda: vec![0.0; n],
            dp: vec![Vec3::zero(); n],
            neigh: Vec::new(),
            rho_pred: vec![0.0; n],
        }
    }

    fn ensure_buffers(&mut self) {
        let n = self.particles.len();
        self.pred.resize(n, Vec3::zero());
        self.lambda.resize(n, 0.0);
        self.dp.resize(n, Vec3::zero());
        self.rho_pred.resize(n, 0.0);
    }

    /// Density at predicted positions into `rho_pred` / `particles.rho`.
    fn density_at_pred(&mut self) {
        let n = self.particles.len();
        let h = self.particles.h;
        let dim = self.particles.dim;
        let kern = self.params.density_kernel;
        let mass = self.particles.mass;
        let rho0 = self.particles.rest_density;
        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                self.rho_pred[i] = rho0;
                self.particles.rho[i] = rho0;
                continue;
            }
            self.neigh.clear();
            // Neighbours from grid on predicted positions — rebuild uses pred.
            self.grid.neighbours(i, &mut self.neigh);
            let xi = self.pred[i];
            let mut rho = mass * w(kern, 0.0, h, dim);
            for &j in &self.neigh {
                if j == i {
                    continue;
                }
                let r = (self.pred[j] - xi).norm();
                let wij = w(kern, r, h, dim);
                match self.particles.kind[j] {
                    ParticleKind::Fluid => rho += mass * wij,
                    ParticleKind::Boundary => rho += rho0 * self.particles.psi[j] * wij,
                }
            }
            self.rho_pred[i] = rho.max(1e-12);
            self.particles.rho[i] = self.rho_pred[i];
        }
    }

    fn compute_lambdas(&mut self) {
        let n = self.particles.len();
        let h = self.particles.h;
        let dim = self.particles.dim;
        let gkern = self.params.grad_kernel;
        let mass = self.particles.mass;
        let rho0 = self.particles.rest_density;
        let eps = self.params.epsilon_lambda;

        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                self.lambda[i] = 0.0;
                continue;
            }
            let ci = self.rho_pred[i] / rho0 - 1.0;
            self.neigh.clear();
            self.grid.neighbours(i, &mut self.neigh);
            let xi = self.pred[i];
            // Σ_k |∇_{p_k} C_i|²
            let mut sum_grad_sq = 0.0;
            let mut grad_i = Vec3::zero();
            for &j in &self.neigh {
                if j == i {
                    continue;
                }
                let rij = self.pred[j] - xi;
                let gw = grad_w(gkern, rij, h, dim); // ∇_i W
                                                     // ∇_{p_j} C_i = − (m_j / ρ₀) ∇_i W   for fluid j
                                                     // ∇_{p_i} C_i = Σ_j (m_j / ρ₀) ∇_i W
                let scale = match self.particles.kind[j] {
                    ParticleKind::Fluid => mass / rho0,
                    ParticleKind::Boundary => (rho0 * self.particles.psi[j]) / rho0, // ψ
                };
                let contrib = gw * scale;
                grad_i += contrib;
                if self.particles.kind[j] == ParticleKind::Fluid {
                    let g_j = contrib * (-1.0);
                    sum_grad_sq += g_j.norm_squared();
                }
            }
            sum_grad_sq += grad_i.norm_squared();
            self.lambda[i] = -ci / (sum_grad_sq + eps);
        }
    }

    fn artificial_pressure(&self, wij: f64, w_dq: f64) -> f64 {
        if w_dq <= 1e-30 || self.params.art_pressure_k == 0.0 {
            return 0.0;
        }
        // Clamp ratio ≤ 1 so |s_corr| ≤ k (pairs closer than Δq do not explode).
        let ratio = (wij / w_dq).min(1.0);
        -self.params.art_pressure_k * pow(ratio, self.params.art_pressure_n)
    }

    fn compute_delta_p(&mut self) {
        let n = self.particles.len();
        let h = self.particles.h;
        let dim = self.particles.dim;
        let dkern = self.params.density_kernel;
        let gkern = self.params.grad_kernel;
        let mass = self.particles.mass;
        let rho0 = self.particles.rest_density;
        let dq = self.params.art_pressure_dq_over_h * h;
        let w_dq = w(dkern, dq, h, dim);

        for d in &mut self.dp {
            *d = Vec3::zero();
        }

        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                continue;
            }
            self.neigh.clear();
            self.grid.neighbours(i, &mut self.neigh);
            let xi = self.pred[i];
            let mut corr = Vec3::zero();
            for &j in &self.neigh {
                if j == i {
                    continue;
                }
                let rij = self.pred[j] - xi;
                let r = rij.norm();
                let wij = w(dkern, r, h, dim);
                let gw = grad_w(gkern, rij, h, dim);
                let s_corr = self.artificial_pressure(wij, w_dq);
                match self.particles.kind[j] {
                    ParticleKind::Fluid => {
                        let coeff = self.lambda[i] + self.lambda[j] + s_corr;
                        corr += gw * (mass / rho0 * coeff);
                    }
                    ParticleKind::Boundary => {
                        // One-sided: λ_i only + artificial pressure
                        let coeff = self.lambda[i] + s_corr;
                        corr += gw * (self.particles.psi[j] * coeff);
                    }
                }
            }
            // Clamp |Δp| per iteration (engineering stabiliser).
            let max_corr = 0.4 * h;
            let mag = corr.norm();
            if mag > max_corr && mag > 0.0 {
                corr *= max_corr / mag;
            }
            self.dp[i] = corr;
        }
    }

    fn apply_viscosity_and_vorticity(&mut self, dt: f64) {
        let n = self.particles.len();
        let h = self.particles.h;
        let dim = self.particles.dim;
        let dkern = self.params.density_kernel;
        let mass = self.particles.mass;
        let visc_eps = self.params.viscosity_eps;
        let vort_eps = self.params.vorticity_eps;

        // XSPH-like viscosity on velocities
        if visc_eps != 0.0 {
            let mut dv = vec![Vec3::zero(); n];
            for i in 0..n {
                if self.particles.kind[i] != ParticleKind::Fluid {
                    continue;
                }
                self.neigh.clear();
                self.grid.neighbours(i, &mut self.neigh);
                let xi = self.particles.x[i];
                let vi = self.particles.v[i];
                let mut corr = Vec3::zero();
                for &j in &self.neigh {
                    if j == i || self.particles.kind[j] != ParticleKind::Fluid {
                        continue;
                    }
                    let r = (self.particles.x[j] - xi).norm();
                    let wij = w(dkern, r, h, dim);
                    let rho_bar = 0.5 * (self.particles.rho[i] + self.particles.rho[j]);
                    corr += (self.particles.v[j] - vi) * (mass / rho_bar.max(1e-12) * wij);
                }
                dv[i] = corr * visc_eps;
            }
            for i in 0..n {
                self.particles.v[i] += dv[i];
            }
        }

        if vort_eps == 0.0 || dt == 0.0 {
            return;
        }
        // ω_i = ∇ × v ≈ Σ_j m_j (v_j − v_i) × ∇W / ρ̄
        let gkern = self.params.grad_kernel;
        let mut omega = vec![Vec3::zero(); n];
        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                continue;
            }
            self.neigh.clear();
            self.grid.neighbours(i, &mut self.neigh);
            let xi = self.particles.x[i];
            let vi = self.particles.v[i];
            let mut w_i = Vec3::zero();
            for &j in &self.neigh {
                if j == i || self.particles.kind[j] != ParticleKind::Fluid {
                    continue;
                }
                let rij = self.particles.x[j] - xi;
                let gw = grad_w(gkern, rij, h, dim);
                let rho_bar = 0.5 * (self.particles.rho[i] + self.particles.rho[j]);
                w_i += (self.particles.v[j] - vi).cross(gw) * (mass / rho_bar.max(1e-12));
            }
            omega[i] = w_i;
        }
        // η = ∇|ω|, N = η/|η|, force ∝ ε (N × ω)
        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                continue;
            }
            self.neigh.clear();
            self.grid.neighbours(i, &mut self.neigh);
            let xi = self.particles.x[i];
            let mut eta = Vec3::zero();
            let om_i = omega[i].norm();
            for &j in &self.neigh {
                if j == i || self.particles.kind[j] != ParticleKind::Fluid {
                    continue;
                }
                let rij = self.particles.x[j] - xi;
                let gw = grad_w(gkern, rij, h, dim);
                let rho_bar = 0.5 * (self.particles.rho[i] + self.particles.rho[j]);
                eta += gw * (mass / rho_bar.max(1e-12) * (omega[j].norm() - om_i));
            }
            let nrm = eta.norm();
            if nrm > 1e-12 {
                let n_hat = eta / nrm;
                let f = n_hat.cross(omega[i]) * vort_eps;
                self.particles.v[i] += f * dt;
            }
        }
    }

    /// One PBF step of size `dt`.
    pub fn step(&mut self, dt: f64) {
        self.ensure_buffers();
        let n = self.particles.len();
        let g = self.params.gravity;

        // Apply external forces + predict
        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                self.pred[i] = self.particles.x[i];
                continue;
            }
            self.particles.v[i] += g * dt;
            self.pred[i] = self.particles.x[i] + self.particles.v[i] * dt;
            if self.particles.dim == crate::Dim::D2 {
                self.pred[i].z = 0.0;
                self.particles.v[i].z = 0.0;
            }
        }

        // Build grid on predicted positions
        self.grid.rebuild(&self.pred);

        for _ in 0..self.params.iterations {
            self.density_at_pred();
            self.compute_lambdas();
            self.compute_delta_p();
            for i in 0..n {
                if self.particles.kind[i] == ParticleKind::Fluid {
                    self.pred[i] += self.dp[i];
                    if self.particles.dim == crate::Dim::D2 {
                        self.pred[i].z = 0.0;
                    }
                }
            }
            self.grid.rebuild(&self.pred);
        }

        // Update velocity from position change; write positions
        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                continue;
            }
            if dt > 0.0 {
                self.particles.v[i] = (self.pred[i] - self.particles.x[i]) * (1.0 / dt);
            }
            self.particles.x[i] = self.pred[i];
        }

        self.grid.rebuild(&self.particles.x);
        // Refresh density at final positions for diagnostics
        for i in 0..n {
            self.pred[i] = self.particles.x[i];
        }
        self.density_at_pred();
        self.apply_viscosity_and_vorticity(dt);
        self.t += dt;
    }

    /// Mean |ρ/ρ₀ − 1| after the last density evaluation.
    pub fn mean_density_error(&self) -> f64 {
        let mut s = 0.0;
        let mut c = 0usize;
        for i in 0..self.particles.len() {
            if self.particles.kind[i] == ParticleKind::Fluid {
                s += fabs(self.particles.rho[i] / self.particles.rest_density - 1.0);
                c += 1;
            }
        }
        if c == 0 {
            0.0
        } else {
            s / c as f64
        }
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
