//! Weakly compressible SPH (WCSPH) with Tait EOS, Monaghan viscosity, XSPH,
//! gravity, and Akinci boundary forces.
//!
//! Pair forces are accumulated with `j > i` so internal fluid–fluid forces are
//! antisymmetric (momentum conserving for equal-mass pairs).
//!
//! # References
//! - Monaghan (1992/1994); Becker & Teschner (2007) WCSPH.
//! - Akinci, N., Ihmsen, M., Akinci, G., Solenthaler, B. & Teschner, M. (2012).
//!   Versatile rigid–fluid coupling for incompressible SPH. *ACM TOG* **31**.

use pe_math::libm_ops::sqrt;
use pe_math::Vec3;

use crate::eos::{ensure_sound_speed, TaitEos};
use crate::grid::UniformGrid;
use crate::kernels::{grad_w, support_radius, w, KernelKind};
use crate::particles::{ParticleKind, Particles};
use crate::viscosity::{max_speed, pi_ij, MonaghanViscosity, Xsph};

/// WCSPH simulation parameters.
#[derive(Clone, Debug)]
pub struct WcsphParams {
    /// Density / viscosity kernel (default cubic spline).
    pub density_kernel: KernelKind,
    /// Tait EOS.
    pub eos: TaitEos,
    /// Artificial viscosity.
    pub viscosity: MonaghanViscosity,
    /// XSPH (set `epsilon = 0` to disable).
    pub xsph: Xsph,
    /// Gravity acceleration vector [m/s²] (default −g ê_z).
    pub gravity: Vec3,
    /// Clamp sound speed with `c ≥ 10 v_max` each step when true.
    pub adapt_sound_speed: bool,
}

impl WcsphParams {
    /// Defaults for water-like WCSPH at the given rest density and sound speed.
    pub fn water(rest_density: f64, sound_speed: f64, gravity: Vec3) -> Self {
        Self {
            density_kernel: KernelKind::CubicSpline,
            eos: TaitEos::new(rest_density, sound_speed, crate::eos::TAIT_GAMMA_WATER),
            viscosity: MonaghanViscosity::default(),
            xsph: Xsph::default(),
            gravity,
            adapt_sound_speed: true,
        }
    }
}

/// WCSPH world: particles + neighbour grid + parameters.
#[derive(Clone, Debug)]
pub struct WcsphWorld {
    /// Particles.
    pub particles: Particles,
    /// Neighbour grid.
    pub grid: UniformGrid,
    /// Parameters.
    pub params: WcsphParams,
    /// Simulation time [s].
    pub t: f64,
    /// Scratch neighbour list.
    neigh: Vec<usize>,
    /// Scratch XSPH delta-v.
    xsph_dv: Vec<Vec3>,
}

impl WcsphWorld {
    /// Create from particles and parameters.
    pub fn new(particles: Particles, params: WcsphParams) -> Self {
        let support = support_radius(params.density_kernel, particles.h);
        let grid = UniformGrid::new(support, particles.dim);
        Self {
            particles,
            grid,
            params,
            t: 0.0,
            neigh: Vec::new(),
            xsph_dv: Vec::new(),
        }
    }

    /// Rebuild the neighbour grid from current positions.
    pub fn rebuild_grid(&mut self) {
        self.grid.rebuild(&self.particles.x);
    }

    /// Density summation (fluid + Akinci boundary contribution).
    pub fn compute_density(&mut self) {
        let n = self.particles.len();
        let h = self.particles.h;
        let dim = self.particles.dim;
        let kern = self.params.density_kernel;
        let mass = self.particles.mass;
        let rho0 = self.particles.rest_density;

        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                self.particles.rho[i] = rho0;
                continue;
            }
            self.neigh.clear();
            self.grid.neighbours(i, &mut self.neigh);
            let xi = self.particles.x[i];
            let mut rho = 0.0;
            // Self contribution
            rho += mass * w(kern, 0.0, h, dim);
            for &j in &self.neigh {
                if j == i {
                    continue;
                }
                let r = (self.particles.x[j] - xi).norm();
                let wij = w(kern, r, h, dim);
                match self.particles.kind[j] {
                    ParticleKind::Fluid => rho += mass * wij,
                    ParticleKind::Boundary => {
                        // Akinci: ρ += ρ₀ ψ_b W
                        rho += rho0 * self.particles.psi[j] * wij;
                    }
                }
            }
            self.particles.rho[i] = rho.max(1e-12);
        }
    }

    /// Pressures from Tait EOS (fluid only).
    pub fn compute_pressure(&mut self) {
        let mut c = self.params.eos.sound_speed;
        if self.params.adapt_sound_speed {
            let fluid: Vec<bool> = self
                .particles
                .kind
                .iter()
                .map(|k| *k == ParticleKind::Fluid)
                .collect();
            let vmax = max_speed(&self.particles.v, &fluid);
            c = ensure_sound_speed(c, vmax);
            self.params.eos.sound_speed = c;
        }
        for i in 0..self.particles.len() {
            if self.particles.kind[i] == ParticleKind::Fluid {
                self.particles.p[i] = self.params.eos.pressure(self.particles.rho[i]);
            } else {
                // Mirror: use neighbouring fluid pressure average later; store 0.
                self.particles.p[i] = 0.0;
            }
        }
    }

    /// Accelerations: pressure + viscosity + gravity + Akinci boundary pressure.
    pub fn compute_forces(&mut self) {
        let n = self.particles.len();
        let h = self.particles.h;
        let dim = self.particles.dim;
        let kern = self.params.density_kernel;
        let mass = self.particles.mass;
        let rho0 = self.particles.rest_density;
        let c0 = self.params.eos.sound_speed;
        let visc = self.params.viscosity;
        let g = self.params.gravity;

        for a in &mut self.particles.a {
            *a = Vec3::zero();
        }

        // Pairwise fluid–fluid (j > i) for momentum symmetry.
        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                continue;
            }
            self.neigh.clear();
            self.grid.neighbours(i, &mut self.neigh);
            let xi = self.particles.x[i];
            let vi = self.particles.v[i];
            let rho_i = self.particles.rho[i];
            let p_i = self.particles.p[i];

            for &j in &self.neigh {
                if j == i {
                    continue;
                }
                let xj = self.particles.x[j];
                let rij = xj - xi; // for grad_w convention
                let r = rij.norm();
                if r < 1e-30 {
                    continue;
                }
                let gw = grad_w(kern, rij, h, dim); // ∇_i W

                match self.particles.kind[j] {
                    ParticleKind::Fluid => {
                        if j < i {
                            continue; // handled when roles reversed
                        }
                        let rho_j = self.particles.rho[j];
                        let p_j = self.particles.p[j];
                        let vj = self.particles.v[j];
                        // Pressure: a_i += − m_j (p_i/ρ_i² + p_j/ρ_j²) ∇_i W
                        let press = p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j);
                        let f_press = gw * (-mass * press);
                        // Viscosity Π
                        let pi = pi_ij(visc, h, c0, xi, xj, vi, vj, rho_i, rho_j);
                        let f_visc = gw * (-mass * pi);
                        let f = f_press + f_visc;
                        self.particles.a[i] += f;
                        self.particles.a[j] -= f; // equal mass ⇒ opposite accel
                    }
                    ParticleKind::Boundary => {
                        // Akinci pressure: a_i += − ρ₀ ψ_b (p_i/ρ_i²) ∇_i W
                        // (boundary rest pressure ≈ fluid pressure for impermeability)
                        let press = p_i / (rho_i * rho_i);
                        let f = gw * (-(rho0 * self.particles.psi[j]) * press);
                        self.particles.a[i] += f;
                    }
                }
            }
            self.particles.a[i] += g;
        }
    }

    /// Apply XSPH velocity smoothing (fluid only).
    pub fn apply_xsph(&mut self) {
        let eps = self.params.xsph.epsilon;
        if eps == 0.0 {
            return;
        }
        let n = self.particles.len();
        let h = self.particles.h;
        let dim = self.particles.dim;
        let kern = self.params.density_kernel;
        let mass = self.particles.mass;
        self.xsph_dv.resize(n, Vec3::zero());
        for d in &mut self.xsph_dv {
            *d = Vec3::zero();
        }
        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                continue;
            }
            self.neigh.clear();
            self.grid.neighbours(i, &mut self.neigh);
            let xi = self.particles.x[i];
            let vi = self.particles.v[i];
            let rho_i = self.particles.rho[i];
            let mut dv = Vec3::zero();
            for &j in &self.neigh {
                if j == i || self.particles.kind[j] != ParticleKind::Fluid {
                    continue;
                }
                let r = (self.particles.x[j] - xi).norm();
                let wij = w(kern, r, h, dim);
                let rho_bar = 0.5 * (rho_i + self.particles.rho[j]);
                dv += (self.particles.v[j] - vi) * (mass / rho_bar * wij);
            }
            self.xsph_dv[i] = dv * eps;
        }
        for i in 0..n {
            if self.particles.kind[i] == ParticleKind::Fluid {
                self.particles.v[i] += self.xsph_dv[i];
            }
        }
    }

    /// One symplectic Euler step: density → pressure → forces → kick → drift → XSPH.
    pub fn step(&mut self, dt: f64) {
        self.rebuild_grid();
        self.compute_density();
        self.compute_pressure();
        self.compute_forces();
        let n = self.particles.len();
        for i in 0..n {
            if self.particles.kind[i] != ParticleKind::Fluid {
                continue;
            }
            self.particles.v[i] += self.particles.a[i] * dt;
            self.particles.x[i] += self.particles.v[i] * dt;
            if self.particles.dim == crate::Dim::D2 {
                self.particles.x[i].z = 0.0;
                self.particles.v[i].z = 0.0;
            }
        }
        self.rebuild_grid();
        self.compute_density(); // refresh ρ for XSPH ρ̄
        self.apply_xsph();
        self.t += dt;
    }

    /// Max fluid speed [m/s].
    pub fn v_max(&self) -> f64 {
        let fluid: Vec<bool> = self
            .particles
            .kind
            .iter()
            .map(|k| *k == ParticleKind::Fluid)
            .collect();
        max_speed(&self.particles.v, &fluid)
    }

    /// Max |a| [m/s²].
    pub fn a_max(&self) -> f64 {
        let mut m: f64 = 0.0;
        for i in 0..self.particles.len() {
            if self.particles.kind[i] == ParticleKind::Fluid {
                m = m.max(self.particles.a[i].norm_squared());
            }
        }
        sqrt(m)
    }
}
