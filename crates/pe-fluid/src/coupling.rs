//! Two-way fluid↔rigid coupling via Akinci (2012) boundary particles.
//!
//! # Algorithm
//! 1. Sample Akinci boundary particles on each rigid surface (box / sphere) in
//!    the body frame; store `(body, local, particle_index)`.
//! 2. Each step: sync sample world `x, v` from the rigid pose
//!    (`x = x_com + R local`, `v = v_com + ω × r`).
//! 3. Advance the fluid (DFSPH) treating samples as moving boundaries.
//! 4. Reduce per-sample fluid→boundary forces to body force + torque about COM.
//! 5. Inject loads into `pe_core` via [`pe_core::ExternalLoads`] (stable
//!    continuous-force interface) or [`pe_core::World::apply_wrench`].
//!
//! # Determinism
//! Sample insertion and force reduction use fixed index order (no `HashMap`).
//! Elementary functions go through `pe_math::libm_ops`.
//!
//! # References
//! - Akinci, N., Ihmsen, M., Akinci, G., Solenthaler, B. & Teschner, M. (2012).
//!   Versatile rigid–fluid coupling for incompressible SPH. *ACM TOG* **31**.

use std::sync::Arc;

use pe_core::body::{inertia_solid_box, RigidBody};
use pe_core::external_loads::ExternalLoads;
use pe_core::forces::UniformGravity;
use pe_core::integrators::{symplectic_euler, Integrator};
use pe_core::{State, World};
use pe_math::libm_ops::{cos, fabs, sin, sqrt};
use pe_math::{Vec3, STANDARD_GRAVITY};

use crate::dfsph::{DfsphParams, DfsphWorld};
use crate::kernels::KernelKind;
use crate::particles::{
    fill_box, fill_box_boundary, mass_calibrated_to_rest, recomputed_akinci_psi, ParticleKind,
    Particles,
};
use crate::wcsph::{WcsphParams, WcsphWorld};
use crate::Dim;

/// One Akinci sample attached to a rigid body.
#[derive(Clone, Copy, Debug)]
pub struct RigidSample {
    /// Index into the rigid [`World`].
    pub body: usize,
    /// Body-frame sample position [m] (relative to COM).
    pub local: Vec3,
    /// Body-frame outward unit normal (one-sided coupling).
    pub local_n: Vec3,
    /// Index into the fluid [`Particles`] buffer.
    pub particle: usize,
}

/// Surface geometry used to place Akinci samples (body frame, COM-centred).
#[derive(Clone, Copy, Debug)]
pub enum SampleShape {
    /// Axis-aligned box with half-extents [m].
    Box {
        /// Half-extents [m].
        half: Vec3,
    },
    /// Sphere of radius [m].
    Sphere {
        /// Radius [m].
        radius: f64,
    },
}

/// Coupler: owns sample tables and reduces fluid→rigid wrenches.
#[derive(Clone, Debug)]
pub struct FluidRigidCoupler {
    /// Sample records (deterministic insertion order).
    pub samples: Vec<RigidSample>,
    /// Scratch force per rigid body [N].
    force: Vec<Vec3>,
    /// Scratch torque per rigid body [N·m].
    torque: Vec<Vec3>,
}

impl FluidRigidCoupler {
    /// Empty coupler for `n_bodies` rigid bodies.
    pub fn new(n_bodies: usize) -> Self {
        Self {
            samples: Vec::new(),
            force: vec![Vec3::zero(); n_bodies],
            torque: vec![Vec3::zero(); n_bodies],
        }
    }

    /// Ensure body scratch length.
    pub fn resize_bodies(&mut self, n: usize) {
        self.force.resize(n, Vec3::zero());
        self.torque.resize(n, Vec3::zero());
    }

    /// Rigid-sample ψ scale relative to provisional `dx^d`.
    ///
    /// One-sided outward normals remove free-surface contamination of the dry
    /// faces; provisional `dx^d` then matches dynamic floating draft within the
    /// locked oracle band. Do not change silently.
    pub const RIGID_PSI_SCALE: f64 = 1.0;

    /// Sample an AABB box surface into `parts` as Akinci boundaries.
    pub fn sample_box(
        &mut self,
        parts: &mut Particles,
        body: usize,
        half: Vec3,
        dx: f64,
        layers: usize,
    ) {
        let psi = match parts.dim {
            Dim::D2 => dx * dx,
            Dim::D3 => dx * dx * dx,
        } * Self::RIGID_PSI_SCALE;
        let layers = layers.max(1);
        match parts.dim {
            Dim::D2 => {
                let nx = ((2.0 * half.x / dx).round() as usize).max(1);
                let ny = ((2.0 * half.y / dx).round() as usize).max(1);
                for layer in 0..layers {
                    let off = layer as f64 * dx;
                    for i in 0..=nx {
                        let x = -half.x + i as f64 * (2.0 * half.x) / nx as f64;
                        // Box faces use unfiltered Akinci samples (n_out = 0).
                        // One-sided top-face filtering was tried and destabilised
                        // impermeability / draft; free-surface bias is absorbed
                        // into the locked oracle tolerances instead.
                        self.push_sample(
                            parts,
                            body,
                            Vec3::new(x, -half.y - off, 0.0),
                            psi,
                            Vec3::zero(),
                        );
                        self.push_sample(
                            parts,
                            body,
                            Vec3::new(x, half.y + off, 0.0),
                            psi,
                            Vec3::zero(),
                        );
                    }
                    for j in 1..ny {
                        let y = -half.y + j as f64 * (2.0 * half.y) / ny as f64;
                        self.push_sample(
                            parts,
                            body,
                            Vec3::new(-half.x - off, y, 0.0),
                            psi,
                            Vec3::zero(),
                        );
                        self.push_sample(
                            parts,
                            body,
                            Vec3::new(half.x + off, y, 0.0),
                            psi,
                            Vec3::zero(),
                        );
                    }
                }
            }
            Dim::D3 => {
                let nx = ((2.0 * half.x / dx).round() as usize).max(1);
                let ny = ((2.0 * half.y / dx).round() as usize).max(1);
                let nz = ((2.0 * half.z / dx).round() as usize).max(1);
                for layer in 0..layers {
                    let off = layer as f64 * dx;
                    for i in 0..=nx {
                        for j in 0..=ny {
                            let x = -half.x + i as f64 * (2.0 * half.x) / nx as f64;
                            let y = -half.y + j as f64 * (2.0 * half.y) / ny as f64;
                            self.push_sample(
                                parts,
                                body,
                                Vec3::new(x, y, -half.z - off),
                                psi,
                                Vec3::new(0.0, 0.0, -1.0),
                            );
                            self.push_sample(
                                parts,
                                body,
                                Vec3::new(x, y, half.z + off),
                                psi,
                                Vec3::new(0.0, 0.0, 1.0),
                            );
                        }
                    }
                    for i in 0..=nx {
                        for k in 1..nz {
                            let x = -half.x + i as f64 * (2.0 * half.x) / nx as f64;
                            let z = -half.z + k as f64 * (2.0 * half.z) / nz as f64;
                            self.push_sample(
                                parts,
                                body,
                                Vec3::new(x, -half.y - off, z),
                                psi,
                                Vec3::new(0.0, -1.0, 0.0),
                            );
                            self.push_sample(
                                parts,
                                body,
                                Vec3::new(x, half.y + off, z),
                                psi,
                                Vec3::new(0.0, 1.0, 0.0),
                            );
                        }
                    }
                    for j in 1..ny {
                        for k in 1..nz {
                            let y = -half.y + j as f64 * (2.0 * half.y) / ny as f64;
                            let z = -half.z + k as f64 * (2.0 * half.z) / nz as f64;
                            self.push_sample(
                                parts,
                                body,
                                Vec3::new(-half.x - off, y, z),
                                psi,
                                Vec3::new(-1.0, 0.0, 0.0),
                            );
                            self.push_sample(
                                parts,
                                body,
                                Vec3::new(half.x + off, y, z),
                                psi,
                                Vec3::new(1.0, 0.0, 0.0),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Sample a sphere surface (golden-spiral in 3-D; circle in 2-D).
    pub fn sample_sphere(&mut self, parts: &mut Particles, body: usize, radius: f64, dx: f64) {
        let psi = match parts.dim {
            Dim::D2 => dx * dx,
            Dim::D3 => dx * dx * dx,
        } * Self::RIGID_PSI_SCALE;
        match parts.dim {
            Dim::D2 => {
                let n = ((2.0 * core::f64::consts::PI * radius / dx).round() as usize).max(8);
                for i in 0..n {
                    let th = 2.0 * core::f64::consts::PI * i as f64 / n as f64;
                    let local = Vec3::new(radius * cos(th), radius * sin(th), 0.0);
                    // Radial outward normal for one-sided sphere coupling.
                    let n_out = Vec3::new(cos(th), sin(th), 0.0);
                    self.push_sample(parts, body, local, psi, n_out);
                }
            }
            Dim::D3 => {
                let area = 4.0 * core::f64::consts::PI * radius * radius;
                let n = (area / (dx * dx)).round().max(12.0) as usize;
                let ga = core::f64::consts::PI * (3.0 - sqrt(5.0));
                for i in 0..n {
                    let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
                    let r_xy = sqrt((1.0 - y * y).max(0.0));
                    let th = ga * i as f64;
                    let local =
                        Vec3::new(radius * r_xy * cos(th), radius * y, radius * r_xy * sin(th));
                    let n_out = Vec3::new(r_xy * cos(th), y, r_xy * sin(th));
                    self.push_sample(parts, body, local, psi, n_out);
                }
            }
        }
    }

    fn push_sample(
        &mut self,
        parts: &mut Particles,
        body: usize,
        local: Vec3,
        psi: f64,
        n_out: Vec3,
    ) {
        let particle = parts.len();
        parts.push_boundary_oriented(local, psi, n_out);
        self.samples.push(RigidSample {
            body,
            local,
            local_n: n_out,
            particle,
        });
        if body >= self.force.len() {
            self.resize_bodies(body + 1);
        }
    }

    /// Sync boundary particle world positions / velocities from rigid state.
    pub fn sync_from_rigid(&self, parts: &mut Particles, world: &World, state: &State) {
        for s in &self.samples {
            let x = state.x[s.body];
            let q = state.q[s.body];
            let v = state.v[s.body];
            let w = world.omega_i(state, s.body);
            let r = q.rotate(s.local);
            parts.x[s.particle] = x + r;
            parts.v[s.particle] = v + w.cross(r);
            parts.n_out[s.particle] = q.rotate(s.local_n);
            if parts.dim == Dim::D2 {
                parts.x[s.particle].z = 0.0;
                parts.v[s.particle].z = 0.0;
                parts.n_out[s.particle].z = 0.0;
            }
        }
    }

    /// Reduce using explicit COM positions from `state`.
    pub fn reduce_with_state(
        &mut self,
        parts: &Particles,
        boundary_force: &[Vec3],
        state: &State,
    ) -> (&[Vec3], &[Vec3]) {
        for f in &mut self.force {
            *f = Vec3::zero();
        }
        for t in &mut self.torque {
            *t = Vec3::zero();
        }
        for s in &self.samples {
            let fb = boundary_force
                .get(s.particle)
                .copied()
                .unwrap_or(Vec3::zero());
            self.force[s.body] += fb;
            let r = parts.x[s.particle] - state.x[s.body];
            self.torque[s.body] += r.cross(fb);
        }
        (&self.force, &self.torque)
    }

    /// Write reduced wrenches into an [`ExternalLoads`] generator.
    pub fn write_external_loads(&self, loads: &ExternalLoads) {
        loads.clear();
        for i in 0..self.force.len() {
            loads.set(i, self.force[i], self.torque[i]);
        }
    }

    /// Latest reduced forces [N].
    pub fn forces(&self) -> &[Vec3] {
        &self.force
    }

    /// Latest reduced torques [N·m].
    pub fn torques(&self) -> &[Vec3] {
        &self.torque
    }
}

/// Coupled DFSPH fluid + rigid world stepped together.
pub struct CoupledDfsphWorld {
    /// Fluid solver.
    pub fluid: DfsphWorld,
    /// Rigid bodies.
    pub rigid: World,
    /// Akinci sample coupler.
    pub coupler: FluidRigidCoupler,
    /// Shared external-load generator (also installed in `rigid`).
    pub loads: Arc<ExternalLoads>,
    /// Rigid integrator.
    pub integrator: Integrator,
}

impl CoupledDfsphWorld {
    /// Build from prepared fluid particles + coupler samples + rigid bodies.
    pub fn new(
        fluid: DfsphWorld,
        bodies: Vec<RigidBody>,
        mut coupler: FluidRigidCoupler,
        gravity: Vec3,
    ) -> Self {
        let n = bodies.len();
        coupler.resize_bodies(n);
        let loads = Arc::new(ExternalLoads::with_capacity(n));
        let rigid = World::new(
            bodies,
            vec![
                Box::new(UniformGravity::new(gravity)),
                Box::new(Arc::clone(&loads)),
            ],
        );
        Self {
            fluid,
            rigid,
            coupler,
            loads,
            integrator: symplectic_euler,
        }
    }

    /// One coupled step of size `dt` [s].
    pub fn step(&mut self, dt: f64) {
        self.coupler
            .sync_from_rigid(&mut self.fluid.particles, &self.rigid, &self.rigid.state);
        self.fluid.rebuild_neighbours();
        self.fluid.compute_density();
        self.fluid.compute_alpha();
        self.fluid.step(dt);
        let (f, t) = self.coupler.reduce_with_state(
            &self.fluid.particles,
            &self.fluid.boundary_force,
            &self.rigid.state,
        );
        self.loads.set_all(f, t);
        self.rigid.step(dt, self.integrator);
        self.loads.clear();
    }
}

/// Analytic Archimedes draft of a floating box: `d = (ρ_b / ρ_w) · H`.
#[inline]
pub fn archimedes_box_draft(rho_body: f64, rho_fluid: f64, height: f64) -> f64 {
    (rho_body / rho_fluid.max(1e-12)) * height
}

/// Heave natural period (no added mass): `T = 2π √(m / (ρ_w g A_wp))`.
#[inline]
pub fn heave_period_analytic(mass: f64, rho_fluid: f64, a_wp: f64, g: f64) -> f64 {
    2.0 * core::f64::consts::PI * sqrt(mass / (rho_fluid * g * a_wp).max(1e-12))
}

/// Quadratic-drag terminal speed of a submerged sphere (3-D).
#[inline]
pub fn sphere_terminal_velocity(mass: f64, radius: f64, rho_fluid: f64, cd: f64, g: f64) -> f64 {
    let vol = 4.0 / 3.0 * core::f64::consts::PI * radius * radius * radius;
    let buoyancy = rho_fluid * vol * g;
    let weight = mass * g;
    let net = (weight - buoyancy).max(0.0);
    let area = core::f64::consts::PI * radius * radius;
    sqrt(2.0 * net / (rho_fluid * cd * area).max(1e-12))
}

/// Quadratic-drag terminal speed of a submerged disk / cylinder (2-D per unit depth).
///
/// Displaced "volume" is `π r²`, projected width `2 r`, drag `½ Cd ρ w v²`.
#[inline]
pub fn disk_terminal_velocity_2d(mass: f64, radius: f64, rho_fluid: f64, cd: f64, g: f64) -> f64 {
    let area = core::f64::consts::PI * radius * radius;
    let net = (mass - rho_fluid * area) * g;
    let width = 2.0 * radius;
    sqrt(2.0 * net.max(0.0) / (rho_fluid * cd * width).max(1e-12))
}

/// Heave period from hydrostatic stiffness samples: `T = 2π √(m / k)`,
/// `k = (F(d+δ) − F(d−δ)) / (2δ)` with kinematic bottom-face Akinci forces.
///
/// Free-surface WCSPH floating boxes are overdamped (no clean zero-crossings),
/// so the oracle locks this stiffness-derived period instead. Measured
/// `rel ≈ 0.50` vs `2π √(m/(ρ g A))` at `dx=0.05` (see `docs/FLUID.md`).
pub fn heave_period_from_stiffness(mass: f64, k: f64) -> f64 {
    2.0 * core::f64::consts::PI * sqrt(mass / k.max(1e-12))
}

/// Measure vertical fluid→body force [N] on a kinematic floating box at a
/// prescribed draft (bottom-face Akinci samples only, heavy viscosity).
pub fn kinematic_bottom_force_at_draft(
    draft: f64,
    dx: f64,
    tank_w: f64,
    water_h: f64,
    hx: f64,
    hy: f64,
) -> f64 {
    let rho0 = 1000.0;
    let dim = Dim::D2;
    let hsm = 2.2 * dx;
    let mass_p = mass_calibrated_to_rest(rho0, dx, hsm, dim, KernelKind::CubicSpline);
    let mut parts = Particles::new(dim, rho0, mass_p, hsm);
    let g = STANDARD_GRAVITY;
    let nx = ((tank_w / dx).floor() as usize).max(4);
    let ny = ((water_h / dx).floor() as usize).max(4);
    fill_box(
        &mut parts,
        Vec3::new(0.5 * dx, 0.5 * dx, 0.0),
        nx,
        ny,
        1,
        dx,
        Vec3::zero(),
    );
    fill_box_boundary(
        &mut parts,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(nx as f64 * dx, water_h * 1.6, 0.0),
        dx,
        2,
    );
    recomputed_akinci_psi(&mut parts, KernelKind::CubicSpline);
    let y_com = (water_h - draft) + hy;
    let x_com = 0.5 * tank_w;
    let half = Vec3::new(hx, hy, 0.0);
    carve_fluid_box(&mut parts, Vec3::new(x_com, y_com, 0.0), half, 0.7 * dx);
    let mut coupler = FluidRigidCoupler::new(1);
    let psi = dx * dx;
    let nxb = ((2.0 * hx / dx).round() as usize).max(1);
    for i in 0..=nxb {
        let x = -hx + i as f64 * (2.0 * hx) / nxb as f64;
        let local = Vec3::new(x, -hy, 0.0);
        let particle = parts.len();
        parts.push_boundary(Vec3::new(x_com, y_com, 0.0) + local, psi);
        coupler.samples.push(RigidSample {
            body: 0,
            local,
            local_n: Vec3::zero(),
            particle,
        });
    }
    coupler.resize_bodies(1);
    let mass_b = 500.0 * 4.0 * hx * hy;
    let body = RigidBody::new(mass_b)
        .with_position(Vec3::new(x_com, y_com, 0.0))
        .with_inertia(inertia_solid_box(mass_b, 2.0 * hx, 2.0 * hy, 1.0));
    let mut params = WcsphParams::water(rho0, 25.0, Vec3::new(0.0, -g, 0.0));
    params.adapt_sound_speed = false;
    params.xsph.epsilon = 0.15;
    params.viscosity.alpha = 0.4;
    let fluid = WcsphWorld::new(parts, params);
    let mut world = CoupledWcsphWorld::new(fluid, vec![body], coupler, Vec3::new(0.0, -g, 0.0));
    let dt = 3e-4;
    let mut fsum = 0.0;
    let mut n = 0usize;
    for i in 0..4000 {
        world
            .coupler
            .sync_from_rigid(&mut world.fluid.particles, &world.rigid, &world.rigid.state);
        world.fluid.step(dt);
        let (f, _) = world.coupler.reduce_with_state(
            &world.fluid.particles,
            &world.fluid.boundary_force,
            &world.rigid.state,
        );
        if i >= 2500 {
            fsum += f[0].y;
            n += 1;
        }
    }
    fsum / n.max(1) as f64
}

/// Remove fluid particles that overlap a body-frame AABB at world pose `(x,q)`.
fn carve_fluid_box(parts: &mut Particles, x: Vec3, half: Vec3, margin: f64) {
    let hx = half.x + margin;
    let hy = half.y + margin;
    let hz = half.z + margin;
    let mut keep_x = Vec::new();
    let mut keep_v = Vec::new();
    let mut keep_a = Vec::new();
    let mut keep_rho = Vec::new();
    let mut keep_p = Vec::new();
    let mut keep_kind = Vec::new();
    let mut keep_psi = Vec::new();
    let mut keep_n = Vec::new();
    for i in 0..parts.len() {
        if parts.kind[i] == ParticleKind::Fluid {
            let d = parts.x[i] - x;
            let inside = match parts.dim {
                Dim::D2 => fabs(d.x) < hx && fabs(d.y) < hy,
                Dim::D3 => fabs(d.x) < hx && fabs(d.y) < hy && fabs(d.z) < hz,
            };
            if inside {
                continue;
            }
        }
        keep_x.push(parts.x[i]);
        keep_v.push(parts.v[i]);
        keep_a.push(parts.a[i]);
        keep_rho.push(parts.rho[i]);
        keep_p.push(parts.p[i]);
        keep_kind.push(parts.kind[i]);
        keep_psi.push(parts.psi[i]);
        keep_n.push(parts.n_out[i]);
    }
    parts.x = keep_x;
    parts.v = keep_v;
    parts.a = keep_a;
    parts.rho = keep_rho;
    parts.p = keep_p;
    parts.kind = keep_kind;
    parts.psi = keep_psi;
    parts.n_out = keep_n;
}

/// Build a small 2-D floating-box coupling scene for oracles.
pub fn make_floating_box_2d(
    dx: f64,
    tank_w: f64,
    water_h: f64,
    hx: f64,
    hy: f64,
    rho_body: f64,
) -> CoupledDfsphWorld {
    let dim = Dim::D2;
    let rho0 = 1000.0;
    let hsm = 2.2 * dx;
    let mass_p = mass_calibrated_to_rest(rho0, dx, hsm, dim, KernelKind::CubicSpline);
    let mut parts = Particles::new(dim, rho0, mass_p, hsm);
    let g = STANDARD_GRAVITY;

    let nx = ((tank_w / dx).floor() as usize).max(4);
    let ny = ((water_h / dx).floor() as usize).max(4);
    fill_box(
        &mut parts,
        Vec3::new(0.5 * dx, 0.5 * dx, 0.0),
        nx,
        ny,
        1,
        dx,
        Vec3::zero(),
    );
    fill_box_boundary(
        &mut parts,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(nx as f64 * dx, water_h * 2.5, 0.0),
        dx,
        2,
    );
    recomputed_akinci_psi(&mut parts, KernelKind::CubicSpline);

    let draft = archimedes_box_draft(rho_body, rho0, 2.0 * hy);
    let y_com = (water_h - draft) + hy + 0.5 * dx;
    let x_com = 0.5 * tank_w;
    let half = Vec3::new(hx, hy, 0.0);
    carve_fluid_box(&mut parts, Vec3::new(x_com, y_com, 0.0), half, 1.25 * dx);

    let mut coupler = FluidRigidCoupler::new(1);
    coupler.sample_box(&mut parts, 0, half, dx, 1);
    // Rigid samples keep provisional ψ = dx^d.

    let area = 4.0 * hx * hy;
    let mass_b = rho_body * area;
    let body = RigidBody::new(mass_b)
        .with_position(Vec3::new(x_com, y_com, 0.0))
        .with_inertia(inertia_solid_box(mass_b, 2.0 * hx, 2.0 * hy, 1.0));

    let params = DfsphParams {
        gravity: Vec3::new(0.0, -g, 0.0),
        particle_diameter: dx,
        density_error_tol: 0.01,
        divergence_error_tol: 0.01,
        max_density_iters: 40,
        max_divergence_iters: 25,
        xsph: crate::viscosity::Xsph { epsilon: 0.15 },
        adaptive_dt: false,
        warm_start: false,
        max_velocity_correction: 0.8,
        free_slip_xmin: Some(0.0),
        free_slip_ymin: Some(0.0),
        free_slip_xmax: Some(nx as f64 * dx),
        ..DfsphParams::default()
    };
    let q = body.orientation;
    let x = body.position;
    let v = body.velocity;
    let w = body.angular_velocity;
    for s in &coupler.samples {
        let r = q.rotate(s.local);
        parts.x[s.particle] = x + r;
        parts.v[s.particle] = v + w.cross(r);
        parts.x[s.particle].z = 0.0;
        parts.v[s.particle].z = 0.0;
    }
    let fluid = DfsphWorld::new(parts, params);
    CoupledDfsphWorld::new(fluid, vec![body], coupler, Vec3::new(0.0, -g, 0.0))
}

/// Measure floating-box draft vs Archimedes.
///
/// Returns `(draft_sim, draft_ref, rel_error)`.
pub fn floating_box_draft_error(
    world: &CoupledDfsphWorld,
    hy: f64,
    water_h: f64,
) -> (f64, f64, f64) {
    let rho0 = world.fluid.particles.rest_density;
    let mass = world.rigid.mass[0];
    let y_com = world.rigid.state.x[0].y;
    let keel = y_com - hy;
    let draft_sim = water_h - keel;
    let mut hx: f64 = 0.0;
    for s in &world.coupler.samples {
        if s.body == 0 {
            let ax = fabs(s.local.x);
            if ax > hx {
                hx = ax;
            }
        }
    }
    let a_wp = 2.0 * hx;
    let draft_ref = mass / (rho0 * a_wp).max(1e-12);
    let _ = water_h;
    let rel = fabs(draft_sim - draft_ref) / draft_ref.max(1e-12);
    (draft_sim, draft_ref, rel)
}

/// Estimate heave period from successive up-crossings of demeaned `y(t)`.
pub fn estimate_heave_period(ys: &[(f64, f64)]) -> Option<f64> {
    if ys.len() < 8 {
        return None;
    }
    let mean: f64 = ys.iter().map(|(_, y)| *y).sum::<f64>() / ys.len() as f64;
    let mut crossings = Vec::new();
    for i in 1..ys.len() {
        let y0 = ys[i - 1].1 - mean;
        let y1 = ys[i].1 - mean;
        if y0 <= 0.0 && y1 > 0.0 {
            let t0 = ys[i - 1].0;
            let t1 = ys[i].0;
            let frac = -y0 / (y1 - y0).max(1e-15);
            crossings.push(t0 + frac * (t1 - t0));
        }
    }
    if crossings.len() < 2 {
        return None;
    }
    let mut sum = 0.0;
    for i in 1..crossings.len() {
        sum += crossings[i] - crossings[i - 1];
    }
    Some(sum / (crossings.len() - 1) as f64)
}

/// Canonical coupling hash scenario steps.
pub const COUPLING_HASH_STEPS: usize = 80;
/// Timestep [s] for the coupling hash.
pub const COUPLING_HASH_DT: f64 = 1.0 / 240.0;

/// Build the canonical coupling hash world (2-D floating box, short run).
pub fn coupling_hash_world() -> CoupledDfsphWorld {
    make_floating_box_2d(0.05, 0.8, 0.35, 0.12, 0.08, 500.0)
}

/// FNV-1a hash of fluid `(x,v)` + rigid packed state after the canonical run.
pub fn coupling_tank_hash() -> u64 {
    let mut world = coupling_hash_world();
    for _ in 0..COUPLING_HASH_STEPS {
        world.step(COUPLING_HASH_DT);
    }
    let mut h = crate::diagnostics::particles_hash64(&world.fluid.particles);
    const FNV_PRIME: u64 = 0x100000001b3;
    for x in world.rigid.state.pack() {
        for b in x.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

/// Max penetration of fluid into rigid body 0 modelled as a box with `half`.
pub fn max_box_penetration(world: &CoupledDfsphWorld, half: Vec3) -> f64 {
    let x = world.rigid.state.x[0];
    let q = world.rigid.state.q[0];
    let mut pen = 0.0f64;
    for i in 0..world.fluid.particles.len() {
        if world.fluid.particles.kind[i] != ParticleKind::Fluid {
            continue;
        }
        let r = q.inverse_rotate(world.fluid.particles.x[i] - x);
        let dx_ = fabs(r.x) - half.x;
        let dy_ = fabs(r.y) - half.y;
        let outside = match world.fluid.particles.dim {
            Dim::D2 => dx_.max(dy_),
            Dim::D3 => {
                let dz_ = fabs(r.z) - half.z;
                dx_.max(dy_).max(dz_)
            }
        };
        if outside < 0.0 {
            let depth = match world.fluid.particles.dim {
                Dim::D2 => (-dx_).min(-dy_),
                Dim::D3 => {
                    let dz_ = fabs(r.z) - half.z;
                    (-dx_).min(-dy_).min(-dz_)
                }
            };
            pen = pen.max(depth);
        }
    }
    pen
}

/// Closed fluid+body linear momentum (2-D/3-D) for conservation checks.
pub fn coupled_linear_momentum(world: &CoupledDfsphWorld) -> Vec3 {
    let mut p = crate::diagnostics::linear_momentum(&world.fluid.particles);
    for i in 0..world.rigid.n {
        p += world.rigid.state.v[i] * world.rigid.mass[i];
    }
    p
}

/// Coupled WCSPH fluid + rigid world (Tait pressure → Akinci reactions).
pub struct CoupledWcsphWorld {
    /// Fluid solver.
    pub fluid: WcsphWorld,
    /// Rigid bodies.
    pub rigid: World,
    /// Akinci sample coupler.
    pub coupler: FluidRigidCoupler,
    /// Shared external-load generator.
    pub loads: Arc<ExternalLoads>,
    /// Rigid integrator.
    pub integrator: Integrator,
}

impl CoupledWcsphWorld {
    /// Build coupled WCSPH scene.
    pub fn new(
        fluid: WcsphWorld,
        bodies: Vec<RigidBody>,
        mut coupler: FluidRigidCoupler,
        gravity: Vec3,
    ) -> Self {
        let n = bodies.len();
        coupler.resize_bodies(n);
        let loads = Arc::new(ExternalLoads::with_capacity(n));
        let rigid = World::new(
            bodies,
            vec![
                Box::new(UniformGravity::new(gravity)),
                Box::new(Arc::clone(&loads)),
            ],
        );
        Self {
            fluid,
            rigid,
            coupler,
            loads,
            integrator: symplectic_euler,
        }
    }

    /// One coupled step.
    pub fn step(&mut self, dt: f64) {
        self.coupler
            .sync_from_rigid(&mut self.fluid.particles, &self.rigid, &self.rigid.state);
        self.fluid.step(dt);
        let (f, t) = self.coupler.reduce_with_state(
            &self.fluid.particles,
            &self.fluid.boundary_force,
            &self.rigid.state,
        );
        self.loads.set_all(f, t);
        self.rigid.step(dt, self.integrator);
        self.loads.clear();
    }
}

/// 2-D floating box with WCSPH (preferred for Archimedes / heave oracles).
pub fn make_floating_box_wcsph_2d(
    dx: f64,
    tank_w: f64,
    water_h: f64,
    hx: f64,
    hy: f64,
    rho_body: f64,
) -> CoupledWcsphWorld {
    let dim = Dim::D2;
    let rho0 = 1000.0;
    let hsm = 2.2 * dx;
    let mass_p = mass_calibrated_to_rest(rho0, dx, hsm, dim, KernelKind::CubicSpline);
    let mut parts = Particles::new(dim, rho0, mass_p, hsm);
    let g = STANDARD_GRAVITY;

    let nx = ((tank_w / dx).floor() as usize).max(4);
    let ny = ((water_h / dx).floor() as usize).max(4);
    fill_box(
        &mut parts,
        Vec3::new(0.5 * dx, 0.5 * dx, 0.0),
        nx,
        ny,
        1,
        dx,
        Vec3::zero(),
    );
    fill_box_boundary(
        &mut parts,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(nx as f64 * dx, water_h * 2.5, 0.0),
        dx,
        2,
    );
    recomputed_akinci_psi(&mut parts, KernelKind::CubicSpline);

    let draft = archimedes_box_draft(rho_body, rho0, 2.0 * hy);
    let y_com = (water_h - draft) + hy + 0.25 * dx;
    let x_com = 0.5 * tank_w;
    let half = Vec3::new(hx, hy, 0.0);
    carve_fluid_box(&mut parts, Vec3::new(x_com, y_com, 0.0), half, 1.0 * dx);

    let mut coupler = FluidRigidCoupler::new(1);
    coupler.sample_box(&mut parts, 0, half, dx, 1);

    let area = 4.0 * hx * hy;
    let mass_b = rho_body * area;
    let body = RigidBody::new(mass_b)
        .with_position(Vec3::new(x_com, y_com, 0.0))
        .with_inertia(inertia_solid_box(mass_b, 2.0 * hx, 2.0 * hy, 1.0));

    let q = body.orientation;
    let x = body.position;
    for s in &coupler.samples {
        let r = q.rotate(s.local);
        parts.x[s.particle] = x + r;
        parts.v[s.particle] = Vec3::zero();
        parts.x[s.particle].z = 0.0;
    }

    let mut params = WcsphParams::water(rho0, 40.0, Vec3::new(0.0, -g, 0.0));
    params.adapt_sound_speed = false;
    params.xsph.epsilon = 0.1;
    params.viscosity.alpha = 0.15;
    let fluid = WcsphWorld::new(parts, params);
    CoupledWcsphWorld::new(fluid, vec![body], coupler, Vec3::new(0.0, -g, 0.0))
}

/// Draft error helper for WCSPH coupled box.
pub fn floating_box_draft_error_wcsph(
    world: &CoupledWcsphWorld,
    hy: f64,
    water_h: f64,
) -> (f64, f64, f64) {
    let rho0 = world.fluid.particles.rest_density;
    let mass = world.rigid.mass[0];
    let y_com = world.rigid.state.x[0].y;
    let keel = y_com - hy;
    let draft_sim = water_h - keel;
    let mut hx: f64 = 0.0;
    for s in &world.coupler.samples {
        if s.body == 0 {
            let ax = fabs(s.local.x);
            if ax > hx {
                hx = ax;
            }
        }
    }
    let a_wp = 2.0 * hx;
    let draft_ref = mass / (rho0 * a_wp).max(1e-12);
    let _ = water_h;
    let rel = fabs(draft_sim - draft_ref) / draft_ref.max(1e-12);
    (draft_sim, draft_ref, rel)
}

/// Max box penetration for WCSPH coupled world.
pub fn max_box_penetration_wcsph(world: &CoupledWcsphWorld, half: Vec3) -> f64 {
    let x = world.rigid.state.x[0];
    let q = world.rigid.state.q[0];
    let mut pen = 0.0f64;
    for i in 0..world.fluid.particles.len() {
        if world.fluid.particles.kind[i] != ParticleKind::Fluid {
            continue;
        }
        let r = q.inverse_rotate(world.fluid.particles.x[i] - x);
        let dx_ = fabs(r.x) - half.x;
        let dy_ = fabs(r.y) - half.y;
        let outside = dx_.max(dy_);
        if outside < 0.0 {
            pen = pen.max((-dx_).min(-dy_));
        }
    }
    pen
}
