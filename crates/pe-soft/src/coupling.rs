//! Two-way soft↔rigid coupling (XPBD particles ↔ `pe-core` rigid bodies).
//!
//! # Method
//!
//! Each outer step `dt` is split into `N` XPBD substeps `h = dt/N` (Macklin 2019).
//! Per substep:
//! 1. Predict soft particles and rigid bodies under gravity (symplectic).
//! 2. Project soft XPBD constraints.
//! 3. Project **attachment** constraints (particle pinned to a body-frame point)
//!    with a shared 3×3 XPBD solve using the rigid generalized inverse mass at
//!    the attach point (`W = m⁻¹I − [r×] I⁻¹ [r×]`).
//! 4. Resolve **contacts** of soft particles against dynamic rigid boxes/spheres
//!    (unilateral XPBD along the contact normal; equal-and-opposite positional
//!    impulse on the rigid).
//! 5. Soft static/self collisions (unchanged one-way statics).
//! 6. Rigid↔static plane contacts (so boxes can rest on the ground).
//! 7. Reconstruct soft and rigid velocities from positions / orientations.
//!
//! Attachment and contact exchange momentum both ways. Isolated systems without
//! external impulses conserve total linear momentum to solver tolerance.
//! Constraint force estimates use Macklin XPBD `f ≈ (λ / h²) ∇C`.
//!
//! # Shapes
//!
//! Dynamic rigid colliders: **OBB box** and **sphere** (particle vs convex).
//! Static planes affect soft particles (via [`crate::collision`]) and rigid
//! bodies (plane–box / plane–sphere positional contacts in this module).
//!
//! # Determinism
//!
//! Fixed body/attachment/collider index order; no `HashMap` on the step path;
//! portable `libm` only; no FMA. See [`docs/DETERMINISM.md`](../../docs/DETERMINISM.md).
//!
//! # References (algorithms only — clean-room)
//!
//! - Macklin, Müller, Chentanez (2016). XPBD.
//! - Macklin (2019). Small Steps in Physics Simulation.
//! - Müller et al. (2007). Position Based Dynamics (rigid generalized mass).
//! - Deul, Charrier, Bender (2014/2016). Direct position-based rigid/soft coupling ideas.
//!
//! No third-party source was copied; see [`THIRD_PARTY.md`](../../THIRD_PARTY.md).

use pe_core::body::{inertia_solid_box, inertia_solid_sphere, mat_mul, mat_vec, RigidBody};
use pe_core::integrators::drift_rotation;
use pe_core::World;
use pe_math::libm_ops::atan2;
use pe_math::{Quat, Vec3};

use crate::builders::{make_rope, make_soft_cube};
use crate::collision::{
    collide_self_with_scratch, collide_statics, SelfCollisionScratch, StaticPlane,
};
use crate::constraint::{project_all, DistanceConstraint, Multiplier};
use crate::material::material_by_name;
use crate::particle::Particle;
use crate::solver::{soft_state_hash64, SoftParams, SoftWorld};

/// Dynamic rigid collider attached to a `pe-core` body (body-frame, COM-centred).
#[derive(Clone, Copy, Debug)]
pub enum RigidShape {
    /// Axis-aligned box in the body frame with half-extents [m].
    Box {
        /// Half-extents [m].
        half: Vec3,
    },
    /// Sphere of radius [m] centred on the COM.
    Sphere {
        /// Radius [m].
        radius: f64,
    },
}

/// Soft particle vs dynamic rigid collider.
#[derive(Clone, Copy, Debug)]
pub struct RigidCollider {
    /// Index into the rigid [`World`].
    pub body: usize,
    /// Shape in the body frame.
    pub shape: RigidShape,
    /// Coulomb friction μ (tangential velocity damp on the soft particle).
    pub friction: f64,
    /// Restitution e (normal velocity response on the soft particle).
    pub restitution: f64,
}

/// Pin a soft particle to a point fixed in a rigid body's frame.
///
/// Constraint `C = x_p − (x_com + R local)` (3-vector). Compliance `α` [m/N]
/// per axis (`α = 0` is the hard pin). Multipliers reset each substep.
#[derive(Clone, Debug)]
pub struct AttachmentConstraint {
    /// Soft particle index.
    pub particle: usize,
    /// Rigid body index.
    pub body: usize,
    /// Body-frame attach point relative to COM [m].
    pub local: Vec3,
    /// Compliance α [m/N] (same on each axis).
    pub compliance: f64,
    /// Accumulated XPBD multipliers (x, y, z) for the current substep.
    pub lambda: Vec3,
}

impl AttachmentConstraint {
    /// Hard (α = 0) attachment.
    pub fn hard(particle: usize, body: usize, local: Vec3) -> Self {
        Self {
            particle,
            body,
            local,
            compliance: 0.0,
            lambda: Vec3::zero(),
        }
    }

    /// Soft attachment with compliance α [m/N].
    pub fn soft(particle: usize, body: usize, local: Vec3, compliance: f64) -> Self {
        Self {
            particle,
            body,
            local,
            compliance,
            lambda: Vec3::zero(),
        }
    }
}

/// Contact / attachment force accumulator for oracles (world-frame).
#[derive(Clone, Debug, Default)]
pub struct CouplingDiagnostics {
    /// Sum of ground→rigid contact forces over the last outer step [N].
    pub ground_force: Vec3,
    /// Sum of soft→rigid contact+attachment forces over the last outer step [N].
    pub soft_on_rigid_force: Vec3,
}

/// Coupled soft XPBD + rigid world.
pub struct SoftRigidWorld {
    /// Soft particles / XPBD constraints / static+self collision.
    pub soft: SoftWorld,
    /// Dynamic rigid bodies (`pe-core`).
    pub rigid: World,
    /// Dynamic colliders (fixed index order).
    pub colliders: Vec<RigidCollider>,
    /// Attachment constraints (fixed index order).
    pub attachments: Vec<AttachmentConstraint>,
    /// Static planes that also support rigid bodies (in addition to soft statics).
    pub rigid_planes: Vec<StaticPlane>,
    /// Last-step force diagnostics.
    pub diagnostics: CouplingDiagnostics,
    /// Scratch: rigid COM at substep start.
    rigid_x_prev: Vec<Vec3>,
    /// Scratch: rigid quaternions at substep start.
    rigid_q_prev: Vec<Quat>,
    /// Reused soft self-collision spatial-hash buffers.
    self_collision_scratch: SelfCollisionScratch,
}

impl SoftRigidWorld {
    /// Build from a soft world and rigid bodies (no gravity generators — gravity
    /// comes from [`SoftParams::gravity`] applied to both sides).
    pub fn new(soft: SoftWorld, bodies: Vec<RigidBody>) -> Self {
        let n = bodies.len();
        // Gravity is applied explicitly in `predict_rigid` so the same `g` vector
        // drives soft and rigid. Register an empty generator list.
        let rigid = World::new(bodies, Vec::new());
        Self {
            soft,
            rigid,
            colliders: Vec::new(),
            attachments: Vec::new(),
            rigid_planes: Vec::new(),
            diagnostics: CouplingDiagnostics::default(),
            rigid_x_prev: vec![Vec3::zero(); n],
            rigid_q_prev: vec![Quat::identity(); n],
            self_collision_scratch: SelfCollisionScratch::default(),
        }
    }

    /// Convenience: [`new`] with `soft.params.gravity` set to `gravity`.
    ///
    /// Gravity is applied explicitly in prediction (same `g` for soft and rigid);
    /// no `ForceGenerator` is registered.
    pub fn with_gravity(mut soft: SoftWorld, bodies: Vec<RigidBody>, gravity: Vec3) -> Self {
        soft.params.gravity = gravity;
        Self::new(soft, bodies)
    }

    /// Add a dynamic rigid collider.
    pub fn add_collider(&mut self, c: RigidCollider) {
        self.colliders.push(c);
    }

    /// Add an attachment constraint.
    pub fn add_attachment(&mut self, a: AttachmentConstraint) {
        self.attachments.push(a);
    }

    /// Add a static plane that supports both soft particles and rigid bodies.
    pub fn add_ground_plane(&mut self, plane: StaticPlane) {
        self.soft.colliders.planes.push(plane);
        self.rigid_planes.push(plane);
    }

    /// Total soft particle mass [kg] (pinned contribute 0).
    pub fn soft_mass(&self) -> f64 {
        self.soft
            .particles
            .iter()
            .filter(|p| p.inv_mass > 0.0)
            .map(|p| 1.0 / p.inv_mass)
            .sum()
    }

    /// Total rigid mass [kg].
    pub fn rigid_mass(&self) -> f64 {
        self.rigid.mass.iter().sum()
    }

    /// Combined centre of mass [m] (soft dynamic particles + rigid COMs).
    pub fn combined_com(&self) -> Vec3 {
        let mut m_tot = 0.0;
        let mut mx = Vec3::zero();
        for p in &self.soft.particles {
            if p.inv_mass == 0.0 {
                continue;
            }
            let m = 1.0 / p.inv_mass;
            m_tot += m;
            mx += p.x * m;
        }
        for i in 0..self.rigid.n {
            let m = self.rigid.mass[i];
            m_tot += m;
            mx += self.rigid.state.x[i] * m;
        }
        if m_tot > 0.0 {
            mx * (1.0 / m_tot)
        } else {
            Vec3::zero()
        }
    }

    /// Total linear momentum [kg m/s] (soft + rigid).
    pub fn total_linear_momentum(&self) -> Vec3 {
        let mut p = Vec3::zero();
        for part in &self.soft.particles {
            if part.inv_mass == 0.0 {
                continue;
            }
            p += part.v * (1.0 / part.inv_mass);
        }
        for i in 0..self.rigid.n {
            p += self.rigid.state.v[i] * self.rigid.mass[i];
        }
        p
    }

    /// Kinetic energy [J] of soft + rigid.
    pub fn total_kinetic_energy(&self) -> f64 {
        let mut ke = crate::energy::kinetic_energy(&self.soft);
        for i in 0..self.rigid.n {
            let m = self.rigid.mass[i];
            ke += 0.5 * m * self.rigid.state.v[i].norm_squared();
            let w = self.rigid.omega_i(&self.rigid.state, i);
            // ½ ω · L
            ke += 0.5 * w.dot(self.rigid.state.l[i]);
        }
        ke
    }

    /// Total mechanical energy estimate [J] (KE + soft gravity PE + stretch PE +
    /// rigid gravity PE). Rigid rotational KE is included in [`total_kinetic_energy`].
    pub fn total_energy(&self) -> f64 {
        let mut e = crate::energy::total_energy(&self.soft);
        // Soft total_energy already has soft KE + soft grav + stretch; add rigid
        // translational/rotational KE and rigid grav (subtract soft KE already in
        // total_energy and replace with combined KE).
        e -= crate::energy::kinetic_energy(&self.soft);
        e += self.total_kinetic_energy();
        let g = self.soft.params.gravity;
        for i in 0..self.rigid.n {
            e += -self.rigid.mass[i] * g.dot(self.rigid.state.x[i]);
        }
        e
    }

    /// One outer coupled step of size `dt` [s].
    pub fn step(&mut self, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        let n_sub = self.soft.params.substeps.max(1);
        let h = dt / n_sub as f64;
        self.diagnostics = CouplingDiagnostics::default();
        let mut ground_acc = Vec3::zero();
        let mut soft_on_rigid_acc = Vec3::zero();
        for _ in 0..n_sub {
            let (dg, ds) = self.substep(h);
            // XPBD force ≈ λ/h²; we accumulate λ-scale impulses and convert once.
            ground_acc += dg;
            soft_on_rigid_acc += ds;
        }
        // Positional impulses over the outer step → average force F ≈ Σp / (h·dt)
        // with p = Δλ (kg·m). Equivalent: F = Σ(λ/h²)·(h/n_sub wait):
        // Per substep force estimate f = λ_final/h²; average F = mean_sub(f).
        // We store sum of λ over substeps in acc; F = acc / (h² * n_sub).
        let scale = 1.0 / (h * h * n_sub as f64);
        self.diagnostics.ground_force = ground_acc * scale;
        self.diagnostics.soft_on_rigid_force = soft_on_rigid_acc * scale;
        self.soft.t += dt;
        self.rigid.t += dt;
    }

    /// Integrate until `t_end` with fixed outer `dt`.
    pub fn run_for(&mut self, t_end: f64, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        while self.soft.t + dt <= t_end + 1e-15 {
            self.step(dt);
        }
    }

    /// Returns `(ground_λ_sum, soft_on_rigid_λ_sum)` for force diagnostics.
    fn substep(&mut self, h: f64) -> (Vec3, Vec3) {
        self.predict_soft(h);
        self.predict_rigid(h);

        for a in &mut self.attachments {
            a.lambda = Vec3::zero();
        }
        self.soft.constraints.reset_lambdas();

        let mut ground_lambda = Vec3::zero();
        let mut soft_rigid_lambda = Vec3::zero();
        let iters = self.soft.params.iterations.max(1);
        for _ in 0..iters {
            project_all(&mut self.soft.particles, &mut self.soft.constraints, h);
            soft_rigid_lambda += self.project_attachments(h);
            soft_rigid_lambda += self.collide_soft_rigid(h);
            collide_statics(&mut self.soft.particles, &self.soft.colliders);
            collide_self_with_scratch(
                &mut self.soft.particles,
                &self.soft.self_collision,
                &mut self.self_collision_scratch,
            );
            ground_lambda += self.collide_rigid_planes(h);
        }

        self.update_soft_velocities(h);
        self.update_rigid_velocities(h);
        (ground_lambda, soft_rigid_lambda)
    }

    fn predict_soft(&mut self, h: f64) {
        let g = self.soft.params.gravity;
        let damp = self.soft.params.damping;
        for p in &mut self.soft.particles {
            p.x_prev = p.x;
            if p.inv_mass == 0.0 {
                continue;
            }
            let a = g + p.ext_force * p.inv_mass;
            p.v += a * h;
            if damp > 0.0 {
                p.v *= 1.0 / (1.0 + damp * h);
            }
            p.x += p.v * h;
        }
    }

    fn predict_rigid(&mut self, h: f64) {
        let g = self.soft.params.gravity;
        let n = self.rigid.n;
        if self.rigid_x_prev.len() != n {
            self.rigid_x_prev.resize(n, Vec3::zero());
            self.rigid_q_prev.resize(n, Quat::identity());
        }
        // If UniformGravity is registered, derivative already includes it — but
        // we apply g explicitly here and skip force generators to keep soft/rigid
        // gravity identical. Clear any generator contribution by not calling them.
        for i in 0..n {
            self.rigid_x_prev[i] = self.rigid.state.x[i];
            self.rigid_q_prev[i] = self.rigid.state.q[i];
            self.rigid.state.v[i] += g * h;
            self.rigid.state.x[i] += self.rigid.state.v[i] * h;
        }
        // Orientation drift with frozen L (same helper as pe-core contact path).
        let q_new = drift_rotation(&self.rigid, &self.rigid.state.q, &self.rigid.state.l, h);
        self.rigid.state.q = q_new;
    }

    fn update_soft_velocities(&mut self, h: f64) {
        let inv_h = 1.0 / h;
        for p in &mut self.soft.particles {
            if p.inv_mass == 0.0 {
                p.v = Vec3::zero();
                p.x_prev = p.x;
                continue;
            }
            p.v = (p.x - p.x_prev) * inv_h;
        }
    }

    fn update_rigid_velocities(&mut self, h: f64) {
        let inv_h = 1.0 / h;
        for i in 0..self.rigid.n {
            self.rigid.state.v[i] = (self.rigid.state.x[i] - self.rigid_x_prev[i]) * inv_h;
            let q_prev = self.rigid_q_prev[i];
            let q = self.rigid.state.q[i];
            let dq = q.multiply(q_prev.conjugate());
            let rv = quat_to_rotvec(dq);
            let w = rv * inv_h;
            // L = I_world ω
            let r = q.to_matrix();
            let i_w = mat_mul(
                &mat_mul(&r, &self.rigid.inertia_body[i]),
                &pe_core::mat_transpose(&r),
            );
            self.rigid.state.l[i] = mat_vec(&i_w, w);
        }
    }

    /// Project attachments; returns sum of positional impulses on rigid (−Δλ).
    fn project_attachments(&mut self, h: f64) -> Vec3 {
        let mut impulse_on_rigid = Vec3::zero();
        let alpha_scale = if h > 0.0 { 1.0 / (h * h) } else { 0.0 };
        // Collect indices first to satisfy borrow checker via split borrows.
        let n_att = self.attachments.len();
        for ai in 0..n_att {
            let particle = self.attachments[ai].particle;
            let body = self.attachments[ai].body;
            let local = self.attachments[ai].local;
            let compliance = self.attachments[ai].compliance;
            let lambda = self.attachments[ai].lambda;

            if particle >= self.soft.particles.len() || body >= self.rigid.n {
                continue;
            }
            let w_p = self.soft.particles[particle].inv_mass;
            let x_p = self.soft.particles[particle].x;
            let x_com = self.rigid.state.x[body];
            let q = self.rigid.state.q[body];
            let r = q.rotate(local);
            let x_a = x_com + r;
            let c_vec = x_p - x_a; // C = x_p - x_a
            let alpha_tilde = compliance * alpha_scale;

            // W_r at attach point
            let inv_m = 1.0 / self.rigid.mass[body];
            let inv_i = self.rigid.inv_inertia_world(q, body);
            let w_r = point_inv_mass_matrix(inv_m, &inv_i, r);

            // Solve (w_p I + W_r + α̃ I) Δλ = −C − α̃ λ
            let mut a = w_r;
            for k in 0..3 {
                a[k][k] += w_p + alpha_tilde;
            }
            let rhs = Vec3::new(
                -c_vec.x - alpha_tilde * lambda.x,
                -c_vec.y - alpha_tilde * lambda.y,
                -c_vec.z - alpha_tilde * lambda.z,
            );
            let Some(a_inv) = pe_core::try_invert3(&a).ok() else {
                continue;
            };
            let dl = mat_vec(&a_inv, rhs);

            if w_p != 0.0 {
                self.soft.particles[particle].x += dl * w_p;
            }
            // Rigid receives −dl as positional impulse at attach point.
            apply_positional_impulse(&mut self.rigid, body, x_a, dl * (-1.0));
            self.attachments[ai].lambda += dl;
            impulse_on_rigid += dl * (-1.0);
        }
        impulse_on_rigid
    }

    /// Soft particle ↔ rigid convex contacts; returns sum of impulses on rigid.
    fn collide_soft_rigid(&mut self, h: f64) -> Vec3 {
        let _ = h;
        let mut impulse_on_rigid = Vec3::zero();
        let n_col = self.colliders.len();
        let n_part = self.soft.particles.len();
        for ci in 0..n_col {
            let col = self.colliders[ci];
            let body = col.body;
            if body >= self.rigid.n {
                continue;
            }
            let x_com = self.rigid.state.x[body];
            let q = self.rigid.state.q[body];
            let inv_m = 1.0 / self.rigid.mass[body];
            let inv_i = self.rigid.inv_inertia_world(q, body);

            for pi in 0..n_part {
                let w_p = self.soft.particles[pi].inv_mass;
                if w_p == 0.0 {
                    continue;
                }
                let x_p = self.soft.particles[pi].x;
                let hit = match col.shape {
                    RigidShape::Sphere { radius } => sphere_contact(x_p, x_com, radius),
                    RigidShape::Box { half } => box_contact(x_p, x_com, q, half),
                };
                let Some((n, pen, r_hit)) = hit else {
                    continue;
                };
                if pen <= 0.0 {
                    continue;
                }
                // Unilateral C = −pen ≤ 0 when penetrating; push along +n (outward).
                let r = r_hit;
                let rxn = r.cross(n);
                let w_r = inv_m + rxn.dot(mat_vec(&inv_i, rxn));
                let w_sum = w_p + w_r;
                if w_sum < 1e-30 {
                    continue;
                }
                let dl = pen / w_sum; // Δλ > 0 separates
                self.soft.particles[pi].x += n * (w_p * dl);
                let p_imp = n * (-dl); // on rigid
                let point = x_com + r;
                apply_positional_impulse(&mut self.rigid, body, point, p_imp);
                impulse_on_rigid += p_imp;

                // Velocity-level friction / restitution on the particle (PBD-style).
                let v_body =
                    self.rigid.state.v[body] + self.rigid.omega_i(&self.rigid.state, body).cross(r);
                let v_rel = self.soft.particles[pi].v - v_body;
                let vn = v_rel.dot(n);
                if vn < 0.0 {
                    let v_n = n * vn;
                    let v_t = v_rel - v_n;
                    let v_new = v_t * (1.0 - col.friction).max(0.0) - v_n * col.restitution;
                    // Assign particle velocity relative correction (rigid untouched here).
                    self.soft.particles[pi].v = v_body + v_new;
                }
            }
        }
        impulse_on_rigid
    }

    /// Rigid bodies vs static planes; returns sum of ground λ n (positional).
    fn collide_rigid_planes(&mut self, h: f64) -> Vec3 {
        let _ = h;
        let mut ground_lambda = Vec3::zero();
        let planes = self.rigid_planes.clone();
        let colliders = self.colliders.clone();
        for plane in &planes {
            for col in &colliders {
                let body = col.body;
                if body >= self.rigid.n {
                    continue;
                }
                let x_com = self.rigid.state.x[body];
                let q = self.rigid.state.q[body];
                let inv_m = 1.0 / self.rigid.mass[body];
                let inv_i = self.rigid.inv_inertia_world(q, body);
                match col.shape {
                    RigidShape::Sphere { radius } => {
                        let gap = x_com.dot(plane.normal) - plane.offset - radius;
                        if gap >= 0.0 {
                            continue;
                        }
                        let pen = -gap;
                        let n = plane.normal;
                        let r = n * (-radius);
                        let rxn = r.cross(n);
                        let w_r = inv_m + rxn.dot(mat_vec(&inv_i, rxn));
                        if w_r < 1e-30 {
                            continue;
                        }
                        let dl = pen / w_r;
                        let p_imp = n * dl;
                        let point = x_com + r;
                        apply_positional_impulse(&mut self.rigid, body, point, p_imp);
                        ground_lambda += n * dl;
                        // Kill penetrating normal velocity.
                        let w = self.rigid.omega_i(&self.rigid.state, body);
                        let v_pt = self.rigid.state.v[body] + w.cross(r);
                        let vn = v_pt.dot(n);
                        if vn < 0.0 {
                            // Apply velocity impulse through COM approx for resting.
                            let j = -vn / w_r;
                            self.rigid.state.v[body] += n * (j * inv_m);
                            let dw = mat_vec(&inv_i, r.cross(n * j));
                            // Update L from new ω.
                            let w2 = w + dw;
                            let rm = q.to_matrix();
                            let i_w = mat_mul(
                                &mat_mul(&rm, &self.rigid.inertia_body[body]),
                                &pe_core::mat_transpose(&rm),
                            );
                            self.rigid.state.l[body] = mat_vec(&i_w, w2);
                        }
                    }
                    RigidShape::Box { half } => {
                        // Support point: lowest vertex along plane normal.
                        let axes = [
                            q.rotate(Vec3::new(half.x, 0.0, 0.0)),
                            q.rotate(Vec3::new(0.0, half.y, 0.0)),
                            q.rotate(Vec3::new(0.0, 0.0, half.z)),
                        ];
                        let mut support = x_com;
                        let mut r_s = Vec3::zero();
                        for &ax in &axes {
                            if ax.dot(plane.normal) < 0.0 {
                                support += ax;
                                r_s += ax;
                            } else {
                                support -= ax;
                                r_s -= ax;
                            }
                        }
                        let gap = support.dot(plane.normal) - plane.offset;
                        if gap >= 0.0 {
                            continue;
                        }
                        let pen = -gap;
                        let n = plane.normal;
                        let r = r_s;
                        let rxn = r.cross(n);
                        let w_r = inv_m + rxn.dot(mat_vec(&inv_i, rxn));
                        if w_r < 1e-30 {
                            continue;
                        }
                        let dl = pen / w_r;
                        let p_imp = n * dl;
                        apply_positional_impulse(&mut self.rigid, body, support, p_imp);
                        ground_lambda += n * dl;
                        let w = self.rigid.omega_i(&self.rigid.state, body);
                        let v_pt = self.rigid.state.v[body] + w.cross(r);
                        let vn = v_pt.dot(n);
                        if vn < 0.0 {
                            let j = -vn * (1.0 - plane.restitution) / w_r;
                            self.rigid.state.v[body] += n * (j * inv_m);
                            let dw = mat_vec(&inv_i, r.cross(n * j));
                            let w2 = w + dw;
                            let rm = q.to_matrix();
                            let i_w = mat_mul(
                                &mat_mul(&rm, &self.rigid.inertia_body[body]),
                                &pe_core::mat_transpose(&rm),
                            );
                            self.rigid.state.l[body] = mat_vec(&i_w, w2);
                            // Tangential friction damp at contact.
                            let v_t = (self.rigid.state.v[body] + w2.cross(r)) - n * 0.0;
                            let vt = v_t - n * v_t.dot(n);
                            let friction = plane.friction;
                            if friction > 0.0 && vt.norm_squared() > 1e-20 {
                                let damp = (1.0 - friction).max(0.0);
                                // Reduce tangential COM velocity proportionally (simple).
                                self.rigid.state.v[body] = n * self.rigid.state.v[body].dot(n)
                                    + vt * damp
                                    - n * vt.dot(n) * damp;
                            }
                        }
                    }
                }
            }
        }
        ground_lambda
    }
}

fn apply_positional_impulse(world: &mut World, body: usize, point: Vec3, impulse: Vec3) {
    let x = world.state.x[body];
    let r = point - x;
    let inv_m = 1.0 / world.mass[body];
    world.state.x[body] += impulse * inv_m;
    let inv_i = world.inv_inertia_world(world.state.q[body], body);
    let dtheta = mat_vec(&inv_i, r.cross(impulse));
    world.state.q[body] = Quat::exp_rotvec(dtheta)
        .multiply(world.state.q[body])
        .normalize();
}

/// Generalized inverse-mass matrix at a point offset `r` from COM.
fn point_inv_mass_matrix(inv_m: f64, inv_i: &[[f64; 3]; 3], r: Vec3) -> [[f64; 3]; 3] {
    // W = inv_m I − [r×] inv_I [r×]
    let k = cross_matrix(r);
    let k_inv_i = mat_mul(&k, inv_i);
    let k_inv_i_k = mat_mul(&k_inv_i, &k);
    let mut w = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let diag = if i == j { inv_m } else { 0.0 };
            w[i][j] = diag - k_inv_i_k[i][j];
        }
    }
    w
}

fn cross_matrix(r: Vec3) -> [[f64; 3]; 3] {
    [[0.0, -r.z, r.y], [r.z, 0.0, -r.x], [-r.y, r.x, 0.0]]
}

fn quat_to_rotvec(q: Quat) -> Vec3 {
    let q = if q.w < 0.0 { q.scale(-1.0) } else { q };
    let v = q.vec();
    let s = v.norm();
    if s < 1e-12 {
        return v * 2.0;
    }
    let half = atan2(s, q.w);
    v * (2.0 * half / s)
}

/// Sphere contact: outward normal from sphere centre, penetration depth, r from COM.
fn sphere_contact(x_p: Vec3, x_com: Vec3, radius: f64) -> Option<(Vec3, f64, Vec3)> {
    let d = x_p - x_com;
    let dist = d.norm();
    if dist >= radius || dist < 1e-15 {
        return None;
    }
    let n = d * (1.0 / dist);
    let pen = radius - dist;
    let r = n * dist; // contact on particle side ≈ particle location relative COM
    Some((n, pen, r))
}

/// OBB box contact: push out along least-penetration body axis (transformed).
fn box_contact(x_p: Vec3, x_com: Vec3, q: Quat, half: Vec3) -> Option<(Vec3, f64, Vec3)> {
    let local = q.inverse_rotate(x_p - x_com);
    if local.x.abs() > half.x || local.y.abs() > half.y || local.z.abs() > half.z {
        return None;
    }
    let px = half.x - local.x.abs();
    let py = half.y - local.y.abs();
    let pz = half.z - local.z.abs();
    let (local_n, pen) = if px <= py && px <= pz {
        (
            Vec3::new(if local.x >= 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0),
            px,
        )
    } else if py <= pz {
        (
            Vec3::new(0.0, if local.y >= 0.0 { 1.0 } else { -1.0 }, 0.0),
            py,
        )
    } else {
        (
            Vec3::new(0.0, 0.0, if local.z >= 0.0 { 1.0 } else { -1.0 }),
            pz,
        )
    };
    let n = q.rotate(local_n);
    let r = x_p - x_com; // approximate contact at particle
    Some((n, pen, r))
}

// ---------------------------------------------------------------------------
// Canonical hash + oracle scene builders
// ---------------------------------------------------------------------------

/// Outer steps for the coupling hash.
pub const COUPLING_HASH_STEPS: usize = 180;
/// Outer timestep [s] for the coupling hash.
pub const COUPLING_HASH_DT: f64 = 1.0 / 120.0;

/// Build the canonical soft↔rigid coupling scene for the locked hash.
///
/// Rope (8 seg) hanging from a dynamic box; soft cube dropping onto a second
/// dynamic box on a ground plane; gravity −z; 20 XPBD substeps.
pub fn soft_coupling_hash_world() -> SoftRigidWorld {
    let g = Vec3::new(0.0, 0.0, -9.80665);
    // Soft: rope (not pinned — top attaches to body 0) + jelly cube.
    let mut rope = make_rope(
        Vec3::new(0.0, 0.0, 1.2),
        Vec3::new(0.0, 0.0, 0.6),
        8,
        0.3,
        1e-6,
        false,
    );
    let jelly_mat = material_by_name("jelly").expect("jelly");
    let mut jelly = make_soft_cube(Vec3::new(0.6, 0.0, 0.55), 0.2, 2, &jelly_mat, true);

    let mut soft = SoftWorld::new();
    soft.params = SoftParams {
        substeps: 20,
        iterations: 2,
        gravity: g,
        damping: 0.05,
    };
    soft.self_collision.enabled = false;
    let rope_n = rope.particles.len();
    append_soft(&mut soft, &mut rope);
    append_soft(&mut soft, &mut jelly);

    // Body 0: hanging support box for the rope.
    let m0 = 2.0;
    let half0 = Vec3::new(0.1, 0.1, 0.1);
    let b0 = RigidBody::new(m0)
        .with_position(Vec3::new(0.0, 0.0, 1.35))
        .with_inertia(inertia_solid_box(m0, 0.2, 0.2, 0.2));
    // Body 1: ground platform box that the jelly lands on.
    let m1 = 5.0;
    let half1 = Vec3::new(0.35, 0.35, 0.08);
    let b1 = RigidBody::new(m1)
        .with_position(Vec3::new(0.6, 0.0, 0.08))
        .with_inertia(inertia_solid_box(m1, 0.7, 0.7, 0.16));

    let mut world = SoftRigidWorld::new(soft, vec![b0, b1]);
    world.add_collider(RigidCollider {
        body: 0,
        shape: RigidShape::Box { half: half0 },
        friction: 0.4,
        restitution: 0.0,
    });
    world.add_collider(RigidCollider {
        body: 1,
        shape: RigidShape::Box { half: half1 },
        friction: 0.5,
        restitution: 0.0,
    });
    world.add_attachment(AttachmentConstraint::hard(0, 0, Vec3::new(0.0, 0.0, -0.1)));
    // Bottom of rope lightly attached to a free sphere-mass? keep free hanging.
    let _ = rope_n;
    world.add_ground_plane(StaticPlane::new(Vec3::unit_z(), 0.0, 0.6, 0.0));
    world
}

fn append_soft(dst: &mut SoftWorld, src: &mut SoftWorld) {
    let base = dst.particles.len();
    dst.particles.append(&mut src.particles);
    for c in &mut src.constraints.distances {
        c.i += base;
        c.j += base;
        dst.constraints.distances.push(c.clone());
    }
    for c in &mut src.constraints.distance_bends {
        c.a += base;
        c.b += base;
        dst.constraints.distance_bends.push(c.clone());
    }
    for c in &mut src.constraints.dihedrals {
        c.i0 += base;
        c.i1 += base;
        c.i2 += base;
        c.i3 += base;
        dst.constraints.dihedrals.push(c.clone());
    }
    for c in &mut src.constraints.volumes {
        for i in &mut c.indices {
            *i += base;
        }
        dst.constraints.volumes.push(c.clone());
    }
    for c in &mut src.constraints.tetra_fem {
        for i in &mut c.indices {
            *i += base;
        }
        dst.constraints.tetra_fem.push(c.clone());
    }
    for c in &mut src.constraints.shape_matches {
        for i in &mut c.indices {
            *i += base;
        }
        dst.constraints.shape_matches.push(c.clone());
    }
    for c in &mut src.constraints.pressures {
        for tri in &mut c.triangles {
            for idx in tri.iter_mut() {
                *idx += base;
            }
        }
        dst.constraints.pressures.push(c.clone());
    }
}

/// FNV-1a 64 over soft packed `(x‖v)` then rigid `State::pack()`.
pub fn soft_coupling_state_hash64(world: &SoftRigidWorld) -> u64 {
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = soft_state_hash64(&world.soft);
    for x in world.rigid.state.pack() {
        for b in x.to_bits().to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

/// Run the canonical coupling scene and return the locked hash.
pub fn soft_coupling_canonical_hash() -> u64 {
    let mut world = soft_coupling_hash_world();
    for _ in 0..COUPLING_HASH_STEPS {
        world.step(COUPLING_HASH_DT);
    }
    soft_coupling_state_hash64(&world)
}

/// Rope of total mass `m_rope` attached to a free rigid body of mass `m_body`
/// in free fall (no ground). Used by the CoM free-fall oracle.
pub fn make_rope_on_free_body(
    m_rope: f64,
    m_body: f64,
    segments: usize,
    compliance: f64,
) -> SoftRigidWorld {
    let g = Vec3::new(0.0, 0.0, -9.80665);
    let mut rope = make_rope(
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(0.0, 0.0, 1.0),
        segments,
        m_rope,
        compliance,
        false,
    );
    rope.params.gravity = g;
    rope.params.substeps = 32;
    rope.params.iterations = 2;
    rope.params.damping = 0.0;

    let r = 0.08;
    let body = RigidBody::new(m_body)
        .with_position(Vec3::new(0.0, 0.0, 1.0 - r))
        .with_inertia(inertia_solid_sphere(m_body, r));
    let mut world = SoftRigidWorld::new(rope, vec![body]);
    world.add_collider(RigidCollider {
        body: 0,
        shape: RigidShape::Sphere { radius: r },
        friction: 0.0,
        restitution: 0.0,
    });
    let last = world.soft.particles.len() - 1;
    world.add_attachment(AttachmentConstraint::hard(last, 0, Vec3::zero()));
    world
}

/// Soft block on a rigid box on a ground plane (static-equilibrium oracle).
pub fn make_soft_on_box_on_ground() -> SoftRigidWorld {
    let g = Vec3::new(0.0, 0.0, -9.80665);
    let mat = material_by_name("jelly").expect("jelly");
    let mut soft = make_soft_cube(Vec3::new(0.0, 0.0, 0.45), 0.25, 2, &mat, true);
    soft.params = SoftParams {
        substeps: 24,
        iterations: 2,
        gravity: g,
        damping: 0.5,
    };
    soft.self_collision.enabled = false;

    let m_box = 4.0;
    let half = Vec3::new(0.3, 0.3, 0.1);
    let body = RigidBody::new(m_box)
        .with_position(Vec3::new(0.0, 0.0, 0.1))
        .with_inertia(inertia_solid_box(m_box, 0.6, 0.6, 0.2));
    let mut world = SoftRigidWorld::new(soft, vec![body]);
    world.add_collider(RigidCollider {
        body: 0,
        shape: RigidShape::Box { half },
        friction: 0.8,
        restitution: 0.0,
    });
    world.add_ground_plane(StaticPlane::new(Vec3::unit_z(), 0.0, 0.9, 0.0));
    world
}

/// Heavy rigid bob on a stiff rope (small-angle pendulum oracle).
///
/// Top particle pinned in world; bottom attached to a heavy sphere. Returns
/// `(world, length_to_com)` where length is pin → bob COM at rest construction.
pub fn make_pendulum_bob(
    m_bob: f64,
    m_rope: f64,
    length: f64,
    segments: usize,
    compliance: f64,
) -> (SoftRigidWorld, f64) {
    let g = Vec3::new(0.0, 0.0, -9.80665);
    let r = 0.05;
    let mut rope = make_rope(
        Vec3::new(0.0, 0.0, length),
        Vec3::new(0.0, 0.0, r),
        segments,
        m_rope,
        compliance,
        false,
    );
    // Pin top.
    rope.pin(0);
    rope.params = SoftParams {
        substeps: 40,
        iterations: 2,
        gravity: g,
        damping: 0.0,
    };
    let body = RigidBody::new(m_bob)
        .with_position(Vec3::new(0.0, 0.0, 0.0))
        .with_inertia(inertia_solid_sphere(m_bob, r));
    let mut world = SoftRigidWorld::new(rope, vec![body]);
    world.add_collider(RigidCollider {
        body: 0,
        shape: RigidShape::Sphere { radius: r },
        friction: 0.0,
        restitution: 0.0,
    });
    let last = world.soft.particles.len() - 1;
    world.add_attachment(AttachmentConstraint::hard(last, 0, Vec3::new(0.0, 0.0, r)));
    let l_com = length; // pin at z=L, COM at z=0
    (world, l_com)
}

/// Damped soft spring pinned in world, free end attached to a free rigid body
/// (no contacts / no gravity) — energy-decay oracle.
pub fn make_damped_attachment_oscillator() -> SoftRigidWorld {
    let mut soft = SoftWorld::new();
    soft.params = SoftParams {
        substeps: 32,
        iterations: 2,
        gravity: Vec3::zero(),
        damping: 1.5,
    };
    let _pin = soft.add_particle(Particle::pinned(Vec3::zero()));
    let free = soft.add_particle(Particle::dynamic(Vec3::new(1.15, 0.0, 0.0), 0.5));
    soft.constraints.distances.push(DistanceConstraint {
        i: 0,
        j: free,
        rest_length: 1.0,
        compliance: 0.01,
        lambda: Multiplier::default(),
        broken: false,
    });
    let m_b = 1.0;
    let r = 0.05;
    let body = RigidBody::new(m_b)
        .with_position(Vec3::new(1.15, 0.0, 0.0))
        .with_inertia(inertia_solid_sphere(m_b, r));
    let mut world = SoftRigidWorld::new(soft, vec![body]);
    world.add_attachment(AttachmentConstraint::hard(free, 0, Vec3::zero()));
    // Optional collider unused for energy test (no contacts).
    world.add_collider(RigidCollider {
        body: 0,
        shape: RigidShape::Sphere { radius: r },
        friction: 0.0,
        restitution: 0.0,
    });
    world
}

/// Isolated momentum exchange: soft particle cluster hitting a free box (no g).
pub fn make_momentum_exchange_scene() -> SoftRigidWorld {
    let mut soft = SoftWorld::new();
    soft.params = SoftParams {
        substeps: 16,
        iterations: 2,
        gravity: Vec3::zero(),
        damping: 0.0,
    };
    // Three particles in a line, moving +x.
    let m_p = 0.5;
    for i in 0..3 {
        let mut p = Particle::dynamic(Vec3::new(-0.4 + i as f64 * 0.05, 0.0, 0.0), m_p);
        p.v = Vec3::new(1.0, 0.0, 0.0);
        soft.add_particle(p);
    }
    soft.constraints.distances.push(DistanceConstraint {
        i: 0,
        j: 1,
        rest_length: 0.05,
        compliance: 1e-8,
        lambda: Multiplier::default(),
        broken: false,
    });
    soft.constraints.distances.push(DistanceConstraint {
        i: 1,
        j: 2,
        rest_length: 0.05,
        compliance: 1e-8,
        lambda: Multiplier::default(),
        broken: false,
    });

    let m_b = 2.0;
    let half = Vec3::new(0.1, 0.1, 0.1);
    let body = RigidBody::new(m_b)
        .with_position(Vec3::new(0.25, 0.0, 0.0))
        .with_velocity(Vec3::new(-0.2, 0.0, 0.0))
        .with_inertia(inertia_solid_box(m_b, 0.2, 0.2, 0.2));
    let mut world = SoftRigidWorld::new(soft, vec![body]);
    world.add_collider(RigidCollider {
        body: 0,
        shape: RigidShape::Box { half },
        friction: 0.0,
        restitution: 1.0,
    });
    world
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn point_inv_mass_at_com_is_inv_m() {
        let inv_i = pe_core::invert3(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let w = point_inv_mass_matrix(0.5, &inv_i, Vec3::zero());
        assert!((w[0][0] - 0.5).abs() < 1e-15);
        assert!(w[0][1].abs() < 1e-15);
    }

    #[test]
    fn coupling_hash_bit_stable() {
        let a = soft_coupling_canonical_hash();
        let b = soft_coupling_canonical_hash();
        assert_eq!(a, b);
    }
}
