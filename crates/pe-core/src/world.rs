//! World: structure-of-arrays state for N rigid bodies plus force generators.

use pe_math::{Quat, Vec3};

use crate::body::{invert3, mat_mul, mat_transpose, mat_vec, RigidBody};
use crate::forces::ForceGenerator;
use crate::integrators::Integrator;

/// Full dynamic state. Positions [m], quaternions, velocities [m/s],
/// world-frame angular momenta [kg m²/s].
#[derive(Clone, Debug)]
pub struct State {
    /// COM positions [m], length N.
    pub x: Vec<Vec3>,
    /// Orientation quaternions `[w,x,y,z]`, length N.
    pub q: Vec<Quat>,
    /// Linear velocities [m/s], length N.
    pub v: Vec<Vec3>,
    /// World-frame angular momenta [kg m²/s], length N.
    pub l: Vec<Vec3>,
}

impl State {
    /// Number of bodies.
    pub fn n(&self) -> usize {
        self.x.len()
    }

    /// Deep copy.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Pack into a flat `f64` buffer: `[x…, q…, v…, L…]` (same layout as Python).
    pub fn pack(&self) -> Vec<f64> {
        let n = self.n();
        let mut out = Vec::with_capacity(13 * n);
        for p in &self.x {
            out.extend_from_slice(&p.to_array());
        }
        for q in &self.q {
            out.extend_from_slice(&q.to_array());
        }
        for v in &self.v {
            out.extend_from_slice(&v.to_array());
        }
        for l in &self.l {
            out.extend_from_slice(&l.to_array());
        }
        out
    }
}

/// Simulation world owning bodies (SoA) and ordered force generators.
pub struct World {
    /// Number of bodies.
    pub n: usize,
    /// Masses [kg].
    pub mass: Vec<f64>,
    /// Body-frame inertia tensors [kg m²].
    pub inertia_body: Vec<[[f64; 3]; 3]>,
    /// Inverse body-frame inertia [1/(kg m²)].
    pub inv_inertia_body: Vec<[[f64; 3]; 3]>,
    /// Dynamic state.
    pub state: State,
    /// Simulation time [s].
    pub t: f64,
    generators: Vec<Box<dyn ForceGenerator>>,
}

impl World {
    /// Build a world from body descriptions and force generators (fixed order).
    pub fn new(bodies: Vec<RigidBody>, generators: Vec<Box<dyn ForceGenerator>>) -> Self {
        let n = bodies.len();
        let mass: Vec<f64> = bodies.iter().map(|b| b.mass).collect();
        let inertia_body: Vec<[[f64; 3]; 3]> = bodies.iter().map(|b| b.inertia_body).collect();
        let inv_inertia_body: Vec<_> = inertia_body.iter().map(invert3).collect();

        let mut x = Vec::with_capacity(n);
        let mut q = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);
        let mut l = Vec::with_capacity(n);
        for b in &bodies {
            let qq = b.orientation.normalize();
            let r = qq.to_matrix();
            let i_w = mat_mul(&mat_mul(&r, &b.inertia_body), &mat_transpose(&r));
            x.push(b.position);
            q.push(qq);
            v.push(b.velocity);
            l.push(mat_vec(&i_w, b.angular_velocity));
        }

        Self {
            n,
            mass,
            inertia_body,
            inv_inertia_body,
            state: State { x, q, v, l },
            t: 0.0,
            generators,
        }
    }

    /// World-frame inverse inertia `R I_b⁻¹ Rᵀ` [1/(kg m²)] at orientation `q`.
    pub fn inv_inertia_world(&self, q: Quat, body: usize) -> [[f64; 3]; 3] {
        let r = q.to_matrix();
        mat_mul(
            &mat_mul(&r, &self.inv_inertia_body[body]),
            &mat_transpose(&r),
        )
    }

    /// World-frame angular velocity [rad/s] for every body.
    pub fn omega(&self, s: &State) -> Vec<Vec3> {
        (0..self.n)
            .map(|i| mat_vec(&self.inv_inertia_world(s.q[i], i), s.l[i]))
            .collect()
    }

    /// World-frame angular velocity [rad/s] for body `i`.
    pub fn omega_i(&self, s: &State, i: usize) -> Vec3 {
        mat_vec(&self.inv_inertia_world(s.q[i], i), s.l[i])
    }

    /// Accumulate force [N] and torque about COM [N m] from all generators (fixed order).
    pub fn forces(&self, s: &State, t: f64) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut f = vec![Vec3::zero(); self.n];
        let mut tau = vec![Vec3::zero(); self.n];
        for g in &self.generators {
            g.apply(self, s, t, &mut f, &mut tau);
        }
        (f, tau)
    }

    /// Continuous-time state derivative for RK4 / explicit Euler.
    ///
    /// Returns `(ẋ, q̇, v̇, L̇)` with `q̇ = ½ (0,ω) ⊗ q`, `v̇ = F/m`, `L̇ = τ`.
    pub fn derivative(&self, s: &State, t: f64) -> State {
        let (f, tau) = self.forces(s, t);
        let mut dx = Vec::with_capacity(self.n);
        let mut dq = Vec::with_capacity(self.n);
        let mut dv = Vec::with_capacity(self.n);
        let mut dl = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let w = self.omega_i(s, i);
            let wq = Quat::from_vec(w);
            dx.push(s.v[i]);
            dq.push(wq.multiply(s.q[i]).scale(0.5));
            dv.push(f[i] / self.mass[i]);
            dl.push(tau[i]);
        }
        State {
            x: dx,
            q: dq,
            v: dv,
            l: dl,
        }
    }

    /// Access generators (for potential energy).
    pub fn generators(&self) -> &[Box<dyn ForceGenerator>] {
        &self.generators
    }

    /// Advance one fixed step [s] with the given integrator; renormalize quaternions.
    pub fn step(&mut self, dt: f64, integrator: Integrator) {
        self.state = integrator(self, &self.state, self.t, dt);
        for q in &mut self.state.q {
            *q = q.normalize();
        }
        self.t += dt;
    }

    /// Advance with fixed `dt` until `t_end` [s] (last step shortened to land exactly).
    pub fn run_for(&mut self, t_end: f64, dt: f64, integrator: Integrator) {
        while self.t < t_end - 1e-12 * t_end.abs().max(1.0) {
            let h = dt.min(t_end - self.t);
            self.step(h, integrator);
        }
    }

    /// Like [`run_for`] but collect `record(self)` after each step.
    pub fn run_recorded<R, F>(
        &mut self,
        t_end: f64,
        dt: f64,
        integrator: Integrator,
        mut record: F,
    ) -> Vec<R>
    where
        F: FnMut(&World) -> R,
    {
        let mut out = Vec::new();
        while self.t < t_end - 1e-12 * t_end.abs().max(1.0) {
            let h = dt.min(t_end - self.t);
            self.step(h, integrator);
            out.push(record(self));
        }
        out
    }
}
