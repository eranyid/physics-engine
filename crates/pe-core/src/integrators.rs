//! Fixed-step integrators. Translational DOF use `(x, v)`; rotational use `(q, L)`.

use pe_math::Quat;

use crate::world::{State, World};

/// Integrator function type: `(world, state, t [s], dt [s]) -> new state`.
pub type Integrator = fn(&World, &State, f64, f64) -> State;

/// Named integrators for bindings / CLI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegratorKind {
    /// Forward Euler, order 1.
    ExplicitEuler,
    /// Semi-implicit (symplectic) Euler, order 1.
    SymplecticEuler,
    /// Velocity Verlet / leapfrog, order 2 for position-dependent forces.
    VelocityVerlet,
    /// Classical RK4, order 4.
    Rk4,
}

impl IntegratorKind {
    /// Resolve to a function pointer.
    pub fn as_fn(self) -> Integrator {
        match self {
            Self::ExplicitEuler => explicit_euler,
            Self::SymplecticEuler => symplectic_euler,
            Self::VelocityVerlet => velocity_verlet,
            Self::Rk4 => rk4,
        }
    }

    /// Parse from the Python-style name.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "explicit_euler" => Some(Self::ExplicitEuler),
            "symplectic_euler" => Some(Self::SymplecticEuler),
            "velocity_verlet" => Some(Self::VelocityVerlet),
            "rk4" => Some(Self::Rk4),
            _ => None,
        }
    }

    /// Python-style name.
    pub fn name(self) -> &'static str {
        match self {
            Self::ExplicitEuler => "explicit_euler",
            Self::SymplecticEuler => "symplectic_euler",
            Self::VelocityVerlet => "velocity_verlet",
            Self::Rk4 => "rk4",
        }
    }
}

fn axpy(s: &State, d: &State, h: f64) -> State {
    let n = s.n();
    let mut x = Vec::with_capacity(n);
    let mut q = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    let mut l = Vec::with_capacity(n);
    for i in 0..n {
        x.push(s.x[i] + d.x[i] * h);
        q.push(s.q[i] + d.q[i].scale(h));
        v.push(s.v[i] + d.v[i] * h);
        l.push(s.l[i] + d.l[i] * h);
    }
    State { x, q, v, l }
}

/// Advance `q` with `L` frozen: `q⁺ = exp(h ω(q_mid)) q`.
///
/// Implicit midpoint (time-reversible) solved by fixed-point iteration (`iters` = 4),
/// matching the Python oracle. Keeps torque-free kinetic energy bounded.
///
/// Used by contact steps (semi-implicit Euler with impulses) as well as free integrators.
pub fn drift_rotation(
    world: &World,
    q: &[pe_math::Quat],
    l: &[pe_math::Vec3],
    h: f64,
) -> Vec<Quat> {
    const ITERS: usize = 4;
    let n = q.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let qi = q[i];
        let li = l[i];
        let omega_at = |qq: Quat| -> pe_math::Vec3 {
            crate::body::mat_vec(&world.inv_inertia_world(qq, i), li)
        };
        let mut w = omega_at(qi);
        for _ in 0..ITERS {
            let q_mid = Quat::exp_rotvec(w * (0.5 * h)).multiply(qi);
            w = omega_at(q_mid);
        }
        out.push(Quat::exp_rotvec(w * h).multiply(qi).normalize());
    }
    out
}

/// Forward Euler, order 1, not symplectic (energy grows on oscillators).
pub fn explicit_euler(world: &World, s: &State, t: f64, dt: f64) -> State {
    axpy(s, &world.derivative(s, t), dt)
}

/// Semi-implicit Euler: kick velocities with forces at `x_n`, then drift with new velocities.
pub fn symplectic_euler(world: &World, s: &State, t: f64, dt: f64) -> State {
    let (f, tau) = world.forces(s, t);
    let n = world.n;
    let mut v = Vec::with_capacity(n);
    let mut l = Vec::with_capacity(n);
    for i in 0..n {
        v.push(s.v[i] + f[i] * (dt / world.mass[i]));
        l.push(s.l[i] + tau[i] * dt);
    }
    let q = drift_rotation(world, &s.q, &l, dt);
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        x.push(s.x[i] + v[i] * dt);
    }
    State { x, q, v, l }
}

/// Kick-drift-kick leapfrog, order 2 and symplectic for position-dependent forces.
/// Closing kick evaluates velocity-dependent forces at the half-step velocity.
pub fn velocity_verlet(world: &World, s: &State, t: f64, dt: f64) -> State {
    let h = 0.5 * dt;
    let (f, tau) = world.forces(s, t);
    let n = world.n;
    let mut vh = Vec::with_capacity(n);
    let mut lh = Vec::with_capacity(n);
    for i in 0..n {
        vh.push(s.v[i] + f[i] * (h / world.mass[i]));
        lh.push(s.l[i] + tau[i] * h);
    }
    let q_mid = drift_rotation(world, &s.q, &lh, dt);
    let mut x_mid = Vec::with_capacity(n);
    for i in 0..n {
        x_mid.push(s.x[i] + vh[i] * dt);
    }
    let mid = State {
        x: x_mid,
        q: q_mid,
        v: vh.clone(),
        l: lh.clone(),
    };
    let (f2, tau2) = world.forces(&mid, t + dt);
    let mut v = Vec::with_capacity(n);
    let mut l = Vec::with_capacity(n);
    for i in 0..n {
        v.push(vh[i] + f2[i] * (h / world.mass[i]));
        l.push(lh[i] + tau2[i] * h);
    }
    State {
        x: mid.x,
        q: mid.q,
        v,
        l,
    }
}

/// Classical 4th-order Runge–Kutta on the full state (`q` renormalised by [`World::step`]).
pub fn rk4(world: &World, s: &State, t: f64, dt: f64) -> State {
    let k1 = world.derivative(s, t);
    let k2 = world.derivative(&axpy(s, &k1, dt / 2.0), t + dt / 2.0);
    let k3 = world.derivative(&axpy(s, &k2, dt / 2.0), t + dt / 2.0);
    let k4 = world.derivative(&axpy(s, &k3, dt), t + dt);
    let n = s.n();
    let mut x = Vec::with_capacity(n);
    let mut q = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    let mut l = Vec::with_capacity(n);
    for i in 0..n {
        x.push(s.x[i] + (k1.x[i] + k2.x[i] * 2.0 + k3.x[i] * 2.0 + k4.x[i]) * (dt / 6.0));
        q.push(
            s.q[i] + (k1.q[i] + k2.q[i].scale(2.0) + k3.q[i].scale(2.0) + k4.q[i]).scale(dt / 6.0),
        );
        v.push(s.v[i] + (k1.v[i] + k2.v[i] * 2.0 + k3.v[i] * 2.0 + k4.v[i]) * (dt / 6.0));
        l.push(s.l[i] + (k1.l[i] + k2.l[i] * 2.0 + k3.l[i] * 2.0 + k4.l[i]) * (dt / 6.0));
    }
    State { x, q, v, l }
}
