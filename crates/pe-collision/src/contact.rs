//! Contact dynamics: sequential impulses (projected Gauss–Seidel) with restitution
//! and Coulomb friction, embedded in a semi-implicit Euler step, plus non-energetic
//! positional correction — and an optional Soft Step solver (M5a).
//!
//! Step (`dt`), PGS mode (oracle default):
//! 1. kick `v += dt F/m`, `L += dt T`
//! 2. detect: broad phase (sorted pairs) + narrow phase
//! 3. solve: sequential impulses on `(v, L)`
//! 4. drift `x`, `q` (stage-1 implicit-midpoint exponential map)
//! 5. correct: re-detect, translate bodies apart by `β(depth − slop)`
//!
//! Soft-step mode: sub-steps with soft constraints (Hertz/ζ), relax pass, then
//! restitution from pre-solve relative velocity — see `soft_step` / `softness`.

use std::collections::BTreeMap;

use pe_core::body::mat_vec;
use pe_core::integrators::drift_rotation;
use pe_core::{State, World};
use pe_math::libm_ops::{fabs, sqrt};
use pe_math::{Quat, Vec3};

use crate::aabb_tree::DynamicAabbTree;
use crate::broadphase::{BroadPhase, BruteForce, SweepAndPrune};
use crate::ccd::{self, CcdParams};
use crate::islands::{angular_speed, SleepState};
use crate::joints::{self, Joint};
use crate::manifold::{ContactPoint, FeatureId};
use crate::narrowphase::{DedicatedNarrowPhase, NarrowPhase};
use crate::shapes::{Material, Shape};
use crate::soft_step::solve_soft_step;
use crate::softness::{SoftStepParams, SolverMode};

/// Infinite-mass static collider (plane, fixed box, …).
#[derive(Clone, Debug)]
pub struct StaticCollider {
    /// Collision shape.
    pub shape: Shape,
    /// World position [m].
    pub position: Vec3,
    /// Orientation body→world.
    pub orientation: Quat,
    /// Contact material.
    pub material: Material,
}

impl StaticCollider {
    /// Construct a static collider at the origin with default material.
    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            position: Vec3::zero(),
            orientation: Quat::identity(),
            material: Material::default(),
        }
    }

    /// Builder: world position [m].
    pub fn with_position(mut self, p: Vec3) -> Self {
        self.position = p;
        self
    }

    /// Builder: orientation.
    pub fn with_orientation(mut self, q: Quat) -> Self {
        self.orientation = q.normalize();
        self
    }

    /// Builder: material.
    pub fn with_material(mut self, m: Material) -> Self {
        self.material = m;
        self
    }
}

fn friction_basis(n: Vec3) -> (Vec3, Vec3) {
    let t = if fabs(n.x) < 0.9 {
        n.cross(Vec3::unit_x())
    } else {
        n.cross(Vec3::unit_y())
    };
    let t1 = t / t.norm();
    let t2 = n.cross(t1);
    (t1, t2)
}

/// Cached warm-start impulse for a feature.
#[derive(Clone, Copy, Debug)]
struct WarmEntry {
    normal_impulse: f64,
    tangent_impulse: Vec3,
    point: Vec3,
}

/// Public warm-cache row for scene I/O (`pe-scene` / `pe-ffi`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WarmCacheRow {
    /// Body / collider index A.
    pub a: usize,
    /// Body / collider index B.
    pub b: usize,
    /// Feature id.
    pub feature: u32,
    /// Accumulated normal impulse.
    pub normal_impulse: f64,
    /// Accumulated tangent impulse [3].
    pub tangent_impulse: [f64; 3],
    /// Contact point [m].
    pub point: [f64; 3],
}

/// Broad-phase backend selection (deterministic pair order for all variants).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BroadPhaseKind {
    /// Sweep-and-prune on x (M3 default).
    #[default]
    Sap,
    /// O(N²) reference.
    Brute,
    /// Dynamic AABB tree (SAH insert, fat AABBs, rotations).
    DynamicTree,
}

/// Wraps a stage-1 [`World`]; body `i` gets `shapes[i]` and `materials[i]`.
/// Statics have infinite mass.
pub struct ContactWorld {
    /// Owned dynamics world.
    pub world: World,
    /// Dynamic body shapes (length `world.n`).
    pub shapes: Vec<Shape>,
    /// Static colliders (appended after dynamics in detect indices).
    pub statics: Vec<StaticCollider>,
    /// Per-dynamic-body materials.
    pub materials: Vec<Material>,
    /// PGS iterations (default 20).
    pub iterations: usize,
    /// Below this approach speed [m/s], restitution is ignored (PGS path).
    pub restitution_threshold: f64,
    /// Penetration slop [m] (default 0.1 mm).
    pub slop: f64,
    /// Baumgarte-like positional correction fraction (default 0.2); velocities untouched.
    pub beta: f64,
    /// Re-apply previous impulses for persistent features.
    pub warm_start: bool,
    /// Solver mode: [`SolverMode::Pgs`] (oracle default) or [`SolverMode::SoftStep`].
    pub solver_mode: SolverMode,
    /// Soft-step tuning (used when `solver_mode == SoftStep`).
    pub soft_params: SoftStepParams,
    /// Broad-phase backend (default SAP).
    pub broad_phase: BroadPhaseKind,
    /// Island sleeping bookkeeping.
    pub sleep: SleepState,
    /// Maximal-coordinate joints (solved with contacts).
    pub joints: Vec<Joint>,
    /// Per-dynamic-body bullet flag (TOI CCD vs statics).
    pub bullet: Vec<bool>,
    /// Continuous collision detection parameters.
    pub ccd: CcdParams,
    /// Last detected contacts.
    pub contacts: Vec<ContactPoint>,
    /// Previous step's broad-phase pairs (for wake-on-new-contact).
    prev_pairs: Vec<(usize, usize)>,
    /// Warm-start cache keyed by `(a, b, feature)` — `BTreeMap` so there is never
    /// unordered hash-map iteration on the step path.
    cache: BTreeMap<(usize, usize, u32), WarmEntry>,
    /// Soft-step warm cache `(a, b, λn, jt)` in contact order.
    soft_warm: Vec<(usize, usize, f64, Vec3)>,
    is_static: Vec<bool>,
    static_x: Vec<Vec3>,
    static_r: Vec<[[f64; 3]; 3]>,
}

impl ContactWorld {
    /// Build a contact world. `shapes.len()` must equal `world.n`.
    pub fn new(
        world: World,
        shapes: Vec<Shape>,
        statics: Vec<StaticCollider>,
        materials: Option<Vec<Material>>,
    ) -> Self {
        assert_eq!(shapes.len(), world.n, "one shape per dynamic body");
        let n = world.n;
        let s = statics.len();
        let materials = materials.unwrap_or_else(|| vec![Material::default(); n]);
        assert_eq!(materials.len(), n);
        let mut is_static = vec![false; n];
        is_static.extend(std::iter::repeat_n(true, s));
        let static_x: Vec<Vec3> = statics.iter().map(|c| c.position).collect();
        let static_r: Vec<[[f64; 3]; 3]> =
            statics.iter().map(|c| c.orientation.to_matrix()).collect();
        let sleep = SleepState::new(n);
        Self {
            world,
            shapes,
            statics,
            materials,
            iterations: 20,
            restitution_threshold: 0.2,
            slop: 1e-4,
            beta: 0.2,
            warm_start: true,
            solver_mode: SolverMode::Pgs,
            soft_params: SoftStepParams::default(),
            broad_phase: BroadPhaseKind::Sap,
            sleep,
            joints: Vec::new(),
            bullet: vec![false; n],
            ccd: CcdParams::default(),
            contacts: Vec::new(),
            prev_pairs: Vec::new(),
            cache: BTreeMap::new(),
            soft_warm: Vec::new(),
            is_static,
            static_x,
            static_r,
        }
    }

    /// Enable or disable island sleeping.
    pub fn set_allow_sleep(&mut self, allow: bool) {
        self.sleep.enabled = allow;
        if !allow {
            self.sleep.awake.fill(true);
        }
    }

    /// Per-dynamic-body sleep flags (length `world.n`).
    pub fn body_awake(&self) -> &[bool] {
        &self.sleep.awake
    }

    /// Wake dynamic body `i` (and its island).
    pub fn wake_body(&mut self, i: usize) {
        self.sleep.wake_island_of(i);
    }

    /// Append a joint; wakes both endpoints.
    pub fn add_joint(&mut self, joint: Joint) -> usize {
        let id = self.joints.len();
        if joint.body_a < self.world.n {
            self.wake_body(joint.body_a);
        }
        if joint.body_b < self.world.n {
            self.wake_body(joint.body_b);
        }
        self.joints.push(joint);
        id
    }

    /// Mark dynamic body `i` as a CCD bullet (TOI vs statics).
    pub fn set_bullet(&mut self, i: usize, bullet: bool) {
        if i < self.bullet.len() {
            self.bullet[i] = bullet;
        }
    }

    /// Reaction force [N] on the last step for joint `id`.
    pub fn joint_reaction_force(&self, id: usize) -> Vec3 {
        self.joints
            .get(id)
            .map(|j| j.reaction_force)
            .unwrap_or(Vec3::zero())
    }

    /// Reaction torque [N·m] on A for joint `id` (last step).
    pub fn joint_reaction_torque(&self, id: usize) -> Vec3 {
        self.joints
            .get(id)
            .map(|j| j.reaction_torque)
            .unwrap_or(Vec3::zero())
    }

    /// Whether joint `id` has broken.
    pub fn joint_broken(&self, id: usize) -> bool {
        self.joints.get(id).map(|j| j.broken).unwrap_or(true)
    }

    fn mats_ext(&self) -> Vec<Material> {
        let mut m = self.materials.clone();
        m.extend(self.statics.iter().map(|s| s.material));
        m
    }

    fn detect_pairs_and_contacts(&self, st: &State) -> (Vec<(usize, usize)>, Vec<ContactPoint>) {
        let (x, r) = self.poses(st);
        let shapes = self.all_shapes();
        let aabbs: Vec<_> = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| s.aabb(x[i], &r[i]))
            .collect();
        let mut pairs = match self.broad_phase {
            BroadPhaseKind::Sap => SweepAndPrune.pairs(&aabbs, &self.is_static),
            BroadPhaseKind::Brute => BruteForce.pairs(&aabbs, &self.is_static),
            BroadPhaseKind::DynamicTree => {
                DynamicAabbTree::default().pairs(&aabbs, &self.is_static)
            }
        };
        pairs.sort_unstable();
        pairs.dedup();
        let narrow = DedicatedNarrowPhase;
        let mut out = Vec::new();
        for &(i, j) in &pairs {
            let mut cs = narrow.collide(
                &crate::narrowphase::ColliderPose::new(shapes[i], x[i], r[i]),
                &crate::narrowphase::ColliderPose::new(shapes[j], x[j], r[j]),
            );
            for (k, c) in cs.iter_mut().enumerate() {
                c.feature = FeatureId(k as u32);
                out.push(ContactPoint {
                    a: i,
                    b: j,
                    point: c.point,
                    normal: c.normal,
                    depth: c.depth,
                    feature: c.feature,
                });
            }
        }
        (pairs, out)
    }

    /// Simulation time [s].
    pub fn t(&self) -> f64 {
        self.world.t
    }

    fn mat(&self, i: usize) -> Material {
        let n = self.world.n;
        if i < n {
            self.materials[i]
        } else {
            self.statics[i - n].material
        }
    }

    fn poses(&self, st: &State) -> (Vec<Vec3>, Vec<[[f64; 3]; 3]>) {
        let n = self.world.n;
        let mut x = Vec::with_capacity(n + self.statics.len());
        let mut r = Vec::with_capacity(n + self.statics.len());
        for i in 0..n {
            x.push(st.x[i]);
            r.push(st.q[i].to_matrix());
        }
        x.extend_from_slice(&self.static_x);
        r.extend_from_slice(&self.static_r);
        (x, r)
    }

    fn all_shapes(&self) -> Vec<Shape> {
        let mut s = self.shapes.clone();
        s.extend(self.statics.iter().map(|c| c.shape));
        s
    }

    /// Broad + narrow phase. Pairs are sorted by `(min id, max id)`.
    pub fn detect(&self, st: Option<&State>) -> Vec<ContactPoint> {
        let st = st.unwrap_or(&self.world.state);
        self.detect_pairs_and_contacts(st).1
    }

    /// Detect with pluggable broad/narrow backends (extension hooks for AABB trees / GJK).
    ///
    /// Pair list is always sorted by `(i, j)` before narrow phase, regardless of backend.
    pub fn detect_with<B: BroadPhase, N: NarrowPhase>(
        &self,
        broad: &B,
        narrow: &N,
        st: &State,
    ) -> Vec<ContactPoint> {
        let (x, r) = self.poses(st);
        let shapes = self.all_shapes();
        let aabbs: Vec<_> = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| s.aabb(x[i], &r[i]))
            .collect();
        let mut pairs = broad.pairs(&aabbs, &self.is_static);
        pairs.sort_unstable();
        pairs.dedup();
        let mut out = Vec::new();
        for (i, j) in pairs {
            let mut cs = narrow.collide(
                &crate::narrowphase::ColliderPose::new(shapes[i], x[i], r[i]),
                &crate::narrowphase::ColliderPose::new(shapes[j], x[j], r[j]),
            );
            for (k, c) in cs.iter_mut().enumerate() {
                c.feature = FeatureId(k as u32);
                out.push(ContactPoint {
                    a: i,
                    b: j,
                    point: c.point,
                    normal: c.normal,
                    depth: c.depth,
                    feature: c.feature,
                });
            }
        }
        out
    }

    /// Default detect path (selected broad phase + dedicated pairs) — used by `step`.
    fn detect_default(&self, st: &State) -> Vec<ContactPoint> {
        self.detect_pairs_and_contacts(st).1
    }

    /// Soft / hard PGS joint solve (maximal coordinates) for the PGS path.
    fn solve_joints_pgs(&mut self, st: &mut State, dt: f64) {
        if self.joints.is_empty() {
            return;
        }
        let n = self.world.n;
        let n_total = n + self.statics.len();
        let mut im = vec![0.0; n_total];
        for i in 0..n {
            im[i] = 1.0 / self.world.mass[i];
        }
        let mut iinv = vec![[[0.0; 3]; 3]; n_total];
        for i in 0..n {
            iinv[i] = self.world.inv_inertia_world(st.q[i], i);
        }
        let mut x = Vec::with_capacity(n_total);
        x.extend_from_slice(&st.x);
        x.extend_from_slice(&self.static_x);
        let mut rmat: Vec<_> = st.q.iter().map(|q| q.to_matrix()).collect();
        rmat.extend_from_slice(&self.static_r);
        let mut v = vec![Vec3::zero(); n_total];
        v[..n].copy_from_slice(&st.v);
        let mut l = vec![Vec3::zero(); n_total];
        l[..n].copy_from_slice(&st.l);
        let awake = if self.sleep.enabled {
            Some(self.sleep.awake.as_slice())
        } else {
            None
        };
        let mut rows = joints::build_joint_rows(
            &self.joints,
            n,
            &x,
            &rmat,
            &self.static_x,
            &self.static_r,
            &im,
            &iinv,
            dt,
            true,
        );
        joints::warm_start_joints(&mut rows, &self.joints, n, awake, &mut v, &mut l, &im);
        for _ in 0..self.iterations {
            joints::solve_joint_rows(&mut rows, n, awake, &mut v, &mut l, &iinv, &im);
        }
        joints::finalize_joints(&mut self.joints, &rows, dt);
        st.v.copy_from_slice(&v[..n]);
        st.l.copy_from_slice(&l[..n]);
    }

    /// Sequential-impulse velocity solve. Writes `st.v` / `st.l` for dynamics.
    ///
    /// `dt` is used for speculative contact bias (`sep/dt`).
    pub fn solve_velocities(&mut self, st: &mut State, contacts: &[ContactPoint], dt: f64) {
        let n = self.world.n;
        let dt_for_speculative = if dt > 0.0 { dt } else { 1.0 };
        let n_total = n + self.statics.len();
        let mut im = vec![0.0; n_total];
        for i in 0..n {
            im[i] = 1.0 / self.world.mass[i];
        }
        let mut iinv = vec![[[0.0; 3]; 3]; n_total];
        for i in 0..n {
            iinv[i] = self.world.inv_inertia_world(st.q[i], i);
        }
        let mut x = Vec::with_capacity(n_total);
        x.extend_from_slice(&st.x);
        x.extend_from_slice(&self.static_x);
        let mut v = vec![Vec3::zero(); n_total];
        v[..n].copy_from_slice(&st.v);
        let mut l = vec![Vec3::zero(); n_total];
        l[..n].copy_from_slice(&st.l);

        struct Row {
            a: usize,
            b: usize,
            point: Vec3,
            normal: Vec3,
            ra: Vec3,
            rb: Vec3,
            mu: f64,
            kn: f64,
            k1: f64,
            k2: f64,
            t1: Vec3,
            t2: Vec3,
            bias: f64,
            lam: f64,
            jt: [f64; 2],
            key: (usize, usize, u32),
        }

        let mut rows = Vec::with_capacity(contacts.len());
        let mut counter: BTreeMap<(usize, usize), u32> = BTreeMap::new();
        for c in contacts {
            let k = *counter.get(&(c.a, c.b)).unwrap_or(&0);
            counter.insert((c.a, c.b), k + 1);
            // Sequential index `k` matches the Python warm-start key `(a, b, k)`.
            let feat = k;
            let ra = c.point - x[c.a];
            let rb = c.point - x[c.b];
            let ma = self.mat(c.a);
            let mb = self.mat(c.b);
            let mu = sqrt(ma.friction * mb.friction);
            let e = ma.restitution.max(mb.restitution);
            // Effective mass: 1 / (1/m_a + 1/m_b + n·((I⁻¹(r×n))×r)_a+b)
            // Matches Python `cross(Iinv @ cross(r, d), r)`.
            let kmass = |d: Vec3| -> f64 {
                let ka = mat_vec(&iinv[c.a], ra.cross(d)).cross(ra);
                let kb = mat_vec(&iinv[c.b], rb.cross(d)).cross(rb);
                1.0 / (im[c.a] + im[c.b] + d.dot(ka + kb))
            };
            let (t1, t2) = friction_basis(c.normal);
            let va = v[c.a] + mat_vec(&iinv[c.a], l[c.a]).cross(ra);
            let vb = v[c.b] + mat_vec(&iinv[c.b], l[c.b]).cross(rb);
            let vrel = vb - va;
            let vn0 = vrel.dot(c.normal);
            let bias = if c.depth < 0.0 {
                // Speculative: target vn ≥ −sep/dt with sep = −depth.
                c.depth / dt_for_speculative
            } else if vn0 < -self.restitution_threshold {
                -e * vn0
            } else {
                0.0
            };
            let mu = if c.depth < 0.0 { 0.0 } else { mu };
            rows.push(Row {
                a: c.a,
                b: c.b,
                point: c.point,
                normal: c.normal,
                ra,
                rb,
                mu,
                kn: kmass(c.normal),
                k1: kmass(t1),
                k2: kmass(t2),
                t1,
                t2,
                bias,
                lam: 0.0,
                jt: [0.0, 0.0],
                key: (c.a, c.b, feat),
            });
        }

        let apply = |a: usize,
                     b: usize,
                     ra: Vec3,
                     rb: Vec3,
                     p: Vec3,
                     v: &mut [Vec3],
                     l: &mut [Vec3],
                     im: &[f64]| {
            v[a] -= p * im[a];
            if a < n {
                l[a] -= ra.cross(p);
            }
            v[b] += p * im[b];
            if b < n {
                l[b] += rb.cross(p);
            }
        };

        let old_cache = std::mem::take(&mut self.cache);
        if self.warm_start {
            for r in &mut rows {
                if let Some(old) = old_cache.get(&r.key) {
                    if (old.point - r.point).norm() > 0.01 {
                        continue;
                    }
                    r.lam = old.normal_impulse;
                    r.jt = [old.tangent_impulse.dot(r.t1), old.tangent_impulse.dot(r.t2)];
                    let p = r.normal * r.lam + r.t1 * r.jt[0] + r.t2 * r.jt[1];
                    apply(r.a, r.b, r.ra, r.rb, p, &mut v, &mut l, &im);
                }
            }
        }

        for _ in 0..self.iterations {
            for r in &mut rows {
                let va = v[r.a] + mat_vec(&iinv[r.a], l[r.a]).cross(r.ra);
                let vb = v[r.b] + mat_vec(&iinv[r.b], l[r.b]).cross(r.rb);
                let mut vrel = vb - va;
                // Friction
                let old = r.jt;
                let mut jt = [
                    old[0] - r.k1 * vrel.dot(r.t1),
                    old[1] - r.k2 * vrel.dot(r.t2),
                ];
                let lim = r.mu * r.lam;
                let nj = sqrt(jt[0] * jt[0] + jt[1] * jt[1]);
                if nj > lim {
                    jt[0] *= lim / nj;
                    jt[1] *= lim / nj;
                }
                let d0 = jt[0] - old[0];
                let d1 = jt[1] - old[1];
                r.jt = jt;
                apply(
                    r.a,
                    r.b,
                    r.ra,
                    r.rb,
                    r.t1 * d0 + r.t2 * d1,
                    &mut v,
                    &mut l,
                    &im,
                );
                let va = v[r.a] + mat_vec(&iinv[r.a], l[r.a]).cross(r.ra);
                let vb = v[r.b] + mat_vec(&iinv[r.b], l[r.b]).cross(r.rb);
                vrel = vb - va;
                // Normal
                let lam = (r.lam + r.kn * (r.bias - vrel.dot(r.normal))).max(0.0);
                apply(
                    r.a,
                    r.b,
                    r.ra,
                    r.rb,
                    r.normal * (lam - r.lam),
                    &mut v,
                    &mut l,
                    &im,
                );
                r.lam = lam;
            }
        }

        for r in &rows {
            self.cache.insert(
                r.key,
                WarmEntry {
                    normal_impulse: r.lam,
                    tangent_impulse: r.t1 * r.jt[0] + r.t2 * r.jt[1],
                    point: r.point,
                },
            );
        }
        st.v.copy_from_slice(&v[..n]);
        st.l.copy_from_slice(&l[..n]);
    }

    /// Positional correction: translate bodies apart; velocities untouched.
    pub fn correct_positions(&mut self, st: &mut State) {
        let n = self.world.n;
        let mut im = Vec::with_capacity(n + self.statics.len());
        for i in 0..n {
            im.push(1.0 / self.world.mass[i]);
        }
        im.extend(std::iter::repeat_n(0.0, self.statics.len()));

        // Insertion-ordered worst depth per pair (BTreeMap sorted by key — Python 3.7+
        // dict preserves detection order; for equal physics the correction commute for
        // disjoint pairs. Using BTreeMap is deterministic; pairs are already sorted.)
        let mut worst: BTreeMap<(usize, usize), (f64, Vec3)> = BTreeMap::new();
        for c in self.detect_default(st) {
            let e = worst.entry((c.a, c.b)).or_insert((0.0, Vec3::zero()));
            if c.depth > e.0 {
                *e = (c.depth, c.normal);
            }
        }
        for ((a, b), (d, nrm)) in worst {
            let c = self.beta * (d - self.slop).max(0.0);
            let s = im[a] + im[b];
            if c <= 0.0 || s == 0.0 {
                continue;
            }
            if a < n {
                st.x[a] -= nrm * (c * im[a] / s);
            }
            if b < n {
                st.x[b] += nrm * (c * im[b] / s);
            }
        }
    }

    /// One contact step [s].
    pub fn step(&mut self, dt: f64) {
        let (f, tau) = self.world.forces(&self.world.state, self.world.t);
        let n = self.world.n;
        self.sleep.resize(n);

        // Wake islands that receive non-gravitational forces/torques.
        // Uniform gravity alone must not prevent sleeping (Box2D-style).
        if self.sleep.enabled {
            use pe_math::STANDARD_GRAVITY;
            for i in 0..n {
                let weight = Vec3::new(0.0, 0.0, -STANDARD_GRAVITY * self.world.mass[i]);
                let excess = f[i] - weight;
                // Torque always wakes; excess linear force (springs, user forces, …) wakes.
                self.sleep.wake_from_force(i, excess, tau[i]);
            }
        }

        let mut st = self.world.state.copy();
        for i in 0..n {
            if !self.sleep.is_awake(i) {
                // Keep sleeping bodies at rest.
                st.v[i] = Vec3::zero();
                st.l[i] = Vec3::zero();
            }
        }

        // PGS integrates forces over the full `dt` before contact solve.
        // Soft-step integrates forces inside each sub-step with `h = dt/N`
        // (Box2D Soft Step / Solver2D), so skip the full-step inject here.
        if self.solver_mode == SolverMode::Pgs {
            for i in 0..n {
                if !self.sleep.is_awake(i) {
                    continue;
                }
                st.v[i] += f[i] * (dt / self.world.mass[i]);
                st.l[i] += tau[i] * dt;
            }
        }

        // Bullet TOI vs statics: advance fast bodies to first impact, shrink dt.
        let mut step_dt = dt;
        let mut toi_contacts: Vec<ContactPoint> = Vec::new();
        if self.ccd.toi && self.bullet.iter().any(|&b| b) {
            self.bullet.resize(n, false);
            let statics: Vec<_> = self
                .statics
                .iter()
                .map(|s| (s.shape, s.position, s.orientation))
                .collect();
            let (t_hit, hits) = ccd::advance_bullets_to_toi(
                &self.bullet,
                &self.shapes,
                &mut st.x,
                &st.q,
                &st.v,
                &statics,
                dt,
                &self.ccd,
            );
            if !hits.is_empty() {
                toi_contacts = hits;
                step_dt = (dt - t_hit).max(0.0);
            }
        }

        let (pairs, mut contacts) = self.detect_pairs_and_contacts(&st);
        // Speculative contacts for fast approaches (skip pairs already discrete).
        {
            let statics: Vec<_> = self
                .statics
                .iter()
                .map(|s| (s.shape, s.position, s.orientation))
                .collect();
            let spec = ccd::generate_speculative_contacts(
                &self.shapes,
                &st.x,
                &st.q,
                &st.v,
                &statics,
                &contacts,
                &self.ccd,
            );
            contacts.extend(spec);
            contacts.extend(toi_contacts);
        }

        let joint_pairs = joints::joint_island_pairs(&self.joints, n);
        let island_pairs = joints::merge_island_pairs(&pairs, &joint_pairs);
        self.sleep
            .wake_from_new_contacts(n, &self.prev_pairs, &island_pairs);
        self.sleep.rebuild_islands(n, &island_pairs);
        // Re-apply awake flags after rebuild (islands rebuilt; keep awake vector).
        self.prev_pairs = island_pairs;

        match self.solver_mode {
            SolverMode::Pgs => {
                // Filter contacts involving only sleeping dynamics.
                let awake = self.sleep.awake.clone();
                let active: Vec<ContactPoint> = contacts
                    .iter()
                    .cloned()
                    .filter(|c| {
                        let a_ok = c.a >= n || awake[c.a];
                        let b_ok = c.b >= n || awake[c.b];
                        a_ok && b_ok && (c.a >= n || c.b >= n || awake[c.a] || awake[c.b])
                    })
                    .collect();
                self.solve_velocities(&mut st, &active, step_dt);
                self.solve_joints_pgs(&mut st, step_dt);
                self.contacts = contacts;
                for i in 0..n {
                    if !self.sleep.is_awake(i) {
                        continue;
                    }
                    st.x[i] += st.v[i] * step_dt;
                }
                // Drift rotation only for awake bodies.
                let q_new = drift_rotation(&self.world, &st.q, &st.l, step_dt);
                for i in 0..n {
                    if self.sleep.is_awake(i) {
                        st.q[i] = q_new[i].normalize();
                    }
                }
                self.correct_positions(&mut st);
            }
            SolverMode::SoftStep => {
                let mats = self.mats_ext();
                let awake_slice = if self.sleep.enabled {
                    Some(self.sleep.awake.as_slice())
                } else {
                    None
                };
                let result = solve_soft_step(
                    &self.world,
                    &mats,
                    &self.static_x,
                    &self.static_r,
                    &self.soft_params,
                    step_dt,
                    &mut st,
                    &contacts,
                    awake_slice,
                    &self.soft_warm,
                    &f,
                    &tau,
                    &mut self.joints,
                );
                self.soft_warm = result.warm;
                self.contacts = result.contacts;
                // Positional polish for residual overlap.
                let beta_save = self.beta;
                self.beta = self.beta.max(0.5);
                self.correct_positions(&mut st);
                self.beta = beta_save;
            }
        }

        // Sleep update.
        if self.sleep.enabled {
            let mut lin = vec![0.0; n];
            let mut ang = vec![0.0; n];
            let mut has_c = vec![false; n];
            for i in 0..n {
                lin[i] = st.v[i].norm();
                let iinv = self.world.inv_inertia_world(st.q[i], i);
                ang[i] = angular_speed(st.l[i], &iinv);
            }
            for c in &self.contacts {
                if c.a < n {
                    has_c[c.a] = true;
                }
                if c.b < n {
                    has_c[c.b] = true;
                }
            }
            self.sleep.update_sleep(dt, &lin, &ang, &has_c);
            for i in 0..n {
                if !self.sleep.awake[i] {
                    st.v[i] = Vec3::zero();
                    st.l[i] = Vec3::zero();
                }
            }
        }

        self.world.state = st;
        self.world.t += dt;
    }

    /// Fixed-`dt` run to `t_end` [s].
    pub fn run_for(&mut self, t_end: f64, dt: f64) {
        while self.world.t < t_end - 1e-12 * t_end.abs().max(1.0) {
            let h = dt.min(t_end - self.world.t);
            self.step(h);
        }
    }

    /// Like [`run_for`] but collect `record(self)` after each step.
    pub fn run_recorded<R, F>(&mut self, t_end: f64, dt: f64, mut record: F) -> Vec<R>
    where
        F: FnMut(&ContactWorld) -> R,
    {
        let mut out = Vec::new();
        while self.world.t < t_end - 1e-12 * t_end.abs().max(1.0) {
            let h = dt.min(t_end - self.world.t);
            self.step(h);
            out.push(record(self));
        }
        out
    }

    // --- M8 / pe-scene / pe-ffi additive hooks (do not reorder private fields) ---

    /// Append a dynamic rigid body with shape and material. Returns the new body index.
    ///
    /// Rebuilds the static-flag table so broad-phase indices stay consistent.
    pub fn add_dynamic_body(
        &mut self,
        body: pe_core::RigidBody,
        shape: Shape,
        material: Material,
    ) -> usize {
        use pe_core::body::{invert3, mat_mul, mat_transpose, mat_vec};
        let i = self.world.n;
        let qq = body.orientation.normalize();
        let r = qq.to_matrix();
        let i_w = mat_mul(&mat_mul(&r, &body.inertia_body), &mat_transpose(&r));
        self.world.mass.push(body.mass);
        self.world.inertia_body.push(body.inertia_body);
        self.world
            .inv_inertia_body
            .push(invert3(&body.inertia_body));
        self.world.state.x.push(body.position);
        self.world.state.q.push(qq);
        self.world.state.v.push(body.velocity);
        self.world
            .state
            .l
            .push(mat_vec(&i_w, body.angular_velocity));
        self.world.n += 1;
        self.shapes.push(shape);
        self.materials.push(material);
        self.bullet.push(false);
        self.sleep.resize(self.world.n);
        self.rebuild_static_tables();
        i
    }

    /// Append a static collider and refresh static pose caches.
    pub fn add_static(&mut self, collider: StaticCollider) -> usize {
        let id = self.statics.len();
        self.statics.push(collider);
        self.rebuild_static_tables();
        id
    }

    /// Rebuild `is_static` / static pose caches after body or static mutations.
    pub fn rebuild_static_tables(&mut self) {
        let n = self.world.n;
        let s = self.statics.len();
        self.is_static = vec![false; n];
        self.is_static.extend(std::iter::repeat_n(true, s));
        self.static_x = self.statics.iter().map(|c| c.position).collect();
        self.static_r = self
            .statics
            .iter()
            .map(|c| c.orientation.to_matrix())
            .collect();
    }

    /// Export warm-start cache entries in deterministic `(a,b,feature)` order.
    ///
    /// Tuple layout: `(a, b, feature, λn, jt[3], point[3])`.
    pub fn warm_cache_entries(&self) -> Vec<WarmCacheRow> {
        self.cache
            .iter()
            .map(|(&(a, b, f), e)| WarmCacheRow {
                a,
                b,
                feature: f,
                normal_impulse: e.normal_impulse,
                tangent_impulse: e.tangent_impulse.to_array(),
                point: e.point.to_array(),
            })
            .collect()
    }

    /// Replace the warm-start cache (clears first). Used by scene restore.
    pub fn set_warm_cache_entries(&mut self, entries: &[WarmCacheRow]) {
        self.cache.clear();
        for e in entries {
            self.cache.insert(
                (e.a, e.b, e.feature),
                WarmEntry {
                    normal_impulse: e.normal_impulse,
                    tangent_impulse: Vec3::from_slice(&e.tangent_impulse),
                    point: Vec3::from_slice(&e.point),
                },
            );
        }
    }

    /// Soft-step warm impulses in contact order.
    pub fn soft_warm_entries(&self) -> &[(usize, usize, f64, Vec3)] {
        &self.soft_warm
    }

    /// Replace soft-step warm impulses.
    pub fn set_soft_warm_entries(&mut self, entries: Vec<(usize, usize, f64, Vec3)>) {
        self.soft_warm = entries;
    }

    /// Previous-step broad-phase pairs (wake-on-new-contact).
    pub fn prev_pairs(&self) -> &[(usize, usize)] {
        &self.prev_pairs
    }

    /// Replace previous-step broad-phase pairs.
    pub fn set_prev_pairs(&mut self, pairs: Vec<(usize, usize)>) {
        self.prev_pairs = pairs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use pe_core::diagnostics::{
        angular_momentum, body_angular_velocity, kinetic_energy, linear_momentum,
    };
    use pe_core::forces::UniformGravity;
    use pe_core::{inertia_solid_box, RigidBody};
    use pe_math::{Quat, Vec3, STANDARD_GRAVITY as G};

    fn ball(m: f64, r: f64, x: Vec3, v: Vec3, w: Vec3) -> RigidBody {
        RigidBody::new(m)
            .with_inertia(crate::shapes::Sphere::new(r).inertia(m))
            .with_position(x)
            .with_velocity(v)
            .with_angular_velocity(w)
    }

    fn make(
        bodies: Vec<RigidBody>,
        shapes: Vec<Shape>,
        statics: Vec<StaticCollider>,
        mats: Option<Vec<Material>>,
        gravity: bool,
    ) -> ContactWorld {
        let gens: Vec<Box<dyn pe_core::ForceGenerator>> = if gravity {
            vec![Box::new(UniformGravity::earth())]
        } else {
            vec![]
        };
        ContactWorld::new(World::new(bodies, gens), shapes, statics, mats)
    }

    #[test]
    fn head_on_elastic_equal_mass_exchanges_velocity() {
        let m = 1.0;
        let mut cw = make(
            vec![
                ball(
                    m,
                    0.5,
                    Vec3::new(-1.0, 0.0, 0.0),
                    Vec3::new(2.0, 0.0, 0.0),
                    Vec3::zero(),
                ),
                ball(
                    m,
                    0.5,
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(-2.0, 0.0, 0.0),
                    Vec3::zero(),
                ),
            ],
            vec![Shape::Sphere(crate::shapes::Sphere::new(0.5)); 2],
            vec![],
            Some(vec![Material::new(1.0, 0.0); 2]),
            false,
        );
        cw.restitution_threshold = 0.0;
        cw.run_for(1.0, 1e-3);
        assert_abs_diff_eq!(cw.world.state.v[0].x, -2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cw.world.state.v[1].x, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn head_on_spheres_momentum_energy_restitution() {
        for e in [1.0, 0.5, 0.0] {
            let m1 = 2.0;
            let m2 = 3.0;
            let v1 = 4.0;
            let v2 = -1.0;
            let mut cw = make(
                vec![
                    ball(
                        m1,
                        0.5,
                        Vec3::new(-1.0, 0.0, 0.0),
                        Vec3::new(v1, 0.0, 0.0),
                        Vec3::zero(),
                    ),
                    ball(
                        m2,
                        0.5,
                        Vec3::new(1.0, 0.0, 0.0),
                        Vec3::new(v2, 0.0, 0.0),
                        Vec3::zero(),
                    ),
                ],
                vec![Shape::Sphere(crate::shapes::Sphere::new(0.5)); 2],
                vec![],
                Some(vec![Material::new(e, 0.0); 2]),
                false,
            );
            cw.restitution_threshold = 0.0;
            let p0 = linear_momentum(&cw.world, &cw.world.state);
            let t0 = kinetic_energy(&cw.world, &cw.world.state);
            cw.run_for(1.0, 1e-3);
            let p1 = linear_momentum(&cw.world, &cw.world.state);
            let t1 = kinetic_energy(&cw.world, &cw.world.state);
            let u1 = (m1 * v1 + m2 * v2 + m2 * e * (v2 - v1)) / (m1 + m2);
            let u2 = (m1 * v1 + m2 * v2 + m1 * e * (v1 - v2)) / (m1 + m2);
            assert_abs_diff_eq!(cw.world.state.v[0].x, u1, epsilon = 1e-12);
            assert_abs_diff_eq!(cw.world.state.v[1].x, u2, epsilon = 1e-12);
            assert_abs_diff_eq!(p1.x, p0.x, epsilon = 1e-12);
            assert_abs_diff_eq!(p1.y, p0.y, epsilon = 1e-12);
            assert_abs_diff_eq!(p1.z, p0.z, epsilon = 1e-12);
            assert_abs_diff_eq!(
                (cw.world.state.v[1].x - cw.world.state.v[0].x) / (v1 - v2),
                e,
                epsilon = 1e-12
            );
            let dt = 0.5 * m1 * m2 / (m1 + m2) * (1.0 - e * e) * (v1 - v2) * (v1 - v2);
            assert_abs_diff_eq!(t0 - t1, dt, epsilon = 1e-10);
        }
    }

    #[test]
    fn block_slides_with_coulomb_acceleration() {
        let theta = 30.0_f64.to_radians();
        let mu = 0.3;
        let n = Vec3::new(theta.sin(), 0.0, theta.cos());
        let tdown = Vec3::new(theta.cos(), 0.0, -theta.sin());
        let hb = Vec3::new(0.2, 0.2, 0.05);
        let q = Quat::from_axis_angle(Vec3::unit_y(), theta);
        let box_b = RigidBody::new(1.0)
            .with_inertia(crate::shapes::BoxShape::new(hb.x, hb.y, hb.z).inertia(1.0))
            .with_position(n * (hb.z - 1e-6))
            .with_orientation(q);
        let mut cw = make(
            vec![box_b],
            vec![Shape::Box(crate::shapes::BoxShape::new(hb.x, hb.y, hb.z))],
            vec![
                StaticCollider::new(Shape::Plane(crate::shapes::Plane::new(n, 0.0)))
                    .with_material(Material::new(0.0, mu)),
            ],
            Some(vec![Material::new(0.0, mu)]),
            true,
        );
        let t = 1.0;
        cw.run_for(t, 1.0 / 480.0);
        let a = G * (theta.sin() - mu * theta.cos());
        assert_relative_eq!(cw.world.state.v[0].dot(tdown), a * t, max_relative = 1e-9);
    }

    #[test]
    fn block_holds_when_tan_below_mu() {
        let theta = 30.0_f64.to_radians();
        let mu = 0.6;
        assert!(theta.tan() < mu);
        let n = Vec3::new(theta.sin(), 0.0, theta.cos());
        let tdown = Vec3::new(theta.cos(), 0.0, -theta.sin());
        let hb = Vec3::new(0.2, 0.2, 0.05);
        let q = Quat::from_axis_angle(Vec3::unit_y(), theta);
        let make_cw = || {
            let box_b = RigidBody::new(1.0)
                .with_inertia(crate::shapes::BoxShape::new(hb.x, hb.y, hb.z).inertia(1.0))
                .with_position(n * (hb.z - 1e-6))
                .with_orientation(q);
            make(
                vec![box_b],
                vec![Shape::Box(crate::shapes::BoxShape::new(hb.x, hb.y, hb.z))],
                vec![
                    StaticCollider::new(Shape::Plane(crate::shapes::Plane::new(n, 0.0)))
                        .with_material(Material::new(0.0, mu)),
                ],
                Some(vec![Material::new(0.0, mu)]),
                true,
            )
        };
        let mut cw1 = make_cw();
        cw1.run_for(0.5, 1.0 / 240.0);
        let x1 = cw1.world.state.x[0];
        let mut cw2 = make_cw();
        cw2.run_for(2.0, 1.0 / 240.0);
        assert!(x1.dot(tdown).abs() < 1e-4);
        assert!((cw2.world.state.x[0] - x1).dot(tdown).abs() < 1e-9);
        assert!(cw2.world.state.v[0].norm() < 1e-6);
    }

    #[test]
    fn sliding_sphere_reaches_five_sevenths_v0() {
        let v0 = 3.0;
        let mu = 0.3;
        let r = 0.1;
        let mut cw = make(
            vec![ball(
                1.0,
                r,
                Vec3::new(0.0, 0.0, r - 1e-6),
                Vec3::new(v0, 0.0, 0.0),
                Vec3::zero(),
            )],
            vec![Shape::Sphere(crate::shapes::Sphere::new(r))],
            vec![
                StaticCollider::new(Shape::Plane(crate::shapes::Plane::new(Vec3::unit_z(), 0.0)))
                    .with_material(Material::new(0.0, mu)),
            ],
            Some(vec![Material::new(0.0, mu)]),
            true,
        );
        let tstar = 2.0 * v0 / (7.0 * mu * G);
        let rec = cw.run_recorded(1.0, 1e-3, |c| {
            (
                c.t(),
                c.world.state.v[0].x,
                body_angular_velocity(&c.world, &c.world.state)[0].y,
            )
        });
        for &(t, v, _) in &rec {
            if t < tstar - 2e-3 {
                assert_abs_diff_eq!(v, v0 - mu * G * t, epsilon = 1e-9);
            }
            if t > tstar + 2e-3 {
                assert_relative_eq!(v, 5.0 / 7.0 * v0, max_relative = 1e-5);
            }
        }
        let (_, v_end, w_end) = *rec.last().unwrap();
        assert_relative_eq!(w_end * r, v_end, max_relative = 1e-5);
    }

    #[test]
    fn box_resting_on_plane_does_not_drift() {
        let hb = (0.5, 0.3, 0.2);
        let mut cw = make(
            vec![RigidBody::new(10.0)
                .with_inertia(inertia_solid_box(10.0, 1.0, 0.6, 0.4))
                .with_position(Vec3::new(0.0, 0.0, 0.2))],
            vec![Shape::Box(crate::shapes::BoxShape::new(hb.0, hb.1, hb.2))],
            vec![
                StaticCollider::new(Shape::Plane(crate::shapes::Plane::new(Vec3::unit_z(), 0.0)))
                    .with_material(Material::new(0.0, 1.0)),
            ],
            None,
            true,
        );
        cw.run_for(5.0, 1.0 / 240.0);
        let s = &cw.world.state;
        assert!(s.x[0].x.abs() < 1e-9 && s.x[0].y.abs() < 1e-9);
        assert_abs_diff_eq!(s.x[0].z, 0.2, epsilon = 2.0 * cw.slop);
        assert!(s.v[0].norm() < G / 240.0 + 1e-9);
    }

    #[test]
    fn contact_simulation_is_bit_deterministic() {
        fn run() -> Vec<f64> {
            // Fixed pseudo-random poses
            let mut state: u64 = 1;
            let mut next = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                state
            };
            let uni = |lo: f64, hi: f64, next: &mut dyn FnMut() -> u64| {
                let u = (next() >> 11) as f64 / ((1u64 << 53) as f64);
                lo + (hi - lo) * u
            };
            let mut bodies = Vec::new();
            let mut shapes = Vec::new();
            for _ in 0..6 {
                let p = Vec3::new(
                    uni(-1.0, 1.0, &mut next),
                    uni(-1.0, 1.0, &mut next),
                    uni(0.5, 2.0, &mut next),
                );
                let q = Quat::new(
                    uni(-1.0, 1.0, &mut next),
                    uni(-1.0, 1.0, &mut next),
                    uni(-1.0, 1.0, &mut next),
                    uni(-1.0, 1.0, &mut next),
                )
                .normalize();
                bodies.push(
                    RigidBody::new(1.0)
                        .with_inertia(crate::shapes::BoxShape::new(0.2, 0.2, 0.2).inertia(1.0))
                        .with_position(p)
                        .with_orientation(q),
                );
                shapes.push(Shape::Box(crate::shapes::BoxShape::new(0.2, 0.2, 0.2)));
            }
            let mut cw = make(
                bodies,
                shapes,
                vec![StaticCollider::new(Shape::Plane(crate::shapes::Plane::new(
                    Vec3::unit_z(),
                    0.0,
                )))
                .with_material(Material::new(0.0, 1.0))],
                Some(vec![Material::new(0.3, 0.5); 6]),
                true,
            );
            cw.run_for(0.5, 1.0 / 240.0);
            cw.world.state.pack()
        }
        assert_eq!(run(), run());
    }

    #[test]
    fn off_centre_hit_conserves_p_and_l() {
        let mb = 4.0;
        let ms = 1.0;
        let box_b = RigidBody::new(mb)
            .with_inertia(crate::shapes::BoxShape::new(0.5, 0.5, 0.5).inertia(mb))
            .with_position(Vec3::zero());
        let sph = ball(
            ms,
            0.2,
            Vec3::new(-2.0, 0.3, 0.1),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 20.0),
        );
        let mut cw = make(
            vec![box_b, sph],
            vec![
                Shape::Box(crate::shapes::BoxShape::new(0.5, 0.5, 0.5)),
                Shape::Sphere(crate::shapes::Sphere::new(0.2)),
            ],
            vec![],
            Some(vec![Material::new(0.6, 0.4); 2]),
            false,
        );
        cw.beta = 0.0;
        let o = Vec3::new(-0.5, 0.3, 0.1);
        let h0 = angular_momentum(&cw.world, &cw.world.state, o);
        let p0 = linear_momentum(&cw.world, &cw.world.state);
        let t0 = kinetic_energy(&cw.world, &cw.world.state);
        cw.run_for(0.6, 1e-3);
        let h1 = angular_momentum(&cw.world, &cw.world.state, o);
        let p1 = linear_momentum(&cw.world, &cw.world.state);
        assert_abs_diff_eq!(p1.x, p0.x, epsilon = 1e-12);
        assert_abs_diff_eq!(p1.y, p0.y, epsilon = 1e-12);
        assert_abs_diff_eq!(p1.z, p0.z, epsilon = 1e-12);
        assert_abs_diff_eq!(h1.x, h0.x, epsilon = 1e-12);
        assert_abs_diff_eq!(h1.y, h0.y, epsilon = 1e-12);
        assert_abs_diff_eq!(h1.z, h0.z, epsilon = 1e-12);
        assert!(body_angular_velocity(&cw.world, &cw.world.state)[0].z.abs() > 0.1);
        assert!(kinetic_energy(&cw.world, &cw.world.state) < t0);
    }
}
