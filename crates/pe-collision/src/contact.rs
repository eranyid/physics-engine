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
use std::time::Instant;

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
use crate::narrowphase::{collide as narrow_collide, NarrowPhase};
use crate::scratch::{StepPhaseTimes, StepScratch};
use crate::shapes::{Material, Shape};
use crate::soft_step::solve_soft_step;
use crate::softness::{SoftStepParams, SolverMode};
use crate::solver_soa::ContactRowsSoa;

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

#[inline]
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

/// Effective mass along direction `d` at levers `ra`/`rb` (matches Python
/// `cross(Iinv @ cross(r, d), r)`).
#[inline]
fn effective_mass(
    d: Vec3,
    ra: Vec3,
    rb: Vec3,
    ima: f64,
    imb: f64,
    iinva: &[[f64; 3]; 3],
    iinvb: &[[f64; 3]; 3],
) -> f64 {
    let ka = mat_vec(iinva, ra.cross(d)).cross(ra);
    let kb = mat_vec(iinvb, rb.cross(d)).cross(rb);
    1.0 / (ima + imb + d.dot(ka + kb))
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
    /// Warm-start / contact-manifold impulse cache keyed by `(a, b, feature)` —
    /// `BTreeMap` so there is never unordered hash-map iteration on the step path.
    /// Geometric narrow-phase reuse (skipping SAT/GJK) is intentionally **not**
    /// used: it would change contact points and break locked hashes. Impulses are
    /// warm-started when the point moves less than 1 cm (Python oracle parity).
    cache: BTreeMap<(usize, usize, u32), WarmEntry>,
    /// Soft-step warm cache `(a, b, λn, jt)` in contact order.
    soft_warm: Vec<(usize, usize, f64, Vec3)>,
    is_static: Vec<bool>,
    static_x: Vec<Vec3>,
    static_r: Vec<[[f64; 3]; 3]>,
    /// Reused pose / AABB / pair / solver buffers (allocation reuse).
    scratch: StepScratch,
    /// SoA contact rows for the PGS velocity solver.
    rows_soa: ContactRowsSoa,
    /// Persistent dynamic AABB tree (incremental fat-AABB updates).
    aabb_tree: DynamicAabbTree,
    /// Last step phase timings (valid when [`Self::profile_phases`] is set).
    pub last_phase_times: StepPhaseTimes,
    /// When true, fill [`Self::last_phase_times`] with wall-clock phase costs.
    pub profile_phases: bool,
    /// When true, solve contacts per island in ascending island-id order.
    /// Required for bit-identical `parallel`-feature island solves. Default
    /// `false` keeps global PGS (locked oracle hashes).
    pub island_ordered_solve: bool,
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
            scratch: StepScratch::new(),
            rows_soa: ContactRowsSoa::default(),
            aabb_tree: DynamicAabbTree::default(),
            last_phase_times: StepPhaseTimes::default(),
            profile_phases: false,
            island_ordered_solve: false,
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

    /// Fill scratch poses (dynamics + statics) without allocating a new Vec pair.
    fn fill_poses_scratch(&mut self, st: &State) {
        let n = self.world.n;
        let sc = &mut self.scratch;
        sc.x.clear();
        sc.r.clear();
        sc.x.reserve(n + self.statics.len());
        sc.r.reserve(n + self.statics.len());
        for i in 0..n {
            sc.x.push(st.x[i]);
            sc.r.push(st.q[i].to_matrix());
        }
        sc.x.extend_from_slice(&self.static_x);
        sc.r.extend_from_slice(&self.static_r);
    }

    fn detect_pairs_and_contacts(
        &mut self,
        st: &State,
    ) -> (Vec<(usize, usize)>, Vec<ContactPoint>) {
        let profile = self.profile_phases;
        let t_detect0 = if profile { Some(Instant::now()) } else { None };

        self.fill_poses_scratch(st);
        let n = self.world.n;
        let n_total = n + self.statics.len();

        // Build AABBs without cloning shapes.
        self.scratch.aabbs.clear();
        self.scratch.aabbs.reserve(n_total);
        for i in 0..n {
            self.scratch
                .aabbs
                .push(self.shapes[i].aabb(self.scratch.x[i], &self.scratch.r[i]));
        }
        for (si, s) in self.statics.iter().enumerate() {
            let i = n + si;
            self.scratch
                .aabbs
                .push(s.shape.aabb(self.scratch.x[i], &self.scratch.r[i]));
        }

        let t_broad0 = if profile { Some(Instant::now()) } else { None };
        self.scratch.pairs.clear();
        match self.broad_phase {
            BroadPhaseKind::Sap => {
                // SAP already returns sorted unique pairs — skip a second sort.
                self.scratch.pairs = SweepAndPrune.pairs(&self.scratch.aabbs, &self.is_static);
            }
            BroadPhaseKind::Brute => {
                self.scratch.pairs = BruteForce.pairs(&self.scratch.aabbs, &self.is_static);
            }
            BroadPhaseKind::DynamicTree => {
                // Incremental fat-AABB update + tight filter (≡ rebuild pair set).
                self.scratch.pairs = self
                    .aabb_tree
                    .pairs_incremental(&self.scratch.aabbs, &self.is_static);
            }
        }
        // Brute is already sorted; SAP/tree sort internally. Cheap check avoids work
        // when the backend already produced stable ascending order.
        if !self.scratch.pairs.windows(2).all(|w| w[0] <= w[1]) {
            self.scratch.pairs.sort_unstable();
        }
        self.scratch.pairs.dedup();
        if let Some(t0) = t_broad0 {
            self.last_phase_times.broad = t0.elapsed().as_secs_f64();
        }

        let t_narrow0 = if profile { Some(Instant::now()) } else { None };
        self.scratch.contacts.clear();
        // Narrow phase: call `collide` with shape references — no Shape clones.
        // Index-based loop so we can push into `scratch.contacts` while reading pairs.
        let npairs = self.scratch.pairs.len();
        for p in 0..npairs {
            let (i, j) = self.scratch.pairs[p];
            let xa = self.scratch.x[i];
            let xb = self.scratch.x[j];
            let ra = self.scratch.r[i];
            let rb = self.scratch.r[j];
            let mut cs = match (i < n, j < n) {
                (true, true) => narrow_collide(&self.shapes[i], xa, &ra, &self.shapes[j], xb, &rb),
                (true, false) => narrow_collide(
                    &self.shapes[i],
                    xa,
                    &ra,
                    &self.statics[j - n].shape,
                    xb,
                    &rb,
                ),
                (false, true) => narrow_collide(
                    &self.statics[i - n].shape,
                    xa,
                    &ra,
                    &self.shapes[j],
                    xb,
                    &rb,
                ),
                (false, false) => narrow_collide(
                    &self.statics[i - n].shape,
                    xa,
                    &ra,
                    &self.statics[j - n].shape,
                    xb,
                    &rb,
                ),
            };
            for (k, c) in cs.iter_mut().enumerate() {
                if c.feature.0 == 0 {
                    c.feature = FeatureId(k as u32);
                }
                self.scratch.contacts.push(ContactPoint {
                    a: i,
                    b: j,
                    point: c.point,
                    normal: c.normal,
                    depth: c.depth,
                    feature: c.feature,
                });
            }
        }
        if let Some(t0) = t_narrow0 {
            self.last_phase_times.narrow = t0.elapsed().as_secs_f64();
        }
        if let Some(t0) = t_detect0 {
            let _ = t0; // broad+narrow accounted separately; total set in step
        }

        (self.scratch.pairs.clone(), self.scratch.contacts.clone())
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
        s.extend(self.statics.iter().map(|c| c.shape.clone()));
        s
    }

    /// Broad + narrow phase. Pairs are sorted by `(min id, max id)`.
    ///
    /// Allocating path for `&self` callers (tests / tooling). The step hot path
    /// uses [`Self::detect_pairs_and_contacts`] with reused scratch buffers.
    pub fn detect(&self, st: Option<&State>) -> Vec<ContactPoint> {
        let st = st.unwrap_or(&self.world.state);
        self.detect_pairs_and_contacts_alloc(st).1
    }

    /// Allocating detect (no scratch mutation) — same results as the scratch path.
    fn detect_pairs_and_contacts_alloc(
        &self,
        st: &State,
    ) -> (Vec<(usize, usize)>, Vec<ContactPoint>) {
        let (x, r) = self.poses(st);
        let n = self.world.n;
        let n_total = n + self.statics.len();
        let mut aabbs = Vec::with_capacity(n_total);
        for i in 0..n {
            aabbs.push(self.shapes[i].aabb(x[i], &r[i]));
        }
        for (si, s) in self.statics.iter().enumerate() {
            aabbs.push(s.shape.aabb(x[n + si], &r[n + si]));
        }
        let mut pairs = match self.broad_phase {
            BroadPhaseKind::Sap => SweepAndPrune.pairs(&aabbs, &self.is_static),
            BroadPhaseKind::Brute => BruteForce.pairs(&aabbs, &self.is_static),
            BroadPhaseKind::DynamicTree => {
                // Stateless rebuild path (matches incremental pair *set*).
                DynamicAabbTree::default().pairs(&aabbs, &self.is_static)
            }
        };
        if !pairs.windows(2).all(|w| w[0] <= w[1]) {
            pairs.sort_unstable();
        }
        pairs.dedup();
        let mut out = Vec::new();
        for &(i, j) in &pairs {
            let (sa, xa, ra) = if i < n {
                (&self.shapes[i], x[i], &r[i])
            } else {
                (&self.statics[i - n].shape, x[i], &r[i])
            };
            let (sb, xb, rb) = if j < n {
                (&self.shapes[j], x[j], &r[j])
            } else {
                (&self.statics[j - n].shape, x[j], &r[j])
            };
            let mut cs = narrow_collide(sa, xa, ra, sb, xb, rb);
            for (k, c) in cs.iter_mut().enumerate() {
                if c.feature.0 == 0 {
                    c.feature = FeatureId(k as u32);
                }
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
                &crate::narrowphase::ColliderPose::new(shapes[i].clone(), x[i], r[i]),
                &crate::narrowphase::ColliderPose::new(shapes[j].clone(), x[j], r[j]),
            );
            for (k, c) in cs.iter_mut().enumerate() {
                if c.feature.0 == 0 {
                    c.feature = FeatureId(k as u32);
                }
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

    /// Default detect path (selected broad phase + dedicated pairs) — used by correction.
    fn detect_default(&mut self, st: &State) -> Vec<ContactPoint> {
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
    ///
    /// Uses SoA contact rows and reused scratch buffers. Floating-point ops and
    /// contact iteration order match the historical AoS loop (bit-identical).
    pub fn solve_velocities(&mut self, st: &mut State, contacts: &[ContactPoint], dt: f64) {
        let n = self.world.n;
        self.prepare_solver_rows(st, contacts, dt);
        self.apply_warm_start_and_solve(n);
        st.v.copy_from_slice(&self.scratch.v[..n]);
        st.l.copy_from_slice(&self.scratch.l[..n]);
    }

    /// Warm-start + PGS iterations over [`Self::rows_soa`] into scratch `v`/`l`.
    fn apply_warm_start_and_solve(&mut self, n: usize) {
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
            let nrows = self.rows_soa.len();
            for i in 0..nrows {
                let key = (
                    self.rows_soa.key_a[i],
                    self.rows_soa.key_b[i],
                    self.rows_soa.key_f[i],
                );
                if let Some(old) = old_cache.get(&key) {
                    let point = self.rows_soa.point[i];
                    if (old.point - point).norm() > 0.01 {
                        continue;
                    }
                    let t1 = self.rows_soa.t1.get(i);
                    let t2 = self.rows_soa.t2.get(i);
                    let normal = self.rows_soa.normal.get(i);
                    self.rows_soa.lam[i] = old.normal_impulse;
                    self.rows_soa.jt0[i] = old.tangent_impulse.dot(t1);
                    self.rows_soa.jt1[i] = old.tangent_impulse.dot(t2);
                    let p = normal * self.rows_soa.lam[i]
                        + t1 * self.rows_soa.jt0[i]
                        + t2 * self.rows_soa.jt1[i];
                    apply(
                        self.rows_soa.a[i],
                        self.rows_soa.b[i],
                        self.rows_soa.ra.get(i),
                        self.rows_soa.rb.get(i),
                        p,
                        &mut self.scratch.v,
                        &mut self.scratch.l,
                        &self.scratch.im,
                    );
                }
            }
        }

        let iterations = self.iterations;
        let nrows = self.rows_soa.len();
        for _ in 0..iterations {
            for i in 0..nrows {
                let a = self.rows_soa.a[i];
                let b = self.rows_soa.b[i];
                let ra = self.rows_soa.ra.get(i);
                let rb = self.rows_soa.rb.get(i);
                let t1 = self.rows_soa.t1.get(i);
                let t2 = self.rows_soa.t2.get(i);
                let normal = self.rows_soa.normal.get(i);
                let va =
                    self.scratch.v[a] + mat_vec(&self.scratch.iinv[a], self.scratch.l[a]).cross(ra);
                let vb =
                    self.scratch.v[b] + mat_vec(&self.scratch.iinv[b], self.scratch.l[b]).cross(rb);
                let mut vrel = vb - va;
                let old0 = self.rows_soa.jt0[i];
                let old1 = self.rows_soa.jt1[i];
                let mut jt0 = old0 - self.rows_soa.k1[i] * vrel.dot(t1);
                let mut jt1 = old1 - self.rows_soa.k2[i] * vrel.dot(t2);
                let lim = self.rows_soa.mu[i] * self.rows_soa.lam[i];
                let nj = sqrt(jt0 * jt0 + jt1 * jt1);
                if nj > lim {
                    jt0 *= lim / nj;
                    jt1 *= lim / nj;
                }
                let d0 = jt0 - old0;
                let d1 = jt1 - old1;
                self.rows_soa.jt0[i] = jt0;
                self.rows_soa.jt1[i] = jt1;
                apply(
                    a,
                    b,
                    ra,
                    rb,
                    t1 * d0 + t2 * d1,
                    &mut self.scratch.v,
                    &mut self.scratch.l,
                    &self.scratch.im,
                );
                let va =
                    self.scratch.v[a] + mat_vec(&self.scratch.iinv[a], self.scratch.l[a]).cross(ra);
                let vb =
                    self.scratch.v[b] + mat_vec(&self.scratch.iinv[b], self.scratch.l[b]).cross(rb);
                vrel = vb - va;
                let lam = (self.rows_soa.lam[i]
                    + self.rows_soa.kn[i] * (self.rows_soa.bias[i] - vrel.dot(normal)))
                .max(0.0);
                apply(
                    a,
                    b,
                    ra,
                    rb,
                    normal * (lam - self.rows_soa.lam[i]),
                    &mut self.scratch.v,
                    &mut self.scratch.l,
                    &self.scratch.im,
                );
                self.rows_soa.lam[i] = lam;
            }
        }

        for i in 0..nrows {
            let key = (
                self.rows_soa.key_a[i],
                self.rows_soa.key_b[i],
                self.rows_soa.key_f[i],
            );
            let t1 = self.rows_soa.t1.get(i);
            let t2 = self.rows_soa.t2.get(i);
            self.cache.insert(
                key,
                WarmEntry {
                    normal_impulse: self.rows_soa.lam[i],
                    tangent_impulse: t1 * self.rows_soa.jt0[i] + t2 * self.rows_soa.jt1[i],
                    point: self.rows_soa.point[i],
                },
            );
        }
    }

    /// Island id for a contact: dynamic endpoint's island (min if both dynamic).
    #[inline]
    fn contact_island_id(&self, n: usize, a: usize, b: usize) -> i32 {
        let ia = if a < n { self.sleep.island[a] } else { -1 };
        let ib = if b < n { self.sleep.island[b] } else { -1 };
        match (ia >= 0, ib >= 0) {
            (true, true) => ia.min(ib),
            (true, false) => ia,
            (false, true) => ib,
            (false, false) => 0,
        }
    }

    /// Build SoA rows for `contacts` into `rows_soa` and fill scratch `im`/`iinv`/`x`/`v`/`l`.
    fn prepare_solver_rows(&mut self, st: &State, contacts: &[ContactPoint], dt: f64) {
        let n = self.world.n;
        let dt_for_speculative = if dt > 0.0 { dt } else { 1.0 };
        let n_total = n + self.statics.len();

        self.scratch.clear_solver();
        let sc = &mut self.scratch;
        sc.im.resize(n_total, 0.0);
        for i in 0..n {
            sc.im[i] = 1.0 / self.world.mass[i];
        }
        for i in n..n_total {
            sc.im[i] = 0.0;
        }
        sc.iinv.resize(n_total, [[0.0; 3]; 3]);
        for i in 0..n {
            sc.iinv[i] = self.world.inv_inertia_world(st.q[i], i);
        }
        sc.x.clear();
        sc.x.reserve(n_total);
        sc.x.extend_from_slice(&st.x);
        sc.x.extend_from_slice(&self.static_x);
        sc.v.clear();
        sc.v.resize(n_total, Vec3::zero());
        sc.v[..n].copy_from_slice(&st.v);
        sc.l.clear();
        sc.l.resize(n_total, Vec3::zero());
        sc.l[..n].copy_from_slice(&st.l);

        self.rows_soa.clear();
        self.rows_soa.reserve(contacts.len());
        let mut prev_pair = (usize::MAX, usize::MAX);
        let mut feat_k = 0u32;
        for c in contacts {
            if (c.a, c.b) != prev_pair {
                prev_pair = (c.a, c.b);
                feat_k = 0;
            }
            let feat = feat_k;
            feat_k += 1;
            let ra = c.point - self.scratch.x[c.a];
            let rb = c.point - self.scratch.x[c.b];
            let ma = self.mat(c.a);
            let mb = self.mat(c.b);
            let mu0 = sqrt(ma.friction * mb.friction);
            let e = ma.restitution.max(mb.restitution);
            let (t1, t2) = friction_basis(c.normal);
            let ima = self.scratch.im[c.a];
            let imb = self.scratch.im[c.b];
            let kn = effective_mass(
                c.normal,
                ra,
                rb,
                ima,
                imb,
                &self.scratch.iinv[c.a],
                &self.scratch.iinv[c.b],
            );
            let k1 = effective_mass(
                t1,
                ra,
                rb,
                ima,
                imb,
                &self.scratch.iinv[c.a],
                &self.scratch.iinv[c.b],
            );
            let k2 = effective_mass(
                t2,
                ra,
                rb,
                ima,
                imb,
                &self.scratch.iinv[c.a],
                &self.scratch.iinv[c.b],
            );
            let va = self.scratch.v[c.a]
                + mat_vec(&self.scratch.iinv[c.a], self.scratch.l[c.a]).cross(ra);
            let vb = self.scratch.v[c.b]
                + mat_vec(&self.scratch.iinv[c.b], self.scratch.l[c.b]).cross(rb);
            let vrel = vb - va;
            let vn0 = vrel.dot(c.normal);
            let bias = if c.depth < 0.0 {
                c.depth / dt_for_speculative
            } else if vn0 < -self.restitution_threshold {
                -e * vn0
            } else {
                0.0
            };
            let mu = if c.depth < 0.0 { 0.0 } else { mu0 };
            self.rows_soa.push(
                c.a,
                c.b,
                c.point,
                c.normal,
                ra,
                rb,
                mu,
                kn,
                k1,
                k2,
                t1,
                t2,
                bias,
                (c.a, c.b, feat),
            );
        }
    }

    /// PGS solve with contacts grouped by ascending island id.
    ///
    /// Single-thread and `parallel` feature paths share this ordering so results
    /// are bit-identical across rayon thread counts. Not used by locked oracle
    /// hashes (those keep global PGS via `island_ordered_solve = false`).
    pub fn solve_velocities_island_ordered(
        &mut self,
        st: &mut State,
        contacts: &[ContactPoint],
        dt: f64,
    ) {
        let n = self.world.n;
        self.prepare_solver_rows(st, contacts, dt);

        // Partition row indices by island id (ascending).
        let n_islands = self.sleep.island_count.max(1);
        let mut rows_by_island: Vec<Vec<usize>> = vec![Vec::new(); n_islands];
        for i in 0..self.rows_soa.len() {
            let id = self.contact_island_id(n, self.rows_soa.a[i], self.rows_soa.b[i]);
            let id = if id < 0 { 0 } else { id as usize };
            let id = id.min(n_islands - 1);
            rows_by_island[id].push(i);
        }

        // Warm-start all rows into shared v/l (sequential, deterministic).
        self.warm_start_rows(n);

        #[cfg(feature = "parallel")]
        {
            self.solve_islands_parallel(n, &rows_by_island);
        }
        #[cfg(not(feature = "parallel"))]
        {
            for rows in &rows_by_island {
                self.solve_row_indices(n, rows);
            }
        }

        self.write_warm_cache_from_rows();
        st.v.copy_from_slice(&self.scratch.v[..n]);
        st.l.copy_from_slice(&self.scratch.l[..n]);
    }

    fn warm_start_rows(&mut self, n: usize) {
        let old_cache = std::mem::take(&mut self.cache);
        if !self.warm_start {
            return;
        }
        let nrows = self.rows_soa.len();
        for i in 0..nrows {
            let key = (
                self.rows_soa.key_a[i],
                self.rows_soa.key_b[i],
                self.rows_soa.key_f[i],
            );
            if let Some(old) = old_cache.get(&key) {
                let point = self.rows_soa.point[i];
                if (old.point - point).norm() > 0.01 {
                    continue;
                }
                let t1 = self.rows_soa.t1.get(i);
                let t2 = self.rows_soa.t2.get(i);
                let normal = self.rows_soa.normal.get(i);
                self.rows_soa.lam[i] = old.normal_impulse;
                self.rows_soa.jt0[i] = old.tangent_impulse.dot(t1);
                self.rows_soa.jt1[i] = old.tangent_impulse.dot(t2);
                let p = normal * self.rows_soa.lam[i]
                    + t1 * self.rows_soa.jt0[i]
                    + t2 * self.rows_soa.jt1[i];
                let a = self.rows_soa.a[i];
                let b = self.rows_soa.b[i];
                let ra = self.rows_soa.ra.get(i);
                let rb = self.rows_soa.rb.get(i);
                self.scratch.v[a] -= p * self.scratch.im[a];
                if a < n {
                    self.scratch.l[a] -= ra.cross(p);
                }
                self.scratch.v[b] += p * self.scratch.im[b];
                if b < n {
                    self.scratch.l[b] += rb.cross(p);
                }
            }
        }
    }

    fn write_warm_cache_from_rows(&mut self) {
        self.cache.clear();
        for i in 0..self.rows_soa.len() {
            let key = (
                self.rows_soa.key_a[i],
                self.rows_soa.key_b[i],
                self.rows_soa.key_f[i],
            );
            let t1 = self.rows_soa.t1.get(i);
            let t2 = self.rows_soa.t2.get(i);
            self.cache.insert(
                key,
                WarmEntry {
                    normal_impulse: self.rows_soa.lam[i],
                    tangent_impulse: t1 * self.rows_soa.jt0[i] + t2 * self.rows_soa.jt1[i],
                    point: self.rows_soa.point[i],
                },
            );
        }
    }

    /// Run PGS iterations over a subset of SoA row indices (single island).
    #[cfg_attr(feature = "parallel", allow(dead_code))]
    fn solve_row_indices(&mut self, n: usize, rows: &[usize]) {
        let iterations = self.iterations;
        for _ in 0..iterations {
            for &i in rows {
                let a = self.rows_soa.a[i];
                let b = self.rows_soa.b[i];
                let ra = self.rows_soa.ra.get(i);
                let rb = self.rows_soa.rb.get(i);
                let t1 = self.rows_soa.t1.get(i);
                let t2 = self.rows_soa.t2.get(i);
                let normal = self.rows_soa.normal.get(i);
                let va =
                    self.scratch.v[a] + mat_vec(&self.scratch.iinv[a], self.scratch.l[a]).cross(ra);
                let vb =
                    self.scratch.v[b] + mat_vec(&self.scratch.iinv[b], self.scratch.l[b]).cross(rb);
                let mut vrel = vb - va;
                let old0 = self.rows_soa.jt0[i];
                let old1 = self.rows_soa.jt1[i];
                let mut jt0 = old0 - self.rows_soa.k1[i] * vrel.dot(t1);
                let mut jt1 = old1 - self.rows_soa.k2[i] * vrel.dot(t2);
                let lim = self.rows_soa.mu[i] * self.rows_soa.lam[i];
                let nj = sqrt(jt0 * jt0 + jt1 * jt1);
                if nj > lim {
                    jt0 *= lim / nj;
                    jt1 *= lim / nj;
                }
                let d0 = jt0 - old0;
                let d1 = jt1 - old1;
                self.rows_soa.jt0[i] = jt0;
                self.rows_soa.jt1[i] = jt1;
                let p_t = t1 * d0 + t2 * d1;
                self.scratch.v[a] -= p_t * self.scratch.im[a];
                if a < n {
                    self.scratch.l[a] -= ra.cross(p_t);
                }
                self.scratch.v[b] += p_t * self.scratch.im[b];
                if b < n {
                    self.scratch.l[b] += rb.cross(p_t);
                }
                let va =
                    self.scratch.v[a] + mat_vec(&self.scratch.iinv[a], self.scratch.l[a]).cross(ra);
                let vb =
                    self.scratch.v[b] + mat_vec(&self.scratch.iinv[b], self.scratch.l[b]).cross(rb);
                vrel = vb - va;
                let lam = (self.rows_soa.lam[i]
                    + self.rows_soa.kn[i] * (self.rows_soa.bias[i] - vrel.dot(normal)))
                .max(0.0);
                let p_n = normal * (lam - self.rows_soa.lam[i]);
                self.scratch.v[a] -= p_n * self.scratch.im[a];
                if a < n {
                    self.scratch.l[a] -= ra.cross(p_n);
                }
                self.scratch.v[b] += p_n * self.scratch.im[b];
                if b < n {
                    self.scratch.l[b] += rb.cross(p_n);
                }
                self.rows_soa.lam[i] = lam;
            }
        }
    }

    #[cfg(feature = "parallel")]
    fn solve_islands_parallel(&mut self, n: usize, rows_by_island: &[Vec<usize>]) {
        use rayon::prelude::*;

        struct IslandUpdate {
            bodies: Vec<usize>,
            v: Vec<Vec3>,
            l: Vec<Vec3>,
            lam: Vec<f64>,
            jt0: Vec<f64>,
            jt1: Vec<f64>,
        }

        // Islands write disjoint dynamic bodies; solve each into a local (v,l) clone
        // of the post-warm-start state, then merge in ascending island order.
        let base_v = self.scratch.v.clone();
        let base_l = self.scratch.l.clone();
        let iterations = self.iterations;
        let im = self.scratch.im.clone();
        let iinv = self.scratch.iinv.clone();
        let rows = self.rows_soa.clone();

        let updates: Vec<IslandUpdate> = rows_by_island
            .par_iter()
            .map(|island_rows| {
                if island_rows.is_empty() {
                    return IslandUpdate {
                        bodies: Vec::new(),
                        v: Vec::new(),
                        l: Vec::new(),
                        lam: Vec::new(),
                        jt0: Vec::new(),
                        jt1: Vec::new(),
                    };
                }
                let mut v = base_v.clone();
                let mut l = base_l.clone();
                let mut lam = island_rows.iter().map(|&i| rows.lam[i]).collect::<Vec<_>>();
                let mut jt0 = island_rows.iter().map(|&i| rows.jt0[i]).collect::<Vec<_>>();
                let mut jt1 = island_rows.iter().map(|&i| rows.jt1[i]).collect::<Vec<_>>();
                for _ in 0..iterations {
                    for (local, &i) in island_rows.iter().enumerate() {
                        let a = rows.a[i];
                        let b = rows.b[i];
                        let ra = rows.ra.get(i);
                        let rb = rows.rb.get(i);
                        let t1 = rows.t1.get(i);
                        let t2 = rows.t2.get(i);
                        let normal = rows.normal.get(i);
                        let va = v[a] + mat_vec(&iinv[a], l[a]).cross(ra);
                        let vb = v[b] + mat_vec(&iinv[b], l[b]).cross(rb);
                        let mut vrel = vb - va;
                        let old0 = jt0[local];
                        let old1 = jt1[local];
                        let mut j0 = old0 - rows.k1[i] * vrel.dot(t1);
                        let mut j1 = old1 - rows.k2[i] * vrel.dot(t2);
                        let lim = rows.mu[i] * lam[local];
                        let nj = sqrt(j0 * j0 + j1 * j1);
                        if nj > lim {
                            j0 *= lim / nj;
                            j1 *= lim / nj;
                        }
                        let d0 = j0 - old0;
                        let d1 = j1 - old1;
                        jt0[local] = j0;
                        jt1[local] = j1;
                        let p_t = t1 * d0 + t2 * d1;
                        v[a] -= p_t * im[a];
                        if a < n {
                            l[a] -= ra.cross(p_t);
                        }
                        v[b] += p_t * im[b];
                        if b < n {
                            l[b] += rb.cross(p_t);
                        }
                        let va = v[a] + mat_vec(&iinv[a], l[a]).cross(ra);
                        let vb = v[b] + mat_vec(&iinv[b], l[b]).cross(rb);
                        vrel = vb - va;
                        let new_lam =
                            (lam[local] + rows.kn[i] * (rows.bias[i] - vrel.dot(normal))).max(0.0);
                        let p_n = normal * (new_lam - lam[local]);
                        v[a] -= p_n * im[a];
                        if a < n {
                            l[a] -= ra.cross(p_n);
                        }
                        v[b] += p_n * im[b];
                        if b < n {
                            l[b] += rb.cross(p_n);
                        }
                        lam[local] = new_lam;
                    }
                }
                // Collect dynamic bodies touched by this island.
                let mut bodies = Vec::new();
                for &i in island_rows {
                    if rows.a[i] < n {
                        bodies.push(rows.a[i]);
                    }
                    if rows.b[i] < n {
                        bodies.push(rows.b[i]);
                    }
                }
                bodies.sort_unstable();
                bodies.dedup();
                let vv: Vec<_> = bodies.iter().map(|&b| v[b]).collect();
                let ll: Vec<_> = bodies.iter().map(|&b| l[b]).collect();
                IslandUpdate {
                    bodies,
                    v: vv,
                    l: ll,
                    lam,
                    jt0,
                    jt1,
                }
            })
            .collect();

        // Merge in ascending island order (par_iter collect preserves order).
        for (island_idx, upd) in updates.into_iter().enumerate() {
            for (k, &b) in upd.bodies.iter().enumerate() {
                self.scratch.v[b] = upd.v[k];
                self.scratch.l[b] = upd.l[k];
            }
            let island_rows = &rows_by_island[island_idx];
            for (local, &i) in island_rows.iter().enumerate() {
                self.rows_soa.lam[i] = upd.lam[local];
                self.rows_soa.jt0[i] = upd.jt0[local];
                self.rows_soa.jt1[i] = upd.jt1[local];
            }
        }
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
        let t_step0 = if self.profile_phases {
            Some(Instant::now())
        } else {
            None
        };
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
                .map(|s| (s.shape.clone(), s.position, s.orientation))
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
                .map(|s| (s.shape.clone(), s.position, s.orientation))
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
                let t_sol0 = if self.profile_phases {
                    Some(Instant::now())
                } else {
                    None
                };
                if self.island_ordered_solve {
                    self.solve_velocities_island_ordered(&mut st, &active, step_dt);
                } else {
                    self.solve_velocities(&mut st, &active, step_dt);
                }
                self.solve_joints_pgs(&mut st, step_dt);
                if let Some(t0) = t_sol0 {
                    self.last_phase_times.solver = t0.elapsed().as_secs_f64();
                }
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
        if let Some(t0) = t_step0 {
            self.last_phase_times.total = t0.elapsed().as_secs_f64();
        }
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
            vec![
                Shape::Sphere(crate::shapes::Sphere::new(0.5)),
                Shape::Sphere(crate::shapes::Sphere::new(0.5)),
            ],
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
                vec![
                    Shape::Sphere(crate::shapes::Sphere::new(0.5)),
                    Shape::Sphere(crate::shapes::Sphere::new(0.5)),
                ],
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
