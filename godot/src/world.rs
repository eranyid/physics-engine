//! [`PePhysicsWorld`] — owns a `ContactWorld`, fixed-step accumulator, queries, contacts.

use std::collections::BTreeSet;
use std::sync::Arc;

use godot::classes::{INode3D, Node3D};
use godot::prelude::*;
use pe_collision::joints::Joint;
use pe_collision::queries::{overlap_aabb, raycast, sphere_cast};
use pe_collision::shapes::{Aabb, BoxShape, Material, Shape, Sphere};
use pe_collision::{ContactWorld, StaticCollider};
use pe_core::diagnostics::state_hash64;
use pe_core::forces::UniformGravity;
use pe_core::{ExternalLoads, RigidBody, World};
use pe_math::{Quat as PeQuat, Vec3 as PeVec3, STANDARD_GRAVITY};

use crate::body::PeRigidBody3D;
use crate::convert::{to_godot_quat, to_godot_vec3, to_pe_vec3};

/// Root physics world node. Steps the engine each physics frame with a fixed
/// accumulator; optional visual interpolation does not affect the f64 state hash.
#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct PePhysicsWorld {
    base: Base<Node3D>,
    world: Option<ContactWorld>,
    loads: Arc<ExternalLoads>,
    /// Fixed step [s]; if ≤ 0, uses Godot's `delta` directly (non-deterministic vs render rate).
    #[export]
    pub fixed_dt: f64,
    /// Max physics steps per render frame (spiral-of-death guard).
    #[export]
    pub max_substeps: i64,
    /// When true, body visuals lerp/slerp between previous and current poses.
    #[export]
    pub interpolate: bool,
    /// Allow island sleeping (core wake-on-gravity assumes Z-up; default off for Y-up Godot).
    #[export]
    pub allow_sleep: bool,
    accum: f64,
    prev_x: Vec<PeVec3>,
    prev_q: Vec<PeQuat>,
    /// Body-pair contacts from the previous step `(min,max)`.
    prev_contact_pairs: BTreeSet<(usize, usize)>,
    /// Dynamic body index → Godot node (for body_entered / body_exited).
    body_nodes: Vec<Option<Gd<PeRigidBody3D>>>,
}

fn y_up_world(loads: Arc<ExternalLoads>) -> ContactWorld {
    let gens: Vec<Box<dyn pe_core::ForceGenerator>> = vec![
        Box::new(UniformGravity::new(PeVec3::new(
            0.0,
            -STANDARD_GRAVITY,
            0.0,
        ))),
        Box::new(Arc::clone(&loads)),
    ];
    let mut cw = ContactWorld::new(World::new(vec![], gens), vec![], vec![], Some(vec![]));
    cw.set_allow_sleep(false);
    cw
}

#[godot_api]
impl INode3D for PePhysicsWorld {
    fn init(base: Base<Node3D>) -> Self {
        let loads = Arc::new(ExternalLoads::new());
        Self {
            base,
            world: Some(y_up_world(Arc::clone(&loads))),
            loads,
            fixed_dt: 1.0 / 60.0,
            max_substeps: 8,
            interpolate: true,
            allow_sleep: false,
            accum: 0.0,
            prev_x: Vec::new(),
            prev_q: Vec::new(),
            prev_contact_pairs: BTreeSet::new(),
            body_nodes: Vec::new(),
        }
    }

    fn ready(&mut self) {
        if let Some(cw) = self.world.as_mut() {
            cw.set_allow_sleep(self.allow_sleep);
        }
    }

    fn physics_process(&mut self, delta: f64) {
        if self.fixed_dt > 0.0 {
            self.accum += delta;
            let step = self.fixed_dt;
            let max_steps = self.max_substeps.max(1) as usize;
            let mut nsteps = 0usize;
            while self.accum >= step && nsteps < max_steps {
                self.snapshot_poses();
                self.step_internal(step);
                self.accum -= step;
                nsteps += 1;
            }
            // Drop leftover time beyond the cap so we do not accumulate forever.
            if nsteps == max_steps && self.accum >= step {
                self.accum = self.accum.rem_euclid(step);
            }
        } else if delta > 0.0 && delta.is_finite() {
            self.snapshot_poses();
            self.step_internal(delta);
            self.accum = 0.0;
        }
        // Clear per-step external forces after the frame's physics work.
        self.loads.clear();
    }
}

#[godot_api]
impl PePhysicsWorld {
    #[signal]
    fn contact_started(body_a: i64, body_b: i64);

    #[signal]
    fn contact_ended(body_a: i64, body_b: i64);

    /// Add a dynamic sphere body; returns body index.
    #[func]
    #[allow(clippy::too_many_arguments)]
    fn add_sphere_body(
        &mut self,
        mass: f64,
        radius: f64,
        px: f64,
        py: f64,
        pz: f64,
        restitution: f64,
        friction: f64,
    ) -> i64 {
        if !(mass > 0.0 && radius >= 0.0) {
            return -1;
        }
        let shape = Shape::Sphere(Sphere::new(radius));
        let body = RigidBody::new(mass)
            .with_inertia(shape.inertia(mass))
            .with_position(PeVec3::new(px, py, pz));
        self.register_dynamic(body, shape, Material::new(restitution, friction))
    }

    /// Add a dynamic box body (half-extents); returns body index.
    #[func]
    #[allow(clippy::too_many_arguments)]
    fn add_box_body(
        &mut self,
        mass: f64,
        hx: f64,
        hy: f64,
        hz: f64,
        px: f64,
        py: f64,
        pz: f64,
        restitution: f64,
        friction: f64,
    ) -> i64 {
        if !(mass > 0.0 && hx >= 0.0 && hy >= 0.0 && hz >= 0.0) {
            return -1;
        }
        let shape = Shape::Box(BoxShape::new(hx, hy, hz));
        let body = RigidBody::new(mass)
            .with_inertia(shape.inertia(mass))
            .with_position(PeVec3::new(px, py, pz));
        self.register_dynamic(body, shape, Material::new(restitution, friction))
    }

    /// Add a dynamic capsule (radius + half-height along body +Z).
    #[func]
    #[allow(clippy::too_many_arguments)]
    fn add_capsule_body(
        &mut self,
        mass: f64,
        radius: f64,
        half_height: f64,
        px: f64,
        py: f64,
        pz: f64,
        restitution: f64,
        friction: f64,
    ) -> i64 {
        if !(mass > 0.0 && radius >= 0.0 && half_height >= 0.0) {
            return -1;
        }
        let shape = Shape::Capsule(pe_collision::shapes::Capsule::new(radius, half_height));
        let body = RigidBody::new(mass)
            .with_inertia(shape.inertia(mass))
            .with_position(PeVec3::new(px, py, pz));
        self.register_dynamic(body, shape, Material::new(restitution, friction))
    }

    /// Add a dynamic convex hull from a PackedVector3Array of body-frame points.
    #[func]
    #[allow(clippy::too_many_arguments)]
    fn add_convex_body(
        &mut self,
        mass: f64,
        points: PackedVector3Array,
        px: f64,
        py: f64,
        pz: f64,
        restitution: f64,
        friction: f64,
    ) -> i64 {
        if mass <= 0.0 || !mass.is_finite() || points.len() < 4 {
            return -1;
        }
        let mut pts = Vec::with_capacity(points.len());
        for i in 0..points.len() {
            pts.push(to_pe_vec3(points.get(i).unwrap_or(Vector3::ZERO)));
        }
        let Ok(hull) = pe_collision::hull::ConvexHull::from_points(&pts) else {
            return -1;
        };
        let shape = Shape::Convex(hull);
        let body = RigidBody::new(mass)
            .with_inertia(shape.inertia(mass))
            .with_position(PeVec3::new(px, py, pz));
        self.register_dynamic(body, shape, Material::new(restitution, friction))
    }

    /// Add a static ground plane with normal +Y at `y = offset` (Godot Y-up).
    #[func]
    fn add_ground_plane(&mut self, offset: f64, restitution: f64, friction: f64) -> i64 {
        let col = StaticCollider::new(Shape::Plane(pe_collision::shapes::Plane::new(
            PeVec3::unit_y(),
            offset,
        )))
        .with_material(Material::new(restitution, friction));
        self.register_static_collider(col)
    }

    /// Manually step once (headless CI). Runs contact begin/end signals.
    #[func]
    fn step_once(&mut self, dt: f64) {
        if dt > 0.0 && dt.is_finite() {
            self.snapshot_poses();
            self.step_internal(dt);
            self.loads.clear();
        }
    }

    /// FNV-1a 64-bit hash of packed dynamic state as lowercase hex (16 chars).
    #[func]
    fn state_hash_hex(&self) -> GString {
        let Some(cw) = self.world.as_ref() else {
            return GString::from("0");
        };
        GString::from(format!("{:016x}", state_hash64(&cw.world.state)).as_str())
    }

    /// Body count (dynamics only).
    #[func]
    fn body_count(&self) -> i64 {
        self.world.as_ref().map(|w| w.world.n as i64).unwrap_or(0)
    }

    /// Read body position into a Godot `Vector3` (engine pose, not interpolated).
    #[func]
    fn get_body_position(&self, index: i64) -> Vector3 {
        let Some(cw) = self.world.as_ref() else {
            return Vector3::ZERO;
        };
        if index < 0 || index as usize >= cw.world.n {
            return Vector3::ZERO;
        }
        to_godot_vec3(cw.world.state.x[index as usize])
    }

    /// Read body orientation as Godot `Quaternion`.
    #[func]
    fn get_body_quaternion(&self, index: i64) -> Quaternion {
        let Some(cw) = self.world.as_ref() else {
            return Quaternion::IDENTITY;
        };
        if index < 0 || index as usize >= cw.world.n {
            return Quaternion::IDENTITY;
        }
        to_godot_quat(cw.world.state.q[index as usize])
    }

    /// Interpolation alpha in `[0,1]` for the current render frame.
    #[func]
    fn interpolation_alpha(&self) -> f64 {
        if self.fixed_dt > 0.0 {
            (self.accum / self.fixed_dt).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Closest raycast. Returns a Dictionary: hit, position, normal, body, distance.
    #[func]
    fn raycast(&self, origin: Vector3, direction: Vector3, max_distance: f64) -> VarDictionary {
        let mut out = VarDictionary::new();
        out.set("hit", false);
        let Some(cw) = self.world.as_ref() else {
            return out;
        };
        let Some(hit) = raycast(cw, to_pe_vec3(origin), to_pe_vec3(direction), max_distance) else {
            return out;
        };
        out.set("hit", true);
        out.set("position", to_godot_vec3(hit.point));
        out.set("normal", to_godot_vec3(hit.normal));
        out.set("body", hit.body as i64);
        out.set("distance", hit.t);
        out
    }

    /// Sphere cast. Returns Dictionary: hit, position, normal, body, toi.
    #[func]
    fn sphere_cast(
        &self,
        origin: Vector3,
        direction: Vector3,
        radius: f64,
        max_distance: f64,
    ) -> VarDictionary {
        let mut out = VarDictionary::new();
        out.set("hit", false);
        let Some(cw) = self.world.as_ref() else {
            return out;
        };
        let dir = to_pe_vec3(direction);
        let dir = if dir.norm() > 0.0 {
            dir.normalize() * max_distance
        } else {
            return out;
        };
        let Some(hit) = sphere_cast(cw, to_pe_vec3(origin), dir, radius, max_distance) else {
            return out;
        };
        out.set("hit", true);
        out.set("position", to_godot_vec3(hit.point));
        out.set("normal", to_godot_vec3(hit.normal));
        out.set("body", hit.body as i64);
        out.set("toi", hit.toi);
        out
    }

    /// Overlap AABB query → PackedInt32Array of body indices (`n+si` for statics).
    #[func]
    fn overlap_aabb(&self, lo: Vector3, hi: Vector3) -> PackedInt32Array {
        let mut arr = PackedInt32Array::new();
        let Some(cw) = self.world.as_ref() else {
            return arr;
        };
        let mut ids = Vec::new();
        overlap_aabb(
            cw,
            Aabb {
                lo: to_pe_vec3(lo),
                hi: to_pe_vec3(hi),
            },
            &mut ids,
        );
        for id in ids {
            arr.push(id as i32);
        }
        arr
    }

    /// Enable/disable sleeping at runtime.
    #[func]
    fn set_sleep_enabled(&mut self, enabled: bool) {
        self.allow_sleep = enabled;
        if let Some(cw) = self.world.as_mut() {
            cw.set_allow_sleep(enabled);
        }
    }

    /// Add a distance joint. `body_b < 0` anchors to world. Returns joint index.
    #[func]
    #[allow(clippy::too_many_arguments)]
    fn add_distance_joint(
        &mut self,
        body_a: i64,
        body_b: i64,
        ax: f64,
        ay: f64,
        az: f64,
        bx: f64,
        by: f64,
        bz: f64,
        length: f64,
        hertz: f64,
        zeta: f64,
    ) -> i64 {
        if body_a < 0 || !(length >= 0.0 && length.is_finite()) {
            return -1;
        }
        let b = if body_b < 0 {
            pe_collision::joints::WORLD_BODY
        } else {
            body_b as usize
        };
        let mut joint = Joint::distance(
            body_a as usize,
            b,
            PeVec3::new(ax, ay, az),
            PeVec3::new(bx, by, bz),
            length,
        );
        joint.soft = pe_collision::joints::JointSoftParams { hertz, zeta };
        self.add_joint(joint)
    }
}

impl PePhysicsWorld {
    pub(crate) fn register_dynamic(
        &mut self,
        body: RigidBody,
        shape: Shape,
        material: Material,
    ) -> i64 {
        let Some(cw) = self.world.as_mut() else {
            return -1;
        };
        let idx = cw.add_dynamic_body(body, shape, material);
        let n = cw.world.n;
        let _ = cw;
        self.ensure_pose_buffers();
        if self.body_nodes.len() < n {
            self.body_nodes.resize(n, None);
        }
        idx as i64
    }

    pub(crate) fn register_static_collider(&mut self, col: StaticCollider) -> i64 {
        let Some(cw) = self.world.as_mut() else {
            return -1;
        };
        cw.add_static(col) as i64
    }

    pub(crate) fn track_body_node(&mut self, index: usize, node: Gd<PeRigidBody3D>) {
        if index >= self.body_nodes.len() {
            self.body_nodes.resize(index + 1, None);
        }
        self.body_nodes[index] = Some(node);
    }

    pub(crate) fn add_joint(&mut self, joint: Joint) -> i64 {
        let Some(cw) = self.world.as_mut() else {
            return -1;
        };
        cw.add_joint(joint) as i64
    }

    pub(crate) fn set_mouse_joint_target(&mut self, joint_index: i64, target: Vector3) {
        let Some(cw) = self.world.as_mut() else {
            return;
        };
        if joint_index < 0 || joint_index as usize >= cw.joints.len() {
            return;
        }
        cw.joints[joint_index as usize].set_target(to_pe_vec3(target));
    }

    pub(crate) fn set_static_pose(&mut self, static_index: i64, pos: Vector3, quat: Quaternion) {
        let Some(cw) = self.world.as_mut() else {
            return;
        };
        if static_index < 0 || static_index as usize >= cw.statics.len() {
            return;
        }
        let s = &mut cw.statics[static_index as usize];
        s.position = to_pe_vec3(pos);
        s.orientation = crate::convert::to_pe_quat(quat);
        cw.rebuild_static_tables();
    }

    pub(crate) fn apply_force_on(&mut self, index: i64, force: Vector3, torque: Vector3) {
        if index < 0 {
            return;
        }
        self.loads
            .add(index as usize, to_pe_vec3(force), to_pe_vec3(torque));
        self.wake_body(index);
    }

    pub(crate) fn apply_impulse_on(&mut self, index: i64, impulse: Vector3) {
        let Some(cw) = self.world.as_mut() else {
            return;
        };
        if index < 0 || index as usize >= cw.world.n {
            return;
        }
        cw.world
            .apply_linear_impulse(index as usize, to_pe_vec3(impulse));
        cw.wake_body(index as usize);
    }

    pub(crate) fn apply_impulse_at_on(&mut self, index: i64, impulse: Vector3, point: Vector3) {
        let Some(cw) = self.world.as_mut() else {
            return;
        };
        if index < 0 || index as usize >= cw.world.n {
            return;
        }
        cw.world
            .apply_impulse(index as usize, to_pe_vec3(impulse), to_pe_vec3(point));
        cw.wake_body(index as usize);
    }

    pub(crate) fn set_linear_velocity(&mut self, index: i64, v: Vector3) {
        let Some(cw) = self.world.as_mut() else {
            return;
        };
        if index < 0 || index as usize >= cw.world.n {
            return;
        }
        cw.world.state.v[index as usize] = to_pe_vec3(v);
        cw.wake_body(index as usize);
    }

    pub(crate) fn set_angular_velocity(&mut self, index: i64, w: Vector3) {
        let Some(cw) = self.world.as_mut() else {
            return;
        };
        if index < 0 || index as usize >= cw.world.n {
            return;
        }
        let i = index as usize;
        let q = cw.world.state.q[i];
        let iw = cw.world.inv_inertia_world(q, i);
        // L = I ω; store by inverting: use body inertia in world frame.
        // pe stores L; set L = I_world * ω where I_world = R I_b Rᵀ.
        let r = q.to_matrix();
        let ib = cw.world.inertia_body[i];
        let i_w = pe_core::body::mat_mul(
            &pe_core::body::mat_mul(&r, &ib),
            &pe_core::body::mat_transpose(&r),
        );
        let omega = to_pe_vec3(w);
        cw.world.state.l[i] = pe_core::body::mat_vec(&i_w, omega);
        let _ = iw; // silence if unused depending on path
        cw.wake_body(i);
    }

    pub(crate) fn sleep_body(&mut self, index: i64) {
        let Some(cw) = self.world.as_mut() else {
            return;
        };
        if index < 0 || index as usize >= cw.world.n {
            return;
        }
        let i = index as usize;
        if i < cw.sleep.awake.len() {
            cw.sleep.awake[i] = false;
            cw.world.state.v[i] = PeVec3::zero();
            cw.world.state.l[i] = PeVec3::zero();
        }
    }

    pub(crate) fn wake_body(&mut self, index: i64) {
        let Some(cw) = self.world.as_mut() else {
            return;
        };
        if index < 0 || index as usize >= cw.world.n {
            return;
        }
        cw.wake_body(index as usize);
    }

    pub(crate) fn is_body_sleeping(&self, index: i64) -> bool {
        let Some(cw) = self.world.as_ref() else {
            return false;
        };
        if index < 0 || index as usize >= cw.world.n {
            return false;
        }
        !cw.sleep.is_awake(index as usize)
    }

    /// Visual pose for a dynamic body (interpolated when enabled).
    pub(crate) fn body_visual_pose(&self, index: i64) -> (Vector3, Quaternion) {
        let Some(cw) = self.world.as_ref() else {
            return (Vector3::ZERO, Quaternion::IDENTITY);
        };
        if index < 0 || index as usize >= cw.world.n {
            return (Vector3::ZERO, Quaternion::IDENTITY);
        }
        let i = index as usize;
        let cur_x = cw.world.state.x[i];
        let cur_q = cw.world.state.q[i];
        if !self.interpolate || self.prev_x.len() != cw.world.n {
            return (to_godot_vec3(cur_x), to_godot_quat(cur_q));
        }
        let a = self.interpolation_alpha();
        let prev_x = self.prev_x[i];
        let prev_q = self.prev_q[i];
        let x = prev_x * (1.0 - a) + cur_x * a;
        let q = slerp_quat(prev_q, cur_q, a);
        (to_godot_vec3(x), to_godot_quat(q))
    }

    fn ensure_pose_buffers(&mut self) {
        let n = self.world.as_ref().map(|w| w.world.n).unwrap_or(0);
        if self.prev_x.len() != n {
            if let Some(cw) = self.world.as_ref() {
                self.prev_x = cw.world.state.x.clone();
                self.prev_q = cw.world.state.q.clone();
            }
        }
        if self.body_nodes.len() < n {
            self.body_nodes.resize(n, None);
        }
    }

    fn snapshot_poses(&mut self) {
        let Some(cw) = self.world.as_ref() else {
            return;
        };
        self.prev_x = cw.world.state.x.clone();
        self.prev_q = cw.world.state.q.clone();
    }

    fn step_internal(&mut self, dt: f64) {
        let Some(cw) = self.world.as_mut() else {
            return;
        };
        cw.step(dt);
        // Build current contact pair set.
        let mut curr = BTreeSet::new();
        for c in &cw.contacts {
            let a = c.a.min(c.b);
            let b = c.a.max(c.b);
            curr.insert((a, b));
        }
        let started: Vec<(usize, usize)> =
            curr.difference(&self.prev_contact_pairs).copied().collect();
        let ended: Vec<(usize, usize)> =
            self.prev_contact_pairs.difference(&curr).copied().collect();
        self.prev_contact_pairs = curr;

        for (a, b) in started {
            self.signals().contact_started().emit(a as i64, b as i64);
            self.emit_body_entered(a, b);
            self.emit_body_entered(b, a);
        }
        for (a, b) in ended {
            self.signals().contact_ended().emit(a as i64, b as i64);
            self.emit_body_exited(a, b);
            self.emit_body_exited(b, a);
        }
    }

    fn emit_body_entered(&mut self, body: usize, other: usize) {
        if body >= self.body_nodes.len() {
            return;
        }
        if let Some(node) = self.body_nodes[body].as_mut() {
            node.bind_mut().emit_entered(other as i64);
        }
    }

    fn emit_body_exited(&mut self, body: usize, other: usize) {
        if body >= self.body_nodes.len() {
            return;
        }
        if let Some(node) = self.body_nodes[body].as_mut() {
            node.bind_mut().emit_exited(other as i64);
        }
    }
}

fn slerp_quat(a: PeQuat, b: PeQuat, t: f64) -> PeQuat {
    // Visual nlerp (stable; not on the deterministic f64 hash path).
    let mut b = b;
    let dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    if dot < 0.0 {
        b = PeQuat::new(-b.w, -b.x, -b.y, -b.z);
    }
    let w = a.w * (1.0 - t) + b.w * t;
    let x = a.x * (1.0 - t) + b.x * t;
    let y = a.y * (1.0 - t) + b.y * t;
    let z = a.z * (1.0 - t) + b.z * t;
    PeQuat::new(w, x, y, z).normalize()
}
