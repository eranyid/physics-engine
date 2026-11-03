//! Godot body nodes that author shapes / materials and sync transforms.

use godot::classes::{INode3D, Node3D};
use godot::prelude::*;
use pe_collision::shapes::Material;
use pe_collision::{Shape, StaticCollider};
use pe_core::RigidBody;
use pe_math::Vec3 as PeVec3;

use crate::convert::{body_type, shape_kind, to_godot_quat, to_godot_vec3, to_pe_quat, to_pe_vec3};
use crate::shapes_util::build_shape;
use crate::world::PePhysicsWorld;

/// Dynamic / kinematic / static rigid body authored in the scene tree.
///
/// Reads collider parameters from exported properties (and optional child
/// [`PeCollisionShape3D`] override). Registers with a parent [`PePhysicsWorld`]
/// in `_ready`.
#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct PeRigidBody3D {
    base: Base<Node3D>,
    /// Engine body index (−1 before registration). For static/kinematic this is
    /// the static-collider index (not `n+si`).
    #[export]
    pub body_index: i64,
    /// 0=dynamic, 1=static, 2=kinematic (scripted static pose).
    #[export]
    pub body_type: i64,
    /// 0=sphere, 1=box, 2=capsule, 3=convex.
    #[export]
    pub shape_kind: i64,
    #[export]
    pub mass: f64,
    #[export]
    pub radius: f64,
    #[export]
    pub half_extents: Vector3,
    #[export]
    pub capsule_half_height: f64,
    #[export]
    pub convex_points: PackedVector3Array,
    #[export]
    pub friction: f64,
    #[export]
    pub restitution: f64,
    #[export]
    pub linear_velocity: Vector3,
    #[export]
    pub angular_velocity: Vector3,
    /// True once registered with the world.
    registered: bool,
    /// Static/kinematic collider index when not dynamic.
    static_index: i64,
}

#[godot_api]
impl INode3D for PeRigidBody3D {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            body_index: -1,
            body_type: body_type::DYNAMIC,
            shape_kind: shape_kind::BOX,
            mass: 1.0,
            radius: 0.5,
            half_extents: Vector3::new(0.5, 0.5, 0.5),
            capsule_half_height: 0.5,
            convex_points: PackedVector3Array::new(),
            friction: 0.5,
            restitution: 0.0,
            linear_velocity: Vector3::ZERO,
            angular_velocity: Vector3::ZERO,
            registered: false,
            static_index: -1,
        }
    }

    fn ready(&mut self) {
        self.try_register();
    }

    fn process(&mut self, _delta: f64) {
        if !self.registered {
            return;
        }
        let Some(mut world) = self.find_world() else {
            return;
        };
        // Kinematic: push Godot pose into the engine before the next step.
        if self.body_type == body_type::KINEMATIC && self.static_index >= 0 {
            let pos = self.base().get_position();
            let quat = self.base().get_quaternion();
            world
                .bind_mut()
                .set_static_pose(self.static_index, pos, quat);
            return;
        }
        if self.body_type != body_type::DYNAMIC || self.body_index < 0 {
            return;
        }
        let (pos, quat) = world.bind().body_visual_pose(self.body_index);
        self.base_mut().set_position(pos);
        self.base_mut().set_quaternion(quat);
    }
}

#[godot_api]
impl PeRigidBody3D {
    #[signal]
    fn body_entered(other_body: i64);

    #[signal]
    fn body_exited(other_body: i64);

    /// Register with the parent world if not already registered.
    #[func]
    pub fn try_register(&mut self) {
        if self.registered {
            return;
        }
        let Some(mut world) = self.find_world() else {
            return;
        };
        let shape = self.resolve_shape();
        let Some(shape) = shape else {
            godot_error!("PeRigidBody3D: invalid shape parameters");
            return;
        };
        let pos = self.base().get_global_position();
        let quat = self.base().get_quaternion();
        let mat = Material::new(self.restitution, self.friction);
        match self.body_type {
            body_type::STATIC | body_type::KINEMATIC => {
                let col = StaticCollider::new(shape)
                    .with_material(mat)
                    .with_position(to_pe_vec3(pos))
                    .with_orientation(to_pe_quat(quat));
                let idx = world.bind_mut().register_static_collider(col);
                self.static_index = idx;
                self.body_index = idx;
                self.registered = true;
            }
            _ => {
                if !(self.mass > 0.0 && self.mass.is_finite()) {
                    godot_error!("PeRigidBody3D: mass must be positive");
                    return;
                }
                let inertia = shape.inertia(self.mass);
                let body = RigidBody::new(self.mass)
                    .with_inertia(inertia)
                    .with_position(to_pe_vec3(pos))
                    .with_orientation(to_pe_quat(quat))
                    .with_velocity(to_pe_vec3(self.linear_velocity))
                    .with_angular_velocity(to_pe_vec3(self.angular_velocity));
                let idx = world.bind_mut().register_dynamic(body, shape, mat);
                self.body_index = idx;
                self.registered = true;
                let this = self.to_gd();
                world.bind_mut().track_body_node(idx as usize, this);
            }
        }
    }

    /// Apply a world-space force [N] for the next physics step (cleared after step).
    #[func]
    pub fn apply_force(&mut self, force: Vector3) {
        if self.body_index < 0 || self.body_type != body_type::DYNAMIC {
            return;
        }
        let Some(mut world) = self.find_world() else {
            return;
        };
        world
            .bind_mut()
            .apply_force_on(self.body_index, force, Vector3::ZERO);
    }

    /// Apply a world-space torque [N·m] for the next physics step.
    #[func]
    pub fn apply_torque(&mut self, torque: Vector3) {
        if self.body_index < 0 || self.body_type != body_type::DYNAMIC {
            return;
        }
        let Some(mut world) = self.find_world() else {
            return;
        };
        world
            .bind_mut()
            .apply_force_on(self.body_index, Vector3::ZERO, torque);
    }

    /// Instantaneous linear impulse [N·s] at the COM.
    #[func]
    pub fn apply_impulse(&mut self, impulse: Vector3) {
        if self.body_index < 0 || self.body_type != body_type::DYNAMIC {
            return;
        }
        let Some(mut world) = self.find_world() else {
            return;
        };
        world.bind_mut().apply_impulse_on(self.body_index, impulse);
    }

    /// Instantaneous impulse [N·s] at a world point.
    #[func]
    pub fn apply_impulse_at(&mut self, impulse: Vector3, point: Vector3) {
        if self.body_index < 0 || self.body_type != body_type::DYNAMIC {
            return;
        }
        let Some(mut world) = self.find_world() else {
            return;
        };
        world
            .bind_mut()
            .apply_impulse_at_on(self.body_index, impulse, point);
    }

    /// Set linear velocity [m/s].
    #[func]
    pub fn set_linear_velocity_vec(&mut self, v: Vector3) {
        self.linear_velocity = v;
        if self.body_index < 0 || self.body_type != body_type::DYNAMIC {
            return;
        }
        let Some(mut world) = self.find_world() else {
            return;
        };
        world.bind_mut().set_linear_velocity(self.body_index, v);
    }

    /// Set angular velocity [rad/s].
    #[func]
    pub fn set_angular_velocity_vec(&mut self, w: Vector3) {
        self.angular_velocity = w;
        if self.body_index < 0 || self.body_type != body_type::DYNAMIC {
            return;
        }
        let Some(mut world) = self.find_world() else {
            return;
        };
        world.bind_mut().set_angular_velocity(self.body_index, w);
    }

    /// Put the body to sleep (no-op if sleep disabled on the world).
    #[func]
    pub fn sleep_body(&mut self) {
        if self.body_index < 0 || self.body_type != body_type::DYNAMIC {
            return;
        }
        let Some(mut world) = self.find_world() else {
            return;
        };
        world.bind_mut().sleep_body(self.body_index);
    }

    /// Wake the body.
    #[func]
    pub fn wake_body(&mut self) {
        if self.body_index < 0 || self.body_type != body_type::DYNAMIC {
            return;
        }
        let Some(mut world) = self.find_world() else {
            return;
        };
        world.bind_mut().wake_body(self.body_index);
    }

    /// Whether the body is currently awake (dynamics only).
    #[func]
    pub fn is_sleeping(&self) -> bool {
        if self.body_index < 0 || self.body_type != body_type::DYNAMIC {
            return false;
        }
        let Some(world) = self.find_world() else {
            return false;
        };
        let sleeping = world.bind().is_body_sleeping(self.body_index);
        sleeping
    }

    fn resolve_shape(&self) -> Option<Shape> {
        // Optional child PeCollisionShape3D overrides exports.
        let children = self.base().get_children();
        for i in 0..children.len() {
            let Some(node) = children.get(i) else {
                continue;
            };
            if let Ok(shape_node) = node.try_cast::<PeCollisionShape3D>() {
                let b = shape_node.bind();
                return build_shape(
                    b.shape_kind,
                    b.radius,
                    b.half_extents.x as f64,
                    b.half_extents.y as f64,
                    b.half_extents.z as f64,
                    b.capsule_half_height,
                    &b.convex_points,
                );
            }
        }
        build_shape(
            self.shape_kind,
            self.radius,
            self.half_extents.x as f64,
            self.half_extents.y as f64,
            self.half_extents.z as f64,
            self.capsule_half_height,
            &self.convex_points,
        )
    }

    fn find_world(&self) -> Option<Gd<PePhysicsWorld>> {
        let mut node = self.to_gd().upcast::<Node3D>();
        loop {
            let parent = node.get_parent()?;
            match parent.clone().try_cast::<PePhysicsWorld>() {
                Ok(world) => return Some(world),
                Err(parent) => {
                    let Ok(as_node3d) = parent.try_cast::<Node3D>() else {
                        return None;
                    };
                    node = as_node3d;
                }
            }
        }
    }
}

impl PeRigidBody3D {
    pub(crate) fn emit_entered(&mut self, other: i64) {
        self.signals().body_entered().emit(other);
    }

    pub(crate) fn emit_exited(&mut self, other: i64) {
        self.signals().body_exited().emit(other);
    }
}

/// Static collider authored in the scene tree.
#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct PeStaticBody3D {
    base: Base<Node3D>,
    #[export]
    pub static_index: i64,
    #[export]
    pub shape_kind: i64,
    #[export]
    pub radius: f64,
    #[export]
    pub half_extents: Vector3,
    #[export]
    pub capsule_half_height: f64,
    #[export]
    pub convex_points: PackedVector3Array,
    #[export]
    pub friction: f64,
    #[export]
    pub restitution: f64,
    registered: bool,
}

#[godot_api]
impl INode3D for PeStaticBody3D {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            static_index: -1,
            shape_kind: shape_kind::BOX,
            radius: 0.5,
            half_extents: Vector3::new(0.5, 0.5, 0.5),
            capsule_half_height: 0.5,
            convex_points: PackedVector3Array::new(),
            friction: 0.5,
            restitution: 0.0,
            registered: false,
        }
    }

    fn ready(&mut self) {
        self.try_register();
    }
}

#[godot_api]
impl PeStaticBody3D {
    #[func]
    pub fn try_register(&mut self) {
        if self.registered {
            return;
        }
        let Some(mut world) = self.find_world() else {
            return;
        };
        let shape = build_shape(
            self.shape_kind,
            self.radius,
            self.half_extents.x as f64,
            self.half_extents.y as f64,
            self.half_extents.z as f64,
            self.capsule_half_height,
            &self.convex_points,
        );
        let Some(shape) = shape else {
            godot_error!("PeStaticBody3D: invalid shape parameters");
            return;
        };
        let pos = self.base().get_global_position();
        let quat = self.base().get_quaternion();
        let col = StaticCollider::new(shape)
            .with_material(Material::new(self.restitution, self.friction))
            .with_position(to_pe_vec3(pos))
            .with_orientation(to_pe_quat(quat));
        let idx = world.bind_mut().register_static_collider(col);
        self.static_index = idx;
        self.registered = true;
    }

    fn find_world(&self) -> Option<Gd<PePhysicsWorld>> {
        let mut node = self.to_gd().upcast::<Node3D>();
        loop {
            let parent = node.get_parent()?;
            match parent.clone().try_cast::<PePhysicsWorld>() {
                Ok(world) => return Some(world),
                Err(parent) => {
                    let Ok(as_node3d) = parent.try_cast::<Node3D>() else {
                        return None;
                    };
                    node = as_node3d;
                }
            }
        }
    }
}

/// Optional child that overrides the parent body's shape exports.
#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct PeCollisionShape3D {
    base: Base<Node3D>,
    #[export]
    pub shape_kind: i64,
    #[export]
    pub radius: f64,
    #[export]
    pub half_extents: Vector3,
    #[export]
    pub capsule_half_height: f64,
    #[export]
    pub convex_points: PackedVector3Array,
}

#[godot_api]
impl INode3D for PeCollisionShape3D {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            shape_kind: shape_kind::BOX,
            radius: 0.5,
            half_extents: Vector3::new(0.5, 0.5, 0.5),
            capsule_half_height: 0.5,
            convex_points: PackedVector3Array::new(),
        }
    }
}

#[godot_api]
impl PeCollisionShape3D {}

// Silence unused import warnings when pose helpers are only used via world.
#[allow(dead_code)]
fn _pose_helpers(v: PeVec3) -> Vector3 {
    to_godot_vec3(v)
}

#[allow(dead_code)]
fn _quat_helpers(q: pe_math::Quat) -> Quaternion {
    to_godot_quat(q)
}
