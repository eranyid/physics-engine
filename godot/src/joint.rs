//! Joint nodes wrapping pe-collision maximal-coordinate joints.

use godot::classes::{INode3D, Node3D};
use godot::prelude::*;
use pe_collision::joints::{Joint, JointSoftParams, WORLD_BODY};
use pe_math::Vec3 as PeVec3;

use crate::body::PeRigidBody3D;
use crate::convert::to_pe_vec3;
use crate::world::PePhysicsWorld;

fn resolve_body_index(path: &NodePath, from: &Gd<Node3D>) -> Option<usize> {
    if path.is_empty() {
        return Some(WORLD_BODY);
    }
    let node = from.get_node_or_null(path)?;
    if let Ok(body) = node.try_cast::<PeRigidBody3D>() {
        let idx = body.bind().body_index;
        if idx < 0 {
            return None;
        }
        return Some(idx as usize);
    }
    None
}

fn find_world(from: &Gd<Node3D>) -> Option<Gd<PePhysicsWorld>> {
    let mut node = from.clone();
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

macro_rules! joint_soft_defaults {
    () => {
        JointSoftParams {
            hertz: 60.0,
            zeta: 0.7,
        }
    };
}

/// Distance joint between two bodies (or body ↔ world when `body_b` is empty).
#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct PeDistanceJoint3D {
    base: Base<Node3D>,
    #[export]
    pub body_a: NodePath,
    #[export]
    pub body_b: NodePath,
    #[export]
    pub anchor_a: Vector3,
    #[export]
    pub anchor_b: Vector3,
    #[export]
    pub rest_length: f64,
    #[export]
    pub hertz: f64,
    #[export]
    pub zeta: f64,
    #[export]
    pub joint_index: i64,
    registered: bool,
}

#[godot_api]
impl INode3D for PeDistanceJoint3D {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            body_a: NodePath::from(""),
            body_b: NodePath::from(""),
            anchor_a: Vector3::ZERO,
            anchor_b: Vector3::ZERO,
            rest_length: 1.0,
            hertz: 60.0,
            zeta: 0.7,
            joint_index: -1,
            registered: false,
        }
    }

    fn ready(&mut self) {
        self.try_register();
    }
}

#[godot_api]
impl PeDistanceJoint3D {
    #[func]
    pub fn try_register(&mut self) {
        if self.registered {
            return;
        }
        let base = self.to_gd().upcast::<Node3D>();
        let Some(mut world) = find_world(&base) else {
            return;
        };
        let Some(a) = resolve_body_index(&self.body_a, &base) else {
            godot_warn!("PeDistanceJoint3D: body_a not ready");
            return;
        };
        let Some(b) = resolve_body_index(&self.body_b, &base) else {
            godot_warn!("PeDistanceJoint3D: body_b not ready");
            return;
        };
        let length = if self.rest_length > 0.0 {
            self.rest_length
        } else {
            // Auto length from current world anchors if both dynamic.
            1.0
        };
        let mut joint = Joint::distance(
            a,
            b,
            to_pe_vec3(self.anchor_a),
            to_pe_vec3(self.anchor_b),
            length,
        );
        joint.soft = JointSoftParams {
            hertz: self.hertz,
            zeta: self.zeta,
        };
        let idx = world.bind_mut().add_joint(joint);
        self.joint_index = idx;
        self.registered = true;
    }
}

/// Revolute (hinge) joint.
#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct PeRevoluteJoint3D {
    base: Base<Node3D>,
    #[export]
    pub body_a: NodePath,
    #[export]
    pub body_b: NodePath,
    #[export]
    pub anchor_a: Vector3,
    #[export]
    pub anchor_b: Vector3,
    #[export]
    pub axis: Vector3,
    #[export]
    pub hertz: f64,
    #[export]
    pub zeta: f64,
    #[export]
    pub joint_index: i64,
    registered: bool,
}

#[godot_api]
impl INode3D for PeRevoluteJoint3D {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            body_a: NodePath::from(""),
            body_b: NodePath::from(""),
            anchor_a: Vector3::ZERO,
            anchor_b: Vector3::ZERO,
            axis: Vector3::new(0.0, 0.0, 1.0),
            hertz: 60.0,
            zeta: 0.7,
            joint_index: -1,
            registered: false,
        }
    }

    fn ready(&mut self) {
        self.try_register();
    }
}

#[godot_api]
impl PeRevoluteJoint3D {
    #[func]
    pub fn try_register(&mut self) {
        if self.registered {
            return;
        }
        let base = self.to_gd().upcast::<Node3D>();
        let Some(mut world) = find_world(&base) else {
            return;
        };
        let Some(a) = resolve_body_index(&self.body_a, &base) else {
            return;
        };
        let Some(b) = resolve_body_index(&self.body_b, &base) else {
            return;
        };
        let axis = to_pe_vec3(self.axis);
        let axis = if axis.norm() > 0.0 {
            axis.normalize()
        } else {
            PeVec3::unit_z()
        };
        // Build an orthonormal ref perpendicular to axis.
        let ref_a = if axis.x.abs() < 0.9 {
            axis.cross(PeVec3::unit_x()).normalize()
        } else {
            axis.cross(PeVec3::unit_y()).normalize()
        };
        let mut joint = Joint::revolute(
            a,
            b,
            to_pe_vec3(self.anchor_a),
            to_pe_vec3(self.anchor_b),
            axis,
            axis,
            ref_a,
            ref_a,
        );
        joint.soft = JointSoftParams {
            hertz: self.hertz,
            zeta: self.zeta,
        };
        let idx = world.bind_mut().add_joint(joint);
        self.joint_index = idx;
        self.registered = true;
    }
}

/// Prismatic (slider) joint.
#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct PePrismaticJoint3D {
    base: Base<Node3D>,
    #[export]
    pub body_a: NodePath,
    #[export]
    pub body_b: NodePath,
    #[export]
    pub anchor_a: Vector3,
    #[export]
    pub anchor_b: Vector3,
    #[export]
    pub axis: Vector3,
    #[export]
    pub hertz: f64,
    #[export]
    pub zeta: f64,
    #[export]
    pub joint_index: i64,
    registered: bool,
}

#[godot_api]
impl INode3D for PePrismaticJoint3D {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            body_a: NodePath::from(""),
            body_b: NodePath::from(""),
            anchor_a: Vector3::ZERO,
            anchor_b: Vector3::ZERO,
            axis: Vector3::new(0.0, 1.0, 0.0),
            hertz: 60.0,
            zeta: 0.7,
            joint_index: -1,
            registered: false,
        }
    }

    fn ready(&mut self) {
        self.try_register();
    }
}

#[godot_api]
impl PePrismaticJoint3D {
    #[func]
    pub fn try_register(&mut self) {
        if self.registered {
            return;
        }
        let base = self.to_gd().upcast::<Node3D>();
        let Some(mut world) = find_world(&base) else {
            return;
        };
        let Some(a) = resolve_body_index(&self.body_a, &base) else {
            return;
        };
        let Some(b) = resolve_body_index(&self.body_b, &base) else {
            return;
        };
        let axis = to_pe_vec3(self.axis);
        let axis = if axis.norm() > 0.0 {
            axis.normalize()
        } else {
            PeVec3::unit_y()
        };
        let mut joint = Joint::prismatic(
            a,
            b,
            to_pe_vec3(self.anchor_a),
            to_pe_vec3(self.anchor_b),
            axis,
            axis,
            0.0,
        );
        joint.soft = JointSoftParams {
            hertz: self.hertz,
            zeta: self.zeta,
        };
        let idx = world.bind_mut().add_joint(joint);
        self.joint_index = idx;
        self.registered = true;
    }
}

/// Soft weld joint locking relative pose.
#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct PeWeldJoint3D {
    base: Base<Node3D>,
    #[export]
    pub body_a: NodePath,
    #[export]
    pub body_b: NodePath,
    #[export]
    pub anchor_a: Vector3,
    #[export]
    pub anchor_b: Vector3,
    #[export]
    pub hertz: f64,
    #[export]
    pub zeta: f64,
    #[export]
    pub joint_index: i64,
    registered: bool,
}

#[godot_api]
impl INode3D for PeWeldJoint3D {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            body_a: NodePath::from(""),
            body_b: NodePath::from(""),
            anchor_a: Vector3::ZERO,
            anchor_b: Vector3::ZERO,
            hertz: 60.0,
            zeta: 0.7,
            joint_index: -1,
            registered: false,
        }
    }

    fn ready(&mut self) {
        self.try_register();
    }
}

#[godot_api]
impl PeWeldJoint3D {
    #[func]
    pub fn try_register(&mut self) {
        if self.registered {
            return;
        }
        let base = self.to_gd().upcast::<Node3D>();
        let Some(mut world) = find_world(&base) else {
            return;
        };
        let Some(a) = resolve_body_index(&self.body_a, &base) else {
            return;
        };
        let Some(b) = resolve_body_index(&self.body_b, &base) else {
            return;
        };
        let mut joint = Joint::weld(
            a,
            b,
            to_pe_vec3(self.anchor_a),
            to_pe_vec3(self.anchor_b),
            pe_math::Quat::identity(),
        );
        joint.soft = JointSoftParams {
            hertz: self.hertz,
            zeta: self.zeta,
        };
        let idx = world.bind_mut().add_joint(joint);
        self.joint_index = idx;
        self.registered = true;
    }
}

/// Wheel / suspension joint.
#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct PeWheelJoint3D {
    base: Base<Node3D>,
    #[export]
    pub body_a: NodePath,
    #[export]
    pub body_b: NodePath,
    #[export]
    pub anchor_a: Vector3,
    #[export]
    pub anchor_b: Vector3,
    #[export]
    pub axis: Vector3,
    #[export]
    pub spring_hertz: f64,
    #[export]
    pub spring_zeta: f64,
    #[export]
    pub rest_length: f64,
    #[export]
    pub hertz: f64,
    #[export]
    pub zeta: f64,
    #[export]
    pub joint_index: i64,
    registered: bool,
}

#[godot_api]
impl INode3D for PeWheelJoint3D {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            body_a: NodePath::from(""),
            body_b: NodePath::from(""),
            anchor_a: Vector3::ZERO,
            anchor_b: Vector3::ZERO,
            axis: Vector3::new(0.0, 1.0, 0.0),
            spring_hertz: 4.0,
            spring_zeta: 0.7,
            rest_length: 0.5,
            hertz: 60.0,
            zeta: 0.7,
            joint_index: -1,
            registered: false,
        }
    }

    fn ready(&mut self) {
        self.try_register();
    }
}

#[godot_api]
impl PeWheelJoint3D {
    #[func]
    pub fn try_register(&mut self) {
        if self.registered {
            return;
        }
        let base = self.to_gd().upcast::<Node3D>();
        let Some(mut world) = find_world(&base) else {
            return;
        };
        let Some(a) = resolve_body_index(&self.body_a, &base) else {
            return;
        };
        let Some(b) = resolve_body_index(&self.body_b, &base) else {
            return;
        };
        let axis = to_pe_vec3(self.axis);
        let axis = if axis.norm() > 0.0 {
            axis.normalize()
        } else {
            PeVec3::unit_y()
        };
        let mut joint = Joint::wheel(
            a,
            b,
            to_pe_vec3(self.anchor_a),
            to_pe_vec3(self.anchor_b),
            axis,
            axis,
            self.spring_hertz,
            self.spring_zeta,
            self.rest_length,
        );
        joint.soft = JointSoftParams {
            hertz: self.hertz,
            zeta: self.zeta,
        };
        let idx = world.bind_mut().add_joint(joint);
        self.joint_index = idx;
        self.registered = true;
    }
}

/// Mouse / target joint pulling a body anchor toward a world point.
#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct PeMouseJoint3D {
    base: Base<Node3D>,
    #[export]
    pub body_a: NodePath,
    #[export]
    pub anchor_a: Vector3,
    #[export]
    pub target: Vector3,
    #[export]
    pub max_force: f64,
    #[export]
    pub joint_index: i64,
    registered: bool,
}

#[godot_api]
impl INode3D for PeMouseJoint3D {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            body_a: NodePath::from(""),
            anchor_a: Vector3::ZERO,
            target: Vector3::ZERO,
            max_force: 0.0,
            joint_index: -1,
            registered: false,
        }
    }

    fn ready(&mut self) {
        self.try_register();
    }

    fn process(&mut self, _delta: f64) {
        if !self.registered || self.joint_index < 0 {
            return;
        }
        let base = self.to_gd().upcast::<Node3D>();
        let Some(mut world) = find_world(&base) else {
            return;
        };
        world
            .bind_mut()
            .set_mouse_joint_target(self.joint_index, self.target);
    }
}

#[godot_api]
impl PeMouseJoint3D {
    #[func]
    pub fn try_register(&mut self) {
        if self.registered {
            return;
        }
        let base = self.to_gd().upcast::<Node3D>();
        let Some(mut world) = find_world(&base) else {
            return;
        };
        let Some(a) = resolve_body_index(&self.body_a, &base) else {
            return;
        };
        let joint = Joint::mouse(
            a,
            to_pe_vec3(self.anchor_a),
            to_pe_vec3(self.target),
            self.max_force,
        );
        let idx = world.bind_mut().add_joint(joint);
        self.joint_index = idx;
        self.registered = true;
    }

    #[func]
    pub fn set_target_vec(&mut self, target: Vector3) {
        self.target = target;
    }
}

// Keep macro referenced for future joint soft defaults.
#[allow(dead_code)]
fn _soft() -> JointSoftParams {
    joint_soft_defaults!()
}
