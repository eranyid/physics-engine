//! Godot 4 GDExtension for the physics engine.
//!
//! Provides [`PePhysicsWorld`] — a `Node3D` that owns a `ContactWorld` and steps
//! it in `_physics_process`. Child [`PeRigidBody3D`] nodes sync transforms from
//! the simulation.
//!
//! Gravity defaults to Godot Y-up (`(0, −g, 0)`). CI runs a headless drop test
//! — see `docs/GODOT.md`.

use godot::classes::{INode3D, Node3D};
use godot::prelude::*;
use pe_collision::shapes::{BoxShape, Material, Shape, Sphere};
use pe_collision::{ContactWorld, StaticCollider};
use pe_core::diagnostics::state_hash64;
use pe_core::{RigidBody, World};
use pe_math::{Quat as PeQuat, Vec3 as PeVec3, STANDARD_GRAVITY};
use pe_scene::{force_to_box, ForceDesc};

struct PeGodotExtension;

#[gdextension]
unsafe impl ExtensionLibrary for PeGodotExtension {}

/// Root physics world node. Steps the engine each physics frame.
#[derive(GodotClass)]
#[class(base=Node3D)]
struct PePhysicsWorld {
    base: Base<Node3D>,
    world: Option<ContactWorld>,
    /// Fixed step [s]; if ≤ 0, uses Godot's `delta`.
    #[export]
    fixed_dt: f64,
    /// Accumulated leftover time when using fixed_dt.
    accum: f64,
}

fn y_up_gravity_world() -> ContactWorld {
    let gravity = ForceDesc::UniformGravity {
        g: [0.0, -STANDARD_GRAVITY, 0.0],
    };
    let gens: Vec<Box<dyn pe_core::ForceGenerator>> = force_to_box(&gravity).into_iter().collect();
    ContactWorld::new(World::new(vec![], gens), vec![], vec![], Some(vec![]))
}

#[godot_api]
impl INode3D for PePhysicsWorld {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            world: Some(y_up_gravity_world()),
            fixed_dt: 1.0 / 60.0,
            accum: 0.0,
        }
    }

    fn physics_process(&mut self, delta: f64) {
        let Some(cw) = self.world.as_mut() else {
            return;
        };
        if self.fixed_dt > 0.0 {
            self.accum += delta;
            let step = self.fixed_dt;
            while self.accum >= step {
                cw.step(step);
                self.accum -= step;
            }
        } else if delta > 0.0 && delta.is_finite() {
            cw.step(delta);
        }
    }
}

#[godot_api]
impl PePhysicsWorld {
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
        let Some(cw) = self.world.as_mut() else {
            return -1;
        };
        if !(mass > 0.0 && radius >= 0.0) {
            return -1;
        }
        let shape = Shape::Sphere(Sphere::new(radius));
        let body = RigidBody::new(mass)
            .with_inertia(shape.inertia(mass))
            .with_position(PeVec3::new(px, py, pz));
        let idx = cw.add_dynamic_body(body, shape, Material::new(restitution, friction));
        idx as i64
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
        let Some(cw) = self.world.as_mut() else {
            return -1;
        };
        if !(mass > 0.0 && hx >= 0.0 && hy >= 0.0 && hz >= 0.0) {
            return -1;
        }
        let shape = Shape::Box(BoxShape::new(hx, hy, hz));
        let body = RigidBody::new(mass)
            .with_inertia(shape.inertia(mass))
            .with_position(PeVec3::new(px, py, pz));
        let idx = cw.add_dynamic_body(body, shape, Material::new(restitution, friction));
        idx as i64
    }

    /// Add a static ground plane with normal +Y at `y = offset` (Godot Y-up).
    #[func]
    fn add_ground_plane(&mut self, offset: f64, restitution: f64, friction: f64) -> i64 {
        let Some(cw) = self.world.as_mut() else {
            return -1;
        };
        let col = StaticCollider::new(Shape::Plane(pe_collision::shapes::Plane::new(
            PeVec3::unit_y(),
            offset,
        )))
        .with_material(Material::new(restitution, friction));
        cw.add_static(col) as i64
    }

    /// Manually step the simulation once (for headless CI scripts).
    #[func]
    fn step_once(&mut self, dt: f64) {
        let Some(cw) = self.world.as_mut() else {
            return;
        };
        if dt > 0.0 && dt.is_finite() {
            cw.step(dt);
        }
    }

    /// FNV-1a 64-bit hash of packed dynamic state as lowercase hex (16 chars).
    ///
    /// Matches Rust `state_hash64` / `pe_scene::godot_drop_box_result`. Returned as
    /// a string so values above `i64::MAX` are not truncated in GDScript.
    #[func]
    fn state_hash_hex(&self) -> GString {
        let Some(cw) = self.world.as_ref() else {
            return GString::from("0");
        };
        GString::from(format!("{:016x}", state_hash64(&cw.world.state)).as_str())
    }

    /// Body count.
    #[func]
    fn body_count(&self) -> i64 {
        self.world.as_ref().map(|w| w.world.n as i64).unwrap_or(0)
    }

    /// Read body position into a Godot `Vector3`.
    #[func]
    fn get_body_position(&self, index: i64) -> Vector3 {
        let Some(cw) = self.world.as_ref() else {
            return Vector3::ZERO;
        };
        if index < 0 || index as usize >= cw.world.n {
            return Vector3::ZERO;
        }
        let p = cw.world.state.x[index as usize];
        Vector3::new(p.x as f32, p.y as f32, p.z as f32)
    }

    /// Read body orientation as Godot `Quaternion` (Godot is `[x,y,z,w]`; we convert).
    #[func]
    fn get_body_quaternion(&self, index: i64) -> Quaternion {
        let Some(cw) = self.world.as_ref() else {
            return Quaternion::IDENTITY;
        };
        if index < 0 || index as usize >= cw.world.n {
            return Quaternion::IDENTITY;
        }
        let q = cw.world.state.q[index as usize];
        // Godot Quaternion: x,y,z,w — pe uses w,x,y,z.
        Quaternion::new(q.x as f32, q.y as f32, q.z as f32, q.w as f32)
    }
}

/// Helper node that copies a simulated body transform onto its own `Node3D`.
#[derive(GodotClass)]
#[class(base=Node3D)]
struct PeRigidBody3D {
    base: Base<Node3D>,
    /// Index into the parent [`PePhysicsWorld`].
    #[export]
    body_index: i64,
}

#[godot_api]
impl INode3D for PeRigidBody3D {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            body_index: -1,
        }
    }

    fn process(&mut self, _delta: f64) {
        if self.body_index < 0 {
            return;
        }
        let Some(parent) = self.base().get_parent() else {
            return;
        };
        let Ok(world) = parent.try_cast::<PePhysicsWorld>() else {
            return;
        };
        let pos = world.bind().get_body_position(self.body_index);
        let quat = world.bind().get_body_quaternion(self.body_index);
        self.base_mut().set_position(pos);
        self.base_mut().set_quaternion(quat);
    }
}

// Keep PeQuat import used for documentation parity / future set APIs.
#[allow(dead_code)]
fn _pe_quat_identity() -> PeQuat {
    PeQuat::identity()
}
