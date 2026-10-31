//! Opaque world handle wrapping [`pe_collision::ContactWorld`] + scene force list.

use pe_collision::shapes::{BoxShape, Capsule, Plane, Shape, Sphere};
use pe_collision::ContactWorld;
use pe_core::World;
use pe_math::{Vec3, STANDARD_GRAVITY};
use pe_scene::{force_to_box, ForceDesc};

use crate::error::PE_ERR_ARG;

/// Host-owned simulation world.
pub struct PeWorld {
    pub(crate) cw: ContactWorld,
    pub(crate) forces: Vec<ForceDesc>,
    pub(crate) last_contacts: Vec<PeContactEvent>,
}

/// Contact event written to host buffers.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PeContactEvent {
    /// Body A index.
    pub body_a: u32,
    /// Body B index (`n+si` for statics).
    pub body_b: u32,
    /// World contact point [m].
    pub point: [f64; 3],
    /// Normal A→B.
    pub normal: [f64; 3],
    /// Penetration depth [m].
    pub depth: f64,
}

impl PeWorld {
    pub(crate) fn new_empty() -> Self {
        let forces = vec![ForceDesc::UniformGravity {
            g: [0.0, 0.0, -STANDARD_GRAVITY],
        }];
        let gens: Vec<Box<dyn pe_core::ForceGenerator>> =
            forces.iter().filter_map(|f| force_to_box(f).ok()).collect();
        let world = World::new(vec![], gens);
        let cw = ContactWorld::new(world, vec![], vec![], Some(vec![]));
        Self {
            cw,
            forces,
            last_contacts: Vec::new(),
        }
    }

    pub(crate) fn refresh_contacts(&mut self) {
        self.last_contacts.clear();
        for c in &self.cw.contacts {
            self.last_contacts.push(PeContactEvent {
                body_a: c.a as u32,
                body_b: c.b as u32,
                point: c.point.to_array(),
                normal: c.normal.to_array(),
                depth: c.depth,
            });
        }
    }
}

pub(crate) fn parse_shape(kind: i32, a: f64, b: f64, c: f64, d: f64) -> Result<Shape, i32> {
    match kind {
        0 => Ok(Shape::Sphere(Sphere::new(a))),
        1 => Ok(Shape::Box(BoxShape::new(a, b, c))),
        2 => Ok(Shape::Capsule(Capsule::new(a, b))),
        3 => Ok(Shape::Plane(Plane::new(Vec3::new(a, b, c), d))),
        _ => Err(PE_ERR_ARG),
    }
}
