//! 10-box pyramid on a plane; prints final COM heights and state hash.

use pe_collision::{BoxShape, ContactWorld, Material, Plane, Shape, StaticCollider};
use pe_core::diagnostics::state_hash64;
use pe_core::forces::UniformGravity;
use pe_core::{RigidBody, World};
use pe_math::Vec3;

fn main() {
    let h = 0.5;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    let mut mats = Vec::new();
    let mut placed = 0usize;
    let mut row = 0usize;
    while placed < 10 {
        let n = row + 1;
        let x0 = -(n as f64 - 1.0) * h;
        for i in 0..n {
            if placed >= 10 {
                break;
            }
            let x = x0 + 2.0 * h * (i as f64);
            let z = h + 2.0 * h * (row as f64);
            bodies.push(
                RigidBody::new(1.0)
                    .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
                    .with_position(Vec3::new(x, 0.0, z)),
            );
            shapes.push(Shape::Box(BoxShape::new(h, h, h)));
            mats.push(Material::new(0.0, 0.5));
            placed += 1;
        }
        row += 1;
    }
    let mut cw = ContactWorld::new(
        World::new(bodies, vec![Box::new(UniformGravity::earth())]),
        shapes,
        vec![
            StaticCollider::new(Shape::Plane(Plane::new(Vec3::unit_z(), 0.0)))
                .with_material(Material::new(0.0, 1.0)),
        ],
        Some(mats),
    );
    for _ in 0..600 {
        cw.step(1.0 / 240.0);
    }
    let zs: Vec<f64> = cw.world.state.x.iter().map(|p| p.z).collect();
    println!("pyramid: z = {zs:?}");
    println!(
        "pyramid: state_hash64 = {:016x}",
        state_hash64(&cw.world.state)
    );
}
