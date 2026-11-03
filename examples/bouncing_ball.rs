//! Sphere bouncing on a plane (e=0.8). Prints apex heights for a few bounces.

use pe_collision::{ContactWorld, Material, Plane, Shape, Sphere, StaticCollider};
use pe_core::forces::UniformGravity;
use pe_core::{RigidBody, World};
use pe_math::Vec3;

fn main() {
    let r = 0.1;
    let e = 0.8;
    let body = RigidBody::new(1.0)
        .with_inertia(Sphere::new(r).inertia(1.0))
        .with_position(Vec3::new(0.0, 0.0, 1.0));
    let mut cw = ContactWorld::new(
        World::new(vec![body], vec![Box::new(UniformGravity::earth())]),
        vec![Shape::Sphere(Sphere::new(r))],
        vec![
            StaticCollider::new(Shape::Plane(Plane::new(Vec3::unit_z(), 0.0)))
                .with_material(Material::new(e, 0.0)),
        ],
        Some(vec![Material::new(e, 0.0)]),
    );
    cw.set_allow_sleep(false);

    let dt = 1e-3;
    let mut last_vz = 0.0;
    let mut apexes = Vec::new();
    for _ in 0..8_000 {
        cw.step(dt);
        let z = cw.world.state.x[0].z;
        let vz = cw.world.state.v[0].z;
        if last_vz > 0.0 && vz <= 0.0 && z > r + 0.02 {
            apexes.push(z);
            if apexes.len() >= 4 {
                break;
            }
        }
        last_vz = vz;
    }
    println!("bouncing_ball: apex heights (m) = {apexes:?}");
    println!("expected ratio ~ e^2 = {}", e * e);
}
