//! Canonical soft-body determinism scenario (cross-OS hash).

use pe_math::Vec3;

use crate::builders::{make_cloth, make_rope, make_soft_cube};
use crate::collision::{StaticColliders, StaticPlane};
use crate::material::material_by_name;
use crate::solver::{soft_state_hash64, SoftWorld};

/// Outer steps in the canonical hash scenario.
pub const SOFT_HASH_STEPS: usize = 120;
/// Outer timestep [s].
pub const SOFT_HASH_DT: f64 = 1.0 / 60.0;

/// Build the canonical soft scene: rope + cloth + jelly cube above a ground plane.
pub fn soft_canonical_world() -> SoftWorld {
    let mut rope = make_rope(
        Vec3::new(-0.5, 0.0, 1.0),
        Vec3::new(0.5, 0.0, 1.0),
        8,
        0.4,
        1e-6,
        true,
    );
    // Cloth beside the rope
    let mut cloth = make_cloth(
        Vec3::new(0.0, 0.8, 1.2),
        4,
        4,
        0.1,
        0.1,
        0.2,
        1e-7,
        1e-5,
        true,
    );
    let jelly_mat = material_by_name("jelly").expect("jelly material");
    let mut jelly = make_soft_cube(Vec3::new(0.8, 0.0, 0.6), 0.25, 2, &jelly_mat, true);

    // Merge into one world (deterministic append order: rope, cloth, jelly).
    let mut world = SoftWorld::new();
    world.params.substeps = 10;
    world.params.iterations = 1;
    world.params.gravity = Vec3::new(0.0, 0.0, -9.80665);
    world.colliders = StaticColliders {
        planes: vec![StaticPlane::new(Vec3::unit_z(), 0.0, 0.4, 0.0)],
        spheres: vec![],
        boxes: vec![],
    };

    append_world(&mut world, &mut rope);
    append_world(&mut world, &mut cloth);
    append_world(&mut world, &mut jelly);
    world
}

fn append_world(dst: &mut SoftWorld, src: &mut SoftWorld) {
    let base = dst.particles.len();
    dst.particles.append(&mut src.particles);
    for c in &mut src.constraints.distances {
        c.i += base;
        c.j += base;
        dst.constraints.distances.push(c.clone());
    }
    for c in &mut src.constraints.distance_bends {
        c.a += base;
        c.b += base;
        dst.constraints.distance_bends.push(c.clone());
    }
    for c in &mut src.constraints.dihedrals {
        c.i0 += base;
        c.i1 += base;
        c.i2 += base;
        c.i3 += base;
        dst.constraints.dihedrals.push(c.clone());
    }
    for c in &mut src.constraints.volumes {
        for i in &mut c.indices {
            *i += base;
        }
        dst.constraints.volumes.push(c.clone());
    }
    for c in &mut src.constraints.tetra_fem {
        for i in &mut c.indices {
            *i += base;
        }
        dst.constraints.tetra_fem.push(c.clone());
    }
    for c in &mut src.constraints.shape_matches {
        for i in &mut c.indices {
            *i += base;
        }
        dst.constraints.shape_matches.push(c.clone());
    }
    for c in &mut src.constraints.pressures {
        for t in &mut c.triangles {
            t[0] += base;
            t[1] += base;
            t[2] += base;
        }
        dst.constraints.pressures.push(c.clone());
    }
}

/// FNV-1a hash after [`SOFT_HASH_STEPS`] outer steps.
pub fn soft_canonical_hash() -> u64 {
    let mut world = soft_canonical_world();
    for _ in 0..SOFT_HASH_STEPS {
        world.step(SOFT_HASH_DT);
    }
    soft_state_hash64(&world)
}
