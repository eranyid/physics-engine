//! Canonical blast determinism scenario (cross-OS hash).
//!
//! Pyramid of boxes hit by a Friedlander blast load; FNV-1a hash of packed
//! world state — must stay in sync with `docs/DETERMINISM.md` and the
//! `blast_hash` binary / Python binding.

use pe_core::diagnostics::state_hash64;
use pe_core::forces::UniformGravity;
use pe_core::{inertia_solid_box, RigidBody, World};
use pe_math::{Vec3, STANDARD_GRAVITY};

use crate::kinney::peak_overpressure;
use crate::loading::{apply_blast_impulse, BlastSource, ExposedFace};
use crate::scaling::scaled_distance;

/// Number of contact-free RK4 steps in the canonical blast hash scenario.
pub const BLAST_HASH_STEPS: usize = 200;
/// Timestep [s].
pub const BLAST_HASH_DT: f64 = 1.0 / 240.0;

/// Build the canonical blast-on-pyramid world (no contact solver — impulse loads only).
pub fn blast_pyramid_world() -> (World, BlastSource, Vec<ExposedFace>) {
    // 6-box pyramid (rows 1+2+3) of half-extent 0.25 m boxes.
    let h = 0.25;
    let mass = 10.0;
    let inertia = inertia_solid_box(mass, 2.0 * h, 2.0 * h, 2.0 * h);
    let mut bodies = Vec::new();
    let mut faces = Vec::new();
    let mut id = 0usize;
    for row in 0..3 {
        let n = row + 1;
        for j in 0..n {
            let x = (j as f64 - 0.5 * (n as f64 - 1.0)) * (2.0 * h + 0.02);
            let z = h + row as f64 * (2.0 * h + 0.01);
            let pos = Vec3::new(x, 0.0, z);
            bodies.push(
                RigidBody::new(mass)
                    .with_position(pos)
                    .with_inertia(inertia),
            );
            faces.push(ExposedFace {
                body: id,
                center: pos + Vec3::new(-h, 0.0, 0.0),
                normal: Vec3::new(-1.0, 0.0, 0.0),
                area: (2.0 * h) * (2.0 * h),
            });
            id += 1;
        }
    }
    let gens: Vec<Box<dyn pe_core::ForceGenerator>> = vec![Box::new(UniformGravity::new(
        Vec3::new(0.0, 0.0, -STANDARD_GRAVITY),
    ))];
    let world = World::new(bodies, gens);
    let src = BlastSource {
        origin: Vec3::new(-2.0, 0.0, 0.5),
        w_eq_kg: 1.0,
        t0: 0.0,
        p0: 101_325.0,
        td_at_z1: 0.008,
        decay_b: 1.0,
    };
    (world, src, faces)
}

/// Run the canonical scenario and return the FNV-1a state hash.
pub fn blast_pyramid_hash() -> u64 {
    let (mut world, src, faces) = blast_pyramid_world();
    let rk4 = pe_core::integrators::rk4;
    for _ in 0..BLAST_HASH_STEPS {
        let t = world.t;
        for f in &faces {
            apply_blast_impulse(&mut world, &src, f, t, BLAST_HASH_DT);
        }
        world.step(BLAST_HASH_DT, rk4);
    }
    state_hash64(&world.state)
}

/// Sanity helper used by tests: peak overpressure at pyramid front.
pub fn blast_pyramid_front_overpressure() -> f64 {
    let (_, src, faces) = blast_pyramid_world();
    let z = scaled_distance((faces[0].center - src.origin).norm(), src.w_eq_kg, 1.0).unwrap();
    peak_overpressure(z, src.p0)
}

/// Grid of the canonical 3-D blast hash scenario.
pub const BLAST3D_HASH_DIMS: (usize, usize, usize) = (32, 24, 16);
/// Cell size [m] of the 3-D hash scenario.
pub const BLAST3D_HASH_DX: f64 = 0.25;
/// Steps of the 3-D hash scenario.
pub const BLAST3D_HASH_STEPS: usize = 60;

/// Canonical 3-D scene: 8 × 6 × 4 m, ground, an L-shaped wall with a gap,
/// 0.2 kg TNT-eq at 0.5 m height.
pub fn blast3d_corridor_scene() -> crate::blast3d::BlastScene {
    use crate::blast3d::BlastScene;
    let (nx, ny, nz) = BLAST3D_HASH_DIMS;
    let dx = BLAST3D_HASH_DX;
    let max = Vec3::new(nx as f64 * dx, ny as f64 * dx, nz as f64 * dx);
    let mut s = BlastScene::new(Vec3::zero(), max, dx);
    s.fill_ground(0.25);
    // Wall along y at x = 4.5 m, 0.25 m thick, 2.5 m high, with a 1 m gap.
    s.fill_aabb(Vec3::new(4.4, 0.0, 0.0), Vec3::new(4.6, 6.0, 2.5));
    s.clear_aabb(Vec3::new(4.3, 2.5, 0.25), Vec3::new(4.7, 3.5, 2.0));
    // Return wall along x at y = 4.5 m.
    s.fill_aabb(Vec3::new(4.6, 4.4, 0.0), Vec3::new(7.0, 4.6, 2.5));
    s.detonate(Vec3::new(2.0, 3.0, 0.75), 0.2);
    s
}

/// FNV-1a hash of the 3-D scene after [`BLAST3D_HASH_STEPS`] CFL steps.
pub fn blast3d_corridor_hash() -> u64 {
    let mut s = blast3d_corridor_scene();
    for _ in 0..BLAST3D_HASH_STEPS {
        s.step();
    }
    crate::blast3d::grid_hash(&s.grid)
}
