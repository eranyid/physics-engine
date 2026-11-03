//! Canonical blast determinism scenarios (cross-OS hashes).
//!
//! Pyramid of boxes hit by a Friedlander blast load; FNV-1a hash of packed
//! world state — must stay in sync with `docs/DETERMINISM.md` and the
//! `blast_hash` binary / Python binding.
//!
//! Additional M20 lines: ExternalLoads KB loading, Gurney/Mott fragment volley,
//! Voronoi cohesive destruction under blast.

use std::sync::Arc;

use pe_ballistics::projectile::ProjectileSystem;
use pe_collision::{ContactWorld, JointSoftParams, Material, Shape, StaticCollider};
use pe_core::diagnostics::state_hash64;
use pe_core::forces::UniformGravity;
use pe_core::{inertia_solid_box, ExternalLoads, RigidBody, World};
use pe_math::{Vec3, STANDARD_GRAVITY};

use crate::compound::{
    chunks_to_bodies, chunks_to_box_shapes, cohesive_joints_from_partition, voronoi_box_partition,
};
use crate::fragments::{detonate_cased_charge, GurneyGeometry};
use crate::kinney::peak_overpressure;
use crate::loading::{
    accumulate_blast_external_all, apply_blast_impulse, BlastLoadParams, BlastSource, ExposedFace,
    OverpressureModel, ReflectionMix,
};
use crate::occlusion::OccluderAabb;
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
///
/// **Locked** — must remain `b51e11e89d4dd055` across OS (legacy Kinney impulse path).
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

/// M20: same pyramid geometry driven through [`ExternalLoads`] with Kingery–Bulmash
/// incident+reflected + dynamic-pressure drag (no occlusion).
pub fn blast_external_loads_hash() -> u64 {
    let (base, src, faces) = blast_pyramid_world();
    let n = base.n;
    let loads = Arc::new(ExternalLoads::with_capacity(n));
    let bodies: Vec<RigidBody> = (0..n)
        .map(|i| {
            RigidBody::new(base.mass[i])
                .with_position(base.state.x[i])
                .with_velocity(base.state.v[i])
                .with_orientation(base.state.q[i])
                .with_inertia(base.inertia_body[i])
        })
        .collect();
    let gens: Vec<Box<dyn pe_core::ForceGenerator>> = vec![
        Box::new(UniformGravity::new(Vec3::new(0.0, 0.0, -STANDARD_GRAVITY))),
        Box::new(Arc::clone(&loads)),
    ];
    let mut world = World::new(bodies, gens);
    let params = BlastLoadParams {
        overpressure: OverpressureModel::KingeryBulmash,
        reflection: ReflectionMix::IncidentReflected,
        drag_cd: 1.0,
        shadow_factor: 0.0,
        gamma: crate::GAMMA_AIR,
    };
    let occluders: [OccluderAabb; 0] = [];
    let rk4 = pe_core::integrators::rk4;
    for _ in 0..BLAST_HASH_STEPS {
        let t = world.t;
        loads.clear();
        accumulate_blast_external_all(&loads, &world, &src, &faces, t, &params, &occluders);
        world.step(BLAST_HASH_DT, rk4);
    }
    state_hash64(&world.state)
}

/// M20: Gurney/Mott fragments into [`ProjectileSystem`] (analytic drag, no spin).
pub fn blast_fragments_hash() -> u64 {
    let mut sys = ProjectileSystem::new(0.0);
    sys.enable_spin_effects = false;
    let (_v, specs) = detonate_cased_charge(
        &mut sys,
        Vec3::zero(),
        2440.0,
        0.5,
        GurneyGeometry::Cylindrical,
        1.0,
        0.02,
        32,
        0xFBA5_7001,
        1,
        0.01,
    )
    .expect("gurney");
    let _ = specs;
    let dt = 1.0 / 120.0;
    for _ in 0..60 {
        sys.step(dt, &[], &[]);
    }
    // FNV-1a over (id, x, v, mass, alive) in ascending shot-id order.
    let mut h: u64 = 0xcbf29ce484222325;
    const P: u64 = 0x100000001b3;
    let mut feed = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(P);
    };
    for p in &sys.projectiles {
        feed(p.id.0);
        for c in p.x.to_array() {
            feed(c.to_bits());
        }
        for c in p.v.to_array() {
            feed(c.to_bits());
        }
        feed(p.mass.to_bits());
        feed(u64::from(p.alive));
    }
    h
}

/// M20: Voronoi 4-chunk compound on a plane, blast load, breakable bonds.
pub fn blast_destruction_hash() -> u64 {
    let half = Vec3::new(0.5, 0.5, 0.5);
    let part = voronoi_box_partition(Vec3::new(0.0, 0.0, 1.0), half, 4, 2000.0, 6, 0xDE57_0001);
    let bodies = chunks_to_bodies(&part);
    let boxes = chunks_to_box_shapes(&part);
    let shapes: Vec<Shape> = boxes.into_iter().map(Shape::Box).collect();
    let mats = vec![
        Material {
            restitution: 0.0,
            friction: 0.4,
        };
        bodies.len()
    ];
    let loads = Arc::new(ExternalLoads::with_capacity(bodies.len()));
    let gens: Vec<Box<dyn pe_core::ForceGenerator>> = vec![
        Box::new(UniformGravity::new(Vec3::new(0.0, 0.0, -STANDARD_GRAVITY))),
        Box::new(Arc::clone(&loads)),
    ];
    let world = World::new(bodies, gens);
    let statics = vec![StaticCollider::new(Shape::Plane(pe_collision::Plane::new(
        Vec3::new(0.0, 0.0, 1.0),
        0.0,
    )))
    .with_material(Material::new(0.0, 0.8))];
    let mut cw = ContactWorld::new(world, shapes, statics, Some(mats));
    cw.set_allow_sleep(true);
    // Map site → body (= chunk list index, since chunks_to_bodies preserves order).
    let site_to_body = part.site_to_chunk.clone();
    let joints = cohesive_joints_from_partition(
        &part,
        |site| site_to_body[site],
        5.0e6, // Pa — nominal brittle demo (unverified)
        JointSoftParams {
            hertz: 120.0,
            zeta: 0.8,
        },
    );
    for j in joints {
        cw.add_joint(j);
    }
    // Exposed −x faces at each chunk COM.
    let faces: Vec<ExposedFace> = part
        .chunks
        .iter()
        .enumerate()
        .map(|(i, c)| ExposedFace {
            body: i,
            center: c.com + Vec3::new(-c.half_extents.x, 0.0, 0.0),
            normal: Vec3::new(-1.0, 0.0, 0.0),
            area: (2.0 * c.half_extents.y) * (2.0 * c.half_extents.z),
        })
        .collect();
    let src = BlastSource {
        origin: Vec3::new(-3.0, 0.0, 1.0),
        w_eq_kg: 5.0,
        t0: 0.0,
        p0: 101_325.0,
        td_at_z1: 0.008,
        decay_b: 1.0,
    };
    let params = BlastLoadParams::default();
    let occluders: [OccluderAabb; 0] = [];
    let dt = 1.0 / 120.0;
    for _ in 0..180 {
        let t = cw.world.t;
        loads.clear();
        accumulate_blast_external_all(&loads, &cw.world, &src, &faces, t, &params, &occluders);
        cw.step(dt);
    }
    state_hash64(&cw.world.state)
}
