//! Canonical soft-body and materials-track determinism scenarios (cross-OS hash).

use pe_math::Vec3;

use crate::builders::{make_cloth, make_rope, make_soft_cube};
use crate::collision::{StaticColliders, StaticPlane};
use crate::constraint::{DistanceConstraint, Multiplier};
use crate::fracture::{apply_fracture, FractureCriterion};
use crate::material::material_by_name;
use crate::particle::Particle;
use crate::plasticity::{return_map_uniaxial, J2Params, UniaxialPlasticState};
use crate::solver::{soft_state_hash64, SoftWorld};
use crate::thermal::{init_rod_sine_mode, ThermalMesh, ThermalProps};

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

/// Outer steps in the materials-track hash scenario.
pub const MATERIALS_HASH_STEPS: usize = 80;
/// Outer timestep [s] for materials-track hash.
pub const MATERIALS_HASH_DT: f64 = 1.0 / 120.0;

/// FNV-1a over a mixed plasticity + thermal + fracture scenario.
///
/// Packed payload (little-endian IEEE bits, fixed order):
/// 1. Uniaxial J2 stress history (41 samples, ε = 0…2ε_y).
/// 2. 1-D rod temperatures after conduction (17 nodes).
/// 3. Soft bar particle `(x‖v)` after stretch-to-fracture + a few XPBD steps.
pub fn materials_canonical_hash() -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    let mut feed = |x: f64| {
        for b in x.to_bits().to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };

    // --- plasticity ---
    let p = J2Params {
        youngs_modulus: 2.0e11,
        poisson_ratio: 0.3,
        yield_strength: 2.5e8,
        hardening_modulus: 2.0e9,
    };
    let eps_y = p.yield_strength / p.youngs_modulus;
    let mut state = UniaxialPlasticState::default();
    for k in 0..=40 {
        let eps = (k as f64 / 40.0) * (2.0 * eps_y);
        state = return_map_uniaxial(&p, state, eps).state;
        feed(state.stress);
        feed(state.plastic_strain);
        feed(state.eq_plastic_strain);
    }

    // --- thermal rod ---
    let props = ThermalProps {
        density: 8960.0,
        specific_heat: 385.0,
        conductivity: 400.0,
        expansion: 16.5e-6,
    };
    let mut mesh = ThermalMesh::rod_1d(17, 1.0, 0.01, props, 300.0);
    init_rod_sine_mode(&mut mesh, 1.0, 50.0);
    let mut fixed = vec![false; mesh.len()];
    fixed[0] = true;
    fixed[mesh.len() - 1] = true;
    let n_th = mesh.len();
    mesh.temperature[0] = 0.0;
    mesh.temperature[n_th - 1] = 0.0;
    let dt_th = mesh.max_stable_dt();
    for _ in 0..MATERIALS_HASH_STEPS {
        let _ = mesh.step_explicit(dt_th, &fixed);
    }
    for &t in &mesh.temperature {
        feed(t);
    }
    feed(mesh.total_energy());

    // --- fracture bar ---
    let e = 1.0e7;
    let l0 = 1.0;
    let thresh = 5.0e5; // Pa
    let mut w = SoftWorld::new();
    w.params.substeps = 8;
    w.params.iterations = 1;
    w.params.gravity = Vec3::zero();
    let i0 = w.add_particle(Particle::pinned(Vec3::zero()));
    let i1 = w.add_particle(Particle::dynamic(Vec3::new(l0, 0.0, 0.0), 1.0));
    w.constraints.distances.push(DistanceConstraint {
        i: i0,
        j: i1,
        rest_length: l0,
        compliance: 1.0 / (e * 0.01 / l0), // A=0.01
        lambda: Multiplier::default(),
        broken: false,
    });
    // Displace free end so axial stress exceeds threshold, then fracture.
    let eps_break = thresh / e * 1.05;
    w.particles[i1].x.x = l0 * (1.0 + eps_break);
    let _events = apply_fracture(
        &w.particles,
        &mut w.constraints,
        FractureCriterion::MaxPrincipalStress(thresh),
        e,
    );
    for _ in 0..MATERIALS_HASH_STEPS {
        w.step(MATERIALS_HASH_DT);
    }
    for x in w.pack() {
        feed(x);
    }
    feed(if w.constraints.distances[0].broken {
        1.0
    } else {
        0.0
    });

    hash
}
