//! Fast-mode M26 CI microbench: fixed workloads, wall-clock medians (ms).
//!
//! Construction is outside the timed region. Prints one JSON object to stdout.
//! Compared against `docs/baselines/ci_bench.json` by `scripts/check_ci_bench.py`.

use pe_collision::{
    BoxShape, BroadPhaseKind, ContactWorld, Joint, JointSoftParams, Material, Plane, Shape,
    SoftStepParams, SolverMode, Sphere, StaticCollider, WORLD_BODY,
};
use pe_core::forces::UniformGravity;
use pe_core::{RigidBody, World};
use pe_fluid::dfsph::{DfsphParams, DfsphWorld};
use pe_fluid::particles::{fill_box, fill_box_boundary, mass_from_spacing, Particles};
use pe_fluid::pbf::{PbfParams, PbfWorld};
use pe_fluid::Dim;
use pe_math::{Vec3, STANDARD_GRAVITY};
use pe_soft::builders::{make_cloth, make_rope};
use std::time::Instant;

fn median_ms(samples: &[f64]) -> f64 {
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2] * 1e3
}

fn time_ms(repeats: usize, warmup: usize, mut body: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        body();
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t0 = Instant::now();
        body();
        samples.push(t0.elapsed().as_secs_f64());
    }
    median_ms(&samples)
}

fn pyramid(rows: usize, soft: bool, sleep: bool) -> ContactWorld {
    let h = 0.5;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    let mut mats = Vec::new();
    for row in 0..rows {
        let n = row + 1;
        let y0 = -(n as f64 - 1.0) * h;
        for i in 0..n {
            let x = y0 + 2.0 * h * (i as f64);
            let z = h + 2.0 * h * (row as f64);
            bodies.push(
                RigidBody::new(1.0)
                    .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
                    .with_position(Vec3::new(x, 0.0, z)),
            );
            shapes.push(Shape::Box(BoxShape::new(h, h, h)));
            mats.push(Material::new(0.0, 0.5));
        }
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
    cw.iterations = 20;
    if soft {
        cw.solver_mode = SolverMode::SoftStep;
        cw.soft_params = SoftStepParams::default();
    }
    cw.set_allow_sleep(sleep);
    cw
}

fn falling(n: usize) -> ContactWorld {
    let h = 0.25;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    let mut mats = Vec::new();
    let side = (n as f64).cbrt().ceil() as usize;
    let mut placed = 0usize;
    'outer: for ix in 0..side {
        for iy in 0..side {
            for iz in 0..side {
                if placed >= n {
                    break 'outer;
                }
                let x = (ix as f64 - side as f64 * 0.5) * 3.0 * h;
                let y = (iy as f64 - side as f64 * 0.5) * 3.0 * h;
                let z = 2.0 + iz as f64 * 3.0 * h;
                bodies.push(
                    RigidBody::new(1.0)
                        .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
                        .with_position(Vec3::new(x, y, z)),
                );
                shapes.push(Shape::Box(BoxShape::new(h, h, h)));
                mats.push(Material::new(0.0, 0.3));
                placed += 1;
            }
        }
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
    cw.broad_phase = BroadPhaseKind::DynamicTree;
    cw
}

fn ragdoll(links: usize) -> ContactWorld {
    let r = 0.08;
    let spacing = 0.25;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    for i in 0..links {
        bodies.push(
            RigidBody::new(1.0)
                .with_inertia(Sphere::new(r).inertia(1.0))
                .with_position(Vec3::new(0.0, 0.0, -spacing * (i as f64 + 1.0))),
        );
        shapes.push(Shape::Sphere(Sphere::new(r)));
    }
    let mut cw = ContactWorld::new(
        World::new(bodies, vec![Box::new(UniformGravity::earth())]),
        shapes,
        vec![],
        None,
    );
    cw.set_allow_sleep(false);
    for i in 0..links {
        let parent = if i == 0 { WORLD_BODY } else { i - 1 };
        let anchor_parent = if i == 0 {
            Vec3::zero()
        } else {
            Vec3::new(0.0, 0.0, -spacing * 0.5)
        };
        let anchor_child = Vec3::new(0.0, 0.0, spacing * 0.5);
        cw.joints.push(
            Joint::distance(parent, i, anchor_parent, anchor_child, spacing)
                .with_soft(JointSoftParams::default()),
        );
    }
    cw
}

fn main() {
    let mut results: Vec<(&str, f64)> = Vec::new();

    // Rigid PGS pyramid (matches Criterion scaling_pyramid/10).
    {
        let mut cw = pyramid(10, false, false);
        for _ in 0..3 {
            cw.step(1.0 / 240.0);
        }
        let ms = time_ms(5, 1, || {
            // Rebuild each sample for Criterion-like fairness on short runs.
            let mut cw = pyramid(10, false, false);
            for _ in 0..60 {
                cw.step(1.0 / 240.0);
            }
        });
        results.push(("pyramid_10_pgs_60", ms));
    }

    {
        let ms = time_ms(4, 1, || {
            let mut cw = pyramid(10, true, false);
            for _ in 0..60 {
                cw.step(1.0 / 240.0);
            }
        });
        results.push(("pyramid_10_softstep_60", ms));
    }

    {
        let ms = time_ms(4, 1, || {
            let mut cw = pyramid(10, false, true);
            for _ in 0..240 {
                cw.step(1.0 / 240.0);
            }
        });
        results.push(("pyramid_10_pgs_sleep_240", ms));
    }

    for &(n, steps, name) in &[
        (100usize, 20usize, "falling_100_tree_20"),
        (1_000, 20, "falling_1000_tree_20"),
        (10_000, 10, "falling_10000_tree_10"),
    ] {
        let ms = time_ms(3, 1, || {
            let mut cw = falling(n);
            for _ in 0..steps {
                cw.step(1.0 / 240.0);
            }
        });
        results.push((name, ms));
    }

    {
        let ms = time_ms(5, 1, || {
            let mut cw = ragdoll(16);
            for _ in 0..120 {
                cw.step(1.0 / 240.0);
            }
        });
        results.push(("ragdoll_16_120", ms));
    }

    // GJK / EPA micro (1k queries).
    {
        use pe_collision::epa::{epa_from_overlap, EpaTol};
        use pe_collision::gjk::{gjk, GjkTol};
        let a = Shape::Sphere(Sphere::new(0.5));
        let b = Shape::Sphere(Sphere::new(0.5));
        let xa = Vec3::new(-0.8, 0.0, 0.0);
        let xb = Vec3::new(0.8, 0.0, 0.0);
        let ra = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let ms = time_ms(8, 2, || {
            for _ in 0..1000 {
                let _ = std::hint::black_box(gjk(&a, xa, &ra, &b, xb, &ra, GjkTol::default()));
            }
        });
        results.push(("gjk_sphere_sphere_1k", ms));

        let box_a = Shape::Box(BoxShape::new(0.5, 0.5, 0.5));
        let box_b = Shape::Box(BoxShape::new(0.5, 0.5, 0.5));
        let xa = Vec3::new(0.0, 0.0, 0.0);
        let xb = Vec3::new(0.4, 0.0, 0.0);
        let ms = time_ms(5, 1, || {
            for _ in 0..1000 {
                let _ = std::hint::black_box(epa_from_overlap(
                    &box_a,
                    xa,
                    &ra,
                    &box_b,
                    xb,
                    &ra,
                    EpaTol::default(),
                ));
            }
        });
        results.push(("epa_box_box_1k", ms));
    }

    // Soft
    {
        let ms = time_ms(4, 1, || {
            let mut w = make_rope(
                Vec3::new(-0.5, 0.0, 1.0),
                Vec3::new(0.5, 0.0, 1.0),
                16,
                0.5,
                1e-6,
                true,
            );
            w.params.substeps = 20;
            for _ in 0..60 {
                w.step(1.0 / 60.0);
            }
        });
        results.push(("soft_rope_hang_1s", ms));

        let ms = time_ms(3, 1, || {
            let mut w = make_cloth(
                Vec3::new(0.0, 0.0, 1.0),
                8,
                8,
                0.08,
                0.08,
                0.3,
                1e-6,
                1e-4,
                true,
            );
            w.params.substeps = 10;
            w.self_collision.enabled = true;
            w.self_collision.radius = 0.05;
            for _ in 0..60 {
                w.step(1.0 / 60.0);
            }
        });
        results.push(("soft_cloth_8x8_selfcol_60", ms));
    }

    // Fluids
    {
        let ms = time_ms(3, 1, || {
            let dim = Dim::D3;
            let dx = 0.05;
            let h = 2.2 * dx;
            let rho0 = 1000.0;
            let mass = mass_from_spacing(rho0, dx, dim);
            let mut parts = Particles::new(dim, rho0, mass, h);
            fill_box(
                &mut parts,
                Vec3::new(0.1, 0.1, 0.1),
                8,
                8,
                8,
                dx,
                Vec3::zero(),
            );
            fill_box_boundary(
                &mut parts,
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.6, 0.6, 0.6),
                dx,
                1,
            );
            let params = PbfParams {
                gravity: Vec3::new(0.0, 0.0, -STANDARD_GRAVITY),
                iterations: 2,
                ..PbfParams::default()
            };
            let mut world = PbfWorld::new(parts, params);
            for _ in 0..10 {
                world.step(1.0 / 120.0);
            }
        });
        results.push(("pbf_512_10_steps", ms));

        let ms = time_ms(2, 1, || {
            let dim = Dim::D3;
            let dx = 0.05;
            let h = 2.2 * dx;
            let rho0 = 1000.0;
            let mass = mass_from_spacing(rho0, dx, dim);
            let mut parts = Particles::new(dim, rho0, mass, h);
            fill_box(
                &mut parts,
                Vec3::new(0.1, 0.1, 0.1),
                6,
                6,
                6,
                dx,
                Vec3::zero(),
            );
            fill_box_boundary(
                &mut parts,
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.5, 0.5, 0.5),
                dx,
                1,
            );
            let params = DfsphParams {
                gravity: Vec3::new(0.0, 0.0, -STANDARD_GRAVITY),
                max_density_iters: 10,
                max_divergence_iters: 10,
                ..DfsphParams::default()
            };
            let mut world = DfsphWorld::new(parts, params);
            for _ in 0..10 {
                world.step(1.0 / 120.0);
            }
        });
        results.push(("dfsph_216_10_steps", ms));
    }

    print!("{{");
    print!("\"host\":\"ci-vm\",\"unit\":\"ms_median\"");
    for (name, ms) in &results {
        print!(",\"{name}\":{ms:.6}");
    }
    println!("}}");
}
