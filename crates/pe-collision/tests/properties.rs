//! Property / fuzz-style tests for contact worlds, joints, sleep, and GJK/EPA (M22).
//!
//! Case count defaults stay CI-friendly; override with `PROPTEST_CASES`.

use pe_collision::{
    epa, gjk, BoxShape, Capsule, ContactWorld, ConvexHull, GjkTol, Joint, Material, Plane, Shape,
    Sphere, StaticCollider, WORLD_BODY,
};
use pe_core::diagnostics::{kinetic_energy, linear_momentum, state_hash64};
use pe_core::forces::UniformGravity;
use pe_core::{RigidBody, World};
use pe_math::{Quat, Vec3, Xoshiro256StarStar};
use proptest::prelude::*;

fn eye() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn state_finite(cw: &ContactWorld) -> bool {
    cw.world.state.all_finite()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn random_mixed_shapes_world_stays_finite(
        seed in any::<u64>(),
        n_dyn in 1usize..5,
        steps in 5usize..40,
        allow_sleep in any::<bool>(),
        with_joint in any::<bool>(),
    ) {
        let mut rng = Xoshiro256StarStar::seed(seed);
        let mut bodies = Vec::new();
        let mut shapes = Vec::new();
        let mut mats = Vec::new();
        for i in 0..n_dyn {
            let mass = rng.uniform(0.5, 2.0);
            let kind = (rng.next_u64() % 3) as u8;
            let (shape, inertia) = match kind {
                0 => {
                    let r = rng.uniform(0.08, 0.35);
                    let s = Shape::Sphere(Sphere::new(r));
                    (s, Sphere::new(r).inertia(mass))
                }
                1 => {
                    let h = rng.uniform(0.1, 0.35);
                    let s = Shape::Box(BoxShape::new(h, h, h));
                    (s, BoxShape::new(h, h, h).inertia(mass))
                }
                _ => {
                    let r = rng.uniform(0.06, 0.2);
                    let hh = rng.uniform(0.05, 0.25);
                    let s = Shape::Capsule(Capsule::new(r, hh));
                    (s, Capsule::new(r, hh).inertia(mass))
                }
            };
            bodies.push(
                RigidBody::new(mass)
                    .with_inertia(inertia)
                    .with_position(Vec3::new(
                        rng.uniform(-1.0, 1.0),
                        rng.uniform(-1.0, 1.0),
                        0.5 + 0.8 * i as f64,
                    ))
                    .with_orientation(Quat::from_axis_angle(
                        Vec3::unit_y(),
                        rng.uniform(-0.4, 0.4),
                    )),
            );
            shapes.push(shape);
            mats.push(Material::new(0.0, 0.4));
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
        cw.set_allow_sleep(allow_sleep);
        if with_joint && n_dyn >= 2 {
            cw.add_joint(Joint::distance(0, 1, Vec3::zero(), Vec3::zero(), 0.8));
        } else if with_joint {
            cw.add_joint(Joint::distance(0, WORLD_BODY, Vec3::zero(), Vec3::new(0.0, 0.0, 1.0), 1.0));
        }
        for _ in 0..steps {
            cw.step(1.0 / 240.0);
            prop_assert!(state_finite(&cw), "NaN/Inf escaped contact step");
        }
    }

    #[test]
    fn gjk_degenerate_and_extreme_scales_finite(
        ax in -1.0f64..1.0, ay in -1.0f64..1.0, az in -1.0f64..1.0,
        scale_exp in -6i32..6,
        coincident in any::<bool>(),
    ) {
        let scale = pe_math::libm_ops::pow(10.0, scale_exp as f64);
        let ra = 0.5 * scale;
        let rb = 0.3 * scale;
        // Zero-volume / flat box half-extent on one axis (near-degenerate).
        let flat = scale * 1e-14;
        let sa = Shape::Box(BoxShape::new(ra, flat.max(1e-18 * scale.abs().max(1e-30)), ra));
        let sb = Shape::Sphere(Sphere::new(rb.max(1e-18)));
        let xa = Vec3::new(ax * scale, ay * scale, az * scale);
        let xb = if coincident {
            xa
        } else {
            xa + Vec3::new(scale, 0.0, 0.0)
        };
        let g = gjk(&sa, xa, &eye(), &sb, xb, &eye(), GjkTol::default());
        prop_assert!(g.distance.is_finite());
        prop_assert!(g.point_a.is_finite());
        prop_assert!(g.point_b.is_finite());
        prop_assert!(g.normal.is_finite() || g.intersecting);
        if g.intersecting {
            let e = epa(
                &sa,
                xa,
                &eye(),
                &sb,
                xb,
                &eye(),
                GjkTol::default(),
                pe_collision::EpaTol::default(),
            );
            if let Some(er) = e {
                prop_assert!(er.depth.is_finite());
                prop_assert!(er.normal.is_finite());
                prop_assert!(er.point_a.is_finite());
                prop_assert!(er.point_b.is_finite());
            }
        }
    }

    #[test]
    fn gjk_coincident_points_and_tiny_hull_no_panic(
        noise in -1e-15f64..1e-15,
    ) {
        // Nearly coincident vertices → near zero-volume hull attempt.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0 + noise, noise, 0.0),
            Vec3::new(noise, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0 + noise),
        ];
        // Hull build may fail on degeneracy — that is OK (typed error).
        if let Ok(hull) = ConvexHull::from_points_with_radius(&verts, 0.0) {
            let s = Shape::Convex(hull);
            let g = gjk(
                &s,
                Vec3::zero(),
                &eye(),
                &Shape::Sphere(Sphere::new(0.5)),
                Vec3::new(2.0, 0.0, 0.0),
                &eye(),
                GjkTol::default(),
            );
            prop_assert!(g.distance.is_finite());
        }
    }
}

#[test]
fn inelastic_stack_kinetic_energy_does_not_explode() {
    // e=0 resting stack: KE after settle must stay below a documented bound.
    let h = 0.25;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    let mut mats = Vec::new();
    for i in 0..4 {
        bodies.push(
            RigidBody::new(1.0)
                .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
                .with_position(Vec3::new(0.0, 0.0, h + 2.0 * h * i as f64)),
        );
        shapes.push(Shape::Box(BoxShape::new(h, h, h)));
        mats.push(Material::new(0.0, 0.5));
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
    cw.set_allow_sleep(false);
    for _ in 0..1_200 {
        cw.step(1.0 / 240.0);
        assert!(state_finite(&cw));
    }
    let ke = kinetic_energy(&cw.world, &cw.world.state);
    // Bound: after 5 s, stack KE must be < 0.5 J (jitter), not growing.
    assert!(ke < 0.5, "inelastic stack KE={ke} exceeds 0.5 J bound");
}

#[test]
fn symmetric_sphere_collision_momentum_conserved() {
    let bodies = vec![
        RigidBody::new(1.0)
            .with_inertia(Sphere::new(0.2).inertia(1.0))
            .with_position(Vec3::new(-1.0, 0.0, 1.0))
            .with_velocity(Vec3::new(2.0, 0.0, 0.0)),
        RigidBody::new(1.0)
            .with_inertia(Sphere::new(0.2).inertia(1.0))
            .with_position(Vec3::new(1.0, 0.0, 1.0))
            .with_velocity(Vec3::new(-2.0, 0.0, 0.0)),
    ];
    let mut cw = ContactWorld::new(
        World::new(bodies, vec![]),
        vec![
            Shape::Sphere(Sphere::new(0.2)),
            Shape::Sphere(Sphere::new(0.2)),
        ],
        vec![],
        Some(vec![Material::new(1.0, 0.0), Material::new(1.0, 0.0)]),
    );
    cw.set_allow_sleep(false);
    let p0 = linear_momentum(&cw.world, &cw.world.state);
    for _ in 0..200 {
        cw.step(1.0 / 240.0);
    }
    let p1 = linear_momentum(&cw.world, &cw.world.state);
    assert!(
        (p1 - p0).norm() < 1e-9,
        "symmetric contact ΔP={}",
        (p1 - p0).norm()
    );
}

#[test]
fn determinism_hash_repeat_and_mid_run_scene_restore() {
    use pe_scene::{ForceDesc, SceneDocument};

    let build = || {
        let h = 0.25;
        let bodies = vec![
            RigidBody::new(1.0)
                .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
                .with_position(Vec3::new(0.0, 0.0, h + 0.01)),
            RigidBody::new(1.0)
                .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
                .with_position(Vec3::new(0.0, 0.0, 3.0 * h + 0.01)),
        ];
        ContactWorld::new(
            World::new(bodies, vec![Box::new(UniformGravity::earth())]),
            vec![
                Shape::Box(BoxShape::new(h, h, h)),
                Shape::Box(BoxShape::new(h, h, h)),
            ],
            vec![
                StaticCollider::new(Shape::Plane(Plane::new(Vec3::unit_z(), 0.0)))
                    .with_material(Material::new(0.0, 1.0)),
            ],
            Some(vec![Material::new(0.0, 0.5), Material::new(0.0, 0.5)]),
        )
    };

    let mut a = build();
    a.set_allow_sleep(false);
    for _ in 0..150 {
        a.step(1.0 / 240.0);
    }
    let forces = vec![ForceDesc::UniformGravity {
        g: [0.0, 0.0, -pe_math::STANDARD_GRAVITY],
    }];
    let mid = SceneDocument::capture(&a, forces).unwrap();
    let bin = mid.to_binary().unwrap();
    let mut b = SceneDocument::from_binary(&bin)
        .unwrap()
        .instantiate()
        .unwrap();
    for _ in 0..150 {
        a.step(1.0 / 240.0);
        b.step(1.0 / 240.0);
    }
    assert_eq!(state_hash64(&a.world.state), state_hash64(&b.world.state));

    // Fresh repeat
    let mut c = build();
    c.set_allow_sleep(false);
    for _ in 0..300 {
        c.step(1.0 / 240.0);
    }
    assert_eq!(state_hash64(&a.world.state), state_hash64(&c.world.state));
}
