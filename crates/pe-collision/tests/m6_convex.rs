//! M6 analytic oracles: GJK/EPA, hulls, manifolds, convex CCD, determinism.

use approx::assert_abs_diff_eq;
use pe_collision::ccd::{conservative_advancement_toi, CcdParams};
use pe_collision::epa::{epa_from_overlap, EpaTol};
use pe_collision::gjk::{gjk, GjkTol};
use pe_collision::hull::ConvexHull;
use pe_collision::narrowphase::collide;
use pe_collision::{
    canonical_convex_hash, BoxShape, Capsule, ContactWorld, Material, Shape, Sphere,
    StaticCollider, CANONICAL_CONVEX_HASH,
};
use pe_core::forces::UniformGravity;
use pe_core::{RigidBody, World};
use pe_math::{Quat, Vec3};
use proptest::prelude::*;

fn eye() -> [[f64; 3]; 3] {
    Quat::identity().to_matrix()
}

#[test]
fn gjk_sphere_sphere_distance_oracle() {
    let a = Shape::Sphere(Sphere::new(0.3));
    let b = Shape::Sphere(Sphere::new(0.7));
    let r = gjk(
        &a,
        Vec3::zero(),
        &eye(),
        &b,
        Vec3::new(2.0, 0.0, 0.0),
        &eye(),
        GjkTol::default(),
    );
    assert!(!r.intersecting);
    assert_abs_diff_eq!(r.distance, 1.0, epsilon = 1e-9);
}

#[test]
fn gjk_sphere_box_and_box_box_axes() {
    let s = Shape::Sphere(Sphere::new(0.25));
    let b = Shape::Box(BoxShape::new(0.5, 0.5, 0.5));
    let r = gjk(
        &s,
        Vec3::new(0.0, 0.0, 2.0),
        &eye(),
        &b,
        Vec3::zero(),
        &eye(),
        GjkTol::default(),
    );
    // Sphere surface at z=1.75, box top at 0.5 → distance 1.25
    assert_abs_diff_eq!(r.distance, 1.25, epsilon = 1e-8);

    let a = Shape::Box(BoxShape::new(1.0, 0.5, 0.5));
    let c = Shape::Box(BoxShape::new(0.5, 1.0, 0.5));
    let r = gjk(
        &a,
        Vec3::zero(),
        &eye(),
        &c,
        Vec3::new(0.0, 3.0, 0.0),
        &eye(),
        GjkTol::default(),
    );
    // Separated along y: extents 0.5+1.0, centres 3 apart → dist 1.5
    assert_abs_diff_eq!(r.distance, 1.5, epsilon = 1e-8);
}

#[test]
fn epa_box_sphere_known_overlap() {
    let a = Shape::Sphere(Sphere::new(1.0));
    let b = Shape::Box(BoxShape::new(1.0, 1.0, 1.0));
    // Sphere centre at x=1.5 → penetration 0.5 along x
    let r = epa_from_overlap(
        &a,
        Vec3::new(1.5, 0.0, 0.0),
        &eye(),
        &b,
        Vec3::zero(),
        &eye(),
        EpaTol::default(),
    )
    .unwrap();
    assert_abs_diff_eq!(r.depth, 0.5, epsilon = 5e-4);
    assert!(r.normal.x.abs() > 0.9);
}

#[test]
fn capsule_box_contact_normal_and_depth() {
    let cap = Capsule::new(0.2, 0.5);
    let box_s = BoxShape::new(0.5, 0.5, 0.5);
    // Capsule along z at x=0.6: radius 0.2 reaches to 0.4; box to 0.5 → depth 0.1
    let cs = collide(
        &Shape::Capsule(cap),
        Vec3::new(0.6, 0.0, 0.0),
        &eye(),
        &Shape::Box(box_s),
        Vec3::zero(),
        &eye(),
    );
    assert!(!cs.is_empty(), "expected capsule-box contacts");
    assert!(
        cs[0].depth > 0.05 && cs[0].depth < 0.2,
        "depth={}",
        cs[0].depth
    );
    // Capsule at +x, box at origin ⇒ normal A→B points −x.
    assert!(cs[0].normal.x < -0.9, "n={:?}", cs[0].normal);
}

#[test]
fn capsule_capsule_manifold_sane() {
    let cap = Capsule::new(0.1, 0.4);
    let cs = collide(
        &Shape::Capsule(cap),
        Vec3::zero(),
        &eye(),
        &Shape::Capsule(cap),
        Vec3::new(0.15, 0.0, 0.0),
        &eye(),
    );
    assert!(!cs.is_empty());
    assert!(cs.len() <= 4);
    assert!(cs[0].depth > 0.0);
    assert!((cs[0].normal.norm() - 1.0).abs() < 1e-9);
}

#[test]
fn convex_hull_euler_and_validity() {
    let h = ConvexHull::from_box(1.0, 1.0, 1.0, 0.0).unwrap();
    h.validate().unwrap();
    let chi = h.vertices.len() as i32 - h.edges.len() as i32 + h.faces.len() as i32;
    assert_eq!(chi, 2);
    let tet = ConvexHull::from_points(&[
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
    ])
    .unwrap();
    assert_eq!(tet.vertices.len(), 4);
    tet.validate().unwrap();
}

#[test]
fn convex_convex_manifold_point_count() {
    let ha = ConvexHull::from_box(0.5, 0.5, 0.5, 0.0).unwrap();
    let hb = ConvexHull::from_box(0.5, 0.5, 0.5, 0.0).unwrap();
    let cs = collide(
        &Shape::Convex(ha),
        Vec3::zero(),
        &eye(),
        &Shape::Convex(hb),
        Vec3::new(0.6, 0.0, 0.0),
        &eye(),
    );
    assert!(!cs.is_empty());
    assert!(cs.len() <= 4);
    for c in &cs {
        assert!(c.depth >= 0.0);
        assert!((c.normal.norm() - 1.0).abs() < 1e-6);
    }
}

#[test]
fn rounded_hull_inflates_aabb() {
    let h = ConvexHull::from_box(0.5, 0.5, 0.5, 0.1).unwrap();
    let aabb = h.aabb(Vec3::zero(), &eye());
    assert_abs_diff_eq!(aabb.hi.x, 0.6, epsilon = 1e-12);
    assert_abs_diff_eq!(aabb.lo.x, -0.6, epsilon = 1e-12);
}

#[test]
fn ccd_fast_box_vs_thin_static_no_tunnel() {
    // Dynamic 0.2-cube at z=1, vz=−500; thin static box half-height 0.005 at z=0.
    // At 60 Hz without CCD the dynamic would travel ~8.3 m/step and tunnel.
    let mut body = RigidBody::new(1.0)
        .with_inertia(BoxShape::new(0.1, 0.1, 0.1).inertia(1.0))
        .with_position(Vec3::new(0.0, 0.0, 1.0));
    body.velocity = Vec3::new(0.0, 0.0, -500.0);
    let thin = BoxShape::new(2.0, 2.0, 0.005);
    let mut cw = ContactWorld::new(
        World::new(vec![body], vec![]),
        vec![Shape::Box(BoxShape::new(0.1, 0.1, 0.1))],
        vec![StaticCollider::new(Shape::Box(thin)).with_position(Vec3::zero())],
        Some(vec![Material::new(0.0, 0.0)]),
    );
    cw.ccd.toi = true;
    cw.set_bullet(0, true);
    let dt = 1.0 / 60.0;
    cw.step(dt);
    // Must not be below the thin slab (tunnel).
    assert!(
        cw.world.state.x[0].z > -0.05,
        "tunneled: z={}",
        cw.world.state.x[0].z
    );
    // And should have been stopped near the top of the thin box + half-extent.
    assert!(cw.world.state.x[0].z < 0.5, "z={}", cw.world.state.x[0].z);
}

#[test]
fn ccd_conservative_advancement_toi_positive() {
    let a = Shape::Box(BoxShape::new(0.1, 0.1, 0.1));
    let b = Shape::Box(BoxShape::new(1.0, 1.0, 0.01));
    let params = CcdParams {
        toi: true,
        toi_max_iters: 24,
        toi_tolerance: 1e-4,
        ..CcdParams::default()
    };
    let hit = conservative_advancement_toi(
        &a,
        Vec3::new(0.0, 0.0, 1.0),
        &eye(),
        Vec3::new(0.0, 0.0, -200.0),
        Vec3::zero(),
        &b,
        Vec3::zero(),
        &eye(),
        1.0 / 60.0,
        &params,
    );
    assert!(hit.is_some(), "expected TOI");
    let (t, n, _, _) = hit.unwrap();
    assert!(t > 0.0 && t <= 1.0 / 60.0);
    assert!(n.z.abs() > 0.5);
}

#[test]
fn convex_hash_matches_locked() {
    assert_eq!(canonical_convex_hash(), CANONICAL_CONVEX_HASH);
    assert_eq!(canonical_convex_hash(), canonical_convex_hash());
}

#[test]
fn existing_collision_hash_unchanged() {
    assert_eq!(
        pe_collision::canonical_collision_hash(),
        pe_collision::CANONICAL_COLLISION_HASH
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn gjk_distance_nonnegative_and_symmetric(
        ax in -2.0f64..2.0, ay in -2.0f64..2.0, az in -2.0f64..2.0,
        bx in -2.0f64..2.0, by in -2.0f64..2.0, bz in -2.0f64..2.0,
        ra in 0.05f64..0.8, rb in 0.05f64..0.8,
    ) {
        let sa = Shape::Sphere(Sphere::new(ra));
        let sb = Shape::Sphere(Sphere::new(rb));
        let xa = Vec3::new(ax, ay, az);
        let xb = Vec3::new(bx, by, bz);
        let ab = gjk(&sa, xa, &eye(), &sb, xb, &eye(), GjkTol::default());
        let ba = gjk(&sb, xb, &eye(), &sa, xa, &eye(), GjkTol::default());
        prop_assert!(ab.distance >= -1e-12);
        prop_assert!((ab.distance - ba.distance).abs() < 1e-8);
        if !ab.intersecting && ab.distance > 1e-8 {
            prop_assert!((ab.normal + ba.normal).norm() < 1e-5);
        }
    }

    #[test]
    fn collide_swap_flips_normal(
        x in 0.0f64..1.5,
    ) {
        let a = Shape::Box(BoxShape::new(0.5, 0.5, 0.5));
        let b = Shape::Capsule(Capsule::new(0.2, 0.3));
        let xa = Vec3::zero();
        let xb = Vec3::new(x, 0.0, 0.0);
        let ab = collide(&a, xa, &eye(), &b, xb, &eye());
        let ba = collide(&b, xb, &eye(), &a, xa, &eye());
        prop_assert_eq!(ab.len(), ba.len());
        for (c, d) in ab.iter().zip(ba.iter()) {
            prop_assert!((c.depth - d.depth).abs() < 1e-9);
            prop_assert!((c.normal + d.normal).norm() < 1e-8);
        }
    }
}

#[test]
fn gravity_stack_with_convex_hull_stable() {
    let h = 0.25;
    let hull = ConvexHull::from_box(h, h, h, 0.0).unwrap();
    let body = RigidBody::new(1.0)
        .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
        .with_position(Vec3::new(0.0, 0.0, h + 0.01));
    let mut cw = ContactWorld::new(
        World::new(vec![body], vec![Box::new(UniformGravity::earth())]),
        vec![Shape::Convex(hull)],
        vec![
            StaticCollider::new(Shape::Plane(pe_collision::Plane::new(Vec3::unit_z(), 0.0)))
                .with_material(Material::new(0.0, 1.0)),
        ],
        Some(vec![Material::new(0.0, 0.5)]),
    );
    for _ in 0..240 {
        cw.step(1.0 / 240.0);
    }
    assert!(cw.world.state.x[0].z > 0.0);
    assert!(cw.world.state.x[0].z < 0.5);
}

#[test]
fn plane_vs_convex_hull_contact() {
    let hull = ConvexHull::from_box(0.25, 0.25, 0.25, 0.0).unwrap();
    let cs = collide(
        &Shape::Convex(hull),
        Vec3::new(0.0, 0.0, 0.2),
        &eye(),
        &Shape::Plane(pe_collision::Plane::new(Vec3::unit_z(), 0.0)),
        Vec3::zero(),
        &eye(),
    );
    assert!(!cs.is_empty());
    // Match box–plane: dynamic A → static plane B has normal −z.
    assert!(cs[0].normal.z < -0.9, "n={:?}", cs[0].normal);
    assert!(cs[0].depth > 0.04 && cs[0].depth < 0.06);
}
