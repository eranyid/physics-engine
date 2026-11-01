//! Determinism / hash smoke for the rigid-body performance work.
//!
//! - Locked oracle hashes must remain bit-identical (global PGS default).
//! - Island-ordered solve is bit-identical across rayon thread pools 1/2/8
//!   when the `parallel` feature is enabled.

use pe_collision::{
    BoxShape, ContactWorld, Material, Plane, Shape, StaticCollider, CANONICAL_COLLISION_HASH,
    CANONICAL_CONVEX_HASH, CANONICAL_JOINTS_HASH, CANONICAL_SOFT_STEP_COLLISION_HASH,
};
use pe_core::diagnostics::state_hash64;
use pe_core::forces::UniformGravity;
use pe_core::{RigidBody, World};
use pe_math::Vec3;

#[test]
fn locked_hashes_unchanged() {
    assert_eq!(
        pe_collision::canonical_collision_hash(),
        CANONICAL_COLLISION_HASH
    );
    assert_eq!(
        pe_collision::canonical_soft_step_collision_hash(),
        CANONICAL_SOFT_STEP_COLLISION_HASH
    );
    assert_eq!(pe_collision::canonical_joints_hash(), CANONICAL_JOINTS_HASH);
    assert_eq!(pe_collision::canonical_convex_hash(), CANONICAL_CONVEX_HASH);
}

fn two_stack_world() -> ContactWorld {
    // Two disjoint box stacks → two islands under gravity + ground.
    let h = 0.25;
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    let mut mats = Vec::new();
    for stack in 0..2 {
        let x0 = stack as f64 * 4.0;
        for level in 0..3 {
            bodies.push(
                RigidBody::new(1.0)
                    .with_inertia(BoxShape::new(h, h, h).inertia(1.0))
                    .with_position(Vec3::new(x0, 0.0, h + 2.0 * h * (level as f64))),
            );
            shapes.push(Shape::Box(BoxShape::new(h, h, h)));
            mats.push(Material::new(0.0, 0.4));
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
    cw.iterations = 12;
    cw.island_ordered_solve = true;
    cw.set_allow_sleep(false);
    cw
}

fn run_hash(steps: usize) -> u64 {
    let mut cw = two_stack_world();
    for _ in 0..steps {
        cw.step(1.0 / 240.0);
    }
    state_hash64(&cw.world.state)
}

#[test]
fn island_ordered_solve_is_bit_stable_on_repeat() {
    let a = run_hash(120);
    let b = run_hash(120);
    assert_eq!(a, b, "island-ordered path must be bit-stable");
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_island_hashes_match_across_thread_counts() {
    use std::sync::OnceLock;
    static LOCK: OnceLock<()> = OnceLock::new();
    let _ = LOCK.get_or_init(|| ());

    let mut hashes = Vec::new();
    for threads in [1usize, 2, 8] {
        // Serialize pool rebuilds — rayon global pool can only be built once.
        // Use install on a local pool instead.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool");
        let h = pool.install(|| run_hash(180));
        hashes.push((threads, h));
    }
    let h1 = hashes[0].1;
    for &(threads, h) in &hashes {
        assert_eq!(
            h, h1,
            "island-parallel hash with {threads} threads diverged from 1-thread"
        );
    }
}

#[test]
fn aabb_tree_incremental_matches_rebuild_pairs() {
    use pe_collision::{brute_force_pairs, dynamic_tree_pairs, Aabb, DynamicAabbTree};
    use pe_math::Vec3;

    let mut state = 42u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        state
    };
    let uni = |next: &mut dyn FnMut() -> u64, a: f64, b: f64| -> f64 {
        let u = (next() >> 11) as f64 / ((1u64 << 53) as f64);
        a + (b - a) * u
    };
    let n = 80;
    let mut aabbs = Vec::with_capacity(n);
    let mut skip = Vec::with_capacity(n);
    for _ in 0..n {
        let cx = uni(&mut next, 0.0, 6.0);
        let cy = uni(&mut next, 0.0, 6.0);
        let cz = uni(&mut next, 0.0, 6.0);
        let ex = uni(&mut next, 0.05, 0.5);
        let ey = uni(&mut next, 0.05, 0.5);
        let ez = uni(&mut next, 0.05, 0.5);
        aabbs.push(Aabb {
            lo: Vec3::new(cx - ex, cy - ey, cz - ez),
            hi: Vec3::new(cx + ex, cy + ey, cz + ez),
        });
        skip.push(uni(&mut next, 0.0, 1.0) < 0.1);
    }
    let expect = brute_force_pairs(&aabbs, Some(&skip));
    assert_eq!(dynamic_tree_pairs(&aabbs, Some(&skip)), expect);

    let mut tree = DynamicAabbTree::default();
    tree.rebuild(&aabbs);
    // Nudge a few AABBs and update incrementally.
    for i in [0usize, 3, 7, 15, 40] {
        aabbs[i].lo.x += 0.2;
        aabbs[i].hi.x += 0.2;
        tree.update_proxy(i, &aabbs[i]);
    }
    let mut inc = tree.pairs_incremental(&aabbs, &skip);
    let mut rebuilt = dynamic_tree_pairs(&aabbs, Some(&skip));
    inc.sort_unstable();
    rebuilt.sort_unstable();
    assert_eq!(inc, rebuilt);
    assert_eq!(inc, brute_force_pairs(&aabbs, Some(&skip)));
}
