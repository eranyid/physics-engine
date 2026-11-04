//! Soft↔rigid two-way coupling demo (M28).
//!
//! Rope hanging from a free dynamic box + jelly cube landing on a second box
//! resting on the ground. Prints CoM, ground support force, and the locked
//! coupling hash.
//!
//! ```text
//! cargo run -p pe-examples --bin soft_rigid_hang --release
//! ```

use pe_math::Vec3;
use pe_soft::{soft_coupling_canonical_hash, soft_coupling_hash_world, COUPLING_HASH_DT};

fn main() {
    let mut world = soft_coupling_hash_world();
    let dt = COUPLING_HASH_DT;
    println!(
        "soft_rigid_hang: n_soft={} n_rigid={} substeps={}",
        world.soft.len(),
        world.rigid.n,
        world.soft.params.substeps
    );
    for step in 0..180 {
        world.step(dt);
        if step % 60 == 59 {
            let com = world.combined_com();
            let p = world.total_linear_momentum();
            let fz = world.diagnostics.ground_force.z;
            println!(
                "t={:.3} com=({:.3},{:.3},{:.3}) P=({:.3},{:.3},{:.3}) Fz_ground={:.2}",
                world.soft.t, com.x, com.y, com.z, p.x, p.y, p.z, fz
            );
        }
    }
    let hash = soft_coupling_canonical_hash();
    println!("canonical soft_coupling_hash = {hash:016x}");
    // Sanity: rope top attachment should stay near body-0 attach point.
    let attach = world.rigid.state.x[0] + world.rigid.state.q[0].rotate(Vec3::new(0.0, 0.0, -0.1));
    let top = world.soft.particles[0].x;
    println!(
        "attachment residual |x_p - x_a| = {:.3e} m",
        (top - attach).norm()
    );
}
