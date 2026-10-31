//! Print the canonical figure-eight N-body state hash for cross-OS CI.
//!
//! Usage: `cargo run -p pe-space --bin space_hash --release`
//! Scenario must stay in lockstep with `docs/DETERMINISM.md` and
//! `physics_engine_rs.canonical_space_hash()`.

fn main() {
    let hash = pe_space::figure_eight_hash();
    println!("{hash:016x}");
}
