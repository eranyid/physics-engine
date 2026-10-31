//! Print the canonical stage-2 collision state hash for cross-OS CI.
//!
//! Scenario: 10-box pyramid + ground, 600 steps @ dt=1/240. Locked in
//! `docs/DETERMINISM.md` and `physics_engine_rs.canonical_collision_hash()`.

fn main() {
    let h = pe_collision::canonical_collision_hash();
    println!("{h:016x}");
}
