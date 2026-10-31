//! Print the canonical M5a soft-step collision state hash for cross-OS CI.
//!
//! Scenario: same 10-box partial pyramid as stage-2, soft-step solver, 600 steps
//! @ dt=1/240. Locked in `docs/DETERMINISM.md`.

fn main() {
    let h = pe_collision::canonical_soft_step_collision_hash();
    println!("{h:016x}");
}
