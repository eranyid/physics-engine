//! Cross-OS hash of the canonical M34 underwater vehicle scenario.
//!
//! Usage: `cargo run -p pe-hydro --bin underwater_hash --release`
//! Scenario must stay in lockstep with `docs/DETERMINISM.md` and
//! `physics_engine_rs.canonical_underwater_hash()`.

fn main() {
    let h = pe_hydro::canonical_underwater_hash();
    println!("{h:016x}");
}
