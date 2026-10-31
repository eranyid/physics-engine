//! Cross-OS hash for M11 Morison + BEM locked scenario.
//!
//! Usage: `cargo run -p pe-hydro --bin m11_hash --release`
//! Scenario must stay in lockstep with `docs/DETERMINISM.md` and
//! `physics_engine_rs.canonical_m11_hash()`.

fn main() {
    let h = pe_hydro::canonical_m11_hash();
    println!("{h:016x}");
}
