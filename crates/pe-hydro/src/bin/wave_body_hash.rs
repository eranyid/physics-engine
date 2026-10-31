//! Cross-OS hash of a wave-driven floating body (Airy surface + buoyancy).
//!
//! Usage: `cargo run -p pe-hydro --bin wave_body_hash --release`
//! Scenario must stay in lockstep with `docs/DETERMINISM.md` and
//! `physics_engine_rs.canonical_wave_body_hash()`.

fn main() {
    let h = pe_hydro::canonical_wave_body_hash();
    println!("{h:016x}");
}
