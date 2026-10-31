//! Print the canonical fluid-tank state hash for cross-OS CI.
//!
//! Usage: `cargo run -p pe-fluid --bin fluid_hash --release`

fn main() {
    let hash = pe_fluid::fluid_tank_hash();
    println!("{hash:016x}");
}
