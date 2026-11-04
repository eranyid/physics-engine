//! Print the locked M29 implicit-thermal determinism hash (hex).
//!
//! Usage: `cargo run -p pe-soft --bin thermal_hash --release`

fn main() {
    println!("{:016x}", pe_soft::thermal_canonical_hash());
}
