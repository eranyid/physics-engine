//! Print the canonical M30 plastic XPBD coupling hash (hex, lowercase).
//!
//! Usage: `cargo run -p pe-soft --bin plastic_hash --release`

fn main() {
    println!("{:016x}", pe_soft::plastic_canonical_hash());
}
