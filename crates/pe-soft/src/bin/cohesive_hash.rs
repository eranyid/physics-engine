//! Print the canonical M33 cohesive-zone fracture hash.
//!
//! Usage: `cargo run -p pe-soft --bin cohesive_hash --release`

fn main() {
    println!("{:016x}", pe_soft::cohesive_canonical_hash());
}
