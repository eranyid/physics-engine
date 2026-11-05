//! Print the canonical M31 finite-strain plastic hash (hex, lowercase).
//!
//! Usage: `cargo run -p pe-soft --bin fs_plastic_hash --release`

fn main() {
    println!("{:016x}", pe_soft::fs_plastic_canonical_hash());
}
