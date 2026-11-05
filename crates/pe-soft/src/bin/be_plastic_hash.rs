//! Print the canonical M32 spectral Hencky / Kirchhoff (`b_e`) plastic hash.
//!
//! Usage: `cargo run -p pe-soft --bin be_plastic_hash --release`

fn main() {
    println!("{:016x}", pe_soft::be_plastic_canonical_hash());
}
