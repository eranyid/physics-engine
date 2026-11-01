//! Print the canonical soft-body state hash for cross-OS CI.
//!
//! Usage: `cargo run -p pe-soft --bin soft_hash --release`

fn main() {
    let hash = pe_soft::soft_canonical_hash();
    println!("{hash:016x}");
}
