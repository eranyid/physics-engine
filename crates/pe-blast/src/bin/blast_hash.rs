//! Print the canonical blast-pyramid state hash for cross-OS CI.
//!
//! Usage: `cargo run -p pe-blast --bin blast_hash --release`

fn main() {
    let hash = pe_blast::blast_pyramid_hash();
    println!("{hash:016x}");
}
