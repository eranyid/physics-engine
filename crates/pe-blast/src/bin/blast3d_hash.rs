//! Print the canonical 3-D blast (Euler voxel grid) hash for cross-OS CI.
//!
//! Usage: `cargo run -p pe-blast --bin blast3d_hash --release`

fn main() {
    println!("{:016x}", pe_blast::blast3d_corridor_hash());
}
