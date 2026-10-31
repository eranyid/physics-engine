//! Print the locked M6 convex-scene FNV-1a state hash (hex).

fn main() {
    println!("{:x}", pe_collision::canonical_convex_hash());
}
