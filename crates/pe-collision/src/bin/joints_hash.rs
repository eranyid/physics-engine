//! Print the canonical M5b joints FNV-1a hash (hex).

fn main() {
    println!("{:016x}", pe_collision::canonical_joints_hash());
}
