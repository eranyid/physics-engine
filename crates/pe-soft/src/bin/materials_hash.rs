//! Print the locked materials-track determinism hash (hex).

fn main() {
    println!("{:016x}", pe_soft::materials_canonical_hash());
}
