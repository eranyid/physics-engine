//! Print the canonical soft↔rigid coupling hash (hex).

fn main() {
    println!("{:x}", pe_soft::soft_coupling_canonical_hash());
}
