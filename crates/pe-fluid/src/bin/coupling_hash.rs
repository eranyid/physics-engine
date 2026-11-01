//! Print the canonical fluid↔rigid coupling hash (hex).
fn main() {
    println!("{:x}", pe_fluid::coupling_tank_hash());
}
