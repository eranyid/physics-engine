//! Print the canonical DFSPH fluid-tank determinism hash (hex).

fn main() {
    println!("{:x}", pe_fluid::dfsph_tank_hash());
}
