//! Print the canonical PBF fluid-tank determinism hash (hex).

fn main() {
    println!("{:x}", pe_fluid::fluid_tank_hash());
}
