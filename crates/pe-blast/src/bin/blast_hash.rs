//! Print canonical blast determinism hashes for cross-OS CI.
//!
//! Lines (must stay in sync with `docs/DETERMINISM.md`):
//! 1. blast pyramid (legacy Kinney impulse) — locked `b51e11e89d4dd055`
//! 2. ExternalLoads KB / RH loading
//! 3. Gurney/Mott ballistic fragments
//! 4. Voronoi cohesive destruction
//!
//! Usage: `cargo run -p pe-blast --bin blast_hash --release`

fn main() {
    println!("{:016x}", pe_blast::blast_pyramid_hash());
    println!("{:016x}", pe_blast::blast_external_loads_hash());
    println!("{:016x}", pe_blast::blast_fragments_hash());
    println!("{:016x}", pe_blast::blast_destruction_hash());
}
