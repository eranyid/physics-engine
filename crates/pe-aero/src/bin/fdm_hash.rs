//! Print canonical 6-DOF FDM hashes for cross-OS CI.
//!
//! Usage: `cargo run -p pe-aero --bin fdm_hash --release`
//! Line 1: rigid-body FDM hash
//! Line 2: Dryden gust-sequence hash
//! Line 3: FDM + held Dryden gust (RK4) hash

use pe_aero::scenarios::{canonical_dryden_gust_hash, canonical_fdm_gust_hash, canonical_fdm_hash};

fn main() {
    println!("{:016x}", canonical_fdm_hash());
    println!("{:016x}", canonical_dryden_gust_hash());
    println!("{:016x}", canonical_fdm_gust_hash());
}
