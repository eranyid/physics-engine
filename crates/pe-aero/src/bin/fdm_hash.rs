//! Print the canonical 6-DOF FDM state hash (+ Dryden gust hash) for cross-OS CI.
//!
//! Usage: `cargo run -p pe-aero --bin fdm_hash --release`
//! Line 1: rigid-body FDM hash; line 2: Dryden gust-sequence hash.

use pe_aero::scenarios::{canonical_dryden_gust_hash, canonical_fdm_hash};

fn main() {
    println!("{:016x}", canonical_fdm_hash());
    println!("{:016x}", canonical_dryden_gust_hash());
}
