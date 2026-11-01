//! Print canonical Cummins heave-decay hash for cross-OS CI.
//!
//! Usage: `cargo run -p pe-hydro --bin cummins_hash --release`

fn main() {
    let h = pe_hydro::canonical_cummins_hash();
    println!("{h:016x}");
}
