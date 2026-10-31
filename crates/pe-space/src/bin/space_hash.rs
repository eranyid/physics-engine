//! Print canonical pe-space state hashes for cross-OS CI.
//!
//! Usage: `cargo run -p pe-space --bin space_hash --release`
//!
//! Line 1: figure-eight leapfrog hash (locked in `docs/DETERMINISM.md`).
//! Line 2: IAS15 eccentric-Kepler hash (locked in `docs/DETERMINISM.md`).

fn main() {
    let h8 = pe_space::figure_eight_hash();
    let hi = pe_space::ias15_hash();
    println!("{h8:016x}");
    println!("{hi:016x}");
}
