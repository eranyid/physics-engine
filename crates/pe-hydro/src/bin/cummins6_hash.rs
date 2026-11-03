//! Print the canonical 6-DOF Cummins free-decay hash (cross-OS lock).
//!
//! Usage: `cargo run -p pe-hydro --bin cummins6_hash --release`

fn main() {
    let h = pe_hydro::canonical_cummins6_hash();
    println!("{h:016x}");
}
