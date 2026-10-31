//! Print the canonical 100-bullet volley hash for cross-OS CI.
//!
//! Usage: `cargo run -p pe-ballistics --bin ballistics_hash --release`

fn main() {
    let hash = pe_ballistics::ballistics_volley_hash();
    println!("{hash:016x}");
}
