//! Print the canonical scene save/load/continue hash for cross-OS CI.
//!
//! Usage: `cargo run -p pe-scene --bin scene_hash --release`

use pe_scene::canonical_scene_roundtrip_hash;

fn main() {
    let h = canonical_scene_roundtrip_hash();
    println!("{h:016x}");
}
