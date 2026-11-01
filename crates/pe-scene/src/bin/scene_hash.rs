//! Print canonical scene hashes for cross-OS CI.
//!
//! Usage: `cargo run -p pe-scene --bin scene_hash --release`
//!
//! Line 1: pyramid save/load/continue hash
//! Line 2: convex save/load/continue hash
//! Line 3: Godot drop-box state hash (Y-up scenario)

use pe_scene::{
    canonical_scene_convex_roundtrip_hash, canonical_scene_roundtrip_hash, godot_drop_box_result,
};

fn main() {
    let h = canonical_scene_roundtrip_hash();
    println!("{h:016x}");
    let h2 = canonical_scene_convex_roundtrip_hash();
    println!("{h2:016x}");
    let (y, h3) = godot_drop_box_result();
    println!("{h3:016x}");
    eprintln!("godot_drop_y={y:.17}");
}
