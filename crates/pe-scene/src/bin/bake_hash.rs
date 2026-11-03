//! Print the locked canonical bake JSON FNV-1a hash (hex).

fn main() {
    println!("{:016x}", pe_scene::canonical_bake_json_hash());
}
