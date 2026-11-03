//! Bake a projectile scene to stdout / optional file as versioned bake JSON.
//!
//! ```bash
//! cargo run -p pe-examples --bin bake_scene --release
//! cargo run -p pe-examples --bin bake_scene --release -- out.bake.json
//! ```

use pe_scene::{bake_scene, canonical_bake_scene, BakeOptions};

fn main() {
    let out_path = std::env::args().nth(1);
    let scene = canonical_bake_scene();
    let opt = BakeOptions {
        duration: 1.0,
        physics_dt: 1.0 / 240.0,
        sample_fps: 24.0,
        include_velocities: true,
        reduce: None,
    };
    let bake = bake_scene(&scene, &opt).expect("bake");
    let json = bake.to_json_pretty().expect("json");
    match out_path {
        Some(path) => {
            std::fs::write(&path, &json).expect("write");
            eprintln!(
                "bake_scene: wrote {} ({} bodies, {} keys)",
                path,
                bake.bodies.len(),
                bake.bodies[0].times.len()
            );
        }
        None => {
            print!("{json}");
        }
    }
    let hash = pe_scene::bake_json_hash64(&bake).expect("hash");
    eprintln!("bake JSON FNV-1a = {hash:016x}");
}
