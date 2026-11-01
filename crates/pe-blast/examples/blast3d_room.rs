//! 3-D blast in a room with a doorway into a corridor.
//!
//! Prints pressure-gauge peaks / impulses and writes horizontal pressure
//! slices as PPM images (open with any image viewer) into `target/blast3d/`.
//!
//! `cargo run -p pe-blast --release --example blast3d_room`

use std::fs;

use pe_blast::*;
use pe_math::Vec3;

fn main() {
    let dx = 0.1;
    // 8 m × 5 m × 3 m: room (x < 4) + doorway + corridor (y 1.5..3.0).
    let mut s = BlastScene::new(Vec3::zero(), Vec3::new(8.0, 5.0, 3.0), dx);
    s.fill_ground(0.1);
    // Ceiling, then the x = 4 partition with a door.
    s.grid.boundary[5] = Boundary::Wall;
    s.fill_aabb(Vec3::new(3.9, 0.0, 0.0), Vec3::new(4.1, 5.0, 3.0));
    s.clear_aabb(Vec3::new(3.85, 2.0, 0.1), Vec3::new(4.15, 3.0, 2.1));
    // Corridor walls beyond the door.
    s.fill_aabb(Vec3::new(4.1, 0.0, 0.0), Vec3::new(8.0, 1.5, 3.0));
    s.fill_aabb(Vec3::new(4.1, 3.5, 0.0), Vec3::new(8.0, 5.0, 3.0));

    let gauges = [
        ("room, 1.5 m from charge", Vec3::new(3.0, 1.5, 1.2)),
        ("room corner", Vec3::new(0.4, 0.4, 0.4)),
        ("doorway", Vec3::new(4.0, 2.5, 1.2)),
        ("corridor 2 m past door", Vec3::new(6.0, 2.5, 1.2)),
        ("corridor end", Vec3::new(7.7, 2.5, 1.2)),
    ];
    let ids: Vec<_> = gauges.iter().map(|(_, p)| s.add_probe(*p)).collect();

    let rep = s.detonate(Vec3::new(1.8, 2.0, 1.0), 0.25);
    println!(
        "grid {}×{}×{} = {} cells, remap {:?}",
        s.grid.nx,
        s.grid.ny,
        s.grid.nz,
        s.grid.len(),
        rep
    );

    let out = "target/blast3d";
    let _ = fs::create_dir_all(out);
    let k_slice = (1.2 / dx) as usize;
    for (n, t_end) in [2e-3, 5e-3, 10e-3, 20e-3, 40e-3].into_iter().enumerate() {
        s.run_until(t_end, 1_000_000);
        let path = format!("{out}/slice_{n}_{:.0}ms.ppm", t_end * 1e3);
        if fs::write(&path, s.slice_ppm(2, k_slice, 3e5)).is_ok() {
            println!(
                "t = {:5.1} ms  ({} steps)  wrote {path}",
                s.t() * 1e3,
                s.grid.steps
            );
        }
    }

    println!(
        "\n{:<26} {:>10} {:>12} {:>12} {:>10}",
        "gauge", "peak kPa", "I+ Pa·s", "I− Pa·s", "t_a ms"
    );
    for ((name, _), id) in gauges.iter().zip(ids) {
        let p = &s.probes[id];
        println!(
            "{name:<26} {:>10.1} {:>12.1} {:>12.1} {:>10.2}",
            p.peak_overpressure / 1e3,
            p.positive_impulse,
            p.negative_impulse,
            p.arrival_time.map(|t| t * 1e3).unwrap_or(f64::NAN)
        );
    }
}
