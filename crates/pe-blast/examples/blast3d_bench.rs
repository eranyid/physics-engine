//! Throughput of the 3-D solver (cell-updates per second).
//!
//! `cargo run -p pe-blast --release --example blast3d_bench [--features parallel]`
use std::time::Instant;

fn main() {
    for (label, jump) in [("MUSCL-MC", f64::INFINITY), ("first-order", 0.0)] {
        let mut s = pe_blast::blast3d_corridor_scene();
        s.grid.first_order_jump = jump;
        let n = 40;
        let t = Instant::now();
        for _ in 0..n {
            s.step();
        }
        let el = t.elapsed().as_secs_f64();
        let cell_steps = (s.grid.len() * n) as f64;
        println!(
            "{label:<12} {} cells × {n} steps: {:.3} s  →  {:.2} M cell-steps/s ({:.2} µs per cell-step)",
            s.grid.len(),
            el,
            cell_steps / el / 1e6,
            el / cell_steps * 1e6
        );
    }
}
