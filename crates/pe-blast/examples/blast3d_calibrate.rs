//! Free-air octant calibration of the 3-D solver against Kinney–Graham.
//!
//! `cargo run -p pe-blast --release --example blast3d_calibrate -- [n] [jump]`

use pe_blast::kinney::peak_overpressure;
use pe_blast::*;
use pe_math::Vec3;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(32);
    let lim = match args.get(2).map(String::as_str) {
        Some("minmod") => Limiter::Minmod,
        Some("vanleer") => Limiter::VanLeer,
        _ => Limiter::MonotonizedCentral,
    };
    let w: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    let size: f64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let dx = size / n as f64;
    let mut s = BlastScene::new(Vec3::zero(), Vec3::new(size, size, size), dx);
    s.grid.limiter = lim;
    s.grid.boundary = [
        Boundary::Wall,
        Boundary::Outflow,
        Boundary::Wall,
        Boundary::Outflow,
        Boundary::Wall,
        Boundary::Outflow,
    ];
    let remap: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.12);
    println!("scaled dx = {:.3} m/kg^(1/3)", dx / w.cbrt());
    let rep = s.detonate_with(&Detonation::new(Vec3::zero(), w).remap_at(remap));
    println!("remap {rep:?}");
    let radii: Vec<f64> = [1.5, 2.0, 3.0, 5.0, 8.0]
        .iter()
        .map(|z| z * w.cbrt())
        .filter(|&r| r < 0.95 * size)
        .collect();
    let ids: Vec<_> = radii
        .iter()
        .map(|&r| {
            let a = r / 3f64.sqrt();
            (
                s.add_probe(Vec3::new(a, a, a)),
                s.add_probe(Vec3::new(r, 0.5 * dx, 0.5 * dx)),
            )
        })
        .collect();
    let r_max = radii.iter().cloned().fold(0.0, f64::max);
    let steps = s.run_until(2.0 * r_max / 340.0, 1_000_000);
    println!("n={n} dx={dx:.4} limiter={lim:?} steps={steps}");
    for (&r, (diag, axis)) in radii.iter().zip(ids) {
        let z = r / w.cbrt();
        let k = peak_overpressure(z, P_AIR);
        let (pd, pa) = (&s.probes[diag], &s.probes[axis]);
        println!(
            "  Z={z:4.1}  Kinney {:7.1} kPa  diag ratio {:.2}  axis ratio {:.2}  I+ {:.2} Pa·s  t_a {:.3} ms",
            k / 1e3,
            pd.peak_overpressure / k,
            pa.peak_overpressure / k,
            pd.positive_impulse,
            pd.arrival_time.unwrap_or(f64::NAN) * 1e3
        );
    }
}
