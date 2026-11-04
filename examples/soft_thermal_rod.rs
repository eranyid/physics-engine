//! M29 implicit heat conduction + thermo-elastic expansion demo.
//!
//! 1-D Cu-like rod: backward-Euler Fourier decay, then free thermal expansion
//! of XPBD rest lengths. Prints max relative error vs the analytic mode and
//! the locked thermal hash.
//!
//! ```text
//! cargo run -p pe-examples --bin soft_thermal_rod --release
//! ```

use pe_math::Vec3;
use pe_soft::constraint::{DistanceConstraint, Multiplier};
use pe_soft::{
    apply_thermal_expansion_from_cold, free_expansion_delta_l, init_rod_sine_mode,
    rod_fourier_mode1, thermal_canonical_hash, Particle, SoftParams, SoftWorld, ThermalMesh,
    ThermalProps,
};

fn main() {
    let props = ThermalProps {
        density: 8960.0,
        specific_heat: 385.0,
        conductivity: 400.0,
        expansion: 16.5e-6,
    };
    let length = 1.0;
    let area = 0.01;
    let n = 33;
    let t0 = 50.0;
    let mut mesh = ThermalMesh::rod_1d(n, length, area, props, 0.0);
    init_rod_sine_mode(&mut mesh, length, t0);
    let mut fixed = vec![false; n];
    fixed[0] = true;
    fixed[n - 1] = true;
    mesh.temperature[0] = 0.0;
    mesh.temperature[n - 1] = 0.0;

    let dt = 25.0;
    let steps = 20;
    println!(
        "soft_thermal_rod: n={n} BE dt={dt} (explicit CFL≈{:.3})",
        mesh.max_stable_dt()
    );
    for _ in 0..steps {
        assert!(mesh.step_backward_euler(dt, &fixed));
    }
    let t_sim = steps as f64 * dt;
    let kappa = props.diffusivity();
    let dx = length / (n - 1) as f64;
    let mut max_rel = 0.0_f64;
    for i in 1..n - 1 {
        let x = i as f64 * dx;
        let analytic = rod_fourier_mode1(x, t_sim, length, t0, kappa);
        max_rel = max_rel.max((mesh.temperature[i] - analytic).abs() / analytic.abs().max(1e-6));
    }
    println!("Fourier mode-1 max rel err = {max_rel:.4e} at t={t_sim}");

    // Thermo-elastic free expansion on a soft distance bar.
    let alpha = props.expansion;
    let d_t = 80.0;
    let l0 = 1.0;
    let mut world = SoftWorld::new();
    world.params = SoftParams {
        substeps: 16,
        iterations: 1,
        damping: 0.5,
        gravity: Vec3::zero(),
    };
    let i0 = world.add_particle(Particle::dynamic(Vec3::zero(), 1.0));
    let i1 = world.add_particle(Particle::dynamic(Vec3::new(l0, 0.0, 0.0), 1.0));
    world.constraints.distances.push(DistanceConstraint {
        i: i0,
        j: i1,
        rest_length: l0,
        compliance: 1e-6,
        lambda: Multiplier::default(),
        broken: false,
    });
    let mut mesh_ex = ThermalMesh::rod_1d(2, l0, area, props, 300.0);
    mesh_ex.temperature[0] = 300.0 + d_t;
    mesh_ex.temperature[1] = 300.0 + d_t;
    apply_thermal_expansion_from_cold(&mut world.constraints.distances, &[l0], &mesh_ex);
    let expect = l0 + free_expansion_delta_l(l0, alpha, d_t);
    println!(
        "expanded rest length = {:.9} (analytic {:.9})",
        world.constraints.distances[0].rest_length, expect
    );
    for _ in 0..120 {
        world.step(1.0 / 120.0);
    }
    let len = (world.particles[i1].x - world.particles[i0].x).norm();
    println!("free-bar length after settle = {len:.9} m");

    println!("canonical thermal_hash = {:016x}", thermal_canonical_hash());
}
