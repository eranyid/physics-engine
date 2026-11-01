//! Probe DFSPH fixes: under-relaxation, dv clamp, no-warm.
use pe_fluid::{
    dam_break_ritter_geometry, fluid_front_x_height, make_dfsph_dam_break_2d,
    ritter_front_rel_error,
};
use pe_math::libm_ops::sqrt;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dx: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.03);
    let omega: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let dv_max: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let warm: bool = args.get(4).map(|s| s == "warm").unwrap_or(false);
    let h0 = 0.4;
    let (l_res, channel, g) = dam_break_ritter_geometry(h0);
    let (mut world, dam_x, _) = make_dfsph_dam_break_2d(dx, h0, l_res, channel);
    world.params.warm_start = warm;
    world.params.solver_omega = omega;
    world.params.max_velocity_correction = dv_max;
    let c0 = sqrt(g * h0);
    let dt = (0.12 * dx / c0.max(1e-6)).clamp(2e-5, 1.5e-4);
    println!("dx={dx} omega={omega} dv_max={dv_max} warm={warm} dt={dt:.3e}");
    let t_end = 4.0 * sqrt(h0 / g);
    let mut vmax_hist = 0.0f64;
    while world.t < t_end {
        world.step(dt);
        vmax_hist = vmax_hist.max(world.v_max());
        if world.v_max() > 200.0 {
            println!("RUNAWAY t={} vmax={}", world.t, world.v_max());
            break;
        }
    }
    let xf = fluid_front_x_height(&world.particles, dx, h0, channel, 0.02);
    let rel = ritter_front_rel_error(xf, dam_x, g, h0, world.t);
    println!("DONE vmax_hist={vmax_hist:.3} xf={xf:.4} rel={rel:.4}");
}
