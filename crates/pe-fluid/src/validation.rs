//! Validation helpers: dam-break front measurement, hydrostatic Tait profiles.
//!
//! # Dam-break front metrics
//!
//! The discontinuous wet/dry toe is smoothed over `O(h)`. Prefer the near-bed
//! density-threshold toe ([`fluid_front_x_toe`]) over raw max-x ([`fluid_front_x`]),
//! which is splash-outlier sensitive. See `docs/FLUID.md` for the measured
//! PBF / WCSPH / DFSPH series and root-cause analysis.

use pe_math::libm_ops::fabs;
use pe_math::Vec3;

use crate::dfsph::{DfsphParams, DfsphWorld};
use crate::eos::TaitEos;
use crate::oracles::{hydrostatic_pressure, ritter_front};
use crate::particles::{fill_box, mass_from_spacing, ParticleKind, Particles};
use crate::pbf::{PbfParams, PbfWorld};
use crate::wcsph::{WcsphParams, WcsphWorld};
use crate::Dim;

/// Wet-front x-position [m]: max fluid particle x (2-D dam break in x–y).
pub fn fluid_front_x(world: &PbfWorld) -> f64 {
    let mut xf = f64::NEG_INFINITY;
    for i in 0..world.particles.len() {
        if world.particles.kind[i] == ParticleKind::Fluid {
            xf = xf.max(world.particles.x[i].x);
        }
    }
    xf
}

/// Near-bed toe front: max x among fluid particles with `y < y_max` and
/// `ρ > ρ_frac · ρ₀` (rejects splash / under-resolved spray).
pub fn fluid_front_x_toe(world: &PbfWorld, y_max: f64, rho_frac: f64) -> f64 {
    let thr = rho_frac * world.particles.rest_density;
    let mut xf = f64::NEG_INFINITY;
    for i in 0..world.particles.len() {
        if world.particles.kind[i] != ParticleKind::Fluid {
            continue;
        }
        if world.particles.x[i].y < y_max && world.particles.rho[i] > thr {
            xf = xf.max(world.particles.x[i].x);
        }
    }
    xf
}

fn front_x_toe_parts(parts: &Particles, y_max: f64, rho_frac: f64) -> f64 {
    let thr = rho_frac * parts.rest_density;
    let mut xf = f64::NEG_INFINITY;
    for i in 0..parts.len() {
        if parts.kind[i] != ParticleKind::Fluid {
            continue;
        }
        if parts.x[i].y < y_max && parts.rho[i] > thr {
            xf = xf.max(parts.x[i].x);
        }
    }
    xf
}

/// Relative front error vs Ritter: `|x_sim − (dam_x + x_f_ritter)| / |dam_x + x_f_ritter|`.
pub fn ritter_front_rel_error(x_sim: f64, dam_x: f64, g: f64, h0: f64, t: f64) -> f64 {
    let x_ref = dam_x + ritter_front(g, h0, t);
    fabs(x_sim - x_ref) / x_ref.abs().max(1e-9)
}

fn seed_dam_break_particles(
    dx: f64,
    h0: f64,
    l_res: f64,
    channel_length: f64,
    h_over_dx: f64,
    calibrate_cubic: bool,
) -> (Particles, f64) {
    let dim = Dim::D2;
    let hsm = h_over_dx * dx;
    let rho0 = 1000.0;
    let mass = if calibrate_cubic {
        // Calibrated mass so a full cubic-spline lattice sums to ρ₀ (DFSPH).
        crate::particles::mass_calibrated_to_rest(
            rho0,
            dx,
            hsm,
            dim,
            crate::kernels::KernelKind::CubicSpline,
        )
    } else {
        mass_from_spacing(rho0, dx, dim)
    };
    let mut parts = Particles::new(dim, rho0, mass, hsm);
    let ny = ((h0 / dx).round() as usize).max(6);
    let nx = ((l_res / dx).round() as usize).max(6);
    let x0 = 0.0;
    fill_box(
        &mut parts,
        Vec3::new(x0 + 0.5 * dx, 0.5 * dx, 0.0),
        nx,
        ny,
        1,
        dx,
        Vec3::zero(),
    );
    let dam_x = x0 + nx as f64 * dx; // ≈ L_res
    let psi = dx * dx;
    let n_floor = ((channel_length / dx).ceil() as usize) + 4;
    for i in 0..n_floor {
        let x = x0 + i as f64 * dx;
        parts.push_boundary(Vec3::new(x, -0.5 * dx, 0.0), psi);
        parts.push_boundary(Vec3::new(x, -1.5 * dx, 0.0), psi);
    }
    for j in 0..ny + 8 {
        parts.push_boundary(Vec3::new(-0.5 * dx, j as f64 * dx, 0.0), psi);
        parts.push_boundary(Vec3::new(-1.5 * dx, j as f64 * dx, 0.0), psi);
    }
    // Keep provisional ψ = dx² for open dam-break walls. Recomputing
    // Akinci ψ = 1/ΣW on a sparse floor+left-wall set inflates ψ and destabilises
    // DFSPH. Closed tanks should call `recomputed_akinci_psi` separately.
    let _ = calibrate_cubic;
    (parts, dam_x)
}

/// Build a 2-D PBF dam-break: column on the left, long dry floor to the right.
///
/// Fixed geometry in metres (independent of `dx`): reservoir `[0, L_res]×[0, h0]`,
/// dam face at `dam_x = L_res`, floor to `channel_length`. Left wall + floor only.
pub fn make_dam_break_2d(
    dx: f64,
    h0: f64,
    l_res: f64,
    channel_length: f64,
) -> (PbfWorld, f64, f64) {
    let (parts, dam_x) = seed_dam_break_particles(dx, h0, l_res, channel_length, 2.0, false);
    let g = 9.81;
    let params = PbfParams {
        gravity: Vec3::new(0.0, -g, 0.0),
        iterations: 12,
        art_pressure_k: 0.0,
        viscosity_eps: 0.01, // light numerical damping; Ritter is inviscid
        reuse_neighbours: true,
        ..PbfParams::default()
    };
    (PbfWorld::new(parts, params), dam_x, g)
}

/// Build a 2-D DFSPH dam-break with the same geometry as [`make_dam_break_2d`].
pub fn make_dfsph_dam_break_2d(
    dx: f64,
    h0: f64,
    l_res: f64,
    channel_length: f64,
) -> (DfsphWorld, f64, f64) {
    // h = 2.2 dx → ~ support 4.4 dx, typical DFSPH neighbour counts.
    let (parts, dam_x) = seed_dam_break_particles(dx, h0, l_res, channel_length, 2.2, true);
    let g = 9.81;
    let params = DfsphParams {
        gravity: Vec3::new(0.0, -g, 0.0),
        particle_diameter: dx,
        density_error_tol: 0.01, // 1 % avg — dam tip is under-resolved
        divergence_error_tol: 0.01,
        max_density_iters: 20,
        max_divergence_iters: 20,
        xsph: crate::viscosity::Xsph { epsilon: 0.01 },
        adaptive_dt: false,
        warm_start: true,
        ..DfsphParams::default()
    };
    (DfsphWorld::new(parts, params), dam_x, g)
}

/// Run PBF dam-break to time `t_end`, return `(x_front_toe, rel_error_vs_ritter, t_actual)`.
pub fn dam_break_front_at(
    dx: f64,
    h0: f64,
    l_res: f64,
    t_end: f64,
    channel_length: f64,
) -> (f64, f64, f64) {
    let (mut world, dam_x, g) = make_dam_break_2d(dx, h0, l_res, channel_length);
    let c0 = pe_math::libm_ops::sqrt(g * h0);
    let dt = (0.1 * dx / c0).min(4e-4);
    let steps = ((t_end / dt).ceil() as usize).max(1);
    for _ in 0..steps {
        world.step(dt);
    }
    // Toe: near-bed dense fluid (rejects splash). y_max = 0.35 h0.
    let xf = fluid_front_x_toe(&world, 0.35 * h0, 0.4);
    let rel = ritter_front_rel_error(xf, dam_x, g, h0, world.t);
    (xf, rel, world.t)
}

/// Run DFSPH dam-break; return `(x_front_toe, rel_error_vs_ritter, t_actual)`.
pub fn dfsph_dam_break_front_at(
    dx: f64,
    h0: f64,
    l_res: f64,
    t_end: f64,
    channel_length: f64,
) -> (f64, f64, f64) {
    let (mut world, dam_x, g) = make_dfsph_dam_break_2d(dx, h0, l_res, channel_length);
    let c0 = pe_math::libm_ops::sqrt(g * h0);
    // Mildly CFL-limited fixed step (adaptive off for reproducibility).
    let dt = (0.25 * dx / c0.max(1e-6)).clamp(5e-5, 5e-4);
    let steps = ((t_end / dt).ceil() as usize).max(1);
    for _ in 0..steps {
        world.step(dt);
    }
    let xf = front_x_toe_parts(&world.particles, 0.35 * h0, 0.4);
    let rel = ritter_front_rel_error(xf, dam_x, g, h0, world.t);
    (xf, rel, world.t)
}

/// High-resolution WCSPH dam-break toe (self-consistency reference helper).
///
/// Returns `(x_front_toe, rel_error_vs_ritter, t_actual)`. Used as a labelled
/// SPH self-reference when Martin & Moyce tables are unavailable.
pub fn wcsph_dam_break_front_at(
    dx: f64,
    h0: f64,
    l_res: f64,
    t_end: f64,
    channel_length: f64,
) -> (f64, f64, f64) {
    let (parts, dam_x) = seed_dam_break_particles(dx, h0, l_res, channel_length, 2.2, true);
    let g = 9.81;
    let rho0 = 1000.0;
    let mut params = WcsphParams::water(rho0, 50.0, Vec3::new(0.0, -g, 0.0));
    params.adapt_sound_speed = false;
    params.xsph.epsilon = 0.1;
    params.viscosity.alpha = 0.05;
    let mut world = WcsphWorld::new(parts, params);
    let dt = (0.2 * dx / 50.0).min(2e-4);
    let steps = ((t_end / dt).ceil() as usize).max(1);
    for _ in 0..steps {
        world.step(dt);
    }
    let xf = front_x_toe_parts(&world.particles, 0.35 * h0, 0.4);
    let rel = ritter_front_rel_error(xf, dam_x, g, h0, world.t);
    (xf, rel, world.t)
}

/// Convergence study vs Ritter at a fixed physical sample time.
///
/// Returns `(dx, x_front, rel_error)` rows. `t_end` must keep the ideal Ritter
/// front well short of `channel_length` (caller responsibility).
pub fn dam_break_convergence(
    dx_list: &[f64],
    h0: f64,
    l_res: f64,
    t_end: f64,
    channel_length: f64,
) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::new();
    for &dx in dx_list {
        let (xf, rel, _t) = dam_break_front_at(dx, h0, l_res, t_end, channel_length);
        out.push((dx, xf, rel));
    }
    out
}

/// DFSPH dam-break convergence vs Ritter: `(dx, x_front, rel_error)`.
pub fn dfsph_dam_break_convergence(
    dx_list: &[f64],
    h0: f64,
    l_res: f64,
    t_end: f64,
    channel_length: f64,
) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::new();
    for &dx in dx_list {
        let (xf, rel, _t) = dfsph_dam_break_front_at(dx, h0, l_res, t_end, channel_length);
        out.push((dx, xf, rel));
    }
    out
}

/// Max relative error of Tait pressure vs `p = ρ₀ g (H − z)` on a vertical profile.
///
/// Densities are the **linearised** hydrostatic field `ρ = ρ₀ (1 + g (H−z)/c₀²)`.
/// Returns `(max_rel, mean_rel, n_samples)`.
pub fn tait_hydrostatic_profile_error(
    eos: &TaitEos,
    g: f64,
    surface_z: f64,
    z_min: f64,
    n_samples: usize,
) -> (f64, f64, usize) {
    let rho0 = eos.rest_density;
    let c0 = eos.sound_speed;
    let mut max_rel: f64 = 0.0;
    let mut sum: f64 = 0.0;
    let mut n = 0usize;
    for k in 0..n_samples {
        let frac = (k as f64 + 0.5) / n_samples as f64;
        let z = z_min + frac * (surface_z - z_min);
        let depth = surface_z - z;
        if depth < 1e-6 {
            continue;
        }
        let rho = rho0 * (1.0 + g * depth / (c0 * c0));
        let p = eos.pressure(rho);
        let p_ref = hydrostatic_pressure(rho0, g, depth);
        let rel = fabs(p - p_ref) / p_ref;
        max_rel = max_rel.max(rel);
        sum += rel;
        n += 1;
    }
    (max_rel, if n == 0 { 0.0 } else { sum / n as f64 }, n)
}
