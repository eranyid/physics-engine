//! Validation helpers: dam-break height-field fronts, Ritter curves, hydrostatic
//! Tait profiles.
//!
//! # Dam-break front metric
//!
//! Prefer a **binned column-height** front ([`front_x_from_height_field`]): the
//! rightmost `x` where volume-based column height stays at/above `~2% h₀`. This
//! rejects splash outliers that dominate raw max-`x` / near-bed toe probes.
//! Sample at several dimensionless times `τ = t √(g/h₀)` (asymptotic Ritter
//! regime), not a single early collapse instant. See `docs/FLUID.md`.

use pe_math::libm_ops::{fabs, sqrt};
use pe_math::Vec3;

use crate::dfsph::{DfsphParams, DfsphWorld};
use crate::eos::TaitEos;
use crate::oracles::{hydrostatic_pressure, ritter_front, ritter_height};
use crate::particles::{fill_box, mass_from_spacing, ParticleKind, Particles};
use crate::pbf::{PbfParams, PbfWorld};
use crate::wcsph::{WcsphParams, WcsphWorld};
use crate::Dim;

/// Default dimensionless sample times `τ = t √(g/h₀)` for Ritter front curves.
pub const DAM_BREAK_TAU_SAMPLES: [f64; 4] = [1.0, 2.0, 3.0, 4.0];

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

/// Near-bed toe front (legacy / diagnostic): max x among fluid with `y < y_max`
/// and `ρ > ρ_frac · ρ₀`. Prefer [`front_x_from_height_field`] for Ritter checks.
pub fn fluid_front_x_toe(world: &PbfWorld, y_max: f64, rho_frac: f64) -> f64 {
    front_x_toe_parts(&world.particles, y_max, rho_frac)
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

/// Volume-based column heights on bins of width `bin_w`.
///
/// Each fluid particle contributes area `mass/ρ₀` (2-D). Bin height
/// `h = (Σ area) / bin_w`. Returns `(x_bin_centre, h)` sorted by `x`.
pub fn column_heights(parts: &Particles, x_min: f64, x_max: f64, bin_w: f64) -> Vec<(f64, f64)> {
    let bw = bin_w.max(1e-12);
    let n_bins = (((x_max - x_min) / bw).ceil() as usize).max(1);
    let mut mass_sum = vec![0.0; n_bins];
    let area = parts.mass / parts.rest_density.max(1e-12);
    for i in 0..parts.len() {
        if parts.kind[i] != ParticleKind::Fluid {
            continue;
        }
        let x = parts.x[i].x;
        if x < x_min || x >= x_max {
            continue;
        }
        let b = ((x - x_min) / bw).floor() as usize;
        if b < n_bins {
            mass_sum[b] += area;
        }
    }
    let mut out = Vec::with_capacity(n_bins);
    for (b, &a) in mass_sum.iter().enumerate() {
        let xc = x_min + (b as f64 + 0.5) * bw;
        out.push((xc, a / bw));
    }
    out
}

/// Wet–dry front from a binned height field: rightmost bin centre with
/// `h ≥ h_frac · h₀` (default `h_frac = 0.02`). Returns `NEG_INFINITY` if dry.
pub fn front_x_from_height_field(heights: &[(f64, f64)], h0: f64, h_frac: f64) -> f64 {
    let thr = h_frac * h0;
    let mut xf = f64::NEG_INFINITY;
    for &(x, h) in heights {
        if h >= thr {
            xf = x;
        }
    }
    xf
}

/// Measure height-field front on `parts` (bin width ≈ `dx`).
pub fn fluid_front_x_height(
    parts: &Particles,
    dx: f64,
    h0: f64,
    channel_length: f64,
    h_frac: f64,
) -> f64 {
    let bin_w = dx.max(1e-9);
    let heights = column_heights(parts, -2.0 * dx, channel_length + 2.0 * dx, bin_w);
    front_x_from_height_field(&heights, h0, h_frac)
}

/// Mean relative `|h_sim − h_ritter| / h₀` on wet bins (where either height > thr).
pub fn ritter_height_profile_rel_error(
    heights: &[(f64, f64)],
    dam_x: f64,
    g: f64,
    h0: f64,
    t: f64,
    h_frac: f64,
) -> (f64, usize) {
    let thr = h_frac * h0;
    let mut sum = 0.0;
    let mut n = 0usize;
    for &(x, h) in heights {
        let href = ritter_height(g, h0, x - dam_x, t);
        if h < thr && href < thr {
            continue;
        }
        sum += fabs(h - href) / h0.max(1e-12);
        n += 1;
    }
    (if n == 0 { 0.0 } else { sum / n as f64 }, n)
}

fn seed_dam_break_particles(
    dx: f64,
    h0: f64,
    l_res: f64,
    channel_length: f64,
    h_over_dx: f64,
    calibrate_cubic: bool,
    with_akinci_walls: bool,
) -> (Particles, f64) {
    let dim = Dim::D2;
    let hsm = h_over_dx * dx;
    let rho0 = 1000.0;
    let mass = if calibrate_cubic {
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
    let dam_x = x0 + nx as f64 * dx;
    if with_akinci_walls {
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
    }
    let _ = calibrate_cubic;
    (parts, dam_x)
}

/// Build a 2-D PBF dam-break: column on the left, long dry floor to the right.
pub fn make_dam_break_2d(
    dx: f64,
    h0: f64,
    l_res: f64,
    channel_length: f64,
) -> (PbfWorld, f64, f64) {
    let (parts, dam_x) = seed_dam_break_particles(dx, h0, l_res, channel_length, 2.0, false, true);
    let g = 9.81;
    let params = PbfParams {
        gravity: Vec3::new(0.0, -g, 0.0),
        iterations: 12,
        art_pressure_k: 0.0,
        viscosity_eps: 0.0, // inviscid for Ritter comparison
        reuse_neighbours: true,
        ..PbfParams::default()
    };
    (PbfWorld::new(parts, params), dam_x, g)
}

/// Build a 2-D DFSPH dam-break tuned for Ritter.
///
/// Inviscid (`xsph=0`), Akinci boundary samples for near-wall density support,
/// plus **free-slip hard walls** (`x≥0`, `y≥0`) to prevent floor penetration.
/// Controlled study (`docs/FLUID.md`): Akinci-only penetrates; hard-wall-only
/// under-resolves the wall kernel (~20% late-time lag); Akinci+hard-wall
/// recovers late-time Ritter within ~10% at `dx≈0.045`. Finer than ~`0.04`
/// develops a vertical splash instability (documented, not silently hidden).
pub fn make_dfsph_dam_break_2d(
    dx: f64,
    h0: f64,
    l_res: f64,
    channel_length: f64,
) -> (DfsphWorld, f64, f64) {
    let (parts, dam_x) = seed_dam_break_particles(dx, h0, l_res, channel_length, 2.2, true, true);
    let g = 9.81;
    let params = DfsphParams {
        gravity: Vec3::new(0.0, -g, 0.0),
        particle_diameter: dx,
        density_error_tol: 0.01,
        divergence_error_tol: 0.01,
        max_density_iters: 40,
        max_divergence_iters: 30,
        xsph: crate::viscosity::Xsph { epsilon: 0.0 },
        adaptive_dt: false,
        warm_start: true,
        free_slip_xmin: Some(0.0),
        free_slip_ymin: Some(0.0),
        ..DfsphParams::default()
    };
    (DfsphWorld::new(parts, params), dam_x, g)
}

/// Akinci boundary samples only (no free-slip hard walls) — penetration study.
pub fn make_dfsph_dam_break_akinci_only(
    dx: f64,
    h0: f64,
    l_res: f64,
    channel_length: f64,
) -> (DfsphWorld, f64, f64) {
    let (parts, dam_x) = seed_dam_break_particles(dx, h0, l_res, channel_length, 2.2, true, true);
    let g = 9.81;
    let params = DfsphParams {
        gravity: Vec3::new(0.0, -g, 0.0),
        particle_diameter: dx,
        density_error_tol: 0.01,
        divergence_error_tol: 0.01,
        max_density_iters: 30,
        max_divergence_iters: 30,
        xsph: crate::viscosity::Xsph { epsilon: 0.0 },
        adaptive_dt: false,
        warm_start: true,
        free_slip_xmin: None,
        free_slip_ymin: None,
        ..DfsphParams::default()
    };
    (DfsphWorld::new(parts, params), dam_x, g)
}

/// Canonical long-bed dam-break geometry for asymptotic Ritter sampling.
///
/// `L_res` long enough that the left rarefaction has not hit the wall before
/// `τ_max = 4` (`t = τ √(h₀/g)`), channel long enough for `x_f(τ=4) = 2 h₀ τ`.
pub fn dam_break_ritter_geometry(h0: f64) -> (f64, f64, f64) {
    let g = 9.81;
    let c0 = sqrt(g * h0);
    // Left rarefaction travel: need L_res > c0 * t(τ=4) = h0 * 4
    let l_res = (4.5 * h0).max(1.5);
    let t_max = 4.0 * sqrt(h0 / g);
    let x_travel = 2.0 * c0 * t_max; // = 2 h0 * 4
                                     // Extra margin so an overshooting tip does not hit the domain end.
    let channel = l_res + x_travel + 6.0 * h0;
    (l_res, channel, g)
}

fn pbf_dt(dx: f64, g: f64, h0: f64) -> f64 {
    let c0 = sqrt(g * h0);
    (0.1 * dx / c0).min(4e-4)
}

fn dfsph_dt(dx: f64, g: f64, h0: f64) -> f64 {
    let c0 = sqrt(g * h0);
    // CFL vs shallow-water scale. Cap keeps CI / probe runs tractable.
    (0.12 * dx / c0.max(1e-6)).clamp(2e-5, 1.5e-4)
}

/// Advance PBF dam-break, sampling height-field front at each `t` in `t_samples`.
///
/// Returns `(t, x_front, rel_vs_ritter)` rows. World is stepped monotonically.
pub fn dam_break_front_curve_pbf(
    dx: f64,
    h0: f64,
    l_res: f64,
    channel_length: f64,
    t_samples: &[f64],
) -> Vec<(f64, f64, f64)> {
    let (mut world, dam_x, g) = make_dam_break_2d(dx, h0, l_res, channel_length);
    let dt = pbf_dt(dx, g, h0);
    sample_front_curve_pbf(&mut world, dam_x, g, h0, dx, channel_length, dt, t_samples)
}

fn sample_front_curve_pbf(
    world: &mut PbfWorld,
    dam_x: f64,
    g: f64,
    h0: f64,
    dx: f64,
    channel_length: f64,
    dt: f64,
    t_samples: &[f64],
) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::with_capacity(t_samples.len());
    let mut samples: Vec<f64> = t_samples.to_vec();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for &t_target in &samples {
        while world.t + 0.5 * dt < t_target {
            world.step(dt);
        }
        let xf = fluid_front_x_height(&world.particles, dx, h0, channel_length, 0.02);
        let rel = ritter_front_rel_error(xf, dam_x, g, h0, world.t);
        out.push((world.t, xf, rel));
    }
    out
}

/// Advance DFSPH dam-break; sample height-field front at each `t` in `t_samples`.
pub fn dam_break_front_curve_dfsph(
    dx: f64,
    h0: f64,
    l_res: f64,
    channel_length: f64,
    t_samples: &[f64],
) -> Vec<(f64, f64, f64)> {
    let (mut world, dam_x, g) = make_dfsph_dam_break_2d(dx, h0, l_res, channel_length);
    let dt = dfsph_dt(dx, g, h0);
    sample_front_curve_dfsph(&mut world, dam_x, g, h0, dx, channel_length, dt, t_samples)
}

fn sample_front_curve_dfsph(
    world: &mut DfsphWorld,
    dam_x: f64,
    g: f64,
    h0: f64,
    dx: f64,
    channel_length: f64,
    dt: f64,
    t_samples: &[f64],
) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::with_capacity(t_samples.len());
    let mut samples: Vec<f64> = t_samples.to_vec();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for &t_target in &samples {
        while world.t + 0.5 * dt < t_target {
            world.step(dt);
        }
        let xf = fluid_front_x_height(&world.particles, dx, h0, channel_length, 0.02);
        let rel = ritter_front_rel_error(xf, dam_x, g, h0, world.t);
        out.push((world.t, xf, rel));
    }
    out
}

/// Times [s] for dimensionless `τ = t √(g/h₀)`.
pub fn dam_break_times_from_tau(h0: f64, g: f64, taus: &[f64]) -> Vec<f64> {
    let scale = sqrt(h0 / g);
    taus.iter().map(|tau| tau * scale).collect()
}

/// Run PBF dam-break to `t_end`; return `(x_front_height, rel_error, t_actual)`.
pub fn dam_break_front_at(
    dx: f64,
    h0: f64,
    l_res: f64,
    t_end: f64,
    channel_length: f64,
) -> (f64, f64, f64) {
    let rows = dam_break_front_curve_pbf(dx, h0, l_res, channel_length, &[t_end]);
    let (t, xf, rel) = rows[0];
    (xf, rel, t)
}

/// Run DFSPH dam-break; return `(x_front_height, rel_error, t_actual)`.
pub fn dfsph_dam_break_front_at(
    dx: f64,
    h0: f64,
    l_res: f64,
    t_end: f64,
    channel_length: f64,
) -> (f64, f64, f64) {
    let rows = dam_break_front_curve_dfsph(dx, h0, l_res, channel_length, &[t_end]);
    let (t, xf, rel) = rows[0];
    (xf, rel, t)
}

/// High-resolution WCSPH dam-break height-field front (self-consistency helper).
pub fn wcsph_dam_break_front_at(
    dx: f64,
    h0: f64,
    l_res: f64,
    t_end: f64,
    channel_length: f64,
) -> (f64, f64, f64) {
    let (parts, dam_x) = seed_dam_break_particles(dx, h0, l_res, channel_length, 2.2, true, true);
    let g = 9.81;
    let rho0 = 1000.0;
    let mut params = WcsphParams::water(rho0, 50.0, Vec3::new(0.0, -g, 0.0));
    params.adapt_sound_speed = false;
    params.xsph.epsilon = 0.0;
    params.viscosity.alpha = 0.0;
    let mut world = WcsphWorld::new(parts, params);
    let dt = (0.2 * dx / 50.0).min(2e-4);
    let steps = ((t_end / dt).ceil() as usize).max(1);
    for _ in 0..steps {
        world.step(dt);
    }
    let xf = fluid_front_x_height(&world.particles, dx, h0, channel_length, 0.02);
    let rel = ritter_front_rel_error(xf, dam_x, g, h0, world.t);
    (xf, rel, world.t)
}

/// DFSPH height-field front + Ritter `h(x)` profile error at one time.
///
/// Returns `(x_front, front_rel, h_profile_mean_rel, n_bins, t)`.
pub fn dfsph_dam_break_front_and_height_at(
    dx: f64,
    h0: f64,
    l_res: f64,
    t_end: f64,
    channel_length: f64,
) -> (f64, f64, f64, usize, f64) {
    let (mut world, dam_x, g) = make_dfsph_dam_break_2d(dx, h0, l_res, channel_length);
    let dt = dfsph_dt(dx, g, h0);
    while world.t + 0.5 * dt < t_end {
        world.step(dt);
    }
    let bin_w = dx;
    let heights = column_heights(
        &world.particles,
        -2.0 * dx,
        channel_length + 2.0 * dx,
        bin_w,
    );
    let xf = front_x_from_height_field(&heights, h0, 0.02);
    let front_rel = ritter_front_rel_error(xf, dam_x, g, h0, world.t);
    let (h_rel, n) = ritter_height_profile_rel_error(&heights, dam_x, g, h0, world.t, 0.02);
    (xf, front_rel, h_rel, n, world.t)
}

/// Convergence study vs Ritter at fixed physical sample time (height-field front).
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

/// Late-time DFSPH front error at `τ` for each `dx` (height-field metric).
pub fn dfsph_dam_break_late_front_by_dx(
    dx_list: &[f64],
    h0: f64,
    tau: f64,
) -> Vec<(f64, f64, f64)> {
    let (l_res, channel, g) = dam_break_ritter_geometry(h0);
    let t = tau * sqrt(h0 / g);
    dfsph_dam_break_convergence(dx_list, h0, l_res, t, channel)
}

/// Max relative error of Tait pressure vs `p = ρ₀ g (H − z)` on a vertical profile.
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
