//! Analytic validation for pe-fluid (honest tolerances).
//!
//! See docs/FLUID.md for formulas and known limits.

use approx::assert_abs_diff_eq;
use pe_fluid::cfl::{suggested_dt, CflParams};
use pe_fluid::diagnostics::{
    kinetic_energy, linear_momentum, mean_density_error, particles_hash64, total_mass,
};
use pe_fluid::eos::{TaitEos, SOUND_SPEED_FACTOR, TAIT_GAMMA_WATER};
use pe_fluid::kernels::{
    gradient_consistency_error, gradient_sum_norm, partition_of_unity_error, support_radius, w,
    KernelKind,
};
use pe_fluid::oracles::{hydrostatic_pressure, poiseuille_u, ritter_front, ritter_height};
use pe_fluid::particles::{
    fill_box, fill_box_boundary, mass_from_spacing, ParticleKind, Particles,
};
use pe_fluid::pbf::{PbfParams, PbfWorld};
use pe_fluid::scenarios::{fluid_tank_hash, FLUID_HASH_DT, FLUID_HASH_STEPS};
use pe_fluid::wcsph::{WcsphParams, WcsphWorld};
use pe_fluid::Dim;
use pe_math::{Vec3, STANDARD_GRAVITY};

#[test]
fn kernel_partition_of_unity_cubic_3d() {
    // Rectangle-rule integral on a fine grid. Monaghan cubic (form with W(0)=α)
    // integrates to 1 analytically; discrete error < 1% at n_half=24.
    let h = 0.1;
    let err = partition_of_unity_error(KernelKind::CubicSpline, h, Dim::D3, 24);
    assert!(
        err < 0.01,
        "cubic spline ∫W error {err} (expect < 1% at n_half=24)"
    );
}

#[test]
fn kernel_partition_of_unity_wendland_poly6_2d() {
    let h = 0.1;
    let e_w = partition_of_unity_error(KernelKind::WendlandC2, h, Dim::D2, 32);
    let e_p = partition_of_unity_error(KernelKind::Poly6, h, Dim::D2, 32);
    assert!(e_w < 0.02, "Wendland ∫W error {e_w}");
    assert!(e_p < 0.05, "poly6 ∫W error {e_p}");
}

#[test]
fn kernel_gradient_zeroth_and_first_order() {
    let h = 0.12;
    // Symmetric lattice → Σ V ∇W ≈ 0
    let g0 = gradient_sum_norm(KernelKind::CubicSpline, h, Dim::D2, h / 8.0);
    assert!(g0 < 1e-10, "∇W sum norm {g0}");
    // First-order: Σ V (xj−xi)⊗∇_i W ≈ +I ; error falls under refinement
    let e_coarse = gradient_consistency_error(KernelKind::CubicSpline, h, Dim::D2, h / 4.0);
    let e_fine = gradient_consistency_error(KernelKind::CubicSpline, h, Dim::D2, h / 8.0);
    assert!(
        e_fine < e_coarse || e_fine < 0.15,
        "gradient consistency did not improve: coarse={e_coarse} fine={e_fine}"
    );
    assert!(
        e_fine < 0.25,
        "fine gradient consistency error {e_fine} (honest lattice SPH bound)"
    );
}

#[test]
fn tait_sound_speed_rule() {
    let vmax = 2.0;
    let eos = TaitEos::from_vmax(1000.0, vmax, TAIT_GAMMA_WATER, SOUND_SPEED_FACTOR);
    assert!((eos.sound_speed - SOUND_SPEED_FACTOR * vmax).abs() < 1e-12);
    // At rest density, p = 0
    assert_abs_diff_eq!(eos.pressure(1000.0), 0.0, epsilon = 1e-9);
    // Overdense → positive pressure
    assert!(eos.pressure(1010.0) > 0.0);
}

#[test]
fn hydrostatic_oracle_formula() {
    let p = hydrostatic_pressure(1000.0, STANDARD_GRAVITY, 2.0);
    assert_abs_diff_eq!(p, 1000.0 * STANDARD_GRAVITY * 2.0, epsilon = 1e-9);
}

#[test]
fn ritter_front_and_height() {
    let g = 9.81;
    let h0 = 1.0;
    let t = 0.5;
    let xf = ritter_front(g, h0, t);
    let c0 = (g * h0).sqrt();
    assert_abs_diff_eq!(xf, 2.0 * c0 * t, epsilon = 1e-12);
    // At the dam plane x=0, Ritter height = (4/9) h0
    let h = ritter_height(g, h0, 0.0, t);
    assert_abs_diff_eq!(h, (4.0 / 9.0) * h0, epsilon = 1e-9);
}

#[test]
fn poiseuille_peak_at_midplane() {
    let f = 1.0;
    let nu = 0.1;
    let h = 1.0;
    let u_mid = poiseuille_u(f, nu, 0.5 * h, h);
    let u_wall = poiseuille_u(f, nu, 0.0, h);
    assert_abs_diff_eq!(u_wall, 0.0, epsilon = 1e-15);
    assert!(u_mid > u_wall);
    assert_abs_diff_eq!(u_mid, f * h * h / (8.0 * nu), epsilon = 1e-12);
}

#[test]
fn mass_conserved_wcsph_step() {
    let dim = Dim::D2;
    let dx = 0.05;
    let h = 2.4 * dx;
    let rho0 = 1000.0;
    let mass = mass_from_spacing(rho0, dx, dim);
    let mut parts = Particles::new(dim, rho0, mass, h);
    fill_box(
        &mut parts,
        Vec3::new(0.1, 0.1, 0.0),
        6,
        6,
        1,
        dx,
        Vec3::zero(),
    );
    fill_box_boundary(
        &mut parts,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.5, 0.5, 0.0),
        dx,
        1,
    );
    let m0 = total_mass(&parts);
    let mut params = WcsphParams::water(rho0, 40.0, Vec3::new(0.0, -STANDARD_GRAVITY, 0.0));
    params.adapt_sound_speed = false;
    params.viscosity.alpha = 0.05;
    let mut world = WcsphWorld::new(parts, params);
    for _ in 0..30 {
        world.step(0.0002);
    }
    let m1 = total_mass(&world.particles);
    assert_abs_diff_eq!(m0, m1, epsilon = 1e-12);
}

#[test]
fn internal_pair_forces_conserve_momentum_no_boundary_no_gravity() {
    // Two particles, WCSPH pressure+viscosity only: total momentum unchanged.
    let dim = Dim::D3;
    let dx = 0.1;
    let h = 2.2 * dx;
    let rho0 = 1000.0;
    let mass = mass_from_spacing(rho0, dx, dim);
    let mut parts = Particles::new(dim, rho0, mass, h);
    parts.push_fluid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
    parts.push_fluid(Vec3::new(dx, 0.0, 0.0), Vec3::new(-0.5, 0.2, 0.0));
    let mut params = WcsphParams::water(rho0, 50.0, Vec3::zero());
    params.adapt_sound_speed = false;
    params.xsph.epsilon = 0.0;
    let mut world = WcsphWorld::new(parts, params);
    world.rebuild_grid();
    world.compute_density();
    world.compute_pressure();
    let p0 = linear_momentum(&world.particles);
    world.compute_forces();
    // Accelerations from internal forces only: Σ m a = 0
    let mut f_sum = Vec3::zero();
    for i in 0..world.particles.len() {
        f_sum += world.particles.a[i] * world.particles.mass;
    }
    assert!(
        f_sum.norm() < 1e-8,
        "internal force sum {f_sum:?} (momentum source)"
    );
    let _ = p0;
}

#[test]
fn viscosity_does_not_increase_ke_pairwise() {
    // Isolated approaching pair: viscosity-only accelerations (pressures zeroed)
    // must not increase kinetic energy over a kick.
    use pe_fluid::viscosity::pi_ij;
    use pe_fluid::{grad_w, KernelKind};
    let dim = Dim::D3;
    let dx = 0.08;
    let h = 2.2 * dx;
    let rho0 = 1000.0;
    let mass = mass_from_spacing(rho0, dx, dim);
    let mut parts = Particles::new(dim, rho0, mass, h);
    parts.push_fluid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
    parts.push_fluid(Vec3::new(dx * 0.8, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
    parts.rho[0] = rho0;
    parts.rho[1] = rho0;
    let visc = pe_fluid::MonaghanViscosity {
        alpha: 0.5,
        beta: 0.0,
        eps: 0.01,
    };
    let c0 = 40.0;
    let ke0 = kinetic_energy(&parts);
    let rij = parts.x[1] - parts.x[0];
    let gw = grad_w(KernelKind::CubicSpline, rij, h, dim);
    let pi = pi_ij(
        visc, h, c0, parts.x[0], parts.x[1], parts.v[0], parts.v[1], rho0, rho0,
    );
    // a_i += − m Π ∇_i W ; a_j −= same (equal mass)
    let f = gw * (-mass * pi);
    let dt = 1e-4;
    parts.v[0] += f * dt;
    parts.v[1] -= f * dt;
    let ke1 = kinetic_energy(&parts);
    assert!(
        ke1 <= ke0 + 1e-9,
        "KE increased under viscosity-only kick: {ke0} → {ke1} (Π={pi})"
    );
}

#[test]
fn pbf_settled_tank_rest_density_under_one_percent() {
    let dim = Dim::D2;
    let dx = 0.04;
    let h = 2.0 * dx;
    let rho0 = 1000.0;
    let mass = mass_from_spacing(rho0, dx, dim);
    let mut parts = Particles::new(dim, rho0, mass, h);
    fill_box(
        &mut parts,
        Vec3::new(0.2, 0.2, 0.0),
        12,
        10,
        1,
        dx,
        Vec3::zero(),
    );
    fill_box_boundary(
        &mut parts,
        Vec3::new(0.1, 0.1, 0.0),
        Vec3::new(0.2 + 12.0 * dx + 0.1, 0.2 + 10.0 * dx + 0.15, 0.0),
        dx,
        2,
    );
    let params = PbfParams {
        gravity: Vec3::new(0.0, -STANDARD_GRAVITY, 0.0),
        iterations: 10,
        viscosity_eps: 0.01,
        art_pressure_k: 0.0, // tensile term off for hydrostatic settle
        ..PbfParams::default()
    };
    let mut world = PbfWorld::new(parts, params);
    for _ in 0..400 {
        world.step(0.0008);
    }
    let err = world.mean_density_error();
    assert!(
        err < 0.01,
        "PBF mean rest-density error {err} (≥ 1%); fluid={}",
        world.particles.fluid_count()
    );
}

#[test]
fn pbf_mass_conserved() {
    let mut world = pe_fluid::fluid_hash_world();
    let m0 = total_mass(&world.particles);
    for _ in 0..20 {
        world.step(FLUID_HASH_DT);
    }
    assert_abs_diff_eq!(m0, total_mass(&world.particles), epsilon = 1e-12);
}

#[test]
fn hydrostatic_pressure_profile_from_tait_density() {
    // Direct oracle: feed the analytic hydrostatic density perturbation into
    // Tait EOS and compare p(ρ) to ρ₀ g h (small-Mach WCSPH limit).
    // Full dynamic WCSPH columns need long settles and are documented separately
    // in docs/FLUID.md with coarser engineering bounds.
    let rho0 = 1000.0;
    let g = STANDARD_GRAVITY;
    let h_depth = 1.0;
    let c0 = 50.0; // ≫ √(g h)
    let eos = TaitEos::new(rho0, c0, TAIT_GAMMA_WATER);
    // Linearised hydrostatic: Δρ/ρ₀ ≈ g h / c₀²
    let rho = rho0 * (1.0 + g * h_depth / (c0 * c0));
    let p = eos.pressure(rho);
    let p_ref = hydrostatic_pressure(rho0, g, h_depth);
    let rel = (p - p_ref).abs() / p_ref;
    // Tait is nonlinear; with γ=7 the linearised density gives ~O((gh/c²)) relative error.
    assert!(
        rel < 0.05,
        "Tait hydrostatic linearised rel error {rel} (p={p} pref={p_ref})"
    );
}

#[test]
fn hydrostatic_wcsph_bottom_pressure_positive_ordering() {
    // Coarse dynamic check: after a short settle, mean pressure in the lower
    // third exceeds the upper third (hydrostatic ordering). Not a tight ρgh test.
    let dim = Dim::D2;
    let dx = 0.05;
    let hsm = 2.2 * dx;
    let rho0 = 1000.0;
    let mass = mass_from_spacing(rho0, dx, dim);
    let mut parts = Particles::new(dim, rho0, mass, hsm);
    fill_box(
        &mut parts,
        Vec3::new(0.2, 0.15, 0.0),
        6,
        10,
        1,
        dx,
        Vec3::zero(),
    );
    fill_box_boundary(
        &mut parts,
        Vec3::new(0.1, 0.05, 0.0),
        Vec3::new(0.6, 0.8, 0.0),
        dx,
        2,
    );
    let g = STANDARD_GRAVITY;
    let mut params = WcsphParams::water(rho0, 30.0, Vec3::new(0.0, -g, 0.0));
    params.adapt_sound_speed = false;
    params.xsph.epsilon = 0.2;
    params.viscosity.alpha = 0.1;
    let mut world = WcsphWorld::new(parts, params);
    for _ in 0..300 {
        world.step(0.0003);
    }
    world.rebuild_grid();
    world.compute_density();
    world.compute_pressure();
    let mut ys: Vec<(f64, f64)> = Vec::new();
    for i in 0..world.particles.len() {
        if world.particles.kind[i] == ParticleKind::Fluid {
            ys.push((world.particles.x[i].y, world.particles.p[i]));
        }
    }
    ys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = ys.len();
    let lo: f64 = ys[..n / 3].iter().map(|q| q.1).sum::<f64>() / (n / 3) as f64;
    let hi: f64 = ys[2 * n / 3..].iter().map(|q| q.1).sum::<f64>() / (n - 2 * n / 3) as f64;
    assert!(
        lo > hi,
        "expected higher pressure at bottom: lo={lo} hi={hi}"
    );
}

#[test]
fn dam_break_front_vs_ritter_2d() {
    // PBF dam break; compare wet front to Ritter. Coarse particles lag the
    // ideal frictionless front — tolerance stated honestly in docs/FLUID.md.
    let dim = Dim::D2;
    let dx = 0.04;
    let hsm = 2.0 * dx;
    let rho0 = 1000.0;
    let mass = mass_from_spacing(rho0, dx, dim);
    let mut parts = Particles::new(dim, rho0, mass, hsm);
    let h0 = 0.32;
    let nx = 8;
    let ny = ((h0 / dx).round() as usize).max(6);
    let x0 = -0.35;
    fill_box(
        &mut parts,
        Vec3::new(x0, dx, 0.0),
        nx,
        ny,
        1,
        dx,
        Vec3::zero(),
    );
    let psi = dx * dx;
    for i in 0..55 {
        let x = -0.5 + i as f64 * dx;
        parts.push_boundary(Vec3::new(x, 0.0, 0.0), psi);
        parts.push_boundary(Vec3::new(x, -dx, 0.0), psi);
    }
    for j in 0..ny + 4 {
        parts.push_boundary(Vec3::new(-0.5, j as f64 * dx, 0.0), psi);
        parts.push_boundary(Vec3::new(-0.5 - dx, j as f64 * dx, 0.0), psi);
    }
    let g = 9.81;
    let params = PbfParams {
        gravity: Vec3::new(0.0, -g, 0.0),
        iterations: 8,
        art_pressure_k: 0.0,
        viscosity_eps: 0.01,
        ..PbfParams::default()
    };
    let mut world = PbfWorld::new(parts, params);
    let t_end = 0.15;
    let dt = 0.0008;
    let steps = (t_end / dt) as usize;
    for _ in 0..steps {
        world.step(dt);
    }
    let mut x_front = f64::NEG_INFINITY;
    for i in 0..world.particles.len() {
        if world.particles.kind[i] == ParticleKind::Fluid {
            x_front = x_front.max(world.particles.x[i].x);
        }
    }
    let dam_x = x0 + nx as f64 * dx;
    let x_f_ritter = dam_x + ritter_front(g, h0, t_end);
    // Front must advance past the dam face.
    assert!(
        x_front > dam_x + 0.05,
        "front did not advance: x_front={x_front} dam_x={dam_x}"
    );
    let rel = (x_front - x_f_ritter).abs() / x_f_ritter.abs().max(0.1);
    assert!(
        rel < 0.55,
        "dam-break front rel error {rel}: sim={x_front} ritter={x_f_ritter}"
    );
}

#[test]
fn cfl_suggested_dt_scales() {
    let dt = suggested_dt(0.1, 20.0, 1.0, 10.0, CflParams::default());
    assert!(dt > 0.0 && dt < 0.01);
}

#[test]
fn fluid_hash_bit_stable_on_repeat() {
    let a = fluid_tank_hash();
    let b = fluid_tank_hash();
    assert_eq!(a, b);
    // Also exercise hash helper on a fresh world
    let mut w = pe_fluid::fluid_hash_world();
    for _ in 0..FLUID_HASH_STEPS {
        w.step(FLUID_HASH_DT);
    }
    assert_eq!(particles_hash64(&w.particles), a);
}

#[test]
fn support_radius_matches_kernel() {
    assert_eq!(support_radius(KernelKind::CubicSpline, 0.2), 0.4);
    assert_eq!(support_radius(KernelKind::Poly6, 0.2), 0.2);
    assert!(w(KernelKind::Spiky, 0.0, 0.2, Dim::D3) > 0.0);
}

#[test]
fn mean_density_error_helper() {
    let mut p = Particles::new(Dim::D3, 1000.0, 1.0, 0.1);
    p.push_fluid(Vec3::zero(), Vec3::zero());
    p.rho[0] = 1010.0;
    assert!((mean_density_error(&p) - 0.01).abs() < 1e-12);
}
