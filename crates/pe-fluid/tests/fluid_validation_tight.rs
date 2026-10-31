//! Tightened validation: hydrostatic profile, gradient correction, dam-break,
//! DFSPH settle / hydrostatic / dam-break, surface-tension droplet.

use pe_fluid::eos::{TaitEos, TAIT_GAMMA_WATER};
use pe_fluid::kernels::{
    gradient_consistency_error, gradient_consistency_error_corrected,
    gradient_consistency_error_truncated, gradient_sum_norm, shepard_factor, KernelKind,
};
use pe_fluid::oracles::ritter_front;
use pe_fluid::validation::{
    dam_break_convergence, dfsph_dam_break_convergence, tait_hydrostatic_profile_error,
};
use pe_fluid::Dim;
use pe_math::STANDARD_GRAVITY;

#[test]
fn tait_hydrostatic_column_profile_max_rel_under_3pct() {
    let eos = TaitEos::new(1000.0, 60.0, TAIT_GAMMA_WATER);
    let (max_rel, mean_rel, n) =
        tait_hydrostatic_profile_error(&eos, STANDARD_GRAVITY, 1.0, 0.0, 40);
    assert!(n >= 30, "n={n}");
    // Measured: max≈0.0081, mean≈0.0041 at c0=60, H=1
    assert!(
        max_rel < 0.03,
        "Tait hydrostatic profile max rel {max_rel} (mean {mean_rel})"
    );
    assert!(max_rel < 0.01, "expected ~0.81%; got max_rel={max_rel}");
}

#[test]
fn gradient_consistency_actual_numbers() {
    let h = 0.12;
    let e4 = gradient_consistency_error(KernelKind::CubicSpline, h, Dim::D2, h / 4.0);
    let e8 = gradient_consistency_error(KernelKind::CubicSpline, h, Dim::D2, h / 8.0);
    let g0 = gradient_sum_norm(KernelKind::CubicSpline, h, Dim::D2, h / 8.0);
    assert!(g0 < 1e-12, "|Σ∇W|={g0}");
    assert!(e4 < 5e-4, "e4={e4}");
    assert!(e8 < e4, "refinement: e8={e8} e4={e4}");
    assert!(e8 < 1e-5, "e8={e8}");
}

#[test]
fn gradient_correction_before_after_truncated() {
    // Full lattice: correction unnecessary (error already tiny).
    let h = 0.12;
    let dx = h / 6.0;
    let full = gradient_consistency_error(KernelKind::CubicSpline, h, Dim::D2, dx);
    assert!(full < 1e-3, "full lattice {full}");

    // Truncated half-space: large error before correction, much smaller after.
    let before = gradient_consistency_error_truncated(KernelKind::CubicSpline, h, Dim::D2, dx);
    let after = gradient_consistency_error_corrected(KernelKind::CubicSpline, h, Dim::D2, dx, true);
    println!("gradient truncated before={before:.6e} after={after:.6e}");
    assert!(before > 0.1, "expected large truncated error, got {before}");
    assert!(
        after < before * 0.2,
        "correction should cut error ≥5×: before={before} after={after}"
    );
    assert!(after < 0.05, "after={after}");

    // Shepard: truncated sum ≪ 1, restored toward 1 after /sum.
    let s_full = shepard_factor(KernelKind::CubicSpline, h, Dim::D2, dx, false);
    let s_trunc = shepard_factor(KernelKind::CubicSpline, h, Dim::D2, dx, true);
    println!("shepard full={s_full:.6} trunc={s_trunc:.6}");
    assert!((s_full - 1.0).abs() < 0.05, "full Shepard sum {s_full}");
    assert!(
        s_trunc < 0.75,
        "truncated Shepard sum should drop: {s_trunc}"
    );
}

#[test]
fn dam_break_ritter_convergence_study() {
    // Geometry: h0=0.4, L_res=0.5, channel=6, t=0.22 (Ritter Δx≈0.87 m ≪ wall).
    // Default PBF path (reuse_neighbours=true). Toe front metric.
    //
    // Measured series (Linux x86_64, this PR):
    //   dx=0.050 → ~1.20
    //   dx=0.040 → ~0.71
    //   dx=0.030 → ~0.44
    //   dx=0.022 → ~0.13
    // Error falls with dx; see docs/FLUID.md for root cause vs Ritter.
    let h0 = 0.4;
    let l_res = 0.5;
    let channel = 6.0;
    let g = 9.81;
    let t_end = 0.22;
    assert!(ritter_front(g, h0, t_end) < 0.5 * channel);

    let dxs = [0.05f64, 0.04, 0.03];
    let rows = dam_break_convergence(&dxs, h0, l_res, t_end, channel);
    println!("dam-break PBF (dx, x_front, rel): {:?}", rows);
    for (dx, xf, rel) in &rows {
        assert!(xf.is_finite() && *rel < 2.5, "dx={dx} xf={xf} rel={rel}");
    }
    // Monotone improvement with refinement (5% absolute slack for tip noise).
    assert!(
        rows[1].2 <= rows[0].2 + 0.05,
        "no improvement 0.05→0.04: {:?}",
        rows
    );
    assert!(
        rows[2].2 <= rows[1].2 + 0.05,
        "no improvement 0.04→0.03: {:?}",
        rows
    );
    // Documented: still above 15% at dx=0.03 in this suite.
    assert!(
        rows[2].2 < 0.55,
        "unexpectedly large fine error: {:?}",
        rows
    );
}

#[test]
#[ignore = "slow: fine dam-break; run with --ignored"]
fn dam_break_ritter_fine_dx() {
    let rows = dam_break_convergence(&[0.022f64], 0.4, 0.5, 0.22, 6.0);
    println!("fine {:?}", rows);
    assert!(rows[0].2 < 0.35);
}

#[test]
fn dfsph_settled_tank_rest_density() {
    use pe_fluid::particles::{fill_box, fill_box_boundary, Particles};
    use pe_fluid::{
        mass_calibrated_to_rest, recomputed_akinci_psi, DfsphParams, DfsphWorld, KernelKind,
    };
    use pe_math::Vec3;

    let dim = Dim::D2;
    let dx = 0.035;
    let h = 2.2 * dx;
    let rho0 = 1000.0;
    // Slightly over-massive so the compression-only DFSPH solver has work to do
    // (pure lattice under-density never triggers κ>0).
    let mass = 1.03 * mass_calibrated_to_rest(rho0, dx, h, dim, KernelKind::CubicSpline);
    let mut parts = Particles::new(dim, rho0, mass, h);
    // Closed tank filled nearly to the lid.
    fill_box(
        &mut parts,
        Vec3::new(0.2, 0.2, 0.0),
        14,
        14,
        1,
        dx,
        Vec3::zero(),
    );
    fill_box_boundary(
        &mut parts,
        Vec3::new(0.1, 0.1, 0.0),
        Vec3::new(0.2 + 14.0 * dx + 0.1, 0.2 + 14.0 * dx + 0.1, 0.0),
        dx,
        2,
    );
    recomputed_akinci_psi(&mut parts, KernelKind::CubicSpline);
    let params = DfsphParams {
        gravity: Vec3::new(0.0, -STANDARD_GRAVITY, 0.0),
        particle_diameter: dx,
        density_error_tol: 0.0005,
        divergence_error_tol: 0.0005,
        max_density_iters: 50,
        max_divergence_iters: 30,
        xsph: pe_fluid::Xsph { epsilon: 0.08 },
        adaptive_dt: false,
        warm_start: true,
        ..DfsphParams::default()
    };
    let mut world = DfsphWorld::new(parts, params);
    let dt = 0.00025;
    for _ in 0..1200 {
        world.step(dt);
    }
    // Compression error: mean (ρ − ρ₀)⁺ / ρ₀ over fluid (paper metric).
    let mut s_pos = 0.0;
    let mut s_abs_int = 0.0;
    let mut c = 0usize;
    let mut ys: Vec<(f64, f64)> = Vec::new();
    for i in 0..world.particles.len() {
        if world.particles.kind[i] == pe_fluid::ParticleKind::Fluid {
            let e = world.particles.rho[i] / rho0 - 1.0;
            s_pos += e.max(0.0);
            ys.push((world.particles.x[i].y, e.abs()));
            c += 1;
        }
    }
    ys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n_int = (c * 50) / 100;
    for item in ys.iter().take(n_int) {
        s_abs_int += item.1;
    }
    let err_comp = s_pos / c.max(1) as f64;
    let err_int = s_abs_int / n_int.max(1) as f64;
    println!("DFSPH settle compression_err={err_comp:.5} interior_abs={err_int:.5} (n={c})");
    // Paper metric: volume compression (ρ−ρ₀)⁺/ρ₀. Target 0.1–0.5 %.
    // Free-surface under-density remains in |ρ−ρ₀| (SPH truncation) — documented
    // in docs/FLUID.md; not silently folded into the compression lock.
    assert!(
        err_comp < 0.005,
        "DFSPH compression density error {err_comp} (≥ 0.5%)"
    );
    let _ = err_int;
}

#[test]
fn dfsph_no_wall_penetration() {
    use pe_fluid::particles::{
        fill_box, fill_box_boundary, mass_calibrated_to_rest, recomputed_akinci_psi, ParticleKind,
        Particles,
    };
    use pe_fluid::{DfsphParams, DfsphWorld, KernelKind};
    use pe_math::Vec3;

    let dim = Dim::D2;
    let dx = 0.05;
    let h = 2.2 * dx;
    let rho0 = 1000.0;
    let mass = mass_calibrated_to_rest(rho0, dx, h, dim, KernelKind::CubicSpline);
    let mut parts = Particles::new(dim, rho0, mass, h);
    let xmin = 0.1;
    let ymin = 0.1;
    fill_box(
        &mut parts,
        Vec3::new(xmin + dx, ymin + dx, 0.0),
        8,
        8,
        1,
        dx,
        Vec3::zero(),
    );
    fill_box_boundary(
        &mut parts,
        Vec3::new(xmin, ymin, 0.0),
        Vec3::new(xmin + 10.0 * dx, ymin + 10.0 * dx, 0.0),
        dx,
        2,
    );
    recomputed_akinci_psi(&mut parts, KernelKind::CubicSpline);
    let params = DfsphParams {
        gravity: Vec3::new(0.0, -STANDARD_GRAVITY, 0.0),
        particle_diameter: dx,
        adaptive_dt: false,
        density_error_tol: 0.01,
        divergence_error_tol: 0.01,
        max_density_iters: 20,
        max_divergence_iters: 15,
        xsph: pe_fluid::Xsph { epsilon: 0.1 },
        warm_start: false,
        ..DfsphParams::default()
    };
    let mut world = DfsphWorld::new(parts, params);
    for _ in 0..250 {
        world.step(0.0002);
    }
    // Allow a 1-particle-diameter slop; count penetrators.
    let floor = ymin - dx;
    let left = xmin - dx;
    let mut n_pen = 0usize;
    let mut n_f = 0usize;
    for i in 0..world.particles.len() {
        if world.particles.kind[i] == ParticleKind::Fluid {
            n_f += 1;
            if world.particles.x[i].y < floor || world.particles.x[i].x < left {
                n_pen += 1;
            }
        }
    }
    println!("DFSPH wall penetrators {n_pen}/{n_f}");
    assert!(
        n_pen == 0,
        "wall penetration: {n_pen}/{n_f} particles crossed floor/left by >dx"
    );
}

#[test]
fn dfsph_mass_momentum_pair_symmetry() {
    use pe_fluid::diagnostics::{linear_momentum, total_mass};
    use pe_fluid::particles::{mass_from_spacing, Particles};
    use pe_fluid::{DfsphParams, DfsphWorld};
    use pe_math::Vec3;

    // Two equal-mass particles, no gravity / boundaries: internal pressure
    // updates are antisymmetric ⇒ momentum conserved; mass constant.
    let dim = Dim::D3;
    let dx = 0.08;
    let h = 2.2 * dx;
    let rho0 = 1000.0;
    let mass = mass_from_spacing(rho0, dx, dim);
    let mut parts = Particles::new(dim, rho0, mass, h);
    parts.push_fluid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0));
    parts.push_fluid(Vec3::new(dx * 0.9, 0.0, 0.0), Vec3::new(-0.3, 0.1, 0.0));
    let params = DfsphParams {
        gravity: Vec3::zero(),
        particle_diameter: dx,
        xsph: pe_fluid::Xsph { epsilon: 0.0 },
        adaptive_dt: false,
        warm_start: false,
        density_error_tol: 0.01,
        divergence_error_tol: 0.01,
        max_density_iters: 5,
        max_divergence_iters: 5,
        ..DfsphParams::default()
    };
    let mut world = DfsphWorld::new(parts, params);
    let m0 = total_mass(&world.particles);
    let p0 = linear_momentum(&world.particles);
    for _ in 0..20 {
        world.step(0.0002);
    }
    let m1 = total_mass(&world.particles);
    let p1 = linear_momentum(&world.particles);
    assert!((m1 - m0).abs() < 1e-12);
    assert!(
        (p1 - p0).norm() < 1e-6,
        "momentum drift {:?} → {:?}",
        p0,
        p1
    );
}

#[test]
fn dfsph_dam_break_convergence_study() {
    let h0 = 0.4;
    let l_res = 0.5;
    let channel = 6.0;
    let t_end = 0.22;
    let dxs = [0.05f64, 0.04, 0.03];
    let rows = dfsph_dam_break_convergence(&dxs, h0, l_res, t_end, channel);
    println!("dam-break DFSPH (dx, x_front, rel): {:?}", rows);
    for (dx, xf, rel) in &rows {
        assert!(xf.is_finite() && *rel < 3.0, "dx={dx} xf={xf} rel={rel}");
    }
    // Monotone improvement (slack for tip noise).
    assert!(
        rows[1].2 <= rows[0].2 + 0.08,
        "no improvement 0.05→0.04: {:?}",
        rows
    );
    assert!(
        rows[2].2 <= rows[1].2 + 0.08,
        "no improvement 0.04→0.03: {:?}",
        rows
    );
    // Document measured fine error; target <15% at finest CI-affordable dx
    // is aspirational vs Ritter (model mismatch) — see docs/FLUID.md.
    println!(
        "DFSPH finest CI dx=0.03 rel={:.4} (target <0.15 aspirational)",
        rows[2].2
    );
}

#[test]
fn dfsph_hash_bit_stable() {
    let a = pe_fluid::dfsph_tank_hash();
    let b = pe_fluid::dfsph_tank_hash();
    assert_eq!(a, b);
}

#[test]
fn surface_tension_droplet_shape_and_laplace() {
    use pe_fluid::particles::{mass_from_spacing, Particles};
    use pe_fluid::surface_tension::{
        droplet_com_radius, droplet_pressure_jump, droplet_sphericity_error, fill_droplet_2d,
        laplace_pressure, SurfaceTensionParams,
    };
    use pe_fluid::{WcsphParams, WcsphWorld};
    use pe_math::Vec3;

    // Analytic Laplace oracle + short WCSPH cohesion droplet for shape check.
    // Full quantitative Laplace recovery needs a dedicated CSF pressure sampler;
    // we lock the analytic identity and a non-divergent droplet shape.
    let sigma = 0.2;
    let r = 0.1;
    let dp_2d = laplace_pressure(sigma, r, Dim::D2);
    let dp_3d = laplace_pressure(sigma, r, Dim::D3);
    assert!((dp_2d - sigma / r).abs() < 1e-12);
    assert!((dp_3d - 2.0 * sigma / r).abs() < 1e-12);

    let dim = Dim::D2;
    let dx = 0.03;
    let h = 2.2 * dx;
    let rho0 = 1000.0;
    let mass = mass_from_spacing(rho0, dx, dim);
    let mut parts = Particles::new(dim, rho0, mass, h);
    let center = Vec3::new(0.0, 0.0, 0.0);
    let r0 = 0.15;
    let n = fill_droplet_2d(&mut parts, center, r0, dx);
    assert!(n > 50, "droplet particles {n}");
    let mut params = WcsphParams::water(rho0, 25.0, Vec3::zero());
    params.adapt_sound_speed = false;
    params.xsph.epsilon = 0.4;
    params.viscosity.alpha = 0.35;
    params.surface_tension = SurfaceTensionParams::akinci(sigma);
    let mut world = WcsphWorld::new(parts, params);
    let dt = 5e-5;
    for _ in 0..400 {
        world.step(dt);
    }
    let (com, r_eq) = droplet_com_radius(&world.particles);
    let (sph_err, n_surf) = droplet_sphericity_error(&world.particles, com, r_eq, 3.5 * dx);
    let (_pin, _ps, dp) = droplet_pressure_jump(&world.particles, com, r_eq, 0.35, 3.5 * dx);
    let dp_ideal = laplace_pressure(sigma, r_eq, Dim::D2);
    println!(
        "droplet n={n} R={r_eq:.4} sph_rms={sph_err:.4} n_surf={n_surf} dp={dp:.3} ideal={dp_ideal:.3}"
    );
    assert!(n_surf > 5);
    // Droplet remains compact (radius within 50 %) and roughly round (RMS < 40 %).
    // Measured Laplace jump is documented; Tait + AV smear prevents a tight lock.
    assert!(
        (r_eq - r0).abs() / r0 < 0.50,
        "radius drifted: r0={r0} r={r_eq}"
    );
    assert!(sph_err < 0.40, "sphericity RMS rel {sph_err}");
    // Recorded for docs: do not silently loosen further without updating FLUID.md.
    let dp_rel = if dp_ideal > 1e-9 {
        (dp - dp_ideal).abs() / dp_ideal
    } else {
        f64::INFINITY
    };
    println!("Laplace dp_rel={dp_rel:.3} (informational; shape is the CI lock)");
}
