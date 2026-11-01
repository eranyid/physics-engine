//! Two-way fluid↔rigid coupling oracles (Akinci 2012).
//! Measured tolerances locked here and in `docs/FLUID.md`. Do not loosen silently.

use pe_core::body::{inertia_solid_sphere, RigidBody};
use pe_fluid::coupling::{
    archimedes_box_draft, barge_heave_added_mass_2d, coupling_tank_hash, estimate_heave_period,
    floating_box_draft_error_wcsph, heave_period_with_added_mass, make_floating_box_2d,
    make_floating_box_wcsph_2d, max_box_penetration, max_box_penetration_wcsph,
    sphere_freefall_accel_added_mass, CoupledWcsphWorld, FluidRigidCoupler,
};
use pe_fluid::kernels::KernelKind;
use pe_fluid::particles::{
    fill_box, fill_box_boundary, mass_calibrated_to_rest, recomputed_akinci_psi, Particles,
};
use pe_fluid::wcsph::{WcsphParams, WcsphWorld};
use pe_fluid::{Dim, ParticleKind};
use pe_math::libm_ops::fabs;
use pe_math::{Vec3, STANDARD_GRAVITY};

#[test]
fn coupling_hash_bit_stable_on_repeat() {
    let a = coupling_tank_hash();
    let b = coupling_tank_hash();
    assert_eq!(a, b, "coupling hash not bit-stable: {a:016x} vs {b:016x}");
    println!("COUPLING_HASH={a:016x}");
}

#[test]
fn floating_box_archimedes_draft() {
    // WCSPH + Akinci samples. Measured (Linux x86_64): rel ≈ 0.0653 at dx=0.05,
    // fixed 2000×2.5e-4 (t=0.5). Instantaneous draft crosses within ≪1% during
    // settle at dx≤0.05; coarser dx is unstable. Lock <10% ≈ 1.5× measured
    // (do not loosen). See docs/FLUID.md dx notes.
    let dx = 0.05;
    let water_h = 0.40;
    let hx = 0.12;
    let hy = 0.08;
    let rho_b = 500.0;
    let mut world = make_floating_box_wcsph_2d(dx, 0.9, water_h, hx, hy, rho_b);
    let dt = 2.5e-4; // CFL ~0.2 dx/c0 with c0=40, dx=0.05
    for _ in 0..2000 {
        world.step(dt);
    }
    let (d_sim, d_ref, rel) = floating_box_draft_error_wcsph(&world, hy, water_h);
    let d_analytic = archimedes_box_draft(rho_b, 1000.0, 2.0 * hy);
    println!("draft_sim={d_sim:.5} draft_ref={d_ref:.5} analytic={d_analytic:.5} rel={rel:.4}");
    assert!(
        rel < 0.10,
        "Archimedes draft rel {rel} (measured ~0.065; lock <10%)"
    );
    assert!(fabs(d_ref - d_analytic) / d_analytic < 0.02);
}

#[test]
fn floating_box_heave_period() {
    // Underdamped free heave after settle: α=0.01, kick=0.04, zero-crossings vs
    // T = 2π √((m+m_a)/(ρ g A)) with 2-D barge m_a = ρ π B²/8.
    // Measured (Linux x86_64, dx=0.05): T≈0.776 vs 0.838 (rel≈0.074).
    // Finer/coarser dx is not monotone better (sweet spot); lock at CI dx.
    // Lock <0.15 ≈ 2× measured.
    let dx = 0.05;
    let water_h = 0.40;
    let hx = 0.12;
    let hy = 0.08;
    let rho_b = 500.0;
    let kick = 0.04;
    let mut world = make_floating_box_wcsph_2d(dx, 0.9, water_h, hx, hy, rho_b);
    let dt = 2.5e-4;
    for _ in 0..2000 {
        world.step(dt);
    }
    let y_eq = world.rigid.state.x[0].y;
    world.fluid.params.viscosity.alpha = 0.01;
    world.fluid.params.xsph.epsilon = 0.0;
    world.rigid.state.x[0].y = y_eq - kick;
    world.rigid.state.v[0] = Vec3::zero();
    let mut samples = Vec::new();
    for _ in 0..16000 {
        world.step(dt);
        samples.push((world.rigid.t, world.rigid.state.x[0].y));
    }
    let t0 = samples[0].0;
    let win: Vec<_> = samples
        .iter()
        .cloned()
        .filter(|(t, _)| *t >= t0 + 0.1 && *t <= t0 + 4.0)
        .collect();
    let mass = world.rigid.mass[0];
    let a_wp = 2.0 * hx;
    let rho0 = 1000.0;
    let m_a = barge_heave_added_mass_2d(rho0, a_wp);
    let t_ref = heave_period_with_added_mass(mass, m_a, rho0, a_wp, STANDARD_GRAVITY);
    let t_sim = estimate_heave_period(&win).expect("need ≥2 up-crossings for underdamped heave");
    let rel = fabs(t_sim - t_ref) / t_ref;
    let mean: f64 = win.iter().map(|(_, y)| *y).sum::<f64>() / win.len() as f64;
    let ncross = (1..win.len())
        .filter(|&i| win[i - 1].1 - mean <= 0.0 && win[i].1 - mean > 0.0)
        .count();
    println!(
        "heave T_sim={t_sim:.4} T_ref={t_ref:.4} m={mass:.2} m_a={m_a:.2} ncross={ncross} rel={rel:.4}"
    );
    assert!(
        ncross >= 3,
        "expected several zero-crossings for underdamped heave, got {ncross}"
    );
    assert!(
        rel < 0.15,
        "heave free-oscillation rel {rel} (measured ~0.074; lock <15%)"
    );
}

#[test]
fn coupled_momentum_near_conserved_no_walls() {
    let dim = Dim::D2;
    let dx = 0.05;
    let h = 2.2 * dx;
    let rho0 = 1000.0;
    let mass_p = mass_calibrated_to_rest(rho0, dx, h, dim, KernelKind::CubicSpline);
    let mut parts = Particles::new(dim, rho0, mass_p, h);
    fill_box(
        &mut parts,
        Vec3::new(0.0, 0.0, 0.0),
        6,
        6,
        1,
        dx,
        Vec3::new(0.2, 0.0, 0.0),
    );
    let mut coupler = FluidRigidCoupler::new(1);
    coupler.sample_box(&mut parts, 0, Vec3::new(0.08, 0.08, 0.0), dx, 1);
    let mass_b = 2.0;
    let body = RigidBody::new(mass_b)
        .with_position(Vec3::new(0.45, 0.15, 0.0))
        .with_velocity(Vec3::new(-0.1, 0.0, 0.0))
        .with_inertia(inertia_solid_sphere(mass_b, 0.08));
    for s in &coupler.samples {
        let r = body.orientation.rotate(s.local);
        parts.x[s.particle] = body.position + r;
        parts.v[s.particle] = body.velocity;
    }
    let mut params = WcsphParams::water(rho0, 30.0, Vec3::zero());
    params.adapt_sound_speed = false;
    params.xsph.epsilon = 0.0;
    params.viscosity.alpha = 0.0;
    let fluid = WcsphWorld::new(parts, params);
    let mut world = CoupledWcsphWorld::new(fluid, vec![body], coupler, Vec3::zero());
    let p0 = {
        world
            .coupler
            .sync_from_rigid(&mut world.fluid.particles, &world.rigid, &world.rigid.state);
        let mut p = pe_fluid::diagnostics::linear_momentum(&world.fluid.particles);
        p += world.rigid.state.v[0] * world.rigid.mass[0];
        p
    };
    let dt = 2e-4;
    for _ in 0..50 {
        world.step(dt);
    }
    let mut p1 = pe_fluid::diagnostics::linear_momentum(&world.fluid.particles);
    p1 += world.rigid.state.v[0] * world.rigid.mass[0];
    let drift = (p1 - p0).norm();
    println!("momentum drift ‖ΔP‖={drift:.3e}");
    // Measured drift ~1e-14. Lock < 1e-3.
    assert!(drift < 1e-3, "coupled momentum drift {drift}");
}

#[test]
fn floating_box_no_penetration() {
    let dx = 0.05;
    let mut world = make_floating_box_wcsph_2d(dx, 0.9, 0.40, 0.12, 0.08, 500.0);
    let half = Vec3::new(0.12, 0.08, 0.0);
    let dt = 2.5e-4;
    let mut max_pen = 0.0f64;
    for _ in 0..800 {
        world.step(dt);
        max_pen = max_pen.max(max_box_penetration_wcsph(&world, half));
    }
    println!("max_box_penetration={max_pen:.5}");
    // Soft Akinci boundaries: measured max pen ~0.08 on dx=0.05.
    assert!(
        max_pen < 2.0 * dx,
        "penetration {max_pen} exceeds 2 dx={}",
        2.0 * dx
    );
}

/// Build a submerged 3-D sphere in a cubic WCSPH tank (kinematic hold then release).
fn make_submerged_sphere_3d(dx: f64, rho_s: f64, radius: f64, tank: f64) -> CoupledWcsphWorld {
    let dim = Dim::D3;
    let hsm = 2.2 * dx;
    let rho0 = 1000.0;
    let g = STANDARD_GRAVITY;
    let mass_p = mass_calibrated_to_rest(rho0, dx, hsm, dim, KernelKind::CubicSpline);
    let mut parts = Particles::new(dim, rho0, mass_p, hsm);
    let n = ((tank / dx).round() as usize).max(8);
    fill_box(
        &mut parts,
        Vec3::new(0.5 * dx, 0.5 * dx, 0.5 * dx),
        n,
        n,
        n,
        dx,
        Vec3::zero(),
    );
    fill_box_boundary(
        &mut parts,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(n as f64 * dx, n as f64 * dx, n as f64 * dx),
        dx,
        1,
    );
    recomputed_akinci_psi(&mut parts, KernelKind::CubicSpline);
    let mut coupler = FluidRigidCoupler::new(1);
    coupler.sample_sphere(&mut parts, 0, radius, dx);
    for s in &coupler.samples {
        parts.n_out[s.particle] = Vec3::zero();
    }
    let vol = 4.0 / 3.0 * core::f64::consts::PI * radius * radius * radius;
    let mass_b = rho_s * vol;
    let c = 0.5 * n as f64 * dx;
    let body = RigidBody::new(mass_b)
        .with_position(Vec3::new(c, c, c))
        .with_inertia(inertia_solid_sphere(mass_b, radius));
    for s in &coupler.samples {
        parts.x[s.particle] = body.position + s.local;
    }
    let mut keep = vec![true; parts.len()];
    for (i, slot) in keep.iter_mut().enumerate() {
        if parts.kind[i] == ParticleKind::Fluid
            && (parts.x[i] - body.position).norm() < radius + 0.5 * dx
        {
            *slot = false;
        }
    }
    let mut parts2 = Particles::new(dim, rho0, mass_p, hsm);
    let mut remap = vec![usize::MAX; parts.len()];
    for (i, &keep_i) in keep.iter().enumerate() {
        if !keep_i {
            continue;
        }
        remap[i] = parts2.len();
        match parts.kind[i] {
            ParticleKind::Fluid => parts2.push_fluid(parts.x[i], Vec3::zero()),
            ParticleKind::Boundary => parts2.push_boundary(parts.x[i], parts.psi[i]),
        }
    }
    for s in &mut coupler.samples {
        s.particle = remap[s.particle];
    }
    let mut params = WcsphParams::water(rho0, 35.0, Vec3::new(0.0, -g, 0.0));
    params.adapt_sound_speed = false;
    params.xsph.epsilon = 0.08;
    params.viscosity.alpha = 0.35;
    let fluid = WcsphWorld::new(parts2, params);
    CoupledWcsphWorld::new(fluid, vec![body], coupler, Vec3::new(0.0, -g, 0.0))
}

#[test]
fn sphere_freefall_added_mass_accel() {
    // Replaces the old Cd=1 disk-sink "sanity" (rel≈36%, not an oracle).
    // 3-D sphere, hydrostatic hold then release; measure a over first 2.5 ms vs
    // a = g (ρ_s − ρ_f) / (ρ_s + ½ ρ_f). Measured rel ≈ 0.0039 at the CI setup
    // (ρ_s=2200, r=0.12, dx=0.055). Neighbour ρ_s ∈ [2000,2500] stay <3%;
    // 2–3 ms windows <7%. Lock <0.10.
    let dx = 0.055;
    let rho_s = 2200.0;
    let radius = 0.12;
    let rho0 = 1000.0;
    let g = STANDARD_GRAVITY;
    let mut world = make_submerged_sphere_3d(dx, rho_s, radius, 0.50);
    let dt = 1.2e-4;
    for _ in 0..1000 {
        world
            .coupler
            .sync_from_rigid(&mut world.fluid.particles, &world.rigid, &world.rigid.state);
        world.fluid.step(dt);
        world.rigid.state.v[0] = Vec3::zero();
        world.rigid.state.l[0] = Vec3::zero();
    }
    world.fluid.params.viscosity.alpha = 0.01;
    world.fluid.params.xsph.epsilon = 0.0;
    world.rigid.state.v[0] = Vec3::zero();
    let t_meas = 0.0025;
    let n_meas = (t_meas / dt).round() as usize;
    for _ in 0..n_meas {
        world.step(dt);
    }
    let a_dyn = -world.rigid.state.v[0].y / t_meas;
    let a_ref = sphere_freefall_accel_added_mass(rho_s, rho0, g);
    let rel = fabs(a_dyn - a_ref) / a_ref;
    println!("sphere freefall a_dyn={a_dyn:.4} a_ref={a_ref:.4} rel={rel:.4}");
    assert!(
        rel < 0.10,
        "sphere freefall added-mass accel rel={rel} (measured ~0.004; lock <10%)"
    );
}

#[test]
fn dfsph_coupling_penetration_resting() {
    let dx = 0.05;
    let mut world = make_floating_box_2d(dx, 0.8, 0.30, 0.10, 0.06, 400.0);
    let half = Vec3::new(0.10, 0.06, 0.0);
    let dt = 1.0 / 240.0;
    let mut max_pen = 0.0f64;
    for _ in 0..120 {
        world.step(dt);
        max_pen = max_pen.max(max_box_penetration(&world, half));
    }
    println!("DFSPH max_pen={max_pen:.5}");
    assert!(max_pen < 2.0 * dx, "DFSPH penetration {max_pen}");
}
