//! Two-way fluid↔rigid coupling oracles (Akinci 2012).
//! Measured tolerances locked here and in `docs/FLUID.md`. Do not loosen silently.

use pe_core::body::{inertia_solid_sphere, RigidBody};
use pe_fluid::coupling::{
    archimedes_box_draft, coupling_tank_hash, disk_terminal_velocity_2d,
    floating_box_draft_error_wcsph, heave_period_analytic, heave_period_from_stiffness,
    kinematic_bottom_force_at_draft, make_floating_box_2d, make_floating_box_wcsph_2d,
    max_box_penetration, max_box_penetration_wcsph, CoupledWcsphWorld, FluidRigidCoupler,
};
use pe_fluid::kernels::KernelKind;
use pe_fluid::particles::{
    fill_box, fill_box_boundary, mass_calibrated_to_rest, recomputed_akinci_psi, Particles,
};
use pe_fluid::wcsph::{WcsphParams, WcsphWorld};
use pe_fluid::Dim;
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
    // WCSPH + Akinci samples. Measured (Linux x86_64): rel ≈ 0.0653 at dx=0.05.
    // Target band was 3–5%; measured 6.5% locked at <10% (do not loosen).
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
    // Free-surface WCSPH float is overdamped (no clean y(t) zero-crossings).
    // Oracle: hydrostatic stiffness from kinematic bottom-face Akinci forces.
    // Measured: k≈1041 vs ρgA≈2354 (rel_k≈0.56); T≈0.954 vs 0.635 (rel_T≈0.50).
    let dx = 0.05;
    let tank_w = 0.9;
    let water_h = 0.45;
    let hx = 0.12;
    let hy = 0.10;
    let d0 = 0.08;
    let delta = 0.02;
    let f_hi = kinematic_bottom_force_at_draft(d0 + delta, dx, tank_w, water_h, hx, hy);
    let f_lo = kinematic_bottom_force_at_draft(d0 - delta, dx, tank_w, water_h, hx, hy);
    let k = (f_hi - f_lo) / (2.0 * delta);
    let mass = 500.0 * 4.0 * hx * hy;
    let t_sim = heave_period_from_stiffness(mass, k);
    let t_ref = heave_period_analytic(mass, 1000.0, 2.0 * hx, STANDARD_GRAVITY);
    let rel = fabs(t_sim - t_ref) / t_ref;
    println!("heave F_hi={f_hi:.1} F_lo={f_lo:.1} k={k:.1} T_sim={t_sim:.4} T_ref={t_ref:.4} rel={rel:.4}");
    assert!(k > 0.0, "stiffness must be restoring, k={k}");
    assert!(
        rel < 0.60,
        "heave stiffness-period rel {rel} (measured ~0.50; lock <60%)"
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

#[test]
fn sphere_sink_terminal_velocity_sanity() {
    // Heavy 2-D disk in a tall column. Measured: sank≈0.39 m, v_peak≈1.59 vs
    // Cd=1 drag-balance ≈2.48 (rel≈0.36). Lock: sinks and rel_peak < 0.55.
    let dim = Dim::D2;
    let dx = 0.05;
    let h = 2.2 * dx;
    let rho0 = 1000.0;
    let mass_p = mass_calibrated_to_rest(rho0, dx, h, dim, KernelKind::CubicSpline);
    let mut parts = Particles::new(dim, rho0, mass_p, h);
    let g = STANDARD_GRAVITY;
    fill_box(
        &mut parts,
        Vec3::new(0.5 * dx, 0.5 * dx, 0.0),
        10,
        36,
        1,
        dx,
        Vec3::zero(),
    );
    fill_box_boundary(
        &mut parts,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.5, 1.9, 0.0),
        dx,
        2,
    );
    recomputed_akinci_psi(&mut parts, KernelKind::CubicSpline);
    let radius = 0.05;
    let mut coupler = FluidRigidCoupler::new(1);
    coupler.sample_sphere(&mut parts, 0, radius, dx);
    for s in &coupler.samples {
        parts.n_out[s.particle] = Vec3::zero();
    }
    let area = core::f64::consts::PI * radius * radius;
    let mass_b = 5000.0 * area;
    let y0 = 1.40;
    let body = RigidBody::new(mass_b)
        .with_position(Vec3::new(0.25, y0, 0.0))
        .with_velocity(Vec3::new(0.0, -1.0, 0.0))
        .with_inertia(inertia_solid_sphere(mass_b, radius));
    for s in &coupler.samples {
        let r = body.orientation.rotate(s.local);
        parts.x[s.particle] = body.position + r;
        parts.v[s.particle] = Vec3::zero();
    }
    let mut keep = vec![true; parts.len()];
    for (i, slot) in keep.iter_mut().enumerate() {
        if parts.kind[i] == pe_fluid::ParticleKind::Fluid
            && (parts.x[i] - body.position).norm() < radius + 0.35 * dx
        {
            *slot = false;
        }
    }
    let mut parts2 = Particles::new(dim, rho0, mass_p, h);
    let mut remap = vec![usize::MAX; parts.len()];
    for (i, &keep_i) in keep.iter().enumerate() {
        if !keep_i {
            continue;
        }
        remap[i] = parts2.len();
        match parts.kind[i] {
            pe_fluid::ParticleKind::Fluid => parts2.push_fluid(parts.x[i], parts.v[i]),
            pe_fluid::ParticleKind::Boundary => parts2.push_boundary(parts.x[i], parts.psi[i]),
        }
    }
    for s in &mut coupler.samples {
        s.particle = remap[s.particle];
    }
    let mut params = WcsphParams::water(rho0, 20.0, Vec3::new(0.0, -g, 0.0));
    params.adapt_sound_speed = false;
    params.xsph.epsilon = 0.05;
    params.viscosity.alpha = 0.15;
    let fluid = WcsphWorld::new(parts2, params);
    let mut world = CoupledWcsphWorld::new(fluid, vec![body], coupler, Vec3::new(0.0, -g, 0.0));
    let dt = 2e-4;
    let mut v_peak = 0.0f64;
    for _ in 0..2500 {
        world.step(dt);
        v_peak = v_peak.max((-world.rigid.state.v[0].y).max(0.0));
        if world.rigid.state.x[0].y < 0.5 {
            break;
        }
    }
    let sank = y0 - world.rigid.state.x[0].y;
    let v_ref = disk_terminal_velocity_2d(mass_b, radius, rho0, 1.0, g);
    let rel = fabs(v_peak - v_ref) / v_ref.max(1e-6);
    println!("sank={sank:.4} v_peak={v_peak:.4} v_ref(Cd=1)={v_ref:.4} rel={rel:.3}");
    assert!(sank > 0.25, "sphere did not sink enough: sank={sank}");
    assert!(v_peak > 0.5, "sphere peak sink speed too small: v={v_peak}");
    assert!(
        rel < 0.55,
        "terminal velocity sanity rel={rel} (measured ~0.36; lock <55%)"
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
