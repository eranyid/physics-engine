//! Two-way soft↔rigid coupling oracles (M28).
//! Measured tolerances locked here and in `docs/SOFT.md`. Do not loosen silently.

use pe_math::libm_ops::{fabs, sqrt};
use pe_math::Vec3;
use pe_soft::{
    make_damped_attachment_oscillator, make_momentum_exchange_scene, make_pendulum_bob,
    make_rope_on_free_body, make_soft_on_box_on_ground, soft_coupling_canonical_hash,
    SoftRigidWorld, COUPLING_HASH_DT, COUPLING_HASH_STEPS,
};

const PI: f64 = core::f64::consts::PI;
const G: f64 = 9.80665;

#[test]
fn coupling_hash_bit_stable_on_repeat() {
    let a = soft_coupling_canonical_hash();
    let b = soft_coupling_canonical_hash();
    assert_eq!(a, b, "coupling hash not bit-stable: {a:016x} vs {b:016x}");
    println!("SOFT_COUPLING_HASH={a:016x}");
}

/// Isolated system, no gravity: total linear momentum conserved.
#[test]
fn momentum_conserved_isolated_no_gravity() {
    let mut world = make_momentum_exchange_scene();
    let p0 = world.total_linear_momentum();
    let dt = 1.0 / 240.0;
    for _ in 0..400 {
        world.step(dt);
    }
    let p1 = world.total_linear_momentum();
    let err = (p1 - p0).norm();
    let scale = p0.norm().max(1.0);
    let rel = err / scale;
    println!(
        "momentum p0={:?} p1={:?} abs_err={err:.3e} rel={rel:.3e}",
        p0, p1
    );
    // Measured ~2.8e-13 abs on Linux x86_64 (M28). Lock < 1e-10.
    assert!(
        err < 1e-10,
        "linear momentum drift {err} (lock < 1e-10); rel={rel}"
    );
}

/// Soft block on rigid box on ground: ground support ≈ total weight.
#[test]
fn static_equilibrium_support_equals_weight() {
    let mut world = make_soft_on_box_on_ground();
    let dt = 1.0 / 120.0;
    // Settle
    for _ in 0..600 {
        world.step(dt);
    }
    let m_tot = world.soft_mass() + world.rigid_mass();
    let weight = m_tot * G;
    // Average ground force over a window after settle.
    let mut f_sum = 0.0;
    let n_win = 120;
    for _ in 0..n_win {
        world.step(dt);
        f_sum += world.diagnostics.ground_force.z;
    }
    let f_avg = f_sum / n_win as f64;
    let rel = fabs(f_avg - weight) / weight;
    let ke = world.total_kinetic_energy();
    println!(
        "support Fz={f_avg:.4} weight={weight:.4} rel={rel:.4} ke={ke:.3e} m_soft={} m_rigid={}",
        world.soft_mass(),
        world.rigid_mass()
    );
    assert!(
        ke < 0.05,
        "expected near-rest after settle, ke={ke} (lock < 0.05 J)"
    );
    // Measured ~0.36% on Linux x86_64 (M28). Lock < 2%.
    assert!(
        rel < 0.02,
        "support vs weight rel {rel} (measured ~0.36%; lock < 2%)"
    );
}

/// Rope + free rigid body in free fall: combined CoM follows ½ g t².
#[test]
fn rope_body_freefall_com_analytic() {
    let m_rope = 1.0;
    let m_body = 3.0;
    let mut world = make_rope_on_free_body(m_rope, m_body, 6, 1e-6);
    let z0 = world.combined_com().z;
    let dt = 1.0 / 240.0;
    let t_end = 0.5;
    let n = f64::round(t_end / dt) as usize;
    for _ in 0..n {
        world.step(dt);
    }
    let t = n as f64 * dt;
    let z = world.combined_com().z;
    let z_analytic = z0 - 0.5 * G * t * t;
    let err = fabs(z - z_analytic);
    println!("freefall z0={z0:.6} z={z:.6} z_an={z_analytic:.6} abs_err={err:.3e} t={t}");
    // Measured ~3.2e-4 m at t=0.5 s (M28). Lock < 5e-4 m.
    assert!(
        err < 5e-4,
        "CoM freefall abs err {err} (measured ~3.2e-4; lock < 5e-4 m at t=0.5 s)"
    );
}

/// Heavy bob on stiff rope: small-angle period ≈ 2π √(L/g).
#[test]
fn pendulum_bob_small_angle_period() {
    let m_bob = 5.0;
    let m_rope = 0.05;
    let length = 1.0;
    let (mut world, l_com) = make_pendulum_bob(m_bob, m_rope, length, 8, 1e-8);
    // Small horizontal displace of bob + rope particles (except pin).
    let theta0 = 0.08; // rad
    let pin = world.soft.particles[0].x;
    for p in world.soft.particles.iter_mut().skip(1) {
        let dz = pin.z - p.x.z;
        let s = dz / length;
        p.x.x = length * s * theta0; // ≈ L_seg * θ for small angle along hanging rope
        p.v = Vec3::zero();
    }
    // Place bob COM under the last particle attachment.
    let last = world.soft.particles.len() - 1;
    let attach_world = world.soft.particles[last].x;
    world.rigid.state.x[0] = attach_world - Vec3::new(0.0, 0.0, 0.05);
    world.rigid.state.v[0] = Vec3::zero();
    world.rigid.state.l[0] = Vec3::zero();

    let dt = 1.0 / 240.0;
    let mut samples: Vec<(f64, f64)> = Vec::new();
    for _ in 0..4000 {
        world.step(dt);
        samples.push((world.soft.t, world.rigid.state.x[0].x));
    }
    let t_ref = 2.0 * PI * sqrt(l_com / G);
    let t_sim = estimate_period(&samples).expect("need ≥2 zero up-crossings");
    let rel = fabs(t_sim - t_ref) / t_ref;
    println!("pendulum T_sim={t_sim:.5} T_ref={t_ref:.5} L={l_com} rel={rel:.4}");
    // Measured ~0.03% on Linux x86_64 (M28). Lock < 1%.
    assert!(
        rel < 0.01,
        "pendulum period rel {rel} (measured ~0.03%; lock < 1%)"
    );
}

/// Damped attachment oscillator (no contacts): mechanical energy never grows.
#[test]
fn damped_energy_never_grows() {
    let mut world = make_damped_attachment_oscillator();
    let dt = 1.0 / 240.0;
    // One step to let XPBD project the initial stretch into multipliers.
    world.step(dt);
    let e0 = world.total_energy();
    let mut e_max = e0;
    let mut e_end = e0;
    for i in 0..2000 {
        world.step(dt);
        let e = world.total_energy();
        e_max = e_max.max(e);
        if i + 1 == 2000 {
            e_end = e;
        }
    }
    let growth = (e_max - e0) / (1.0 + fabs(e0));
    let drop = (e0 - e_end) / (1.0 + fabs(e0));
    println!(
        "energy e0={e0:.6} e_end={e_end:.6} e_max={e_max:.6} growth={growth:.3e} drop={drop:.4}"
    );
    // Lock: no growth beyond 1e-6 relative (matches pe-soft spring oracle style).
    assert!(
        e_max <= e0 * (1.0 + 1e-6) + 1e-12,
        "energy grew: Emax={e_max} E0={e0}"
    );
    assert!(drop > 0.01, "expected visible damping decay, drop={drop}");
}

fn estimate_period(samples: &[(f64, f64)]) -> Option<f64> {
    let mean: f64 = samples.iter().map(|(_, x)| *x).sum::<f64>() / samples.len() as f64;
    let mut crossings = Vec::new();
    for i in 1..samples.len() {
        let y0 = samples[i - 1].1 - mean;
        let y1 = samples[i].1 - mean;
        if y0 <= 0.0 && y1 > 0.0 {
            let t0 = samples[i - 1].0;
            let t1 = samples[i].0;
            let frac = -y0 / (y1 - y0);
            crossings.push(t0 + frac * (t1 - t0));
        }
    }
    if crossings.len() < 2 {
        return None;
    }
    let mut periods = Vec::new();
    for i in 1..crossings.len() {
        periods.push(crossings[i] - crossings[i - 1]);
    }
    // Use middle periods to avoid start-up transient.
    let start = periods.len() / 4;
    let end = periods
        .len()
        .saturating_sub(periods.len() / 4)
        .max(start + 1);
    let slice = &periods[start..end];
    Some(slice.iter().sum::<f64>() / slice.len() as f64)
}

#[test]
fn hash_scenario_runs_finite() {
    let mut world = pe_soft::soft_coupling_hash_world();
    for _ in 0..COUPLING_HASH_STEPS {
        world.step(COUPLING_HASH_DT);
    }
    assert!(world
        .soft
        .particles
        .iter()
        .all(|p| p.x.is_finite() && p.v.is_finite()));
    assert!(world.rigid.state.all_finite());
    let _ = SoftRigidWorld::new;
}
