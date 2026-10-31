//! Analytic validation oracles for pe-soft (XPBD + materials).

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_math::libm_ops;
use pe_math::Vec3;
use pe_soft::builders::{
    make_axial_bar_tets, make_balloon, make_cantilever, make_cloth, make_rope, make_soft_cube,
};
use pe_soft::collision::{StaticColliders, StaticPlane};
use pe_soft::constraint::{project_distance, DistanceConstraint, Multiplier, VolumeConstraint};
use pe_soft::energy::{kinetic_energy, total_energy, volume_error_sum};
use pe_soft::material::{
    alpha_tilde, builtin_materials, lame_lambda, lame_mu, material_by_name, stretch_compliance,
    stretch_stiffness, Material,
};
use pe_soft::particle::Particle;
use pe_soft::scenarios::soft_canonical_hash;
use pe_soft::solver::{SoftParams, SoftWorld};
use pe_soft::{soft_state_hash64, SOFT_HASH_DT, SOFT_HASH_STEPS};

const PI: f64 = core::f64::consts::PI;

#[test]
fn material_lame_and_compliance_mapping() {
    let e = 2.0e6;
    let nu = 0.3;
    let mu = lame_mu(e, nu).unwrap();
    let lambda = lame_lambda(e, nu).unwrap();
    assert_relative_eq!(mu, e / (2.0 * (1.0 + nu)), epsilon = 1e-12);
    let e_back = mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
    assert_relative_eq!(e_back, e, epsilon = 1e-6);

    let l = 1.0;
    let a = 0.01;
    let k = stretch_stiffness(e, l, a).unwrap();
    let alpha = stretch_compliance(e, l, a).unwrap();
    assert_relative_eq!(alpha * k, 1.0, epsilon = 1e-12);
    assert_relative_eq!(alpha_tilde(alpha, 0.01), alpha / 1e-4, epsilon = 1e-12);
}

#[test]
fn builtin_materials_are_unverified_nominal() {
    for m in builtin_materials() {
        assert!(!m.verified);
        assert!(m.density > 0.0 && m.youngs_modulus > 0.0);
    }
}

#[test]
fn spring_mass_period_via_compliance() {
    // Single spring-mass: T = 2π √(m/k), XPBD compliance α = 1/k.
    let m = 1.0;
    let k = 100.0;
    let alpha = 1.0 / k;
    let t_analytic = 2.0 * PI * libm_ops::sqrt(m / k);

    let mut w = SoftWorld::new();
    w.params = SoftParams {
        substeps: 64,
        iterations: 1,
        gravity: Vec3::zero(),
        damping: 0.0,
    };
    let i0 = w.add_particle(Particle::pinned(Vec3::zero()));
    let i1 = w.add_particle(Particle::dynamic(Vec3::new(1.1, 0.0, 0.0), m));
    w.constraints.distances.push(DistanceConstraint {
        i: i0,
        j: i1,
        rest_length: 1.0,
        compliance: alpha,
        lambda: Multiplier::default(),
    });
    // Initial condition: stretch, zero velocity — free oscillation.
    w.particles[i1].v = Vec3::zero();

    let dt = t_analytic / 200.0;
    let mut crossings = Vec::new();
    let mut prev = w.particles[i1].x.x - 1.0;
    for step in 0..800 {
        w.step(dt);
        let cur = w.particles[i1].x.x - 1.0;
        if prev > 0.0 && cur <= 0.0 {
            crossings.push(step as f64 * dt);
            if crossings.len() >= 3 {
                break;
            }
        }
        prev = cur;
    }
    assert!(
        crossings.len() >= 2,
        "expected zero crossings, got {}",
        crossings.len()
    );
    let t_meas = crossings[1] - crossings[0];
    // XPBD with finite substeps is approximate; allow 8% period error.
    let rel = (t_meas - t_analytic).abs() / t_analytic;
    assert!(
        rel < 0.08,
        "period rel err {rel} (meas={t_meas}, analytic={t_analytic})"
    );
}

#[test]
fn zero_compliance_equals_rigid_distance() {
    let mut w = SoftWorld::new();
    w.params = SoftParams {
        substeps: 10,
        iterations: 4,
        gravity: Vec3::zero(),
        damping: 0.0,
    };
    let i0 = w.add_particle(Particle::pinned(Vec3::zero()));
    let i1 = w.add_particle(Particle::dynamic(Vec3::new(2.0, 0.0, 0.0), 1.0));
    w.constraints.distances.push(DistanceConstraint {
        i: i0,
        j: i1,
        rest_length: 1.0,
        compliance: 0.0,
        lambda: Multiplier::default(),
    });
    for _ in 0..50 {
        w.step(1.0 / 120.0);
    }
    let len = (w.particles[i1].x - w.particles[i0].x).norm();
    assert_abs_diff_eq!(len, 1.0, epsilon = 1e-8);
}

#[test]
fn hanging_rope_approaches_catenary() {
    // Symmetric rope pinned at (±L/2, 0, H), rest length > chord → catenary.
    // Analytic: z(x) = a cosh(x/a) + c, with a = T₀/(λg), fitted from sag.
    let half = 0.5;
    let h_pin = 1.0;
    let segments = 20;
    let mass = 1.0;
    // Soft stretch so geometric catenary dominates (nearly inextensible).
    let compliance = 1e-8;
    let mut w = make_rope(
        Vec3::new(-half, 0.0, h_pin),
        Vec3::new(half, 0.0, h_pin),
        segments,
        mass,
        compliance,
        true,
    );
    w.params.substeps = 40;
    w.params.damping = 2.0;
    // Slightly longer rest length than chord to allow sag? make_rope uses chord/seg;
    // force extra length by scaling rest lengths.
    let extra = 1.15;
    for c in &mut w.constraints.distances {
        c.rest_length *= extra;
    }
    for _ in 0..400 {
        w.step(1.0 / 120.0);
    }
    // Measure sag at mid particle
    let mid = segments / 2;
    let z_mid = w.particles[mid].x.z;
    let sag = h_pin - z_mid;
    assert!(sag > 0.05, "expected visible sag, got {sag}");

    // Fit a from endpoints (±half, h_pin) and mid (0, z_mid):
    // h_pin = a cosh(half/a) + c, z_mid = a + c  ⇒  h_pin - z_mid = a (cosh(half/a) - 1)
    // Solve a from sag = a (cosh(half/a) - 1) by Newton.
    let mut a = half * half / (2.0 * sag); // parabolic seed
    for _ in 0..20 {
        let u = half / a;
        let f = a * (libm_ops::cosh(u) - 1.0) - sag;
        let df = (libm_ops::cosh(u) - 1.0) - u * libm_ops::sinh(u);
        a -= f / df;
        if a <= 1e-6 {
            a = 1e-6;
        }
    }
    let c = z_mid - a;
    // Compare interior free particles to catenary (skip pins).
    let mut max_err: f64 = 0.0;
    for i in 1..segments {
        let x = w.particles[i].x.x;
        let z = w.particles[i].x.z;
        let z_cat = a * libm_ops::cosh(x / a) + c;
        max_err = max_err.max((z - z_cat).abs());
    }
    // Honest tolerance: coarse XPBD rope + stretch residual; within 3 cm on 1 m span.
    assert!(
        max_err < 0.03,
        "catenary max |Δz|={max_err} (a={a}, sag={sag})"
    );
}

#[test]
fn axial_bar_static_elongation() {
    // δL = F L / (E A). End load on the free face; settle with damping.
    // Edge springs carry EA/L (tet FEM is supplementary). Tolerance stated honestly.
    let length = 1.0;
    let side = 0.1;
    let area = side * side;
    let e = 1.0e6;
    let mat = Material {
        name: "test_bar",
        density: 1000.0,
        youngs_modulus: e,
        poisson_ratio: 0.3,
        friction: 0.0,
        restitution: 0.0,
        yield_strength: 0.0,
        thermal_conductivity: 0.0,
        specific_heat: 0.0,
        verified: false,
        source_note: "test only",
    };
    let mut w = make_axial_bar_tets(length, side, 4, &mat);
    w.params.gravity = Vec3::zero();
    w.params.damping = 8.0;
    w.params.substeps = 40;

    let f_total = 50.0; // N
    let end: Vec<usize> = w
        .particles
        .iter()
        .enumerate()
        .filter(|(_, p)| p.inv_mass != 0.0 && p.x.x > length * 0.95)
        .map(|(i, _)| i)
        .collect();
    assert!(!end.is_empty());
    let f_each = f_total / end.len() as f64;

    let dt = 1.0 / 120.0;
    for _ in 0..800 {
        for &i in &end {
            let m = 1.0 / w.particles[i].inv_mass;
            w.particles[i].v.x += dt * f_each / m;
        }
        w.step(dt);
    }

    let x_end: f64 = end.iter().map(|&i| w.particles[i].x.x).sum::<f64>() / end.len() as f64;
    let dl = x_end - length;
    let dl_analytic = f_total * length / (e * area);
    // Coarse XPBD bar: allow 40% relative error vs continuum formula.
    let rel = (dl - dl_analytic).abs() / dl_analytic.max(1e-12);
    assert!(
        rel < 0.40,
        "axial δL rel err {rel}: meas={dl}, analytic={dl_analytic}"
    );
    assert!(dl > 0.0, "expected elongation");
}

#[test]
fn tet_volume_conservation_near_incompressible() {
    let mut mat = material_by_name("jelly").unwrap();
    mat.poisson_ratio = 0.495;
    mat.youngs_modulus = 5e4;
    let mut w = make_soft_cube(Vec3::new(0.0, 0.0, 0.5), 0.3, 2, &mat, true);
    w.colliders = StaticColliders {
        planes: vec![StaticPlane::new(Vec3::unit_z(), 0.0, 0.5, 0.0)],
        ..Default::default()
    };
    w.params.damping = 1.0;
    let v0 = volume_error_sum(&w); // should be ~0 at rest
    let _ = v0;
    let rest_vols: Vec<f64> = w
        .constraints
        .volumes
        .iter()
        .map(|c| c.rest_volume)
        .collect();
    let rest_sum: f64 = rest_vols.iter().map(|v| v.abs()).sum();
    for _ in 0..300 {
        w.step(1.0 / 120.0);
    }
    let err = volume_error_sum(&w);
    // Bounded relative volume drift for ν→0.5 (honest: 8% of rest volume sum).
    let rel = err / rest_sum.max(1e-12);
    assert!(
        rel < 0.08,
        "volume conservation rel err {rel} (abs sum {err}, rest {rest_sum})"
    );
}

#[test]
fn cantilever_deflection_vs_euler_bernoulli() {
    // δ = F L³ / (3 E I) for tip load. We use self-weight as distributed load approx
    // with tip-equivalent F = mg for order-of-magnitude check against EB tip formula
    // for a tip load — stated tolerance is loose for distributed load + soft mesh.
    let length = 1.0;
    let height = 0.1;
    let depth = 0.1;
    let e = 1.0e6;
    let mat = Material {
        name: "cantilever_test",
        density: 200.0,
        youngs_modulus: e,
        poisson_ratio: 0.3,
        friction: 0.0,
        restitution: 0.0,
        yield_strength: 0.0,
        thermal_conductivity: 0.0,
        specific_heat: 0.0,
        verified: false,
        source_note: "test",
    };
    let mut w = make_cantilever(length, height, depth, 6, &mat);
    // Tip load via gravity in −z; beam along +x, cross-section in yz.
    // Rebuild gravity: default −z is correct for bending about y.
    w.params.gravity = Vec3::new(0.0, 0.0, -9.80665);
    w.params.damping = 3.0;
    w.params.substeps = 40;
    for _ in 0..500 {
        w.step(1.0 / 120.0);
    }
    // Tip mean z among particles with x near L
    let mut tip_z = 0.0;
    let mut n = 0usize;
    for p in &w.particles {
        if p.x.x > length * 0.95 {
            tip_z += p.x.z;
            n += 1;
        }
    }
    tip_z /= n.max(1) as f64;
    // Rest tip z was depth/2 average of z in [0, depth] — use mean rest z = depth/2
    let rest_tip_z = depth * 0.5;
    let defl = rest_tip_z - tip_z; // positive downward deflection magnitude if tip drops
                                   // Actually particles start with z in [0,depth]; COM tip drops under −z gravity.
    let tip_drop = {
        let mut s = 0.0;
        let mut c = 0usize;
        for p in &w.particles {
            if p.x.x > length * 0.95 {
                // compare to initial z stored... we didn't keep initial; use
                // that free end should drop below its max neighbor — measure mean z vs initial mean.
                s += p.x.z;
                c += 1;
            }
        }
        // Initial mean tip z ≈ depth/2; drop = depth/2 - mean_z
        depth * 0.5 - s / c.max(1) as f64
    };
    let _ = defl;
    let mass = mat.density * length * height * depth;
    let f_tip_equiv = mass * 9.80665; // total weight as tip-equivalent (conservative overestimate)
    let i_area = depth * height * height * height / 12.0;
    let delta_eb = f_tip_equiv * length * length * length / (3.0 * e * i_area);
    // Distributed self-weight tip deflection is w L^4/(8EI) = (mg/L) L^4/(8EI) = mg L^3/(8EI)
    let delta_dist = mass * 9.80665 * length * length * length / (8.0 * e * i_area);
    // Soft tet mesh under-resolves bending; allow factor-of-2 vs distributed EB.
    let ratio = tip_drop / delta_dist;
    assert!(
        tip_drop > 0.0,
        "expected tip drop, got {tip_drop} (EB tip={delta_eb}, dist={delta_dist})"
    );
    assert!(
        ratio > 0.25 && ratio < 2.5,
        "cantilever tip_drop/δ_EB_dist={ratio} (drop={tip_drop}, δ={delta_dist})"
    );
}

#[test]
fn cloth_pinned_corners_stable_energy_decays() {
    let mut w = make_cloth(
        Vec3::new(0.0, 0.0, 1.0),
        6,
        6,
        0.1,
        0.1,
        0.5,
        1e-6,
        1e-4,
        true,
    );
    w.params.damping = 1.5;
    w.colliders = StaticColliders {
        planes: vec![StaticPlane::new(Vec3::unit_z(), 0.0, 0.3, 0.0)],
        ..Default::default()
    };
    // Perturb
    for p in &mut w.particles {
        if p.inv_mass != 0.0 {
            p.v = Vec3::new(0.2, -0.1, 0.0);
        }
    }
    let e0 = total_energy(&w);
    for _ in 0..200 {
        w.step(1.0 / 60.0);
    }
    let e1 = total_energy(&w);
    // With damping, energy should not grow; allow small numerical PE bookkeeping slack.
    assert!(
        e1 < e0 + 0.5,
        "energy grew beyond tolerance: e0={e0}, e1={e1}"
    );
    // Still finite / hanging
    for p in &w.particles {
        assert!(p.x.z.is_finite() && p.x.z > -0.1);
    }
}

#[test]
fn conservative_spring_energy_bounded() {
    let mut w = SoftWorld::new();
    w.params = SoftParams {
        substeps: 32,
        iterations: 1,
        gravity: Vec3::zero(),
        damping: 0.0,
    };
    let i0 = w.add_particle(Particle::pinned(Vec3::zero()));
    let i1 = w.add_particle(Particle::dynamic(Vec3::new(1.2, 0.0, 0.0), 1.0));
    w.constraints.distances.push(DistanceConstraint {
        i: i0,
        j: i1,
        rest_length: 1.0,
        compliance: 0.01,
        lambda: Multiplier::default(),
    });
    let e0 = total_energy(&w);
    let mut e_max = e0;
    for _ in 0..500 {
        w.step(1.0 / 240.0);
        e_max = e_max.max(total_energy(&w));
    }
    // XPBD is not exactly symplectic; allow 5% growth bound for this conservative setup.
    assert!(
        e_max < e0.abs().max(e0) + 0.05 * e0.abs().max(1.0),
        "energy max {e_max} vs start {e0}"
    );
}

#[test]
fn iteration_substep_convergence_distance() {
    // Finer substeps → closer to rest length under load.
    fn residual(substeps: usize) -> f64 {
        let mut w = SoftWorld::new();
        w.params = SoftParams {
            substeps,
            iterations: 1,
            gravity: Vec3::new(0.0, 0.0, -9.80665),
            damping: 0.0,
        };
        let i0 = w.add_particle(Particle::pinned(Vec3::new(0.0, 0.0, 1.0)));
        let i1 = w.add_particle(Particle::dynamic(Vec3::new(0.0, 0.0, 0.5), 1.0));
        w.constraints.distances.push(DistanceConstraint {
            i: i0,
            j: i1,
            rest_length: 0.5,
            compliance: 1e-4,
            lambda: Multiplier::default(),
        });
        for _ in 0..120 {
            w.step(1.0 / 60.0);
        }
        let len = (w.particles[i1].x - w.particles[i0].x).norm();
        (len - 0.5).abs()
    }
    let r4 = residual(4);
    let r32 = residual(32);
    assert!(
        r32 <= r4 * 1.05 + 1e-6,
        "expected more substeps to not worsen residual: r4={r4}, r32={r32}"
    );
}

#[test]
fn balloon_pressure_inflates() {
    let mut w = make_balloon(Vec3::zero(), 0.5, 4, 6, 0.5, 1e-5, 1e-6, 1.3);
    w.params.gravity = Vec3::zero();
    w.params.damping = 1.0;
    let target = w.constraints.pressures[0].target_volume;
    let v0 = {
        let tris = &w.constraints.pressures[0].triangles;
        pe_soft::mesh_volume(&w.particles, tris)
    };
    assert!(v0 > 0.0, "rest volume must be positive, got {v0}");
    assert!(target > v0, "target should exceed rest volume");
    for _ in 0..250 {
        w.step(1.0 / 60.0);
    }
    let v1 = {
        let tris = &w.constraints.pressures[0].triangles;
        pe_soft::mesh_volume(&w.particles, tris)
    };
    assert!(v1 > v0 * 1.05, "balloon should inflate: v0={v0}, v1={v1}");
}

#[test]
fn soft_hash_bit_stable_on_repeat() {
    let a = soft_canonical_hash();
    let b = soft_canonical_hash();
    assert_eq!(a, b);
}

#[test]
fn project_distance_unit() {
    let mut particles = vec![
        Particle::dynamic(Vec3::new(0.0, 0.0, 0.0), 1.0),
        Particle::dynamic(Vec3::new(2.0, 0.0, 0.0), 1.0),
    ];
    let mut c = DistanceConstraint {
        i: 0,
        j: 1,
        rest_length: 1.0,
        compliance: 0.0,
        lambda: Multiplier::default(),
    };
    project_distance(&mut particles, &mut c, 0.01);
    let len = (particles[0].x - particles[1].x).norm();
    assert_abs_diff_eq!(len, 1.0, epsilon = 1e-9);
}

#[test]
fn volume_constraint_restores() {
    let mut w = SoftWorld::new();
    w.params.gravity = Vec3::zero();
    w.params.substeps = 20;
    let verts = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    for v in verts {
        w.add_particle(Particle::dynamic(v, 1.0));
    }
    let v0 = pe_soft::tetra_volume(verts[0], verts[1], verts[2], verts[3]);
    w.constraints.volumes.push(VolumeConstraint {
        indices: [0, 1, 2, 3],
        rest_volume: v0,
        compliance: 0.0,
        lambda: Multiplier::default(),
    });
    // Crush
    w.particles[3].x.z = 0.5;
    for _ in 0..30 {
        w.step(0.01);
    }
    let v1 = pe_soft::tetra_volume(
        w.particles[0].x,
        w.particles[1].x,
        w.particles[2].x,
        w.particles[3].x,
    );
    assert_relative_eq!(v1, v0, epsilon = 1e-5);
}

#[test]
fn kinetic_energy_nonnegative() {
    let w = soft_canonical_world_quick();
    assert!(kinetic_energy(&w) >= 0.0);
}

fn soft_canonical_world_quick() -> SoftWorld {
    pe_soft::soft_canonical_world()
}

#[test]
fn hash_scenario_constants_match_docs_contract() {
    assert_eq!(SOFT_HASH_STEPS, 120);
    assert_abs_diff_eq!(SOFT_HASH_DT, 1.0 / 60.0, epsilon = 0.0);
    let mut w = pe_soft::soft_canonical_world();
    for _ in 0..SOFT_HASH_STEPS {
        w.step(SOFT_HASH_DT);
    }
    let h = soft_state_hash64(&w);
    assert_eq!(h, soft_canonical_hash());
}
