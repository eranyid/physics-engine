//! Analytic validation oracles for pe-soft (XPBD + materials).

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_math::libm_ops;
use pe_math::Vec3;
use pe_soft::builders::{
    make_axial_bar_grid, make_axial_bar_springs, make_balloon, make_cantilever, make_cloth,
    make_rope, make_soft_cube,
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

/// Run a spring-mass oscillator; return (period, amplitude_peaks).
fn spring_mass_run(substeps: usize, outer_steps_per_period: usize) -> (f64, Vec<f64>) {
    let m = 1.0;
    let k = 100.0;
    let alpha = 1.0 / k;
    let t_analytic = 2.0 * PI * libm_ops::sqrt(m / k);

    let mut w = SoftWorld::new();
    w.params = SoftParams {
        substeps,
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
        broken: false,
    });

    let dt = t_analytic / outer_steps_per_period as f64;
    let mut crossings = Vec::new();
    let mut prev = w.particles[i1].x.x - 1.0;
    let mut prev2 = prev;
    let mut peaks = Vec::new();
    for step in 0..(outer_steps_per_period * 8) {
        w.step(dt);
        let cur = w.particles[i1].x.x - 1.0;
        if prev > 0.0 && cur <= 0.0 {
            crossings.push(step as f64 * dt);
        }
        let a_prev = prev.abs();
        let a_cur = cur.abs();
        let a_prev2 = prev2.abs();
        if a_prev >= a_prev2 && a_prev >= a_cur && a_prev > 0.01 {
            peaks.push(a_prev);
        }
        prev2 = prev;
        prev = cur;
        if crossings.len() >= 3 && peaks.len() >= 8 {
            break;
        }
    }
    let t_meas = crossings[1] - crossings[0];
    (t_meas, peaks)
}

#[test]
fn spring_mass_period_and_damping_vs_substeps() {
    // T = 2π √(m/k) with α = 1/k. XPBD has numerical amplitude damping but the
    // period is accurate once the outer step resolves the orbit.
    let t_analytic = 2.0 * PI * libm_ops::sqrt(1.0 / 100.0);
    let mut period_errs = Vec::new();
    for &sub in &[4usize, 8, 16, 32, 64, 128] {
        let (t_meas, peaks) = spring_mass_run(sub, 400);
        let rel = (t_meas - t_analytic).abs() / t_analytic;
        period_errs.push((sub, rel, peaks));
        // With ≥32 substeps and outer dt = T/400, period error < 2%.
        if sub >= 32 {
            assert!(
                rel < 0.02,
                "sub={sub}: period rel err {rel} (meas={t_meas}, analytic={t_analytic})"
            );
        }
    }
    // Period error is already ~0 at these outer steps; require it does not worsen
    // as substeps increase (monotone non-increasing within noise).
    for w in period_errs.windows(2) {
        assert!(
            w[1].1 <= w[0].1 + 1e-6,
            "period err should not worsen with more substeps: {:?}",
            period_errs
        );
    }
    // Amplitude decay (numerical XPBD damping), reported separately from period.
    let (_, peaks) = spring_mass_run(128, 400);
    assert!(
        peaks.len() >= 4,
        "need amplitude peaks, got {}",
        peaks.len()
    );
    let decay = peaks[peaks.len() - 1] / peaks[0];
    // At 128 substeps, < 1% amplitude loss over ~10 half-cycles in this setup.
    assert!(
        decay > 0.99,
        "unexpectedly strong damping: peak_n/peak_0={decay}, peaks={peaks:?}"
    );
    // And damping decreases with more substeps (first peaks ratio closer to 1).
    let (_, p4) = spring_mass_run(4, 400);
    let (_, p64) = spring_mass_run(64, 400);
    let d4 = p4[3] / p4[0];
    let d64 = p64[3] / p64[0];
    assert!(
        d64 >= d4,
        "more substeps should reduce numerical damping: d4={d4}, d64={d64}"
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
        broken: false,
    });
    for _ in 0..50 {
        w.step(1.0 / 120.0);
    }
    let len = (w.particles[i1].x - w.particles[i0].x).norm();
    assert_abs_diff_eq!(len, 1.0, epsilon = 1e-8);
}

#[test]
fn hanging_rope_approaches_catenary() {
    let half = 0.5;
    let h_pin = 1.0;
    let segments = 20;
    let mass = 1.0;
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
    let extra = 1.15;
    for c in &mut w.constraints.distances {
        c.rest_length *= extra;
    }
    for _ in 0..400 {
        w.step(1.0 / 120.0);
    }
    let mid = segments / 2;
    let z_mid = w.particles[mid].x.z;
    let sag = h_pin - z_mid;
    assert!(sag > 0.05, "expected visible sag, got {sag}");

    let mut a = half * half / (2.0 * sag);
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
    let mut max_err: f64 = 0.0;
    for i in 1..segments {
        let x = w.particles[i].x.x;
        let z = w.particles[i].x.z;
        let z_cat = a * libm_ops::cosh(x / a) + c;
        max_err = max_err.max((z - z_cat).abs());
    }
    assert!(
        max_err < 0.03,
        "catenary max |Δz|={max_err} (a={a}, sag={sag})"
    );
}

#[test]
fn axial_bar_spring_chain_fl_over_ea() {
    // Exact XPBD compliance mapping α = L₀/(E A). Persistent ext_force → static eq.
    let length = 1.0;
    let area = 0.01;
    let e = 1.0e6;
    let f_total = 50.0;
    let dl_analytic = f_total * length / (e * area);

    let mut errs = Vec::new();
    for n_seg in [1usize, 2, 4, 8, 16] {
        let mut w = make_axial_bar_springs(length, area, n_seg, e, 1.0);
        w.params.damping = 10.0;
        w.params.substeps = 40;
        w.params.iterations = 4;
        let end = w.particles.len() - 1;
        w.particles[end].ext_force = Vec3::new(f_total, 0.0, 0.0);
        for _ in 0..3000 {
            w.step(1.0 / 120.0);
        }
        assert!(
            w.particles[end].v.norm() < 1e-6,
            "not settled: |v|={}",
            w.particles[end].v.norm()
        );
        let dl = w.particles[end].x.x - length;
        let rel = (dl - dl_analytic).abs() / dl_analytic;
        errs.push((n_seg, dl, rel));
        assert!(
            rel < 0.005,
            "spring-chain n={n_seg}: δL rel err {rel} (meas={dl}, analytic={dl_analytic})"
        );
    }
    // Mesh independence of the 1-D spring mapping.
    let r_max = errs.iter().map(|e| e.2).fold(0.0_f64, f64::max);
    let r_min = errs.iter().map(|e| e.2).fold(f64::INFINITY, f64::min);
    assert!(
        r_max - r_min < 0.002,
        "spring-chain error should be mesh-independent: {errs:?}"
    );
}

#[test]
fn axial_bar_tet_fem_fl_over_ea_converges() {
    // StVK XPBD: C = √(2Ψ), α = 1/V₀; λ-constraint skipped when ν=0 (never α=0/rigid J).
    // Continuum FL/(EA) within a few percent; isotropic refinement improves coarse → mid.
    let length = 1.0;
    let side = 0.1;
    let area = side * side;
    let e = 1.0e6;
    let f_total = 50.0;
    let dl_analytic = f_total * length / (e * area);

    let mut rels_nu0 = Vec::new();
    for &(nx, ny, nz) in &[(4usize, 1, 1), (8, 2, 2), (12, 3, 3)] {
        let mat = Material {
            name: "bar",
            density: 1000.0,
            youngs_modulus: e,
            poisson_ratio: 0.0,
            friction: 0.0,
            restitution: 0.0,
            yield_strength: 0.0,
            hardening_modulus: 0.0,
            thermal_conductivity: 0.0,
            specific_heat: 0.0,
            thermal_expansion: 0.0,
            fracture_stress: 0.0,
            fracture_strain: 0.0,
            verified: false,
            source_note: "test",
        };
        let mut w = make_axial_bar_grid(length, side, side, nx, ny, nz, &mat);
        w.params.damping = 30.0;
        w.params.substeps = 40;
        w.params.iterations = 6;
        let end: Vec<usize> = w
            .particles
            .iter()
            .enumerate()
            .filter(|(_, p)| p.inv_mass != 0.0 && p.x.x > length * 0.98)
            .map(|(i, _)| i)
            .collect();
        let f_each = f_total / end.len() as f64;
        for &i in &end {
            w.particles[i].ext_force = Vec3::new(f_each, 0.0, 0.0);
        }
        for _ in 0..2000 {
            w.step(1.0 / 120.0);
        }
        let x_end: f64 = end.iter().map(|&i| w.particles[i].x.x).sum::<f64>() / end.len() as f64;
        let v_end: f64 =
            end.iter().map(|&i| w.particles[i].v.norm()).sum::<f64>() / end.len() as f64;
        assert!(
            v_end < 1e-4,
            "ν=0 tet FEM not settled |v|={v_end} at {nx}x{ny}x{nz}"
        );
        let dl = x_end - length;
        let rel = (dl - dl_analytic).abs() / dl_analytic;
        rels_nu0.push((nx, ny, nz, dl, rel));
        assert!(
            rel < 0.05,
            "ν=0 tet FEM rel err {rel} ≥ 5% at {nx}x{ny}x{nz}: {rels_nu0:?}"
        );
    }
    // Coarse → mid isotropic refinement should not worsen the axial error.
    assert!(
        rels_nu0[1].4 <= rels_nu0[0].4 + 1e-6,
        "ν=0 refinement 4x1x1→8x2x2 should improve: {rels_nu0:?}"
    );

    // ν=0.3 + fixed root face (Saint-Venant); free-lateral FL/(EA) still <5% refined.
    let mat = Material {
        name: "bar",
        density: 1000.0,
        youngs_modulus: e,
        poisson_ratio: 0.3,
        friction: 0.0,
        restitution: 0.0,
        yield_strength: 0.0,
        hardening_modulus: 0.0,
        thermal_conductivity: 0.0,
        specific_heat: 0.0,
        thermal_expansion: 0.0,
        fracture_stress: 0.0,
        fracture_strain: 0.0,
        verified: false,
        source_note: "test",
    };
    let mut rels_nu3 = Vec::new();
    for &(nx, ny, nz) in &[(4usize, 1, 1), (8, 2, 2), (12, 3, 3)] {
        let mut w = make_axial_bar_grid(length, side, side, nx, ny, nz, &mat);
        w.params.damping = 30.0;
        w.params.substeps = 40;
        w.params.iterations = 6;
        let end: Vec<usize> = w
            .particles
            .iter()
            .enumerate()
            .filter(|(_, p)| p.inv_mass != 0.0 && p.x.x > length * 0.98)
            .map(|(i, _)| i)
            .collect();
        let f_each = f_total / end.len() as f64;
        for &i in &end {
            w.particles[i].ext_force = Vec3::new(f_each, 0.0, 0.0);
        }
        for _ in 0..2000 {
            w.step(1.0 / 120.0);
        }
        let x_end: f64 = end.iter().map(|&i| w.particles[i].x.x).sum::<f64>() / end.len() as f64;
        let dl = x_end - length;
        let rel = (dl - dl_analytic).abs() / dl_analytic;
        rels_nu3.push((nx, ny, nz, dl, rel));
        assert!(
            rel < 0.05,
            "ν=0.3 tet FEM rel err {rel} ≥ 5% at {nx}x{ny}x{nz}: {rels_nu3:?}"
        );
    }
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
    w.params.damping = 2.0;
    w.params.substeps = 40;
    let rest_sum: f64 = w
        .constraints
        .volumes
        .iter()
        .map(|c| c.rest_volume.abs())
        .sum();
    for _ in 0..400 {
        w.step(1.0 / 120.0);
    }
    let err = volume_error_sum(&w);
    let rel = err / rest_sum.max(1e-12);
    // Measured ~0.9% on this scene; lock under 3%.
    assert!(
        rel < 0.03,
        "volume conservation rel err {rel} (abs sum {err}, rest {rest_sum})"
    );
}

#[test]
fn cantilever_deflection_vs_euler_bernoulli() {
    // Distributed self-weight: δ_EB = mg L³ / (8 E I).
    // With correct StVK energy, linear tets shear-lock on coarse meshes (tip too
    // small). The old ~1.0 EB match was an artifact of the under-stiff FEM bug
    // cancelling locking. Light load keeps the response near-linear; h-refinement
    // increases tip drop toward EB (8→16 along length: ratio ~0.36→~0.62).
    let length = 1.0;
    let height = 0.1;
    let depth = 0.1;
    let e = 1.0e6;
    let nu = 0.3;
    let density = 20.0;
    let mat = Material {
        name: "cantilever_test",
        density,
        youngs_modulus: e,
        poisson_ratio: nu,
        friction: 0.0,
        restitution: 0.0,
        yield_strength: 0.0,
        hardening_modulus: 0.0,
        thermal_conductivity: 0.0,
        specific_heat: 0.0,
        thermal_expansion: 0.0,
        fracture_stress: 0.0,
        fracture_strain: 0.0,
        verified: false,
        source_note: "test",
    };
    let mass = density * length * height * depth;
    let i_area = depth * height * height * height / 12.0;
    let delta_eb = mass * 9.80665 * length * length * length / (8.0 * e * i_area);

    let mut ratios = Vec::new();
    for nx in [8usize, 16] {
        let mut w = make_cantilever(length, height, depth, nx, &mat);
        w.params.gravity = Vec3::new(0.0, 0.0, -9.80665);
        w.params.damping = 15.0;
        w.params.substeps = 40;
        w.params.iterations = 6;
        let end: Vec<usize> = w
            .particles
            .iter()
            .enumerate()
            .filter(|(_, p)| p.x.x > length * 0.95)
            .map(|(i, _)| i)
            .collect();
        let z0: f64 = end.iter().map(|&i| w.particles[i].x.z).sum::<f64>() / end.len() as f64;
        for _ in 0..2000 {
            w.step(1.0 / 120.0);
        }
        let z1: f64 = end.iter().map(|&i| w.particles[i].x.z).sum::<f64>() / end.len() as f64;
        let tip_drop = z0 - z1;
        let ratio = tip_drop / delta_eb;
        assert!(tip_drop > 0.0, "nx={nx}: expected tip drop, EB={delta_eb}");
        ratios.push((nx, tip_drop, ratio));
    }
    // Coarse locks harder; refining along the span must increase tip / EB ratio.
    assert!(
        ratios[1].2 > ratios[0].2 + 0.1,
        "expected h-refinement to unlock shear: {ratios:?}"
    );
    // 16×2×2 band (~0.62 measured); not a <5% continuum claim on this mesh.
    assert!(
        ratios[1].2 > 0.5 && ratios[1].2 < 0.85,
        "refined cantilever ratio out of locking band: {ratios:?}, δ_EB={delta_eb}"
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
    assert!(
        e1 < e0 + 0.5,
        "energy grew beyond tolerance: e0={e0}, e1={e1}"
    );
    for p in &w.particles {
        assert!(p.x.z.is_finite() && p.x.z > -0.1);
    }
}

#[test]
fn conservative_spring_energy_long_run_decay() {
    // XPBD is dissipative (numerical damping): energy does not grow; it decays.
    // Measured on this setup (α=0.01, 64 substeps, dt=1/240):
    //   t≈21 s: ΔE/E₀ ≈ −12.7%
    //   t≈42 s: ΔE/E₀ ≈ −23.8%
    //   E_max = E₀ (no growth).
    let mut w = SoftWorld::new();
    w.params = SoftParams {
        substeps: 64,
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
        broken: false,
    });
    let e0 = total_energy(&w);
    let mut e_max = e0;
    let mut e_mid = e0;
    let mut e_end = e0;
    let nsteps = 10_000;
    let dt = 1.0 / 240.0;
    for i in 0..nsteps {
        w.step(dt);
        let e = total_energy(&w);
        e_max = e_max.max(e);
        if i == nsteps / 2 {
            e_mid = e;
        }
        if i + 1 == nsteps {
            e_end = e;
        }
    }
    // No growth beyond a tiny numerical ceiling.
    assert!(
        e_max <= e0 * (1.0 + 1e-6),
        "energy grew: Emax={e_max} E0={e0}"
    );
    let drop_mid = (e0 - e_mid) / e0;
    let drop_end = (e0 - e_end) / e0;
    // Long-run decay band (honest measured numbers ± slack).
    assert!(
        drop_mid > 0.08 && drop_mid < 0.20,
        "mid-run decay out of band: {drop_mid} (E_mid={e_mid})"
    );
    assert!(
        drop_end > 0.15 && drop_end < 0.35,
        "end-run decay out of band: {drop_end} (E_end={e_end})"
    );
}

#[test]
fn iteration_substep_convergence_distance() {
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
            broken: false,
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
        broken: false,
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
    let w = pe_soft::soft_canonical_world();
    assert!(kinetic_energy(&w) >= 0.0);
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
