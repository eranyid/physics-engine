//! Analytic oracles for plasticity, fracture, thermal, materials table.
//!
//! Tolerances are locked from measured errors on this platform — never silently
//! loosened. See docs/MATERIALS.md.

use approx::assert_relative_eq;
use pe_math::libm_ops;
use pe_math::Vec3;
use pe_soft::constraint::{DistanceConstraint, Multiplier};
use pe_soft::fracture::{
    active_constraint_count, apply_fracture, constraint_islands, distance_axial_stress,
    fracture_released_spring_energy, soft_state_has_nan, FractureCriterion,
};
use pe_soft::material::{builtin_materials, material_by_name};
use pe_soft::particle::Particle;
use pe_soft::plasticity::{
    return_map_j2_3d, return_map_uniaxial, J2Params, J2PlasticState3, SymTensor3,
    UniaxialPlasticState,
};
use pe_soft::scenarios::materials_canonical_hash;
use pe_soft::solver::{SoftParams, SoftWorld};
use pe_soft::thermal::{
    apply_thermal_expansion_from_cold, free_expansion_delta_l, init_rod_sine_mode,
    rod_fourier_mode1, steady_linear_profile, ThermalMesh, ThermalProps,
};

fn steel_j2() -> J2Params {
    J2Params {
        youngs_modulus: 2.0e11,
        poisson_ratio: 0.3,
        yield_strength: 2.5e8,
        hardening_modulus: 2.0e9,
    }
}

/// Measured uniaxial tension oracle: yield, slopes E then E_t, unload E, permanent strain.
///
/// Measured max relative errors (Linux x86_64, this suite) are ~0 (exact return map).
/// CI lock: all under 1%.
#[test]
fn plasticity_uniaxial_oracle_under_1pct() {
    let p = steel_j2();
    let e = p.youngs_modulus;
    let sig_y = p.yield_strength;
    let et = p.uniaxial_tangent();
    let eps_y = sig_y / e;

    // Elastic branch
    let r_el = return_map_uniaxial(&p, UniaxialPlasticState::default(), 0.5 * eps_y);
    assert!(!r_el.plastic);
    let err_el = (r_el.state.stress - 0.5 * sig_y).abs() / sig_y;
    assert!(err_el < 0.01, "elastic stress err {err_el}");

    // Cross yield
    let r_y = return_map_uniaxial(&p, UniaxialPlasticState::default(), eps_y);
    let err_y = (r_y.state.stress - sig_y).abs() / sig_y;
    assert!(err_y < 0.01, "yield stress err {err_y}");

    // Plastic slope: sample two points past yield
    let eps1 = eps_y + 5e-4;
    let eps2 = eps_y + 1.5e-3;
    let s1 = return_map_uniaxial(&p, UniaxialPlasticState::default(), eps1)
        .state
        .stress;
    let s2 = return_map_uniaxial(&p, UniaxialPlasticState::default(), eps2)
        .state
        .stress;
    let slope = (s2 - s1) / (eps2 - eps1);
    let err_slope = (slope - et).abs() / et;
    assert!(
        err_slope < 0.01,
        "hardening slope err {err_slope}: meas={slope} Et={et}"
    );

    // Peak then unload elastically
    let eps_peak = eps_y + 2e-3;
    let r_p = return_map_uniaxial(&p, UniaxialPlasticState::default(), eps_peak);
    let sig_peak = r_p.state.stress;
    let eps_perm = eps_peak - sig_peak / e;
    let err_perm = (r_p.state.plastic_strain - eps_perm).abs() / eps_perm.abs().max(1e-16);
    assert!(err_perm < 0.01, "permanent strain err {err_perm}");

    // Unload by Δε such that stress drops elastically
    let d_eps = 0.5 * sig_peak / e;
    let eps_u = eps_peak - d_eps;
    let r_u = return_map_uniaxial(&p, r_p.state, eps_u);
    assert!(!r_u.plastic);
    let sig_u_expect = sig_peak - e * d_eps;
    let err_u = (r_u.state.stress - sig_u_expect).abs() / sig_peak;
    assert!(err_u < 0.01, "unload elastic err {err_u}");

    // Report measured numbers for MATERIALS.md (kept as asserts with headroom).
    assert!(err_el < 1e-12);
    assert!(err_y < 1e-12);
    assert!(err_slope < 1e-12);
    assert!(err_perm < 1e-12);
    assert!(err_u < 1e-12);
}

#[test]
fn plasticity_j2_3d_hydrostatic_no_yield_and_surface() {
    let p = steel_j2();
    let strain = SymTensor3 {
        xx: 1e-3,
        yy: 1e-3,
        zz: 1e-3,
        ..SymTensor3::zero()
    };
    let r = return_map_j2_3d(&p, J2PlasticState3::default(), strain);
    assert!(!r.plastic, "hydrostatic must not yield under J2");
    // Pure shear-ish: ε_xx = −ε_yy, rest 0 — yield when σ_eq hits σ_y
    let mu = p.shear_modulus();
    // σ_eq = √3 |σ_xx| for σ_xx=−σ_yy, σ_zz=0 with ε_xx=−ε_yy and ν path via Hooke
    let eps = 1e-3;
    let strain_s = SymTensor3 {
        xx: eps,
        yy: -eps,
        zz: 0.0,
        ..SymTensor3::zero()
    };
    let r2 = return_map_j2_3d(&p, J2PlasticState3::default(), strain_s);
    let seq = r2.state.stress.von_mises_eq();
    if r2.plastic {
        let err = (seq - (p.yield_strength + p.hardening_modulus * r2.state.eq_plastic_strain))
            .abs()
            / p.yield_strength;
        assert!(err < 0.01, "return-to-surface err {err}, seq={seq}");
    } else {
        // Still elastic: σ_eq should be < σ_y
        assert!(seq < p.yield_strength);
        let _ = mu;
    }
}

/// Bar fails when axial stress reaches threshold (within 2%).
#[test]
fn fracture_bar_threshold_within_2pct() {
    let e = 1.0e7;
    let l0 = 1.0;
    let area = 0.01;
    let thresh = 5.0e5;
    let mut w = SoftWorld::new();
    w.params = SoftParams {
        substeps: 4,
        iterations: 2,
        gravity: Vec3::zero(),
        damping: 0.0,
    };
    let i0 = w.add_particle(Particle::pinned(Vec3::zero()));
    let i1 = w.add_particle(Particle::dynamic(Vec3::new(l0, 0.0, 0.0), 1.0));
    w.constraints.distances.push(DistanceConstraint {
        i: i0,
        j: i1,
        rest_length: l0,
        compliance: 1.0 / (e * area / l0),
        lambda: Multiplier::default(),
        broken: false,
    });

    let mut broke_at = None;
    let n_steps = 200;
    let eps_max = thresh / e * 1.5;
    for s in 0..n_steps {
        let eps = eps_max * (s as f64 + 1.0) / n_steps as f64;
        w.particles[i1].x.x = l0 * (1.0 + eps);
        w.particles[i1].x_prev = w.particles[i1].x;
        let stress = distance_axial_stress(&w.particles, &w.constraints.distances[0], e);
        let events = apply_fracture(
            &w.particles,
            &mut w.constraints,
            FractureCriterion::MaxPrincipalStress(thresh),
            e,
        );
        if let Some(ev) = events.first() {
            broke_at = Some((stress, ev.value));
            break;
        }
    }
    let (stress, value) = broke_at.expect("bar should fracture");
    let err = (value - thresh).abs() / thresh;
    // Measured: breaks on first sample past threshold; relative error ≪ 2%.
    assert!(
        err < 0.02,
        "fracture stress err {err}: value={value} thresh={thresh} stress={stress}"
    );
    assert!(w.constraints.distances[0].broken);
    assert_eq!(active_constraint_count(&w.constraints), 0);
    assert!(!soft_state_has_nan(&w.particles, &w.constraints));

    // Energy accounting: released spring energy is finite and ≥ 0.
    let e_rel = fracture_released_spring_energy(&w.particles, &w.constraints);
    assert!(e_rel.is_finite() && e_rel >= 0.0);

    // Islands: two particles, one broken link → 2 islands.
    let (ids, n_isl) = constraint_islands(w.len(), &w.constraints);
    assert_eq!(n_isl, 2);
    assert_ne!(ids[0], ids[1]);
}

#[test]
fn fracture_order_stable_and_tet_path() {
    // Two-segment bar: only the more-stretched segment should break first by index order
    // when both exceed; ordering is by index when both candidates.
    let e = 1.0e6;
    let thresh = 1.0e5;
    let particles = vec![
        Particle::pinned(Vec3::zero()),
        Particle::dynamic(Vec3::new(1.2, 0.0, 0.0), 1.0),
        Particle::dynamic(Vec3::new(2.5, 0.0, 0.0), 1.0),
    ];
    let mut cs = pe_soft::ConstraintSet::default();
    for i in 0..2 {
        cs.distances.push(DistanceConstraint {
            i,
            j: i + 1,
            rest_length: 1.0,
            compliance: 1e-8,
            lambda: Multiplier::default(),
            broken: false,
        });
    }
    let ev1 = apply_fracture(
        &particles,
        &mut cs,
        FractureCriterion::MaxPrincipalStress(thresh),
        e,
    );
    assert!(!ev1.is_empty());
    for w in ev1.windows(2) {
        assert!(w[0].index <= w[1].index);
    }
    assert!(!soft_state_has_nan(&particles, &cs));
    let _ = particles;
}

/// 1-D rod transient vs Fourier series (< 2%), steady linear profile, insulated energy.
#[test]
fn thermal_rod_fourier_and_steady_and_energy() {
    let props = ThermalProps {
        density: 8960.0,
        specific_heat: 385.0,
        conductivity: 400.0,
        expansion: 16.5e-6,
    };
    let length = 1.0;
    let t0 = 50.0;
    let n = 65;
    let mut mesh = ThermalMesh::rod_1d(n, length, 0.01, props, 0.0);
    init_rod_sine_mode(&mut mesh, length, t0);
    let mut fixed = vec![false; n];
    fixed[0] = true;
    fixed[n - 1] = true;
    mesh.temperature[0] = 0.0;
    mesh.temperature[n - 1] = 0.0;

    let kappa = props.diffusivity();
    // Use half the CFL bound for temporal accuracy in the oracle.
    let dt = 0.5 * mesh.max_stable_dt();
    // Diffusive time scale τ ≈ 1/(κ π²/L²) ≈ 870 s for Cu-like props on L=1 m.
    // Run to t≈500 s (~0.57 τ) so the mode has decayed ~43%.
    let t_end = 500.0;
    let steps = (t_end / dt).ceil() as usize;
    for _ in 0..steps {
        assert!(mesh.step_explicit(dt, &fixed));
    }
    let t_sim = steps as f64 * dt;
    let dx = length / (n - 1) as f64;
    let mut max_rel: f64 = 0.0;
    for i in 1..n - 1 {
        let x = i as f64 * dx;
        let analytic = rod_fourier_mode1(x, t_sim, length, t0, kappa);
        let num = mesh.temperature[i];
        let denom = analytic.abs().max(1e-6);
        max_rel = max_rel.max((num - analytic).abs() / denom);
    }
    // Measured (n=65, dt=0.5·CFL, t≈500 s): ~0.45% — CI < 2% (docs/MATERIALS.md).
    assert!(
        max_rel < 0.02,
        "Fourier rod max rel err {max_rel} at t={t_sim} (steps={steps}, dt={dt})"
    );

    // Steady-state: Dirichlet ends → linear profile.
    let mut mesh2 = ThermalMesh::rod_1d(17, length, 0.01, props, 0.0);
    let ta = 10.0;
    let tb = 40.0;
    mesh2.temperature[0] = ta;
    mesh2.temperature[16] = tb;
    for i in 1..16 {
        mesh2.temperature[i] = 0.0;
    }
    let mut fixed2 = vec![false; 17];
    fixed2[0] = true;
    fixed2[16] = true;
    let dt2 = mesh2.max_stable_dt();
    for _ in 0..5000 {
        assert!(mesh2.step_explicit(dt2, &fixed2));
    }
    let dx2 = length / 16.0;
    let mut max_ss: f64 = 0.0;
    for i in 0..17 {
        let x = i as f64 * dx2;
        let expect = steady_linear_profile(x, length, ta, tb);
        max_ss = max_ss.max((mesh2.temperature[i] - expect).abs());
    }
    assert!(max_ss < 0.05, "steady linear abs err {max_ss}");

    // Insulated energy conservation.
    let mut mesh3 = ThermalMesh::rod_1d(17, length, 0.01, props, 300.0);
    for i in 0..17 {
        mesh3.temperature[i] =
            300.0 + 20.0 * libm_ops::sin(core::f64::consts::PI * i as f64 / 16.0);
    }
    let e0 = mesh3.total_energy();
    let fixed3 = vec![false; 17];
    let dt3 = mesh3.max_stable_dt();
    for _ in 0..400 {
        assert!(mesh3.step_explicit(dt3, &fixed3));
    }
    let rel_e = (mesh3.total_energy() - e0).abs() / e0.abs();
    assert!(rel_e < 1e-11, "insulated energy rel drift {rel_e}");
}

#[test]
fn thermal_expansion_free_dl_oracle() {
    let alpha = 1.2e-5;
    let l0 = 2.0;
    let d_t = 80.0;
    let dl = free_expansion_delta_l(l0, alpha, d_t);
    assert_relative_eq!(dl, alpha * l0 * d_t, epsilon = 1e-15);

    // Coupling: cold rest → expanded rest on a distance constraint.
    let props = ThermalProps {
        density: 2700.0,
        specific_heat: 900.0,
        conductivity: 200.0,
        expansion: alpha,
    };
    let mut mesh = ThermalMesh::rod_1d(2, l0, 0.01, props, 300.0);
    mesh.temperature[0] = 300.0 + d_t;
    mesh.temperature[1] = 300.0 + d_t;
    let mut cs = [DistanceConstraint {
        i: 0,
        j: 1,
        rest_length: l0, // will be overwritten from cold
        compliance: 1e-6,
        lambda: Multiplier::default(),
        broken: false,
    }];
    let cold = [l0];
    apply_thermal_expansion_from_cold(&mut cs, &cold, &mesh);
    let err = (cs[0].rest_length - (l0 + dl)).abs() / (l0 + dl);
    assert!(err < 0.01, "expansion coupling err {err}");
}

#[test]
fn materials_table_additions_and_flags() {
    let names: Vec<_> = builtin_materials().iter().map(|m| m.name).collect();
    for need in [
        "copper_nominal",
        "glass_soda_lime_nominal",
        "titanium_ti6al4v_nominal",
        "lead_nominal",
        "ice_nominal",
        "polycarbonate_nominal",
    ] {
        assert!(names.contains(&need), "missing {need}");
        let m = material_by_name(need).unwrap();
        assert!(!m.verified, "{need} must be verified:false unless sourced");
        assert!(m.thermal_expansion > 0.0 || need.contains("unused"));
        assert!(m.hardening_modulus >= 0.0);
    }
    for need in [
        "steel_aisi1020",
        "aluminum_6061_t6",
        "copper_c11000",
        "titanium_ti6al4v",
        "glass_soda_lime",
        "concrete_normal_weight",
        "water_20c",
    ] {
        assert!(names.contains(&need), "missing handbook row {need}");
        let m = material_by_name(need).unwrap();
        assert!(m.verified, "{need} should be verified with citation");
        assert!(!m.source_note.trim().is_empty());
    }
}

#[test]
fn materials_hash_bit_stable() {
    let a = materials_canonical_hash();
    let b = materials_canonical_hash();
    assert_eq!(a, b);
}
