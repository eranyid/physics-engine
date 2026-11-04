//! M29 implicit heat conduction + thermo-elastic oracles (measured tolerances).

use pe_math::libm_ops;
use pe_math::Vec3;
use pe_soft::constraint::{DistanceConstraint, Multiplier};
use pe_soft::{
    apply_thermal_expansion_from_cold, clamped_thermal_stress, free_expansion_delta_l,
    init_rod_sine_mode, robin_lumped_exp, rod_fourier_mode1, semi_infinite_erfc,
    steady_linear_profile, thermal_canonical_hash, Particle, SoftParams, SoftWorld, ThermalMesh,
    ThermalProps, THERMAL_HASH_DT, THERMAL_HASH_STEPS,
};

fn copperish() -> ThermalProps {
    ThermalProps {
        density: 8960.0,
        specific_heat: 385.0,
        conductivity: 400.0,
        expansion: 16.5e-6,
    }
}

/// 1-D slab Dirichlet ends vs Fourier mode-1 (backward Euler).
#[test]
fn implicit_slab_fourier_mode1() {
    let props = copperish();
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
    // dt ≫ explicit CFL (max_stable ≈ O(0.1 s) for this mesh).
    let dt = 25.0_f64;
    let t_end = 500.0_f64;
    let steps = (t_end / dt).round() as usize;
    for _ in 0..steps {
        assert!(mesh.step_backward_euler(dt, &fixed));
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
    // Measured: ~5.4e-4 at n=65, dt=25 s, t=500 s — lock < 0.5%.
    assert!(
        max_rel < 0.005,
        "BE Fourier slab max rel err {max_rel} at t={t_sim}"
    );
}

/// Semi-infinite solid surface step vs erfc (Carslaw & Jaeger / Incropera).
#[test]
fn semi_infinite_erfc_oracle() {
    let props = copperish();
    let kappa = props.diffusivity();
    // Long rod; left end Dirichlet T_s, initial T_i; probe near left before right feels heat.
    let length = 4.0;
    let n = 257;
    let t_i = 0.0;
    let t_s = 100.0;
    let mut mesh = ThermalMesh::rod_1d(n, length, 0.01, props, t_i);
    for i in 0..n {
        mesh.temperature[i] = t_i;
    }
    mesh.temperature[0] = t_s;
    let mut fixed = vec![false; n];
    fixed[0] = true;
    // Modest dt: BE is stable at large dt but the front smears; keep temporal error down.
    let dt = 0.25_f64;
    let t_end = 20.0_f64;
    let steps = (t_end / dt).round() as usize;
    for _ in 0..steps {
        assert!(mesh.step_backward_euler(dt, &fixed));
    }
    let t_sim = steps as f64 * dt;
    let dx = length / (n - 1) as f64;
    let depth = 4.0 * libm_ops::sqrt(kappa * t_sim);
    assert!(
        depth < 0.4 * length,
        "domain too short for semi-infinite probe"
    );
    let mut max_abs: f64 = 0.0;
    let mut n_probe = 0usize;
    for i in 1..n {
        let x = i as f64 * dx;
        if x > depth {
            break;
        }
        let analytic = semi_infinite_erfc(x, t_sim, t_i, t_s, kappa);
        max_abs = max_abs.max((mesh.temperature[i] - analytic).abs());
        n_probe += 1;
    }
    assert!(n_probe >= 5, "too few probe nodes");
    // Measured ~0.21 K — lock < 2 K on a 100 K step.
    assert!(
        max_abs < 2.0,
        "semi-infinite max abs err {max_abs} K (probes={n_probe}, depth={depth})"
    );
}

/// Steady linear profile + insulated energy with backward Euler.
#[test]
fn steady_and_insulated_implicit() {
    let props = copperish();
    let length = 1.0;
    let mut mesh = ThermalMesh::rod_1d(17, length, 0.01, props, 0.0);
    let ta = 10.0;
    let tb = 40.0;
    mesh.temperature[0] = ta;
    mesh.temperature[16] = tb;
    for i in 1..16 {
        mesh.temperature[i] = 0.0;
    }
    let mut fixed = vec![false; 17];
    fixed[0] = true;
    fixed[16] = true;
    let dt = 100.0;
    for _ in 0..400 {
        assert!(mesh.step_backward_euler(dt, &fixed));
    }
    let dx = length / 16.0;
    let mut max_ss: f64 = 0.0;
    for i in 0..17 {
        let x = i as f64 * dx;
        let expect = steady_linear_profile(x, length, ta, tb);
        max_ss = max_ss.max((mesh.temperature[i] - expect).abs());
    }
    // Measured ~5e-12 after settle — lock < 1e-9.
    assert!(max_ss < 1e-9, "steady linear abs err {max_ss}");

    let mut mesh3 = ThermalMesh::rod_1d(17, length, 0.01, props, 300.0);
    for i in 0..17 {
        mesh3.temperature[i] =
            300.0 + 20.0 * libm_ops::sin(core::f64::consts::PI * i as f64 / 16.0);
    }
    let e0 = mesh3.total_energy();
    let fixed3 = vec![false; 17];
    let dt3 = 20.0 * mesh3.max_stable_dt();
    for _ in 0..100 {
        assert!(mesh3.step_backward_euler(dt3, &fixed3));
    }
    let rel_e = (mesh3.total_energy() - e0).abs() / e0.abs();
    assert!(rel_e < 1e-11, "insulated BE energy rel drift {rel_e}");
}

/// Backward Euler stays bounded/monotone at dt ≫ CFL; CN shows ~2nd-order in dt.
#[test]
fn be_stability_and_cn_order() {
    let props = copperish();
    let length = 1.0;
    let t0 = 50.0;
    let n = 33;
    let kappa = props.diffusivity();

    // Stability: BE at 100× explicit CFL.
    let mut mesh = ThermalMesh::rod_1d(n, length, 0.01, props, 0.0);
    init_rod_sine_mode(&mut mesh, length, t0);
    let mut fixed = vec![false; n];
    fixed[0] = true;
    fixed[n - 1] = true;
    mesh.temperature[0] = 0.0;
    mesh.temperature[n - 1] = 0.0;
    let dt_cfl = mesh.max_stable_dt();
    let dt = 100.0 * dt_cfl;
    let mut prev_mid = mesh.temperature[n / 2];
    for _ in 0..40 {
        assert!(mesh.step_backward_euler(dt, &fixed));
        let mid = mesh.temperature[n / 2];
        assert!(mid.is_finite());
        assert!(mid >= -1e-9, "BE went negative {mid}");
        assert!(mid <= prev_mid + 1e-9, "BE not monotone {mid} > {prev_mid}");
        prev_mid = mid;
    }

    // CN temporal order via self-convergence vs a fine-dt reference (same mesh),
    // so spatial truncation cancels and the observed rate is temporal.
    let t_end = 200.0_f64;
    let run = |dt: f64| -> Vec<f64> {
        let mut m = ThermalMesh::rod_1d(n, length, 0.01, props, 0.0);
        init_rod_sine_mode(&mut m, length, t0);
        m.temperature[0] = 0.0;
        m.temperature[n - 1] = 0.0;
        let steps = (t_end / dt).round() as usize;
        assert!((steps as f64 * dt - t_end).abs() < 1e-12);
        for _ in 0..steps {
            assert!(m.step_crank_nicolson(dt, &fixed));
        }
        m.temperature
    };
    let fine = run(1.0);
    let e = |dt: f64| -> f64 {
        let t = run(dt);
        let mut err: f64 = 0.0;
        for i in 1..n - 1 {
            err = err.max((t[i] - fine[i]).abs());
        }
        err
    };
    let e1 = e(8.0);
    let e2 = e(4.0);
    let ratio = e1 / e2.max(1e-30);
    // 2nd-order ⇒ ratio ≈ 4; allow [3, 5.5].
    assert!(
        ratio > 3.0 && ratio < 5.5,
        "CN order ratio e(dt)/e(dt/2) = {ratio} (e1={e1}, e2={e2})"
    );
    let _ = kappa;
}

/// Robin lumped Bi≪1 matches exponential cooling.
#[test]
fn robin_lumped_small_biot() {
    // Single-node lump: C dT/dt = hA (Tinf - T).
    // Bi = h (V/A) / k with V/A = 0.1 m → Bi = 0.01 for h=0.1, k=1.
    let props = ThermalProps {
        density: 1000.0,
        specific_heat: 1000.0,
        conductivity: 1.0,
        expansion: 0.0,
    };
    let vol = 0.001;
    let area = 0.01;
    let h = 0.1;
    let c = props.density * props.specific_heat * vol;
    let ha = h * area;
    let tau = c / ha;
    let biot = h * (vol / area) / props.conductivity;
    assert!(biot < 0.05, "Biot={biot} not ≪ 1");

    let mut mesh = ThermalMesh {
        temperature: vec![400.0],
        capacity: vec![c],
        edges: vec![],
        conductance: vec![],
        t_ref: 300.0,
        props,
        neumann_power: vec![0.0],
        robin_ha: vec![ha],
        robin_tinf: vec![300.0],
    };
    let fixed = vec![false];
    let dt = tau / 20.0;
    let steps = 40;
    for _ in 0..steps {
        assert!(mesh.step_backward_euler(dt, &fixed));
    }
    let t_sim = steps as f64 * dt;
    let analytic = robin_lumped_exp(t_sim, 400.0, 300.0, tau);
    let err = (mesh.temperature[0] - analytic).abs() / (400.0 - 300.0);
    // Measured BE truncation ~O(dt) — lock < 2%.
    assert!(
        err < 0.02,
        "Robin lumped rel err {err} (num={} an={analytic})",
        mesh.temperature[0]
    );
}

/// Free expansion ΔL and clamped-bar stress E α ΔT.
#[test]
fn thermoelastic_free_and_clamped() {
    let alpha = 1.2e-5;
    let l0 = 1.0;
    let d_t = 80.0;
    let e_mod = 2.0e11;
    let dl = free_expansion_delta_l(l0, alpha, d_t);
    assert!((dl - alpha * l0 * d_t).abs() < 1e-15);

    let props = ThermalProps {
        density: 7800.0,
        specific_heat: 500.0,
        conductivity: 40.0,
        expansion: alpha,
    };
    // Free expansion rest update
    let mut mesh = ThermalMesh::rod_1d(2, l0, 0.01, props, 300.0);
    mesh.temperature[0] = 300.0 + d_t;
    mesh.temperature[1] = 300.0 + d_t;
    let mut cs = [DistanceConstraint {
        i: 0,
        j: 1,
        rest_length: l0,
        compliance: 1e-6,
        lambda: Multiplier::default(),
        broken: false,
    }];
    apply_thermal_expansion_from_cold(&mut cs, &[l0], &mesh);
    let err = (cs[0].rest_length - (l0 + dl)).abs() / (l0 + dl);
    assert!(err < 0.01, "free expansion coupling err {err}");

    // Clamped bar: both ends pinned, heat, expand rest, settle XPBD → σ ≈ −E α ΔT
    let area = 0.01;
    let compliance = l0 / (e_mod * area);
    let mut w = SoftWorld::new();
    w.params = SoftParams {
        substeps: 64,
        iterations: 2,
        damping: 1.0,
        gravity: Vec3::zero(),
    };
    let i0 = w.add_particle(Particle::pinned(Vec3::zero()));
    let i1 = w.add_particle(Particle::pinned(Vec3::new(l0, 0.0, 0.0)));
    w.constraints.distances.push(DistanceConstraint {
        i: i0,
        j: i1,
        rest_length: l0,
        compliance,
        lambda: Multiplier::default(),
        broken: false,
    });
    let mut mesh2 = ThermalMesh::rod_1d(2, l0, area, props, 300.0);
    mesh2.temperature[0] = 300.0 + d_t;
    mesh2.temperature[1] = 300.0 + d_t;
    let cold = [l0];
    apply_thermal_expansion_from_cold(&mut w.constraints.distances, &cold, &mesh2);
    for _ in 0..200 {
        w.step(1.0 / 240.0);
    }
    let len = (w.particles[i1].x - w.particles[i0].x).norm();
    // σ = E (L − L_rest) / L_cold  (cold reference for continuum match)
    let stress = e_mod * (len - w.constraints.distances[0].rest_length) / l0;
    let expect = clamped_thermal_stress(e_mod, alpha, d_t);
    let rel = (stress - expect).abs() / expect.abs().max(1.0);
    // Measured ~O(1e-3) or better with stiff XPBD — lock < 2%.
    assert!(
        rel < 0.02,
        "clamped stress rel err {rel}: got {stress}, expect {expect}, L={len}, Lrest={}",
        w.constraints.distances[0].rest_length
    );
}

/// Tet mesh assembly produces a usable conduction graph.
#[test]
fn tet_mesh_conduction_steps() {
    let props = copperish();
    // Unit cube split into 5 tets (same corner set as soft cube cell).
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];
    let tets = [
        [0usize, 1, 2, 4],
        [1, 3, 2, 7],
        [1, 4, 5, 7],
        [2, 4, 6, 7],
        [1, 2, 4, 7],
    ];
    let mut mesh = ThermalMesh::from_tets(&positions, &tets, props, 300.0);
    assert!(mesh.len() == 8);
    assert!(!mesh.edges.is_empty());
    assert!(mesh.capacity.iter().all(|&c| c > 0.0));
    mesh.temperature[0] = 400.0;
    let fixed = vec![false; 8];
    for _ in 0..20 {
        assert!(mesh.step_backward_euler(1.0, &fixed));
    }
    assert!(mesh.temperature.iter().all(|t| t.is_finite()));
    // Heat should have diffused: node 0 cooler than start, others warmer than 300.
    assert!(mesh.temperature[0] < 400.0);
    assert!(mesh.temperature[7] > 300.0);
}

#[test]
fn thermal_hash_bit_stable() {
    const {
        assert!(THERMAL_HASH_STEPS > 0);
        assert!(THERMAL_HASH_DT > 0.0);
    }
    let a = thermal_canonical_hash();
    let b = thermal_canonical_hash();
    assert_eq!(a, b);
}
