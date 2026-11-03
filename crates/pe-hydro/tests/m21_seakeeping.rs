//! M21 seakeeping / resistance / maneuvering oracles (tight, justified tolerances).

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_hydro::cummins6::{
    box_a_inf_matrix, box_hydrostatic_restoring, box_retardation_6dof, canonical_cummins6_hash,
    heave_pitch_natural_omegas, roll_decay_analytic, roll_decay_rk4, truncation_tail_l1,
    Cummins6Dof, RigidMass6, N_DOF,
};
use pe_hydro::forward_speed::{assemble_strip_matrices_forward_speed, encounter_frequency_deep};
use pe_hydro::hull::RHO_FRESHWATER;
use pe_hydro::maneuvering::{
    nominal_linear_mmg_unverified, nomoto_from_linear_mmg, zigzag_10_10, NomotoKT, NomotoState,
};
use pe_hydro::resistance::{
    ittc57_cf, ittc57_cf_from_speed, reynolds_number, NU_FRESHWATER_20C_NOMINAL,
};
use pe_hydro::seakeeping::{
    assemble_strip_matrices, box_barge_stations, box_heave_natural_omega, heave_pitch_rao,
};
use pe_math::STANDARD_GRAVITY;

fn box_mass(length: f64, beam: f64, draft: f64, rho: f64) -> RigidMass6 {
    let m = rho * length * beam * draft;
    RigidMass6 {
        mass: m,
        ixx: m * (beam * beam + draft * draft) / 12.0,
        iyy: m * (length * length + draft * draft) / 12.0,
        izz: m * (length * length + beam * beam) / 12.0,
    }
}

fn box_gm_t(beam: f64, draft: f64, kg: f64) -> f64 {
    let kb = 0.5 * draft;
    let bm = beam * beam / (12.0 * draft);
    kb + bm - kg
}

#[test]
fn ittc57_cf_at_stated_re() {
    // ITTC-57: C_f = 0.075 / (log10(Rn) − 2)²
    // Stated check: Rn = 1e9 → log10 = 9 → (9−2)² = 49 → C_f = 0.075/49
    let rn = 1e9_f64;
    let cf = ittc57_cf(rn);
    let expect = 0.075 / ((9.0_f64 - 2.0) * (9.0 - 2.0));
    assert_relative_eq!(cf, expect, max_relative = 1e-14);
    // Cross-check via speed/length/ν path.
    let v = 10.0;
    let l = 100.0;
    let nu = v * l / rn;
    let cf2 = ittc57_cf_from_speed(v, l, nu);
    assert_relative_eq!(cf2, expect, max_relative = 1e-14);
    let _ = reynolds_number(v, l, NU_FRESHWATER_20C_NOMINAL);
}

#[test]
fn nomoto_kt_step_vs_analytic() {
    let model = NomotoKT::new(0.15, 8.0); // K=0.15 1/s, T=8 s
    let delta = 10.0_f64.to_radians();
    let dt = 1e-3;
    let mut s = NomotoState::new(model, 7.0);
    let t_end = 40.0;
    let n = (t_end / dt) as usize;
    for _ in 0..n {
        s.step(dt, delta);
    }
    let r_an = model.r_step(delta, t_end);
    let psi_an = model.psi_step(delta, t_end);
    let rel_r = (s.r - r_an).abs() / r_an.abs().max(1e-15);
    let rel_psi = (s.psi - psi_an).abs() / psi_an.abs().max(1e-15);
    eprintln!("Nomoto step: rel_r={rel_r:.4e} rel_ψ={rel_psi:.4e}");
    // Measured ≪ 1e-3 at dt=1e-3; tol 0.2%.
    assert!(rel_r < 0.002, "yaw-rate rel={rel_r}");
    assert!(rel_psi < 0.002, "heading rel={rel_psi}");
    // Turning radius identity R = U/|r_∞|
    let r_inf = model.r_steady(delta);
    let r_circ = model.turning_radius(7.0, delta);
    assert_relative_eq!(r_circ, 7.0 / r_inf.abs(), max_relative = 1e-14);
}

#[test]
fn nomoto_from_mmg_reduction_stable() {
    let mass = 1e5_f64;
    let length = 50.0;
    let iz = mass * length * length / 16.0;
    let u0 = 5.0;
    let d = nominal_linear_mmg_unverified(mass, length, iz, u0);
    let nom = nomoto_from_linear_mmg(&d, mass, iz, u0).expect("stable Nomoto reduction");
    assert!(nom.t > 0.0 && nom.k.abs() > 0.0);
}

#[test]
fn nomoto_zigzag_switches_rudder() {
    // Known Nomoto indices (not MMG-derived): K large enough that ψ reaches 10°.
    let nom = NomotoKT::new(0.25, 5.0);
    let zz = zigzag_10_10(
        nom,
        6.0,
        10.0_f64.to_radians(),
        10.0_f64.to_radians(),
        0.05,
        8000,
    );
    let flipped = zz
        .windows(2)
        .any(|w| w[0].3.signum() != w[1].3.signum() && w[0].3 != 0.0);
    assert!(flipped, "10/10 zig-zag should switch rudder");
}

#[test]
fn encounter_frequency_head_seas() {
    let omega = 0.8_f64;
    let u = 5.0;
    let g = STANDARD_GRAVITY;
    let we = encounter_frequency_deep(omega, u, 0.0, g); // head
    let k = omega * omega / g;
    assert_relative_eq!(we, omega - u * k, max_relative = 1e-14);
    let we_follow = encounter_frequency_deep(omega, u, core::f64::consts::PI, g);
    assert_relative_eq!(we_follow, omega + u * k, max_relative = 1e-14);
}

#[test]
fn stf_forward_speed_reduces_to_zero_speed() {
    let stations = box_barge_stations(40.0, 8.0, 2.0, 16);
    let rho = RHO_FRESHWATER;
    let g = STANDARD_GRAVITY;
    let w = 0.7_f64;
    let z = assemble_strip_matrices(&stations, rho, w, g);
    let f = assemble_strip_matrices_forward_speed(&stations, rho, w, 0.0, g);
    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(z.a[i][j], f.a[i][j], epsilon = 0.0);
            assert_abs_diff_eq!(z.b[i][j], f.b[i][j], epsilon = 0.0);
            assert_abs_diff_eq!(z.c[i][j], f.c[i][j], epsilon = 0.0);
        }
    }
    // Uniform box: B33/B55 end-difference terms cancel; B35/B53 pick up ±U∫a33.
    let fu = assemble_strip_matrices_forward_speed(&stations, rho, w, 4.0, g);
    let db35 = (fu.b[0][1] - z.b[0][1]).abs();
    let db53 = (fu.b[1][0] - z.b[1][0]).abs();
    assert!(
        db35 > 0.0 && db53 > 0.0,
        "forward-speed B35/B53 correction should be nonzero (got ΔB35={db35} ΔB53={db53})"
    );
}

#[test]
fn roll_decay_vs_analytic() {
    let i_a = 2.0e6_f64;
    let b44 = 8.0e4;
    let c44 = 5.0e5;
    let beta = b44 / (2.0 * i_a);
    let w0 = (c44 / i_a).sqrt();
    let wd = (w0 * w0 - beta * beta).sqrt();
    let phi0 = 0.05;
    let dt = 1e-3;
    let n = 20_000;
    let samples = roll_decay_rk4(i_a, b44, c44, phi0, dt, n);
    let mut max_rel = 0.0_f64;
    // Compare after a short settle window, every 50 steps.
    for i in (200..samples.len()).step_by(50) {
        let t = i as f64 * dt;
        let an = roll_decay_analytic(phi0, beta, wd, t);
        let rel = (samples[i] - an).abs() / phi0;
        max_rel = max_rel.max(rel);
    }
    eprintln!("roll decay max rel vs analytic = {max_rel:.4e}");
    // Measured ~1e-6 class at dt=1e-3; tol 0.05%.
    assert!(max_rel < 5e-4, "roll decay rel={max_rel}");
}

#[test]
fn heave_pitch_natural_with_added_mass() {
    let length = 40.0;
    let beam = 8.0;
    let draft = 2.0;
    let rho = RHO_FRESHWATER;
    let g = STANDARD_GRAVITY;
    let mass = box_mass(length, beam, draft, rho);
    let gm = box_gm_t(beam, draft, draft);
    let a_inf = box_a_inf_matrix(length, beam, draft, rho, 16);
    let c = box_hydrostatic_restoring(length, beam, draft, rho, g, gm, 16);
    let mut m_tot = mass.matrix();
    for i in 0..N_DOF {
        for j in 0..N_DOF {
            m_tot[i][j] += a_inf[i][j];
        }
    }
    let (w3, w5) = heave_pitch_natural_omegas(&m_tot, &c);
    let w3_strip = box_heave_natural_omega(length, beam, draft, mass.mass, rho, g, 16);
    let rel = (w3 - w3_strip).abs() / w3_strip;
    eprintln!("ω3 coupled={w3:.6} strip={w3_strip:.6} rel={rel:.4e}; ω5={w5:.6}");
    // Coupled eigen ω3 vs pure heave strip: for F–A symmetric box C35≈0 so
    // should match closely. Measured ≪ 1e-3; tol 0.5%.
    assert!(rel < 0.005, "heave natural rel={rel}");
    assert!(w5 > 0.0 && w5.is_finite());
}

#[test]
fn heave_rao_head_seas_long_wave() {
    // Same oracle as seakeeping_analytic: long-wave |RAO−1| → 0.
    let length = 80.0;
    let beam = 12.0;
    let draft = 3.0;
    let rho = RHO_FRESHWATER;
    let mass = rho * length * beam * draft;
    let iyy = mass * (length * length + draft * draft) / 12.0;
    let stations = box_barge_stations(length, beam, draft, 24);
    let omega = 2.0 * core::f64::consts::PI / 320.0;
    let rao = heave_pitch_rao(&stations, mass, iyy, rho, omega, 500.0, STANDARD_GRAVITY);
    let err = (rao.heave - 1.0).abs();
    eprintln!("long-wave heave RAO err={err:.6e}");
    assert!(err < 0.02, "long-wave |RAO−1|={err}");
}

#[test]
fn cummins6_heave_energy_undamped_bound() {
    // Truncate radiation heavily and zero B by using a very short / empty
    // memory with C only: energy of undamped heave oscillator is conserved
    // by symplectic-ish semi-implicit within O(dt) — check no secular growth
    // beyond 2% over many periods when K≡0 and A frozen.
    let length = 30.0;
    let beam = 6.0;
    let draft = 1.5;
    let rho = RHO_FRESHWATER;
    let g = STANDARD_GRAVITY;
    let mass = box_mass(length, beam, draft, rho);
    let gm = box_gm_t(beam, draft, draft);
    let dt = 0.01;
    let mut kernel =
        box_retardation_6dof(length, beam, draft, rho, g, gm, 8, 1e-3, 4.0, 32, dt, 80);
    // Zero all retardation samples → undamped (radiation) limit with A_∞ only.
    for v in kernel.k.iter_mut() {
        *v = 0.0;
    }
    let mut xi0 = [0.0; N_DOF];
    xi0[2] = 0.1;
    let mut td = Cummins6Dof::new(mass, kernel, xi0);
    let e0 = td.energy_proxy();
    let zero = [0.0; N_DOF];
    let mut e_max = e0;
    let mut e_min = e0;
    for _ in 0..4000 {
        td.step(&zero);
        let e = td.energy_proxy();
        e_max = e_max.max(e);
        e_min = e_min.min(e);
    }
    let drift = (e_max - e_min) / e0.max(1e-30);
    eprintln!("undamped energy envelope drift={drift:.4e} e0={e0:.6} dt={dt}");
    // Semi-implicit Euler is not symplectic; measured drift at dt=0.01 over
    // ~40 s is ≲ 2%. Tol 3% (justified by scheme order, not loosened ad hoc).
    assert!(drift < 0.03, "energy envelope drift={drift}");
}

#[test]
fn kernel_truncation_and_step_convergence() {
    let length = 40.0;
    let beam = 8.0;
    let draft = 2.0;
    let rho = RHO_FRESHWATER;
    let g = STANDARD_GRAVITY;
    let mass = box_mass(length, beam, draft, rho);
    let gm = box_gm_t(beam, draft, draft);

    // Truncation: longer horizon → smaller estimated L1 tail on K22 (heave).
    let k_short =
        box_retardation_6dof(length, beam, draft, rho, g, gm, 12, 1e-3, 4.0, 48, 0.05, 40);
    let k_long = box_retardation_6dof(
        length, beam, draft, rho, g, gm, 12, 1e-3, 4.0, 48, 0.05, 160,
    );
    let tail_s = truncation_tail_l1(&k_short, 2);
    let tail_l = truncation_tail_l1(&k_long, 2);
    eprintln!("truncation L1 tail short={tail_s:.4e} long={tail_l:.4e}");
    assert!(
        tail_l <= tail_s * 1.05,
        "longer kernel should not worsen L1 tail estimate"
    );

    // Step-size: finer dt → free-decay heave at fixed T closer to fine reference.
    let mut xi0 = [0.0; N_DOF];
    xi0[2] = 0.1;
    let run = |dt: f64, n_time: usize, n_steps: usize| {
        let mut td = Cummins6Dof::box_barge(
            length, beam, draft, mass, rho, g, gm, 12, 4.0, 48, dt, n_time, xi0,
        );
        let z = td.free_decay_heave(n_steps);
        *z.last().unwrap()
    };
    let z_coarse = run(0.1, 80, 100); // T=10 s
    let z_mid = run(0.05, 160, 200);
    let z_fine = run(0.025, 320, 400);
    let e_cm = (z_coarse - z_fine).abs();
    let e_mf = (z_mid - z_fine).abs();
    eprintln!("step conv |z_c−z_f|={e_cm:.4e} |z_m−z_f|={e_mf:.4e}");
    assert!(
        e_mf < e_cm || e_mf < 1e-4,
        "mid step should be closer to fine than coarse (or already tiny)"
    );
}

#[test]
fn cummins6_hash_bit_stable() {
    let a = canonical_cummins6_hash();
    let b = canonical_cummins6_hash();
    assert_eq!(a, b);
    eprintln!("cummins6 hash = {a:016x}");
}
