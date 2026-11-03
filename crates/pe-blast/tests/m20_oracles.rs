//! M20 oracles with tight tolerances (never loosened to pass).
//!
//! Each test states the tolerance and why. Comparisons are against closed-form
//! or published chart evaluations in-tree — never against a non-oracle.

use approx::assert_relative_eq;
use pe_ballistics::ProjectileSystem;
use pe_blast::compound::{
    bond_break_force, bond_breaks_on_impulse, break_bonds_by_impulse, voronoi_box_partition,
};
use pe_blast::fragments::{
    build_fragment_specs, fragments_kinetic_energy, fragments_momentum, gurney_constant_by_name,
    gurney_metal_ke, gurney_velocity, mott_mean_mass, mott_sample_many, spawn_fragments_ballistic,
    GurneyGeometry,
};
use pe_blast::friedlander::{positive_impulse, pressure};
use pe_blast::kingery::{self, incident_overpressure};
use pe_blast::kinney::peak_overpressure;
use pe_blast::loading::{
    face_positive_impulse_params, rigid_wall_reflection_coefficient, BlastLoadParams, BlastSource,
    ExposedFace, OverpressureModel, ReflectionMix,
};
use pe_blast::rankine_hugoniot::{from_overpressure_ratio, GAMMA_AIR};
use pe_blast::scaling::scaled_distance;
use pe_blast::{
    blast_destruction_hash, blast_external_loads_hash, blast_fragments_hash, blast_pyramid_hash,
};
use pe_collision::{Joint, JointBreak};
use pe_math::libm_ops;
use pe_math::Vec3;

/// Friedlander analytic impulse vs closed form.
///
/// Tolerance: **1e-12 abs/rel** — pure algebra + `libm::exp`; exact for `b=0`,
/// closed form for `b≠0`.
#[test]
fn oracle_friedlander_impulse_closed_form() {
    // b=0: I = p_s t_d / 2
    assert_relative_eq!(positive_impulse(2e5, 0.02, 0.0), 2000.0, epsilon = 1e-12);
    // b=1: I = p_s t_d (1 - 1 + e^{-1}) / 1 = p_s t_d / e
    let i = positive_impulse(1e5, 0.01, 1.0);
    let expect = 1e5 * 0.01 * libm_ops::exp(-1.0);
    // Closed form (b−1+e^{−b})/b² at b=1: (0 + e^{-1}) = 1/e
    assert_relative_eq!(i, expect, epsilon = 1e-12 * expect);
    // Cross-check trapezoid (fine grid) within 1e-6 relative — numeric, not the lock.
    let n = 20_000usize;
    let t_d = 0.015;
    let p_s = 3e5;
    let b = 1.7;
    let analytic = positive_impulse(p_s, t_d, b);
    let dt = t_d / n as f64;
    let mut num = 0.0;
    for k in 0..=n {
        let t = k as f64 * dt;
        let w = if k == 0 || k == n { 0.5 } else { 1.0 };
        num += w * pressure(t, p_s, t_d, b);
    }
    num *= dt;
    assert_relative_eq!(analytic, num, epsilon = 1e-6 * analytic);
}

/// Hopkinson–Cranz scaling invariance: same Z ⇒ same Kinney p_s.
///
/// Tolerance: **0 abs** on Z algebra; **1e-14 rel** on p_s (same formula path).
#[test]
fn oracle_hopkinson_scaling_invariance() {
    let z1 = scaled_distance(2.0, 8.0, 1.0).unwrap();
    let z2 = scaled_distance(4.0, 64.0, 1.0).unwrap();
    assert_relative_eq!(z1, 1.0, epsilon = 0.0);
    assert_relative_eq!(z2, 1.0, epsilon = 0.0);
    let p0 = 101_325.0;
    let a = peak_overpressure(z1, p0);
    let b = peak_overpressure(z2, p0);
    assert_relative_eq!(a, b, epsilon = 0.0);
    // Different W,R but equal Z → equal Swisdak incident too.
    let ki = incident_overpressure(z1).value;
    let kj = incident_overpressure(z2).value;
    assert_relative_eq!(ki, kj, epsilon = 0.0);
}

/// Gurney velocity vs published cylindrical C/M case.
///
/// Case: √(2E)=2440 m/s (TNT handbook figure, **unverified** primary cell),
/// C/M=0.5, cylindrical → V = 2440 √(0.5/(1+0.25)) = 2440 √0.4.
/// Tolerance: **1e-12 rel** against the closed Gurney formula (not against a
/// live fire table — the √(2E) itself is labelled unverified in the table).
#[test]
fn oracle_gurney_cylindrical_published_case() {
    let c = gurney_constant_by_name("TNT").unwrap();
    assert!(!c.verified, "TNT √(2E) must stay labelled unverified");
    let v = gurney_velocity(c.sqrt_2e_m_s, 0.5, GurneyGeometry::Cylindrical).unwrap();
    let expect = 2440.0 * libm_ops::sqrt(0.4);
    assert_relative_eq!(v, expect, epsilon = 1e-12 * expect);
}

/// Mott sample mean vs closed form E[m]=2μ.
///
/// Tolerance: **2% rel** on N=50_000 samples (Monte Carlo SE ~ O(1/√N) of a
/// heavy-tailed law); closed-form identity is exact at **0**.
#[test]
fn oracle_mott_mean_vs_closed_form() {
    let mu = 0.01;
    assert_relative_eq!(mott_mean_mass(mu), 2.0 * mu, epsilon = 0.0);
    let samples = mott_sample_many(0xA0_77, mu, 50_000);
    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    let expect = mott_mean_mass(mu);
    let rel = (mean - expect).abs() / expect;
    assert!(
        rel < 0.02,
        "Mott sample mean rel err {rel} (tol 2%, N=50000); mean={mean} expect={expect}"
    );
}

/// Fragment KE vs Gurney metal energy budget; net momentum with antipodal pairs.
///
/// Energy: with mass normalization Σm=M and common V, KE = ½ M V² **exactly**
/// (tol **1e-12 rel**). Momentum: equal-mass antipodal direction pairs cancel
/// exactly (tol **1e-14** on |P|/(M V)) — Fibonacci+Mott is not a momentum
/// oracle (mass-weighted lattice residual is geometry, not Gurney physics).
#[test]
fn oracle_fragment_energy_and_momentum_budget() {
    let m_casing = 2.0;
    let v = gurney_velocity(2440.0, 0.4, GurneyGeometry::Spherical).unwrap();
    let masses = mott_sample_many(0xE7E7, 0.03, 64);
    let specs = build_fragment_specs(Vec3::zero(), v, &masses, Some(m_casing));
    let ke = fragments_kinetic_energy(&specs);
    let budget = gurney_metal_ke(m_casing, v);
    assert_relative_eq!(ke, budget, epsilon = 1e-12 * budget);

    // Antipodal equal-mass pairs → exact momentum cancellation.
    let m_each = m_casing / 8.0;
    let dirs = [
        Vec3::unit_x(),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 1.0, 0.0).normalize(),
    ];
    let mut paired = Vec::new();
    for d in dirs {
        paired.push(pe_blast::FragmentSpec {
            mass_kg: m_each,
            position: Vec3::zero(),
            velocity: d * v,
        });
        paired.push(pe_blast::FragmentSpec {
            mass_kg: m_each,
            position: Vec3::zero(),
            velocity: d * (-v),
        });
    }
    let p = fragments_momentum(&paired);
    let scale = m_casing * v;
    assert!(
        p.norm() / scale < 1e-14,
        "antipodal momentum residual {}",
        p.norm() / scale
    );
    let ke_p = fragments_kinetic_energy(&paired);
    assert_relative_eq!(ke_p, budget, epsilon = 1e-12 * budget);
}

/// Bond-break threshold exactness: strict inequality on impulse.
///
/// Tolerance: **exact** boolean — `|J| > J*` breaks; `|J| = J*` does not.
#[test]
fn oracle_bond_break_threshold_exactness() {
    let j_star = 12.5;
    assert!(!bond_breaks_on_impulse(j_star, j_star));
    assert!(!bond_breaks_on_impulse(j_star - 1e-12, j_star));
    assert!(bond_breaks_on_impulse(j_star + 1e-12, j_star));
    // Joint helper path
    let mut joints = vec![
        Joint::distance(0, 1, Vec3::zero(), Vec3::zero(), 1.0).with_break(JointBreak {
            force: 100.0,
            torque: 0.0,
        }),
    ];
    let broken = break_bonds_by_impulse(&mut joints, &[j_star], &[j_star]);
    assert!(broken.is_empty());
    let broken = break_bonds_by_impulse(&mut joints, &[j_star + 1e-9], &[j_star]);
    assert_eq!(broken, vec![0]);
    assert!(joints[0].broken);
    // Stress × area
    assert_relative_eq!(bond_break_force(1e6, 0.02), 2e4, epsilon = 0.0);
}

/// Rigid-wall reflection coefficient vs RH closed form / KB chart.
///
/// RH limits: weak → 2, strong → 8 at γ=1.4 (**1e-5 / 1e-6** as in unit tests).
/// Mid-strength: RH formula vs helper (**0**). KB reflected/incident ratio at
/// Z=1 compared to RH(ξ) with ξ from KB incident/p0 — engineering charts are
/// not identical to ideal-gas RH; lock **rel < 15%** and document as chart vs
/// ideal-gas residual (not a bug).
#[test]
fn oracle_rigid_wall_reflection_kb_rh() {
    assert_relative_eq!(
        rigid_wall_reflection_coefficient(1e-8, GAMMA_AIR),
        2.0,
        epsilon = 1e-5
    );
    assert_relative_eq!(
        rigid_wall_reflection_coefficient(1e10, GAMMA_AIR),
        8.0,
        epsilon = 1e-6
    );
    let p0 = 101_325.0;
    let z = 1.0;
    let p_i = kingery::incident_overpressure(z).value;
    let p_r_kb = kingery::reflected_overpressure(z).value;
    let ratio_kb = p_r_kb / p_i;
    let xi = p_i / p0;
    let ratio_rh = from_overpressure_ratio(xi, GAMMA_AIR).reflected_overpressure_ratio;
    let rel = (ratio_kb - ratio_rh).abs() / ratio_rh;
    assert!(
        rel < 0.15,
        "KB vs RH reflection rel {rel} (tol 15%); kb={ratio_kb} rh={ratio_rh} Z={z}"
    );
    // Helper consistency
    assert_relative_eq!(
        rigid_wall_reflection_coefficient(xi, GAMMA_AIR),
        ratio_rh,
        epsilon = 0.0
    );
}

/// Face impulse with KB params is positive and finite at Z~1.
#[test]
fn oracle_kb_face_impulse_positive() {
    let src = BlastSource {
        origin: Vec3::zero(),
        w_eq_kg: 1.0,
        t0: 0.0,
        p0: 101_325.0,
        td_at_z1: 0.01,
        decay_b: 1.0,
    };
    let face = ExposedFace {
        body: 0,
        center: Vec3::new(1.0, 0.0, 0.0),
        normal: Vec3::new(-1.0, 0.0, 0.0),
        area: 1.0,
    };
    let params = BlastLoadParams {
        overpressure: OverpressureModel::KingeryBulmash,
        reflection: ReflectionMix::IncidentReflected,
        drag_cd: 1.0,
        shadow_factor: 1.0,
        gamma: GAMMA_AIR,
    };
    let i = face_positive_impulse_params(&src, &face, &params);
    assert!(i.is_finite() && i > 0.0, "impulse={i}");
}

/// Voronoi partition is deterministic and mass-conserving.
#[test]
fn oracle_voronoi_deterministic_mass() {
    let a = voronoi_box_partition(Vec3::zero(), Vec3::new(0.5, 0.5, 0.5), 5, 800.0, 8, 42);
    let b = voronoi_box_partition(Vec3::zero(), Vec3::new(0.5, 0.5, 0.5), 5, 800.0, 8, 42);
    assert_eq!(a.chunks.len(), b.chunks.len());
    for (c1, c2) in a.chunks.iter().zip(b.chunks.iter()) {
        assert_relative_eq!(c1.mass, c2.mass, epsilon = 0.0);
        assert_relative_eq!(c1.com.x, c2.com.x, epsilon = 0.0);
    }
    let m: f64 = a.chunks.iter().map(|c| c.mass).sum();
    assert_relative_eq!(m, 800.0, epsilon = 1e-9 * 800.0);
}

/// Fragment ballistic spawn is deterministic given seed / shot ids.
#[test]
fn oracle_fragment_spawn_deterministic() {
    let masses = mott_sample_many(9, 0.01, 8);
    let v = gurney_velocity(2000.0, 0.3, GurneyGeometry::Cylindrical).unwrap();
    let specs = build_fragment_specs(Vec3::new(1.0, 2.0, 3.0), v, &masses, Some(0.5));
    let mut s1 = ProjectileSystem::new(0.0);
    s1.enable_spin_effects = false;
    spawn_fragments_ballistic(&mut s1, &specs, 100, 0.01);
    let mut s2 = ProjectileSystem::new(0.0);
    s2.enable_spin_effects = false;
    spawn_fragments_ballistic(&mut s2, &specs, 100, 0.01);
    for _ in 0..30 {
        s1.step(1.0 / 120.0, &[], &[]);
        s2.step(1.0 / 120.0, &[], &[]);
    }
    for (a, b) in s1.projectiles.iter().zip(s2.projectiles.iter()) {
        assert_eq!(a.x.to_array(), b.x.to_array());
        assert_eq!(a.v.to_array(), b.v.to_array());
    }
}

/// Locked hashes: pyramid unchanged; new lines bit-stable.
#[test]
fn oracle_hashes_locked_and_stable() {
    assert_eq!(blast_pyramid_hash(), 0xb51e_11e8_9d4d_d055);
    assert_eq!(blast_pyramid_hash(), blast_pyramid_hash());
    let e = blast_external_loads_hash();
    assert_eq!(e, blast_external_loads_hash());
    assert_eq!(e, 0x693b_60be_33f9_b8f2);
    let f = blast_fragments_hash();
    assert_eq!(f, blast_fragments_hash());
    assert_eq!(f, 0x36d3_f61d_77c1_4859);
    let d = blast_destruction_hash();
    assert_eq!(d, blast_destruction_hash());
    assert_eq!(d, 0x38f5_5bac_9ab3_d16f);
}
