//! Analytic / formula tests for pe-blast.

use approx::assert_relative_eq;
use pe_blast::euler::{sod_l1_density_error, Euler1d, ExactRiemann, Prim};
use pe_blast::fragments::{gurney_velocity, GurneyGeometry};
use pe_blast::friedlander::{positive_impulse, pressure};
use pe_blast::kingery::incident_overpressure;
use pe_blast::kinney::peak_overpressure;
use pe_blast::rankine_hugoniot::{from_overpressure_ratio, GAMMA_AIR};
use pe_blast::scaling::scaled_distance;
use pe_blast::sedov::{shock_radius, solve_xi0, SedovGeometry};
use pe_blast::{blast_pyramid_hash, xi0_gamma_14_spherical};

#[test]
fn hopkinson_cranz_cube_root() {
    let z = scaled_distance(2.0, 8.0, 1.0).unwrap();
    assert_relative_eq!(z, 1.0, epsilon = 1e-14);
}

#[test]
fn kinney_z1_z10() {
    let p0 = 101.325e3;
    let ps1 = peak_overpressure(1.0, p0) / 1e3;
    let ps10 = peak_overpressure(10.0, p0) / 1e3;
    assert_relative_eq!(ps1, 1008.7895037932, epsilon = 1e-8);
    assert_relative_eq!(ps10, 9.9853637227, epsilon = 1e-8);
}

#[test]
fn friedlander_impulse_analytic() {
    let i = positive_impulse(1e5, 0.01, 0.0);
    assert_relative_eq!(i, 500.0, epsilon = 1e-12);
    assert_eq!(pressure(0.0, 1e5, 0.01, 1.0), 1e5);
}

#[test]
fn swisdak_incident_in_range_flag() {
    let r = incident_overpressure(1.0);
    assert!(r.in_range);
    let oob = incident_overpressure(0.01);
    assert!(!oob.in_range);
}

#[test]
fn rh_reflection_limits() {
    assert_relative_eq!(
        from_overpressure_ratio(1e-8, GAMMA_AIR).reflected_overpressure_ratio,
        2.0,
        epsilon = 1e-5
    );
    assert_relative_eq!(
        from_overpressure_ratio(1e10, GAMMA_AIR).reflected_overpressure_ratio,
        8.0,
        epsilon = 1e-5
    );
}

#[test]
fn sedov_xi0_and_scaling() {
    let xi = xi0_gamma_14_spherical();
    assert!((xi - 1.033).abs() < 0.05, "xi0={xi}");
    let sol = solve_xi0(1.4, SedovGeometry::Spherical);
    let r1 = shock_radius(&sol, 1e12, 1.225, 1.0);
    let r2 = shock_radius(&sol, 1e12, 1.225, 32.0);
    assert_relative_eq!(r2 / r1, 4.0, epsilon = 1e-12);
}

#[test]
fn sod_l1_converges() {
    let gamma = 1.4;
    let t = 0.2;
    let mut errs = Vec::new();
    for n in [100, 200, 400] {
        let mut s = Euler1d::sod(n, gamma);
        s.run_cfl(t, 0.4);
        errs.push(sod_l1_density_error(&s, t));
    }
    // Expect roughly first-order or better reduction when doubling N.
    let rate = (errs[0] / errs[1]).log2();
    assert!(rate > 0.5, "convergence rate {rate} with errs={errs:?}");
    assert!(errs[2] < errs[0]);
}

#[test]
fn sod_exact_riemann_left_state() {
    let ex = ExactRiemann {
        left: Prim {
            rho: 1.0,
            u: 0.0,
            p: 1.0,
        },
        right: Prim {
            rho: 0.125,
            u: 0.0,
            p: 0.1,
        },
        gamma: 1.4,
    };
    let p = ex.sample(0.1, 0.2, 0.5);
    assert_relative_eq!(p.rho, 1.0, epsilon = 1e-9);
}

#[test]
fn gurney_flat_cm_one() {
    let v = gurney_velocity(1000.0, 1.0, GurneyGeometry::Flat).unwrap();
    assert_relative_eq!(v, 1000.0 * (0.5_f64).sqrt(), epsilon = 1e-12);
}

#[test]
fn blast_hash_bit_stable() {
    let a = blast_pyramid_hash();
    let b = blast_pyramid_hash();
    assert_eq!(a, b);
}

#[test]
fn hllc_identical_states_is_physical_flux() {
    use pe_blast::hllc;
    let wl = Prim {
        rho: 1.0,
        u: 0.0,
        p: 1.0,
    };
    let f = hllc(wl, wl, 1.4);
    assert_relative_eq!(f[0], 0.0, epsilon = 1e-14);
    assert_relative_eq!(f[1], 1.0, epsilon = 1e-14);
    assert_relative_eq!(f[2], 0.0, epsilon = 1e-14);
}

#[test]
fn euler_spherical_sedov_like_radius() {
    use pe_blast::{shock_radius, solve_xi0, Geometry1d, SedovGeometry};
    // Deposit energy near the origin on a coarse spherical mesh; track the
    // density jump radius and compare loosely to Sedov R(t).
    let gamma = 1.4;
    let n = 200;
    let r_max = 1.0;
    let mut s = Euler1d::uniform(
        0.01,
        r_max,
        n,
        Prim {
            rho: 1.0,
            u: 0.0,
            p: 1e-4,
        },
        gamma,
        Geometry1d::Spherical,
    );
    // Hot central cells approximate a point blast.
    let e_blast = 1.0;
    for i in 0..4 {
        let vol = 4.0 / 3.0 * core::f64::consts::PI
            * (s.x[i] + 0.5 * s.dx[i]).powi(3).max(1e-8);
        let p = (gamma - 1.0) * e_blast / (4.0 * vol);
        s.u[i] = Prim {
            rho: 1.0,
            u: 0.0,
            p: p.max(1.0),
        }
        .to_cons(gamma);
    }
    let t_end = 0.05;
    s.run_cfl(t_end, 0.3);
    // Find outermost cell with rho > 2 * ambient as a shock proxy.
    let mut r_num = 0.0;
    for i in 0..n {
        if let Some(p) = Prim::from_cons(s.u[i], gamma) {
            if p.rho > 2.0 {
                r_num = s.x[i];
            }
        }
    }
    let sol = solve_xi0(gamma, SedovGeometry::Spherical);
    let r_sedov = shock_radius(&sol, e_blast, 1.0, t_end);
    // Coarse 1-D Sedov setup is qualitative — allow factor-of-two envelope.
    assert!(
        r_num > 0.05 && r_num < 2.0 * r_sedov.max(0.1),
        "r_num={r_num} r_sedov={r_sedov}"
    );
}
