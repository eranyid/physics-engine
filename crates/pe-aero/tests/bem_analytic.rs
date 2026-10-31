//! BEM / actuator-disk oracles and rotor demos.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_aero::atmosphere::RHO0;
use pe_aero::bem::{
    figure_of_merit, hover_induced_velocity, ideal_hover_power, induced_velocity_momentum,
    nominal_rotor_polar, prandtl_tip_loss, BemRotor, RotorGeometry,
};

#[test]
fn hover_induced_velocity_formula() {
    let t = 10_000.0;
    let a = std::f64::consts::PI * 5.0_f64.powi(2);
    let vh = hover_induced_velocity(t, RHO0, a);
    assert_abs_diff_eq!(vh, (t / (2.0 * RHO0 * a)).sqrt(), epsilon = 1e-12);
    let p = ideal_hover_power(t, RHO0, a);
    assert_abs_diff_eq!(p, t * vh, epsilon = 1e-9);
    assert_abs_diff_eq!(figure_of_merit(t, p, RHO0, a), 1.0, epsilon = 1e-12);
}

#[test]
fn momentum_climb_reduces_induced_velocity() {
    let t = 5000.0;
    let a = 20.0;
    let vh = hover_induced_velocity(t, RHO0, a);
    let vi_climb = induced_velocity_momentum(t, RHO0, a, 5.0);
    assert!(vi_climb < vh);
    // Check quadratic identity: T = 2 ρ A (V+vi) vi
    let check = 2.0 * RHO0 * a * (5.0 + vi_climb) * vi_climb;
    assert_relative_eq!(check, t, max_relative = 1e-10);
}

#[test]
fn prandtl_tip_loss_bounds() {
    let f_mid = prandtl_tip_loss(3, 5.0, 2.5, 0.2);
    let f_tip = prandtl_tip_loss(3, 5.0, 4.99, 0.2);
    assert!(f_mid > 0.5 && f_mid <= 1.0);
    assert!(f_tip < f_mid);
    assert!(f_tip > 0.0);
}

#[test]
fn helicopter_hover_demo_energy_accounting() {
    // Nominal 2-blade rotor, R=4 m, chord 0.2 m, twist 10°→4° (unverified demo).
    let geom = RotorGeometry::uniform_chord(
        2,
        4.0,
        0.3,
        0.2,
        10.0_f64.to_radians(),
        4.0_f64.to_radians(),
        16,
    );
    let polar = nominal_rotor_polar(0.2, 0.012);
    let bem = BemRotor::new(geom.clone(), polar);
    let omega = 35.0; // rad/s
    let r = bem.evaluate(omega, 0.0, RHO0);
    assert!(
        r.converged,
        "BEM failed to converge in {} iters",
        r.iterations
    );
    assert!(r.thrust > 0.0);
    assert!(r.power > 0.0);
    let a = geom.disk_area();
    let fm = figure_of_merit(r.thrust, r.power, RHO0, a);
    // Physical FM typically 0.5–0.8; allow broad band for nominal polar.
    assert!(
        fm > 0.3 && fm < 1.05,
        "FM={fm} thrust={} power={}",
        r.thrust,
        r.power
    );
    // Energy: P should exceed ideal actuator-disk power.
    let p_ideal = ideal_hover_power(r.thrust, RHO0, a);
    assert!(
        r.power >= p_ideal * 0.95,
        "power {} < ~ideal {}",
        r.power,
        p_ideal
    );
    // Induced velocity order of momentum hover.
    let vh = hover_induced_velocity(r.thrust, RHO0, a);
    assert!(
        (r.induced_velocity - vh).abs() / vh < 0.5,
        "vi={} vh={}",
        r.induced_velocity,
        vh
    );
}

#[test]
fn propeller_thrust_falls_with_advance_ratio() {
    let geom = RotorGeometry::uniform_chord(
        3,
        1.0,
        0.15,
        0.08,
        25.0_f64.to_radians(),
        15.0_f64.to_radians(),
        12,
    );
    let polar = nominal_rotor_polar(0.08, 0.015);
    let bem = BemRotor::new(geom, polar);
    let omega = 200.0;
    let t0 = bem.evaluate(omega, 0.0, RHO0).thrust;
    let t1 = bem.evaluate(omega, 10.0, RHO0).thrust;
    let t2 = bem.evaluate(omega, 25.0, RHO0).thrust;
    assert!(t0 > t1 && t1 > t2, "T(J): {t0} > {t1} > {t2}");
    assert!(t0 > 0.0);
}

#[test]
fn tip_loss_reduces_thrust_vs_no_effective_loss_at_root() {
    // Sanity: F→1 well inboard.
    let f = prandtl_tip_loss(4, 10.0, 1.0, 0.3);
    assert!(f > 0.95);
}
