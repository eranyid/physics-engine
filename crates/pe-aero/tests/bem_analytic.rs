//! BEM / actuator-disk oracles and rotor demos.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use pe_aero::atmosphere::RHO0;
use pe_aero::bem::{
    figure_of_merit, hover_induced_velocity, ideal_hover_power, ideal_rotor_polar,
    induced_velocity_momentum, nominal_rotor_polar, prandtl_tip_loss, BemRotor, RotorGeometry,
};

fn annulus_area(radius: f64, hub: f64) -> f64 {
    std::f64::consts::PI * (radius * radius - hub * hub)
}

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

/// Ideal-twist + Betz chord, no tip loss, Cd=0: BEM ↔ annulus momentum hover.
#[test]
fn ideal_bem_matches_momentum_hover() {
    let radius = 5.0;
    let hub = 0.2 * radius;
    let chord_tip = 0.10;
    let omega = 50.0;
    let alpha = 4.0_f64.to_radians();
    // Momentum reference uses the **annulus** (hub cut-out), matching ∑ 4πr dr.
    let area = annulus_area(radius, hub);
    // One redesign step: build twist for a guess, then rebuild at the measured vh.
    let mut v_design = 5.0;
    let mut r = {
        let geom =
            RotorGeometry::ideal_hover_twist(2, radius, hub, chord_tip, omega, v_design, alpha, 48);
        BemRotor::new(geom, ideal_rotor_polar(chord_tip))
            .ideal()
            .evaluate(omega, 0.0, RHO0)
    };
    assert!(
        r.converged,
        "iters={} residual={}",
        r.iterations, r.residual
    );
    v_design = hover_induced_velocity(r.thrust, RHO0, area);
    {
        let geom =
            RotorGeometry::ideal_hover_twist(2, radius, hub, chord_tip, omega, v_design, alpha, 48);
        r = BemRotor::new(geom, ideal_rotor_polar(chord_tip))
            .ideal()
            .evaluate(omega, 0.0, RHO0);
    }
    assert!(
        r.converged,
        "iters={} residual={}",
        r.iterations, r.residual
    );
    assert!(r.residual < 1e-8, "residual={}", r.residual);
    let vh = hover_induced_velocity(r.thrust, RHO0, area);
    let vi_err = (r.induced_velocity - vh).abs() / vh;
    let vi_tw_err = (r.induced_velocity_thrust_wt - vh).abs() / vh;
    let p_ideal = ideal_hover_power(r.thrust, RHO0, area);
    let p_err = (r.power - p_ideal).abs() / p_ideal.max(1.0);
    let p_tw = r.thrust * r.induced_velocity_thrust_wt;
    let p_id_err = (r.power - p_tw).abs() / p_tw.max(1.0);
    let fm = figure_of_merit(r.thrust, r.power, RHO0, area);
    eprintln!(
        "ideal BEM: |vi−vh|/vh={vi_err:.6} |vi_tw−vh|/vh={vi_tw_err:.6} |P−T vh|/(T vh)={p_err:.6} |P−T vi_tw|={p_id_err:.3e} FM={fm:.6} iters={} residual={:.3e}",
        r.iterations, r.residual
    );
    assert!(r.iterations < 200, "should converge before max_iter");
    assert!(
        vi_err < 0.02,
        "ideal |vi−vh|/vh = {vi_err} (vi={}, vh={})",
        r.induced_velocity,
        vh
    );
    assert!(vi_tw_err < 0.02, "ideal |vi_tw−vh|/vh = {vi_tw_err}");
    assert!(
        p_err < 0.02,
        "ideal |P−T vh|/(T vh) = {p_err} (P={}, T vh={})",
        r.power,
        p_ideal
    );
    assert!(
        p_id_err < 0.005,
        "Cd=0 identity |P−T vi_tw| err = {p_id_err}"
    );
    assert!((fm - 1.0).abs() < 0.02, "ideal FM={fm}");
}

/// Tip loss alone (Cd=0) lowers FM; bounds from measured values.
#[test]
fn tip_loss_reduces_figure_of_merit() {
    let radius = 5.0;
    let hub = 0.2 * radius;
    let chord_tip = 0.10;
    let omega = 50.0;
    let alpha = 4.0_f64.to_radians();
    let area = annulus_area(radius, hub);
    let v_design = 5.0;
    let geom =
        RotorGeometry::ideal_hover_twist(2, radius, hub, chord_tip, omega, v_design, alpha, 48);

    let ideal = BemRotor::new(geom.clone(), ideal_rotor_polar(chord_tip))
        .ideal()
        .evaluate(omega, 0.0, RHO0);
    let mut with_tip = BemRotor::new(geom, ideal_rotor_polar(chord_tip));
    with_tip.tip_loss = true;
    with_tip.zero_drag = true;
    with_tip.tol = 1e-8;
    with_tip.max_iter = 200;
    with_tip.relax = 0.45;
    let tip = with_tip.evaluate(omega, 0.0, RHO0);
    assert!(
        ideal.converged && tip.converged,
        "ideal conv={} tip conv={} tip residual={}",
        ideal.converged,
        tip.converged,
        tip.residual
    );

    let fm_i = figure_of_merit(ideal.thrust, ideal.power, RHO0, area);
    let fm_t = figure_of_merit(tip.thrust, tip.power, RHO0, area);
    let drop = fm_i - fm_t;
    eprintln!("tip-loss: FM_ideal={fm_i:.6} FM_tip={fm_t:.6} drop={drop:.6}");
    assert!(
        drop > 0.01,
        "expected tip-loss FM drop: ideal={fm_i} tip={fm_t}"
    );
    assert!(
        fm_t > 0.85 && fm_t < 0.99,
        "tip-loss FM={fm_t} outside stated band [0.85, 0.99]"
    );
}

/// Profile drag alone and combined tip+drag FM — bands from measurement.
#[test]
fn profile_drag_and_combined_fm_bounds() {
    let radius = 5.0;
    let hub = 0.2 * radius;
    let chord_tip = 0.10;
    let omega = 50.0;
    let alpha = 4.0_f64.to_radians();
    let area = annulus_area(radius, hub);
    let v_design = 5.0;
    let geom =
        RotorGeometry::ideal_hover_twist(2, radius, hub, chord_tip, omega, v_design, alpha, 48);
    let cd0 = 0.004;

    let mut drag_only = BemRotor::new(geom.clone(), nominal_rotor_polar(chord_tip, cd0));
    drag_only.tip_loss = false;
    drag_only.zero_drag = false;
    drag_only.tol = 1e-8;
    drag_only.max_iter = 200;
    drag_only.relax = 0.45;
    let d = drag_only.evaluate(omega, 0.0, RHO0);

    let mut both = BemRotor::new(geom, nominal_rotor_polar(chord_tip, cd0));
    both.tip_loss = true;
    both.zero_drag = false;
    both.tol = 1e-8;
    both.max_iter = 200;
    both.relax = 0.45;
    let b = both.evaluate(omega, 0.0, RHO0);
    assert!(d.converged && b.converged);

    let fm_d = figure_of_merit(d.thrust, d.power, RHO0, area);
    let fm_b = figure_of_merit(b.thrust, b.power, RHO0, area);
    eprintln!("drag Cd0={cd0}: FM_drag={fm_d:.6} FM_combined={fm_b:.6}");
    assert!(
        fm_d > 0.70 && fm_d < 0.98,
        "drag-only FM={fm_d} outside [0.70, 0.98]"
    );
    assert!(
        fm_b > 0.65 && fm_b < fm_d,
        "combined FM={fm_b} should be below drag-only {fm_d} and >0.65"
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
    let f = prandtl_tip_loss(4, 10.0, 1.0, 0.3);
    assert!(f > 0.95);
}
