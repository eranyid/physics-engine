//! Canonical 6-DOF FDM scenarios for cross-OS determinism hashes.

use pe_core::diagnostics::state_hash64;
use pe_core::forces::{ForceGenerator, UniformGravity};
use pe_core::integrators::rk4;
use pe_core::{RigidBody, World};
use pe_math::{Quat, Vec3};

use crate::atmosphere::{DensityModel, RHO0};
use crate::fdm::navion::navion_nelson;
use crate::fdm::{ControlInputs, SixDofAero, SixDofModel, StabilityDerivatives};
use crate::turbulence::{DrydenParams, DrydenTurbulence, HeldDrydenGust};

/// Canonical 6-DOF FDM hash scenario (must stay in sync with
/// `docs/DETERMINISM.md`, the `fdm_hash` binary, and
/// `physics_engine_rs.canonical_fdm_hash()`).
///
/// - Navion (Nelson / Teper) SI model at `z = 1000` m
/// - Steady wind `(2, 0.5, 0)` m/s; elevator `δe = −0.02` rad;
///   aileron `0.01`, rudder `−0.005`; thrust 1200 N with 0.1 m z-offset
/// - Integrate **RK4**, `t_end = 1`, `dt = 1e-2`
/// - Hash: FNV-1a 64 over little-endian IEEE bits of `State::pack()`
///
/// Hash `ba6d68e43c69d315` supersedes `fb8c25d5fe2184fd` after correcting
/// body-+y side-force sign for `C_Y` (Dutch-roll damping). Same scenario
/// inputs; only the force algebra changed.
pub fn canonical_fdm_hash() -> u64 {
    let nav = navion_nelson();
    let mut model = nav.model.clone();
    model.density = DensityModel::Constant(RHO0);
    model.thrust = 1200.0;
    model.thrust_offset = Vec3::new(0.0, 0.0, -0.1);
    model.controls = ControlInputs {
        delta_e: -0.02,
        delta_a: 0.01,
        delta_r: -0.005,
    };
    model.wind = Vec3::new(2.0, 0.5, 0.0);

    let alpha0 = nav.alpha_trim;
    let v0 = nav.v_ref;
    let q_att = Quat::from_axis_angle(Vec3::unit_y(), -alpha0);
    let body = RigidBody::new(model.mass)
        .with_inertia(model.inertia_matrix())
        .with_position(Vec3::new(0.0, 0.0, 1000.0))
        .with_velocity(Vec3::new(v0, 0.0, 0.0))
        .with_orientation(q_att);

    let mut world = World::new(
        vec![body],
        vec![
            Box::new(UniformGravity::earth()) as Box<dyn ForceGenerator>,
            Box::new(SixDofAero::new(0, model)),
        ],
    );
    world.run_for(1.0, 1e-2, rk4);
    state_hash64(&world.state)
}

/// Dryden gust-sequence hash (FNV-1a over 100 `(u,v,w)` samples at `V=50`,
/// `dt=0.01`, seed `0xAE80_FD01`). Locked beside the FDM body hash.
///
/// Uses 2nd-order Dryden `v`/`w` (MIL-F-8785C) + 1st-order `u`. Hash
/// `2895b75a14f621c5` supersedes `233148553b199f90` (1st-order surrogate).
pub fn canonical_dryden_gust_hash() -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET;
    let mut d = DrydenTurbulence::new(DrydenParams::light_low_altitude(), 0xAE80_FD01);
    for _ in 0..100 {
        let g = d.step(50.0, 1e-2);
        for x in [g.x, g.y, g.z] {
            for b in x.to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(FNV_PRIME);
            }
        }
    }
    h
}

/// FDM + held Dryden gust inside [`SixDofAero`] (RK4 replay).
///
/// Same rigid-body IC as [`canonical_fdm_hash`], plus
/// [`HeldDrydenGust`] (`dt=1e-2`, seed `0xAE80_FD01`, light-low-altitude
/// params). Gust is piecewise-constant on each major step so all RK4 stages
/// in an interval see identical wind — bit-deterministic replay.
pub fn canonical_fdm_gust_hash() -> u64 {
    let nav = navion_nelson();
    let mut model = nav.model.clone();
    model.density = DensityModel::Constant(RHO0);
    model.thrust = 1200.0;
    model.thrust_offset = Vec3::new(0.0, 0.0, -0.1);
    model.controls = ControlInputs {
        delta_e: -0.02,
        delta_a: 0.01,
        delta_r: -0.005,
    };
    model.wind = Vec3::new(2.0, 0.5, 0.0);

    let alpha0 = nav.alpha_trim;
    let v0 = nav.v_ref;
    let q_att = Quat::from_axis_angle(Vec3::unit_y(), -alpha0);
    let body = RigidBody::new(model.mass)
        .with_inertia(model.inertia_matrix())
        .with_position(Vec3::new(0.0, 0.0, 1000.0))
        .with_velocity(Vec3::new(v0, 0.0, 0.0))
        .with_orientation(q_att);

    let gust = HeldDrydenGust::new(DrydenParams::light_low_altitude(), 0xAE80_FD01, 1e-2);
    let mut world = World::new(
        vec![body],
        vec![
            Box::new(UniformGravity::earth()) as Box<dyn ForceGenerator>,
            Box::new(SixDofAero::with_gust(0, model, gust)),
        ],
    );
    world.run_for(1.0, 1e-2, rk4);
    state_hash64(&world.state)
}

/// Lightweight smoke model — **unverified** (not from a published table).
pub fn unverified_demo_model() -> SixDofModel {
    SixDofModel {
        mass: 1000.0,
        s: 16.0,
        b: 10.0,
        c: 1.6,
        inertia_diag: (2000.0, 3000.0, 4000.0),
        deriv: StabilityDerivatives {
            cl0: 0.2,
            cl_alpha: 5.0,
            cl_alphadot: 0.0,
            cl_q: 4.0,
            cl_de: 0.4,
            cd0: 0.03,
            cd_alpha: 0.2,
            cm0: 0.0,
            cm_alpha: -0.8,
            cm_alphadot: -4.0,
            cm_q: -12.0,
            cm_de: -1.0,
            cy_beta: -0.5,
            cy_p: 0.0,
            cy_r: 0.0,
            cy_da: 0.0,
            cy_dr: 0.15,
            cl_beta: -0.08,
            cl_p: -0.4,
            cl_r: 0.1,
            cl_da: 0.12,
            cl_dr: 0.01,
            cn_beta: 0.06,
            cn_p: -0.05,
            cn_r: -0.1,
            cn_da: -0.01,
            cn_dr: -0.06,
        },
        thrust: 0.0,
        thrust_axis: Vec3::unit_x(),
        thrust_offset: Vec3::zero(),
        density: DensityModel::Constant(RHO0),
        wind: Vec3::zero(),
        controls: ControlInputs::default(),
        alpha_trim: 0.0,
        verified: false,
        ground_height: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fdm_hash_bit_stable_on_repeat() {
        let a = canonical_fdm_hash();
        let b = canonical_fdm_hash();
        assert_eq!(a, b);
        assert_eq!(format!("{a:016x}"), "ba6d68e43c69d315");
    }

    #[test]
    fn dryden_gust_hash_bit_stable() {
        let g = canonical_dryden_gust_hash();
        assert_eq!(g, canonical_dryden_gust_hash());
        assert_eq!(format!("{g:016x}"), "2895b75a14f621c5");
    }

    #[test]
    fn fdm_gust_hash_bit_stable() {
        let a = canonical_fdm_gust_hash();
        assert_eq!(a, canonical_fdm_gust_hash());
        assert_ne!(a, canonical_fdm_hash());
        assert_eq!(format!("{a:016x}"), "58f491fb7f3b34a6");
    }
}
