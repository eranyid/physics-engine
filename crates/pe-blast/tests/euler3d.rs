//! Validation of the 3-D Euler blast solver (`euler3d` / `blast3d`).
//!
//! Default tests are sized for debug-mode CI. `#[ignore]` twins run the
//! finer calibration grids: `cargo test --release -p pe-blast -- --ignored`.

use pe_blast::euler::{ExactRiemann, Prim};
use pe_blast::kinney::peak_overpressure;
use pe_blast::rankine_hugoniot::reflected_overpressure;
use pe_blast::sedov::{shock_radius, solve_xi0, SedovGeometry};
use pe_blast::*;
use pe_math::Vec3;

const GAMMA: f64 = 1.4;

// ------------------------------------------------------------------ Sod

/// Sod tube of `n` cells along axis `d` (other axes 1 cell). Returns the
/// density profile along the tube at t = 0.2.
fn sod_along(d: usize, n: usize) -> Vec<f64> {
    let mut dims = [1usize; 3];
    dims[d] = n;
    let dx = 1.0 / n as f64;
    let mut g = Grid3d::new(
        dims[0],
        dims[1],
        dims[2],
        dx,
        Vec3::zero(),
        Prim3::at_rest(1.0, 1.0),
        GAMMA,
    );
    let right = Prim3::at_rest(0.125, 0.1).to_cons(GAMMA);
    for i in n / 2..n {
        let mut ijk = [0usize; 3];
        ijk[d] = i;
        let c = g.idx(ijk[0], ijk[1], ijk[2]);
        g.u[c] = right;
    }
    g.advance_to(0.2, 0.4, 100_000, |_, _| {});
    (0..n)
        .map(|i| {
            let mut ijk = [0usize; 3];
            ijk[d] = i;
            g.u[g.idx(ijk[0], ijk[1], ijk[2])][0]
        })
        .collect()
}

#[test]
fn sod_matches_exact_riemann_on_every_axis() {
    let n = 200;
    let exact = ExactRiemann {
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
        gamma: GAMMA,
    };
    let x_profile = sod_along(0, n);
    let mut l1 = 0.0;
    for (i, rho) in x_profile.iter().enumerate() {
        let x = (i as f64 + 0.5) / n as f64;
        l1 += (rho - exact.sample(x, 0.2, 0.5).rho).abs() / n as f64;
    }
    assert!(l1 < 0.012, "Sod L1 density error {l1}");
    // Same arithmetic along y and z.
    for d in [1, 2] {
        let p = sod_along(d, n);
        for (a, b) in p.iter().zip(&x_profile) {
            assert!((a - b).abs() <= 1e-12 * b.abs(), "axis {d}");
        }
    }
}

// ------------------------------------------------------------------ conservation & symmetry

#[test]
fn closed_box_conserves_mass_and_energy() {
    let mut s = BlastScene::new(Vec3::zero(), Vec3::new(1.6, 1.2, 1.0), 0.1);
    s.grid.boundary = [Boundary::Wall; 6];
    s.fill_aabb(Vec3::new(0.7, 0.0, 0.0), Vec3::new(0.85, 0.7, 1.0)); // baffle
    let (m0, e0) = s.grid.totals();
    s.deposit_energy(Vec3::new(0.35, 0.6, 0.5), 2.0e5, 0.15);
    let (_, e1) = s.grid.totals();
    assert!(((e1 - e0) - 2.0e5).abs() <= 1e-9 * e1, "deposit exact");
    for _ in 0..80 {
        s.step();
    }
    let (m, e) = s.grid.totals();
    assert!((m - m0).abs() <= 1e-12 * m0, "mass drift {}", (m - m0) / m0);
    assert!(
        (e - e1).abs() <= 1e-12 * e1,
        "energy drift {}",
        (e - e1) / e1
    );
}

#[test]
fn centered_blast_stays_symmetric() {
    let n = 15;
    let dx = 0.1;
    let mut s = BlastScene::new(Vec3::zero(), Vec3::new(1.5, 1.5, 1.5), dx);
    s.deposit_energy(Vec3::new(0.75, 0.75, 0.75), 5.0e4, 0.21);
    for _ in 0..40 {
        s.step();
    }
    let g = &s.grid;
    let p = |i, j, k| g.overpressure(g.idx(i, j, k));
    let p_max = (0..g.len())
        .map(|c| g.overpressure(c).abs())
        .fold(0.0, f64::max);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let a = p(i, j, k);
                for b in [p(j, i, k), p(k, j, i), p(i, k, j), p(n - 1 - i, j, k)] {
                    // Relative to the peak, plus round-off of the ambient pressure.
                    // (The deposit radius 0.21 avoids cells exactly on the ball
                    // edge, whose inclusion would depend on rounding.)
                    let tol = 1e-11 * p_max + 1e-11 * P_AIR;
                    assert!((a - b).abs() <= tol, "({i},{j},{k}) {a} vs {b}");
                }
            }
        }
    }
}

// ------------------------------------------------------------------ Sedov–Taylor

/// Octant (symmetry walls at x−, y−, z−) Sedov blast; returns
/// (numerical shock radius, Sedov radius) at the time Sedov predicts `r_target`.
fn sedov_octant(n: usize, size: f64, energy: f64, r_target: f64) -> (f64, f64) {
    let dx = size / n as f64;
    let mut s = BlastScene::new(Vec3::zero(), Vec3::new(size, size, size), dx);
    s.grid.boundary = [
        Boundary::Wall,
        Boundary::Outflow,
        Boundary::Wall,
        Boundary::Outflow,
        Boundary::Wall,
        Boundary::Outflow,
    ];
    s.deposit_energy(Vec3::zero(), energy / 8.0, 0.12 * size);
    let sol = solve_xi0(GAMMA, SedovGeometry::Spherical);
    let t_target = pe_blast::sedov::time_at_radius(&sol, energy, RHO_AIR, r_target);
    s.run_until(t_target, 1_000_000);
    // Captured shock position: radius of the pressure maximum along each
    // axis; average the three axes.
    let r_sedov = shock_radius(&sol, energy, RHO_AIR, s.t());
    let g = &s.grid;
    let mut r_sum = 0.0;
    for d in 0..3 {
        let (mut best, mut r_axis) = (f64::MIN, 0.0);
        for i in 0..n {
            let mut ijk = [0usize; 3];
            ijk[d] = i;
            let p = g.overpressure(g.idx(ijk[0], ijk[1], ijk[2]));
            if p > best {
                best = p;
                r_axis = (i as f64 + 0.5) * dx;
            }
        }
        r_sum += r_axis;
    }
    (r_sum / 3.0, r_sedov)
}

#[test]
fn sedov_taylor_shock_radius_coarse() {
    let (r, r_ref) = sedov_octant(16, 1.28, 2.0e7, 0.95);
    let err = (r - r_ref).abs() / r_ref;
    assert!(
        err < 0.12,
        "Sedov radius {r:.3} vs {r_ref:.3} ({:.1}%)",
        100.0 * err
    );
}

#[test]
#[ignore = "slow: fine-grid Sedov calibration"]
fn sedov_taylor_shock_radius_fine() {
    let (r, r_ref) = sedov_octant(48, 1.28, 2.0e7, 1.0);
    let err = (r - r_ref).abs() / r_ref;
    println!("Sedov fine: {r:.4} vs {r_ref:.4} ({:.2}%)", 100.0 * err);
    assert!(err < 0.05);
}

// ------------------------------------------------------------------ reflection

#[test]
fn normal_reflection_matches_rankine_hugoniot() {
    // Planar shock tube (x only) closed by a wall at x+.
    let n = 400;
    let dx = 4.0 / n as f64;
    let mut g = Grid3d::air(n, 1, 1, dx, Vec3::zero());
    g.boundary[1] = Boundary::Wall;
    let driver = Prim3::at_rest(4.0 * RHO_AIR, 6.0 * P_AIR).to_cons(1.4);
    for i in 0..n / 8 {
        g.u[i] = driver;
    }
    let exact = ExactRiemann {
        left: Prim {
            rho: 4.0 * RHO_AIR,
            u: 0.0,
            p: 6.0 * P_AIR,
        },
        right: Prim {
            rho: RHO_AIR,
            u: 0.0,
            p: P_AIR,
        },
        gamma: 1.4,
    };
    // Incident overpressure behind the right-moving shock (sample just left of it).
    let p_s = exact.sample(0.6, 1e-3, 0.5).p - P_AIR;
    let pr_ref = reflected_overpressure(p_s, P_AIR, 1.4);
    let mut peak = 0.0f64;
    let wall = n - 1;
    g.advance_to(9e-3, 0.4, 1_000_000, |g, _| {
        peak = peak.max(g.overpressure(wall));
    });
    let err = (peak - pr_ref).abs() / pr_ref;
    assert!(
        err < 0.05,
        "reflected {peak:.0} vs RH {pr_ref:.0} ({:.1}%)",
        100.0 * err
    );
    assert!(peak > 2.0 * p_s);
}

// ------------------------------------------------------------------ shielding & venting

#[test]
fn wall_shields_probe_behind_it() {
    let mut s = BlastScene::new(Vec3::zero(), Vec3::new(4.0, 2.0, 2.0), 0.1);
    s.fill_ground(0.0);
    s.fill_aabb(Vec3::new(2.0, 0.0, 0.0), Vec3::new(2.2, 2.0, 1.2));
    s.detonate(Vec3::new(1.5, 1.0, 0.5), 0.05);
    let behind = s.add_probe(Vec3::new(2.6, 1.0, 0.45));
    let open = s.add_probe(Vec3::new(0.4, 1.0, 0.45));
    s.run_until(5e-3, 100_000);
    let (b, o) = (&s.probes[behind], &s.probes[open]);
    assert!(o.peak_overpressure > 0.0 && b.peak_overpressure > 0.0);
    assert!(
        b.peak_overpressure < 0.6 * o.peak_overpressure,
        "behind {:.0} Pa vs open {:.0} Pa",
        b.peak_overpressure,
        o.peak_overpressure
    );
    assert!(b.arrival_time.unwrap() > o.arrival_time.unwrap());
}

/// Mean overpressure in the room `x < 2` after 20 ms, sealed or with a door.
fn room_pressure(door: bool) -> f64 {
    let mut s = BlastScene::new(Vec3::zero(), Vec3::new(3.2, 2.0, 2.0), 0.2);
    s.grid.boundary = [Boundary::Wall; 6];
    s.grid.boundary[1] = Boundary::Outflow;
    s.fill_aabb(Vec3::new(2.0, 0.0, 0.0), Vec3::new(2.2, 2.0, 2.0));
    if door {
        s.clear_aabb(Vec3::new(1.9, 0.6, 0.0), Vec3::new(2.3, 1.4, 1.5));
    }
    s.detonate(Vec3::new(0.9, 1.0, 1.0), 0.05);
    s.run_until(0.02, 100_000);
    let g = &s.grid;
    let (mut sum, mut n) = (0.0, 0);
    for k in 0..g.nz {
        for j in 0..g.ny {
            for i in 0..g.nx {
                let c = g.idx(i, j, k);
                if !g.solid[c] && g.center(i, j, k).x < 2.0 {
                    sum += g.overpressure(c);
                    n += 1;
                }
            }
        }
    }
    sum / n as f64
}

#[test]
fn door_vents_confined_blast() {
    let sealed = room_pressure(false);
    let vented = room_pressure(true);
    // Sealed: quasi-static gas pressure ≈ (γ−1) E / V_room.
    let qs = 0.4 * 0.05 * TNT_ENERGY_J_PER_KG / (2.0 * 2.0 * 2.0);
    assert!(
        (sealed - qs).abs() < 0.25 * qs,
        "sealed {sealed:.0} vs QS {qs:.0}"
    );
    assert!(
        vented < 0.6 * sealed,
        "vented {vented:.0} vs sealed {sealed:.0}"
    );
}

// ------------------------------------------------------------------ calibration vs Kinney

/// Free-air octant blast of `w` kg with the 1-D → 3-D remap at `remap`;
/// returns (Z, simulated peak, Kinney peak) for diagonal gauges at `radii`.
fn free_air_peaks(n: usize, size: f64, w: f64, remap: f64, radii: &[f64]) -> Vec<(f64, f64, f64)> {
    let dx = size / n as f64;
    let mut s = BlastScene::new(Vec3::zero(), Vec3::new(size, size, size), dx);
    s.grid.boundary = [
        Boundary::Wall,
        Boundary::Outflow,
        Boundary::Wall,
        Boundary::Outflow,
        Boundary::Wall,
        Boundary::Outflow,
    ];
    let ids: Vec<_> = radii
        .iter()
        .map(|&r| {
            let a = r / 3f64.sqrt();
            s.add_probe(Vec3::new(a, a, a))
        })
        .collect();
    let rep = s.detonate_with(&Detonation::new(Vec3::zero(), w).remap_at(remap));
    assert!(rep.remapped);
    // Octant holds 1/8 of the charge energy.
    let e8 = w * TNT_ENERGY_J_PER_KG / 8.0;
    assert!((rep.energy_mapped_j - e8).abs() < 0.03 * e8, "{rep:?}");
    let r_max = radii.iter().cloned().fold(0.0, f64::max);
    s.run_until((2.0 * r_max / 340.0).max(s.t() + 1e-5), 1_000_000);
    radii
        .iter()
        .zip(ids)
        .map(|(&r, id)| {
            let z = r / w.cbrt();
            (
                z,
                s.probes[id].peak_overpressure,
                peak_overpressure(z, P_AIR),
            )
        })
        .collect()
}

/// Gauges inside the remap radius record the fine 1-D solution: that path
/// must track Kinney–Graham closely from Z = 1.5 to 8.
#[test]
fn remap_near_field_tracks_kinney() {
    let w = 1e-3;
    let radii = [0.15, 0.2, 0.3, 0.5, 0.75];
    let mut last = f64::INFINITY;
    for (z, sim, kinney) in free_air_peaks(8, 1.0, w, 0.8, &radii) {
        let ratio = sim / kinney;
        assert!(
            (0.75..1.2).contains(&ratio),
            "Z={z:.1}: sim {sim:.0} vs Kinney {kinney:.0}"
        );
        assert!(sim < last);
        last = sim;
    }
}

/// 3-D far field at scaled cell size 0.25 m·kg^{-1/3}: the measured band of
/// `docs/BLAST.md` (shock smearing lowers peaks; see the resolution table).
#[test]
#[ignore = "slow: 3-D far-field calibration (64k cells)"]
fn far_field_peak_band_at_scaled_dx_quarter() {
    for (z, sim, kinney) in free_air_peaks(40, 1.0, 1e-3, 0.12, &[0.3, 0.5, 0.8]) {
        let ratio = sim / kinney;
        println!("Z={z:.1}: ratio {ratio:.2}");
        assert!((0.35..0.8).contains(&ratio), "Z={z:.1}: ratio {ratio:.2}");
    }
}

// ------------------------------------------------------------------ coupling & determinism

#[test]
fn rigid_box_is_pushed_away_from_charge() {
    use pe_core::{RigidBody, World};
    let mut s = BlastScene::new(Vec3::zero(), Vec3::new(3.0, 2.0, 2.0), 0.1);
    // The body occupies voxels, so the wave reflects off its front and
    // diffracts around it.
    s.fill_aabb(Vec3::new(1.8, 0.8, 0.8), Vec3::new(2.2, 1.2, 1.2));
    s.detonate(Vec3::new(1.0, 1.0, 1.0), 0.02);
    let mut world = World::new(
        vec![RigidBody::new(5.0).with_position(Vec3::new(2.0, 1.0, 1.0))],
        vec![],
    );
    let faces = [
        ExposedFace {
            body: 0,
            center: Vec3::new(1.8, 1.0, 1.0),
            normal: Vec3::new(-1.0, 0.0, 0.0),
            area: 0.16,
        },
        ExposedFace {
            body: 0,
            center: Vec3::new(2.2, 1.0, 1.0),
            normal: Vec3::new(1.0, 0.0, 0.0),
            area: 0.16,
        },
    ];
    // Positive phase on the front face (≈ 0.6–1.8 ms): the box is pushed
    // away from the charge. (Over the whole event the front's suction phase
    // and the late diffracted wave on the back can reverse the net impulse —
    // that is physical, so it is not asserted.)
    let (mut peak_front, mut peak_back) = (0.0f64, 0.0f64);
    s.run_until_with(1.8e-3, 100_000, |s, dt| {
        peak_front = peak_front.max(s.face_pressure(&faces[0]));
        peak_back = peak_back.max(s.face_pressure(&faces[1]));
        s.apply_to_world(&mut world, &faces, dt);
    });
    let v = world.state.v[0];
    assert!(v.x > 0.0, "pushed +x, got {v:?}");
    assert!(v.y.abs() < 1e-9 * v.x.abs().max(1.0) && v.z.abs() < 1e-9 * v.x.abs().max(1.0));
    // Reflection on the front, diffraction on the back.
    assert!(
        peak_front > 3.0 * peak_back,
        "front {peak_front:.0} back {peak_back:.0}"
    );
}

#[test]
fn hash_scene_is_reproducible() {
    let mut a = blast3d_corridor_scene();
    let mut b = blast3d_corridor_scene();
    for _ in 0..10 {
        a.step();
        b.step();
    }
    assert_eq!(grid_hash(&a.grid), grid_hash(&b.grid));
    let img = a.slice_ppm(2, 3, 2e5);
    assert!(img.starts_with(b"P6\n32 24\n255\n"));
    assert_eq!(img.len(), "P6\n32 24\n255\n".len() + 32 * 24 * 3);
}
