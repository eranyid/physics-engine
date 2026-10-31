//! Convenience builders: rope, cloth, soft cube / jelly, pressure balloon.

use pe_math::libm_ops;
use pe_math::Vec3;

use crate::constraint::{
    tetra_dm_inv, tetra_volume, ConstraintSet, DistanceBendConstraint, DistanceConstraint,
    Multiplier, PressureConstraint, ShapeMatchConstraint, TetraFemConstraint, VolumeConstraint,
};
use crate::material::{lame_lambda, lame_mu, Material};
use crate::particle::Particle;
use crate::solver::{SoftParams, SoftWorld};

/// Build a hanging rope along +x then droop under gravity (particles from `a` to `b`).
pub fn make_rope(
    a: Vec3,
    b: Vec3,
    segments: usize,
    mass_total: f64,
    compliance: f64,
    pin_ends: bool,
) -> SoftWorld {
    let n = segments + 1;
    let mut w = SoftWorld::new();
    w.params = SoftParams {
        substeps: 20,
        iterations: 1,
        ..SoftParams::default()
    };
    let dm = if n > 1 {
        mass_total / (n as f64)
    } else {
        mass_total
    };
    for i in 0..n {
        let t = if n == 1 {
            0.0
        } else {
            i as f64 / (n - 1) as f64
        };
        let x = a + (b - a) * t;
        let mut p = Particle::dynamic(x, dm);
        if pin_ends && (i == 0 || i == n - 1) {
            p = Particle::pinned(x);
        }
        w.add_particle(p);
    }
    let rest = (b - a).norm() / segments.max(1) as f64;
    for i in 0..segments {
        w.constraints.distances.push(DistanceConstraint {
            i,
            j: i + 1,
            rest_length: rest,
            compliance,
            lambda: Multiplier::default(),
            broken: false,
        });
    }
    w
}

/// Regular cloth grid in the x–y plane at height `z`, pinned at listed corner indices.
pub fn make_cloth(
    origin: Vec3,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    mass_total: f64,
    stretch_compliance: f64,
    bend_compliance: f64,
    pin_corners: bool,
) -> SoftWorld {
    let mut w = SoftWorld::new();
    w.params.substeps = 25;
    let n = nx * ny;
    let dm = mass_total / n as f64;
    for j in 0..ny {
        for i in 0..nx {
            let x = origin + Vec3::new(i as f64 * dx, j as f64 * dy, 0.0);
            w.add_particle(Particle::dynamic(x, dm));
        }
    }
    if pin_corners {
        let corners = [0, nx - 1, (ny - 1) * nx, ny * nx - 1];
        for &c in &corners {
            w.pin(c);
        }
    }
    // Structural springs
    for j in 0..ny {
        for i in 0..nx {
            let idx = j * nx + i;
            if i + 1 < nx {
                w.constraints.distances.push(DistanceConstraint {
                    i: idx,
                    j: idx + 1,
                    rest_length: dx,
                    compliance: stretch_compliance,
                    lambda: Multiplier::default(),
                    broken: false,
                });
            }
            if j + 1 < ny {
                w.constraints.distances.push(DistanceConstraint {
                    i: idx,
                    j: idx + nx,
                    rest_length: dy,
                    compliance: stretch_compliance,
                    lambda: Multiplier::default(),
                    broken: false,
                });
            }
            // Distance-based bend (skip one)
            if i + 2 < nx {
                w.constraints.distance_bends.push(DistanceBendConstraint {
                    a: idx,
                    b: idx + 2,
                    rest_length: 2.0 * dx,
                    compliance: bend_compliance,
                    lambda: Multiplier::default(),
                });
            }
            if j + 2 < ny {
                w.constraints.distance_bends.push(DistanceBendConstraint {
                    a: idx,
                    b: idx + 2 * nx,
                    rest_length: 2.0 * dy,
                    compliance: bend_compliance,
                    lambda: Multiplier::default(),
                });
            }
        }
    }
    w
}

/// Axis-aligned soft cube discretized into tetrahedra (5 or 6 tets per cell).
pub fn make_soft_cube(
    origin: Vec3,
    size: f64,
    divisions: usize,
    material: &Material,
    neo_hookean: bool,
) -> SoftWorld {
    let n = divisions + 1;
    let mut w = SoftWorld::new();
    w.params.substeps = 30;
    let h = size / divisions.max(1) as f64;
    let vol_total = size * size * size;
    let mass_total = material.density * vol_total;
    let dm = mass_total / (n * n * n) as f64;

    let idx = |i: usize, j: usize, k: usize| -> usize { (k * n + j) * n + i };
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let x = origin + Vec3::new(i as f64 * h, j as f64 * h, k as f64 * h);
                w.add_particle(Particle::dynamic(x, dm));
            }
        }
    }

    let lambda = lame_lambda(material.youngs_modulus, material.poisson_ratio).unwrap_or(0.0);
    let mu = lame_mu(material.youngs_modulus, material.poisson_ratio).unwrap_or(0.0);

    // 5-tet decomposition per cube cell
    for k in 0..divisions {
        for j in 0..divisions {
            for i in 0..divisions {
                let p000 = idx(i, j, k);
                let p100 = idx(i + 1, j, k);
                let p010 = idx(i, j + 1, k);
                let p110 = idx(i + 1, j + 1, k);
                let p001 = idx(i, j, k + 1);
                let p101 = idx(i + 1, j, k + 1);
                let p011 = idx(i, j + 1, k + 1);
                let p111 = idx(i + 1, j + 1, k + 1);
                let tets = [
                    [p000, p100, p010, p001],
                    [p100, p110, p010, p111],
                    [p100, p010, p001, p111],
                    [p010, p001, p011, p111],
                    [p100, p001, p101, p111],
                ];
                for tet in tets {
                    if let Some((dm_inv, vol)) = tetra_dm_inv(&w.particles, tet) {
                        if vol < 1e-18 {
                            continue;
                        }
                        w.constraints.tetra_fem.push(TetraFemConstraint {
                            indices: tet,
                            dm_inv,
                            rest_volume: vol,
                            lame_lambda: lambda,
                            lame_mu: mu,
                            neo_hookean,
                            lambda_dev: Multiplier::default(),
                            lambda_vol: Multiplier::default(),
                            broken: false,
                        });
                        w.constraints.volumes.push(VolumeConstraint {
                            indices: tet,
                            rest_volume: tetra_volume(
                                w.particles[tet[0]].x,
                                w.particles[tet[1]].x,
                                w.particles[tet[2]].x,
                                w.particles[tet[3]].x,
                            ),
                            compliance: material.volume_compliance(vol).unwrap_or(0.0),
                            lambda: Multiplier::default(),
                        });
                    }
                }
            }
        }
    }
    w
}

/// Axial bar along +x from a structured tet mesh (5 tets per cell).
///
/// - Face at `x = 0` is fully pinned (all DOFs) — lateral Poisson contraction is
///   blocked at the root (Saint-Venant end effect).
/// - Free face at `x = length` receives a distributed axial load in tests.
/// - **No** edge-spring overlay (earlier double-counting with FEM made the bar too stiff).
///
/// `n_x`, `n_y`, `n_z` are cell counts along each axis (≥ 1).
pub fn make_axial_bar_tets(
    length: f64,
    area_side: f64,
    n_seg: usize,
    material: &Material,
) -> SoftWorld {
    make_axial_bar_grid(length, area_side, area_side, n_seg, 1, 1, material)
}

/// Structured rectangular bar with independent cell counts (`n_x`, `n_y`, `n_z`).
pub fn make_axial_bar_grid(
    length: f64,
    width: f64,
    height: f64,
    n_x: usize,
    n_y: usize,
    n_z: usize,
    material: &Material,
) -> SoftWorld {
    let n_x = n_x.max(1);
    let n_y = n_y.max(1);
    let n_z = n_z.max(1);
    let mut w = SoftWorld::new();
    w.params.substeps = 50;
    w.params.iterations = 2;
    w.params.gravity = Vec3::zero();

    let nx = n_x + 1;
    let ny = n_y + 1;
    let nz = n_z + 1;
    let dx = length / n_x as f64;
    let dy = width / n_y as f64;
    let dz = height / n_z as f64;
    let vol = length * width * height;
    let mass = material.density * vol;
    let dm = mass / (nx * ny * nz) as f64;

    let idx = |i: usize, j: usize, k: usize| -> usize { (k * ny + j) * nx + i };
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let x = Vec3::new(i as f64 * dx, j as f64 * dy, k as f64 * dz);
                w.add_particle(Particle::dynamic(x, dm));
            }
        }
    }
    // Pin entire root face (blocks lateral Poisson contraction at x=0).
    for k in 0..nz {
        for j in 0..ny {
            w.pin(idx(0, j, k));
        }
    }

    let lambda = lame_lambda(material.youngs_modulus, material.poisson_ratio).unwrap_or(0.0);
    let mu = lame_mu(material.youngs_modulus, material.poisson_ratio).unwrap_or(0.0);

    for k in 0..n_z {
        for j in 0..n_y {
            for i in 0..n_x {
                let p000 = idx(i, j, k);
                let p100 = idx(i + 1, j, k);
                let p010 = idx(i, j + 1, k);
                let p110 = idx(i + 1, j + 1, k);
                let p001 = idx(i, j, k + 1);
                let p101 = idx(i + 1, j, k + 1);
                let p011 = idx(i, j + 1, k + 1);
                let p111 = idx(i + 1, j + 1, k + 1);
                let tets = [
                    [p000, p100, p010, p001],
                    [p100, p110, p010, p111],
                    [p100, p010, p001, p111],
                    [p010, p001, p011, p111],
                    [p100, p001, p101, p111],
                ];
                for mut tet in tets {
                    let v = tetra_volume(
                        w.particles[tet[0]].x,
                        w.particles[tet[1]].x,
                        w.particles[tet[2]].x,
                        w.particles[tet[3]].x,
                    );
                    if v < 0.0 {
                        tet.swap(1, 2);
                    }
                    if let Some((dm_inv, vol)) = tetra_dm_inv(&w.particles, tet) {
                        if vol < 1e-18 {
                            continue;
                        }
                        w.constraints.tetra_fem.push(TetraFemConstraint {
                            indices: tet,
                            dm_inv,
                            rest_volume: vol,
                            lame_lambda: lambda,
                            lame_mu: mu,
                            neo_hookean: false,
                            lambda_dev: Multiplier::default(),
                            lambda_vol: Multiplier::default(),
                            broken: false,
                        });
                    }
                }
            }
        }
    }
    w
}

/// Uniaxial spring-chain bar with XPBD compliance `α = L₀/(E A)` per segment.
///
/// Exact continuum mapping for pure uniaxial response (no Poisson / end effects).
pub fn make_axial_bar_springs(
    length: f64,
    area: f64,
    n_seg: usize,
    youngs_modulus: f64,
    mass_total: f64,
) -> SoftWorld {
    let n_seg = n_seg.max(1);
    let mut w = SoftWorld::new();
    w.params.substeps = 40;
    w.params.iterations = 2;
    w.params.gravity = Vec3::zero();
    let dx = length / n_seg as f64;
    let n = n_seg + 1;
    let dm = mass_total / n as f64;
    // Axial stiffness of one segment: k = E A / dx ⇒ α = dx / (E A).
    let alpha = dx / (youngs_modulus * area);
    for i in 0..n {
        let p = Particle::dynamic(Vec3::new(i as f64 * dx, 0.0, 0.0), dm);
        w.add_particle(p);
    }
    w.pin(0);
    for i in 0..n_seg {
        w.constraints.distances.push(DistanceConstraint {
            i,
            j: i + 1,
            rest_length: dx,
            compliance: alpha,
            lambda: Multiplier::default(),
            broken: false,
        });
    }
    w
}

/// Cantilever beam as a soft tet strip, pinned on the face at `x = 0`.
///
/// Uses StVK tet FEM (`neo_hookean = false`). Linear tets shear-lock on coarse
/// through-thickness meshes; tip deflection approaches Euler–Bernoulli under
/// isotropic refinement (see `SOFT.md`).
pub fn make_cantilever(
    length: f64,
    height: f64,
    depth: f64,
    nx: usize,
    material: &Material,
) -> SoftWorld {
    let mut w = SoftWorld::new();
    w.params.substeps = 40;
    let ny = 2;
    let nz = 2;
    let dx = length / nx as f64;
    let dy = height / ny as f64;
    let dz = depth / nz as f64;
    let vol = length * height * depth;
    let mass = material.density * vol;
    let n_particles = (nx + 1) * (ny + 1) * (nz + 1);
    let dm = mass / n_particles as f64;
    let idx = |i: usize, j: usize, k: usize| (k * (ny + 1) + j) * (nx + 1) + i;
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                let x = Vec3::new(i as f64 * dx, j as f64 * dy, k as f64 * dz);
                w.add_particle(Particle::dynamic(x, dm));
            }
        }
    }
    for k in 0..=nz {
        for j in 0..=ny {
            w.pin(idx(0, j, k));
        }
    }
    let lambda = lame_lambda(material.youngs_modulus, material.poisson_ratio).unwrap_or(0.0);
    let mu = lame_mu(material.youngs_modulus, material.poisson_ratio).unwrap_or(0.0);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p000 = idx(i, j, k);
                let p100 = idx(i + 1, j, k);
                let p010 = idx(i, j + 1, k);
                let p110 = idx(i + 1, j + 1, k);
                let p001 = idx(i, j, k + 1);
                let p101 = idx(i + 1, j, k + 1);
                let p011 = idx(i, j + 1, k + 1);
                let p111 = idx(i + 1, j + 1, k + 1);
                let tets = [
                    [p000, p100, p010, p001],
                    [p100, p110, p010, p111],
                    [p100, p010, p001, p111],
                    [p010, p001, p011, p111],
                    [p100, p001, p101, p111],
                ];
                for tet in tets {
                    if let Some((dm_inv, vol)) = tetra_dm_inv(&w.particles, tet) {
                        if vol < 1e-18 {
                            continue;
                        }
                        w.constraints.tetra_fem.push(TetraFemConstraint {
                            indices: tet,
                            dm_inv,
                            rest_volume: vol,
                            lame_lambda: lambda,
                            lame_mu: mu,
                            neo_hookean: false,
                            lambda_dev: Multiplier::default(),
                            lambda_vol: Multiplier::default(),
                            broken: false,
                        });
                    }
                }
            }
        }
    }
    w
}

/// Spherical balloon: icosahedral-ish latitude/longitude mesh with pressure constraint.
pub fn make_balloon(
    center: Vec3,
    radius: f64,
    n_lat: usize,
    n_lon: usize,
    mass_total: f64,
    stretch_compliance: f64,
    pressure_compliance: f64,
    overpressure_volume_scale: f64,
) -> SoftWorld {
    let mut w = SoftWorld::new();
    w.params.substeps = 25;
    let n_lat = n_lat.max(2);
    let n_lon = n_lon.max(3);
    // poles + rings
    let north = w.add_particle(Particle::dynamic(
        center + Vec3::new(0.0, 0.0, radius),
        mass_total / ((n_lat - 1) * n_lon + 2) as f64,
    ));
    let mut rings: Vec<Vec<usize>> = Vec::new();
    let dm = mass_total / ((n_lat - 1) * n_lon + 2) as f64;
    for i in 1..n_lat {
        let theta = core::f64::consts::PI * i as f64 / n_lat as f64;
        let mut ring = Vec::new();
        for j in 0..n_lon {
            let phi = 2.0 * core::f64::consts::PI * j as f64 / n_lon as f64;
            let x = center
                + Vec3::new(
                    radius * libm_ops::sin(theta) * libm_ops::cos(phi),
                    radius * libm_ops::sin(theta) * libm_ops::sin(phi),
                    radius * libm_ops::cos(theta),
                );
            ring.push(w.add_particle(Particle::dynamic(x, dm)));
        }
        rings.push(ring);
    }
    let south = w.add_particle(Particle::dynamic(center + Vec3::new(0.0, 0.0, -radius), dm));

    let mut tris: Vec<[usize; 3]> = Vec::new();
    // north cap (outward = away from centre ⇒ CCW when viewed from outside)
    for j in 0..n_lon {
        let j2 = (j + 1) % n_lon;
        tris.push([north, rings[0][j2], rings[0][j]]);
        w.constraints.distances.push(DistanceConstraint {
            i: north,
            j: rings[0][j],
            rest_length: (w.particles[north].x - w.particles[rings[0][j]].x).norm(),
            compliance: stretch_compliance,
            lambda: Multiplier::default(),
            broken: false,
        });
    }
    // rings
    for r in 0..rings.len() {
        for j in 0..n_lon {
            let j2 = (j + 1) % n_lon;
            let a = rings[r][j];
            let b = rings[r][j2];
            w.constraints.distances.push(DistanceConstraint {
                i: a,
                j: b,
                rest_length: (w.particles[a].x - w.particles[b].x).norm(),
                compliance: stretch_compliance,
                lambda: Multiplier::default(),
                broken: false,
            });
            if r + 1 < rings.len() {
                let c = rings[r + 1][j];
                let d = rings[r + 1][j2];
                // Outward winding for lat-long strip
                tris.push([a, b, d]);
                tris.push([a, d, c]);
                w.constraints.distances.push(DistanceConstraint {
                    i: a,
                    j: c,
                    rest_length: (w.particles[a].x - w.particles[c].x).norm(),
                    compliance: stretch_compliance,
                    lambda: Multiplier::default(),
                    broken: false,
                });
            }
        }
    }
    // south cap
    let last = rings.len() - 1;
    for j in 0..n_lon {
        let j2 = (j + 1) % n_lon;
        tris.push([south, rings[last][j], rings[last][j2]]);
        w.constraints.distances.push(DistanceConstraint {
            i: south,
            j: rings[last][j],
            rest_length: (w.particles[south].x - w.particles[rings[last][j]].x).norm(),
            compliance: stretch_compliance,
            lambda: Multiplier::default(),
            broken: false,
        });
    }

    let mut v0 = crate::constraint::mesh_volume(&w.particles, &tris);
    if v0 < 0.0 {
        // Flip all triangles if winding was inward.
        for t in &mut tris {
            t.swap(1, 2);
        }
        v0 = -v0;
    }
    w.constraints.pressures.push(PressureConstraint {
        triangles: tris,
        target_volume: v0 * overpressure_volume_scale.max(1.0),
        compliance: pressure_compliance,
        lambda: Multiplier::default(),
    });
    w
}

/// Attach a shape-matching cluster over all dynamic particles (optional rigidification).
pub fn add_shape_match_all(world: &mut SoftWorld, compliance: f64) {
    let indices: Vec<usize> = (0..world.particles.len()).collect();
    let mut com = Vec3::zero();
    let mut msum = 0.0;
    for &i in &indices {
        let m = if world.particles[i].inv_mass == 0.0 {
            1.0
        } else {
            1.0 / world.particles[i].inv_mass
        };
        msum += m;
        com += world.particles[i].x * m;
    }
    com *= 1.0 / msum.max(1e-30);
    let rest_offsets: Vec<Vec3> = indices
        .iter()
        .map(|&i| world.particles[i].x - com)
        .collect();
    world.constraints.shape_matches.push(ShapeMatchConstraint {
        indices,
        rest_offsets,
        rest_com: com,
        compliance,
        lambda: Multiplier::default(),
    });
}

/// Empty constraint set helper.
pub fn empty_constraints() -> ConstraintSet {
    ConstraintSet::default()
}
