//! Soft-body energy diagnostics (kinetic + constraint potential estimates).

use pe_math::STANDARD_GRAVITY;

use crate::constraint::tetra_volume;
use crate::solver::SoftWorld;

/// Kinetic energy Σ ½ m |v|² [J].
pub fn kinetic_energy(world: &SoftWorld) -> f64 {
    let mut ke = 0.0;
    for p in &world.particles {
        if p.inv_mass == 0.0 {
            continue;
        }
        let m = 1.0 / p.inv_mass;
        ke += 0.5 * m * p.v.norm_squared();
    }
    ke
}

/// Gravitational potential with reference z=0 and g = |params.gravity| or STANDARD_GRAVITY.
pub fn gravitational_potential(world: &SoftWorld) -> f64 {
    let g = world.params.gravity.norm();
    let g = if g > 0.0 { g } else { STANDARD_GRAVITY };
    // Assume gravity along −z when using default; use −g · x for general.
    let mut pe = 0.0;
    for p in &world.particles {
        if p.inv_mass == 0.0 {
            continue;
        }
        let m = 1.0 / p.inv_mass;
        pe += -m * world.params.gravity.dot(p.x);
        let _ = g;
    }
    pe
}

/// Distance-spring potential Σ ½ k (ΔL)² with k = 1/α (α=0 skipped as rigid).
pub fn stretch_potential(world: &SoftWorld) -> f64 {
    let mut pe = 0.0;
    for c in &world.constraints.distances {
        if c.compliance <= 0.0 {
            continue;
        }
        let k = 1.0 / c.compliance;
        let len = (world.particles[c.i].x - world.particles[c.j].x).norm();
        let dl = len - c.rest_length;
        pe += 0.5 * k * dl * dl;
    }
    for c in &world.constraints.distance_bends {
        if c.compliance <= 0.0 {
            continue;
        }
        let k = 1.0 / c.compliance;
        let len = (world.particles[c.a].x - world.particles[c.b].x).norm();
        let dl = len - c.rest_length;
        pe += 0.5 * k * dl * dl;
    }
    pe
}

/// Total mechanical energy estimate (KE + gravity PE + stretch PE).
pub fn total_energy(world: &SoftWorld) -> f64 {
    kinetic_energy(world) + gravitational_potential(world) + stretch_potential(world)
}

/// Sum of absolute tetra volume errors |V − V₀| over volume constraints [m³].
pub fn volume_error_sum(world: &SoftWorld) -> f64 {
    let mut e = 0.0;
    for c in &world.constraints.volumes {
        let i = c.indices;
        let v = tetra_volume(
            world.particles[i[0]].x,
            world.particles[i[1]].x,
            world.particles[i[2]].x,
            world.particles[i[3]].x,
        );
        e += (v - c.rest_volume).abs();
    }
    e
}
