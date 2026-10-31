//! Conserved-quantity diagnostics (SI).

use pe_math::Vec3;

use crate::world::{State, World};

/// Total linear momentum `Σ m v` [kg m/s].
pub fn linear_momentum(world: &World, s: &State) -> Vec3 {
    let mut p = Vec3::zero();
    for i in 0..world.n {
        p += s.v[i] * world.mass[i];
    }
    p
}

/// Total angular momentum about `origin` [m]: `Σ r × m v + L` [kg m²/s].
pub fn angular_momentum(world: &World, s: &State, origin: Vec3) -> Vec3 {
    let mut total = Vec3::zero();
    for i in 0..world.n {
        let r = s.x[i] - origin;
        total += r.cross(s.v[i] * world.mass[i]) + s.l[i];
    }
    total
}

/// Translational + rotational kinetic energy [J].
pub fn kinetic_energy(world: &World, s: &State) -> f64 {
    let mut ke = 0.0;
    for i in 0..world.n {
        let w = world.omega_i(s, i);
        ke += 0.5 * world.mass[i] * s.v[i].norm_squared() + 0.5 * w.dot(s.l[i]);
    }
    ke
}

/// Sum of generator potentials [J].
pub fn potential_energy(world: &World, s: &State) -> f64 {
    let mut u = 0.0;
    for g in world.generators() {
        if let Some(p) = g.potential(world, s) {
            u += p;
        }
    }
    u
}

/// Kinetic + potential energy [J].
pub fn total_energy(world: &World, s: &State) -> f64 {
    kinetic_energy(world, s) + potential_energy(world, s)
}

/// Angular velocity in the body frame [rad/s] for every body.
pub fn body_angular_velocity(world: &World, s: &State) -> Vec<Vec3> {
    (0..world.n)
        .map(|i| {
            let r = s.q[i].to_matrix();
            let w_world = world.omega_i(s, i);
            // Rᵀ w_world
            Vec3::new(
                r[0][0] * w_world.x + r[1][0] * w_world.y + r[2][0] * w_world.z,
                r[0][1] * w_world.x + r[1][1] * w_world.y + r[2][1] * w_world.z,
                r[0][2] * w_world.x + r[1][2] * w_world.y + r[2][2] * w_world.z,
            )
        })
        .collect()
}

/// FNV-1a 64-bit hash of the packed state (for cross-OS determinism CI).
///
/// Hashes IEEE-754 bit patterns of every `f64` in [`State::pack`], so identical
/// floats yield identical digests across platforms when the step path is portable.
pub fn state_hash64(s: &State) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for x in s.pack() {
        for b in x.to_bits().to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}
