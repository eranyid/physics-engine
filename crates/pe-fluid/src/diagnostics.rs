//! Conservation diagnostics and FNV-1a state hashing.

use pe_math::Vec3;

use crate::particles::{ParticleKind, Particles};

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

#[inline]
fn fnv1a_f64(hash: &mut u64, x: f64) {
    for b in x.to_bits().to_le_bytes() {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

/// FNV-1a 64-bit hash of packed fluid particle `(x, v)` in index order.
pub fn particles_hash64(p: &Particles) -> u64 {
    let mut hash = FNV_OFFSET;
    for i in 0..p.len() {
        if p.kind[i] != ParticleKind::Fluid {
            continue;
        }
        fnv1a_f64(&mut hash, p.x[i].x);
        fnv1a_f64(&mut hash, p.x[i].y);
        fnv1a_f64(&mut hash, p.x[i].z);
        fnv1a_f64(&mut hash, p.v[i].x);
        fnv1a_f64(&mut hash, p.v[i].y);
        fnv1a_f64(&mut hash, p.v[i].z);
    }
    hash
}

/// Total fluid mass [kg].
pub fn total_mass(p: &Particles) -> f64 {
    p.total_fluid_mass()
}

/// Total linear momentum of fluid particles [kg·m/s].
pub fn linear_momentum(p: &Particles) -> Vec3 {
    let mut mom = Vec3::zero();
    for i in 0..p.len() {
        if p.kind[i] == ParticleKind::Fluid {
            mom += p.v[i] * p.mass;
        }
    }
    mom
}

/// Kinetic energy of fluid particles [J].
pub fn kinetic_energy(p: &Particles) -> f64 {
    let mut ke = 0.0;
    for i in 0..p.len() {
        if p.kind[i] == ParticleKind::Fluid {
            ke += 0.5 * p.mass * p.v[i].norm_squared();
        }
    }
    ke
}

/// Gravitational potential about `z = 0` with `g` downward (−z): `m g z` [J].
pub fn gravitational_potential(p: &Particles, g: f64) -> f64 {
    let mut pe = 0.0;
    for i in 0..p.len() {
        if p.kind[i] == ParticleKind::Fluid {
            pe += p.mass * g * p.x[i].z;
        }
    }
    pe
}

/// Mean relative density error `|ρ − ρ₀| / ρ₀` over fluid particles.
pub fn mean_density_error(p: &Particles) -> f64 {
    let mut s = 0.0;
    let mut n = 0usize;
    for i in 0..p.len() {
        if p.kind[i] == ParticleKind::Fluid {
            s += pe_math::libm_ops::fabs(p.rho[i] - p.rest_density) / p.rest_density;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        s / n as f64
    }
}

/// Max relative density error.
pub fn max_density_error(p: &Particles) -> f64 {
    let mut m = 0.0;
    for i in 0..p.len() {
        if p.kind[i] == ParticleKind::Fluid {
            let e = pe_math::libm_ops::fabs(p.rho[i] - p.rest_density) / p.rest_density;
            if e > m {
                m = e;
            }
        }
    }
    m
}
