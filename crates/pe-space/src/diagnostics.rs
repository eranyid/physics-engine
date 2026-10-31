//! Conserved-quantity diagnostics for N-body states [SI].

use pe_math::Vec3;

use crate::field::GravityField;
use crate::state::NBodyState;

/// Total linear momentum `Σ m v` [kg m/s].
pub fn linear_momentum(state: &NBodyState) -> Vec3 {
    let mut p = Vec3::zero();
    for i in 0..state.n() {
        p += state.v[i] * state.mass[i];
    }
    p
}

/// Total angular momentum about `origin` [m]: `Σ r × m v` [kg m²/s].
pub fn angular_momentum(state: &NBodyState, origin: Vec3) -> Vec3 {
    let mut l = Vec3::zero();
    for i in 0..state.n() {
        let r = state.x[i] - origin;
        l += r.cross(state.v[i] * state.mass[i]);
    }
    l
}

/// Kinetic energy `Σ ½ m |v|²` [J].
pub fn kinetic_energy(state: &NBodyState) -> f64 {
    let mut ke = 0.0;
    for i in 0..state.n() {
        ke += 0.5 * state.mass[i] * state.v[i].norm_squared();
    }
    ke
}

/// Potential energy from the gravity field [J], or `0` if the field has none.
pub fn potential_energy(field: &dyn GravityField, state: &NBodyState) -> f64 {
    field.potential(state).unwrap_or(0.0)
}

/// Kinetic + potential [J].
pub fn total_energy(field: &dyn GravityField, state: &NBodyState) -> f64 {
    kinetic_energy(state) + potential_energy(field, state)
}

/// FNV-1a 64-bit hash of packed `(x, v)` IEEE bits (cross-OS determinism).
pub fn state_hash64(state: &NBodyState) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for x in state.pack() {
        for b in x.to_bits().to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}
