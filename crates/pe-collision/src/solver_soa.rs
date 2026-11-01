//! Structure-of-arrays contact rows for the PGS velocity solver.
//!
//! Iteration order and floating-point ops match the historical AoS loop so
//! packed state hashes stay bit-identical. SoA improves cache locality when
//! streaming the same field across many contacts.

use pe_math::Vec3;

/// SoA contact constraint rows (one contact per index).
#[derive(Clone, Debug, Default)]
pub struct ContactRowsSoa {
    /// Body / collider index A.
    pub a: Vec<usize>,
    /// Body / collider index B.
    pub b: Vec<usize>,
    /// World contact point [m].
    pub point: Vec<Vec3>,
    /// Unit normal A→B.
    pub normal: Vec3Vec,
    /// Lever arm on A.
    pub ra: Vec3Vec,
    /// Lever arm on B.
    pub rb: Vec3Vec,
    /// Combined friction coefficient.
    pub mu: Vec<f64>,
    /// Normal effective mass.
    pub kn: Vec<f64>,
    /// Tangent-1 effective mass.
    pub k1: Vec<f64>,
    /// Tangent-2 effective mass.
    pub k2: Vec<f64>,
    /// Friction basis t1.
    pub t1: Vec3Vec,
    /// Friction basis t2.
    pub t2: Vec3Vec,
    /// Velocity bias (restitution / speculative).
    pub bias: Vec<f64>,
    /// Accumulated normal impulse.
    pub lam: Vec<f64>,
    /// Accumulated tangent impulses `[jt0, jt1]` packed as pairs.
    pub jt0: Vec<f64>,
    /// Second tangent impulse.
    pub jt1: Vec<f64>,
    /// Warm-start keys `(a, b, feature)`.
    pub key_a: Vec<usize>,
    /// Warm-start key body B.
    pub key_b: Vec<usize>,
    /// Warm-start feature index.
    pub key_f: Vec<u32>,
}

/// Parallel arrays for `Vec3` components (SoA).
#[derive(Clone, Debug, Default)]
pub struct Vec3Vec {
    /// x components.
    pub x: Vec<f64>,
    /// y components.
    pub y: Vec<f64>,
    /// z components.
    pub z: Vec<f64>,
}

impl Vec3Vec {
    #[inline]
    fn clear(&mut self) {
        self.x.clear();
        self.y.clear();
        self.z.clear();
    }

    #[inline]
    fn push(&mut self, v: Vec3) {
        self.x.push(v.x);
        self.y.push(v.y);
        self.z.push(v.z);
    }

    /// Read the `i`-th vector.
    #[inline]
    pub fn get(&self, i: usize) -> Vec3 {
        Vec3::new(self.x[i], self.y[i], self.z[i])
    }

    /// Write the `i`-th vector.
    #[inline]
    pub fn set(&mut self, i: usize, v: Vec3) {
        self.x[i] = v.x;
        self.y[i] = v.y;
        self.z[i] = v.z;
    }

    /// Number of stored vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// True when empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

impl ContactRowsSoa {
    /// Clear without freeing capacity.
    pub fn clear(&mut self) {
        self.a.clear();
        self.b.clear();
        self.point.clear();
        self.normal.clear();
        self.ra.clear();
        self.rb.clear();
        self.mu.clear();
        self.kn.clear();
        self.k1.clear();
        self.k2.clear();
        self.t1.clear();
        self.t2.clear();
        self.bias.clear();
        self.lam.clear();
        self.jt0.clear();
        self.jt1.clear();
        self.key_a.clear();
        self.key_b.clear();
        self.key_f.clear();
    }

    /// Number of contact rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.a.len()
    }

    /// True when empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.a.is_empty()
    }

    /// Reserve capacity for `n` contacts.
    pub fn reserve(&mut self, n: usize) {
        self.a.reserve(n);
        self.b.reserve(n);
        self.point.reserve(n);
        self.normal.x.reserve(n);
        self.normal.y.reserve(n);
        self.normal.z.reserve(n);
        self.ra.x.reserve(n);
        self.ra.y.reserve(n);
        self.ra.z.reserve(n);
        self.rb.x.reserve(n);
        self.rb.y.reserve(n);
        self.rb.z.reserve(n);
        self.mu.reserve(n);
        self.kn.reserve(n);
        self.k1.reserve(n);
        self.k2.reserve(n);
        self.t1.x.reserve(n);
        self.t1.y.reserve(n);
        self.t1.z.reserve(n);
        self.t2.x.reserve(n);
        self.t2.y.reserve(n);
        self.t2.z.reserve(n);
        self.bias.reserve(n);
        self.lam.reserve(n);
        self.jt0.reserve(n);
        self.jt1.reserve(n);
        self.key_a.reserve(n);
        self.key_b.reserve(n);
        self.key_f.reserve(n);
    }

    /// Push one contact row.
    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        a: usize,
        b: usize,
        point: Vec3,
        normal: Vec3,
        ra: Vec3,
        rb: Vec3,
        mu: f64,
        kn: f64,
        k1: f64,
        k2: f64,
        t1: Vec3,
        t2: Vec3,
        bias: f64,
        key: (usize, usize, u32),
    ) {
        self.a.push(a);
        self.b.push(b);
        self.point.push(point);
        self.normal.push(normal);
        self.ra.push(ra);
        self.rb.push(rb);
        self.mu.push(mu);
        self.kn.push(kn);
        self.k1.push(k1);
        self.k2.push(k2);
        self.t1.push(t1);
        self.t2.push(t2);
        self.bias.push(bias);
        self.lam.push(0.0);
        self.jt0.push(0.0);
        self.jt1.push(0.0);
        self.key_a.push(key.0);
        self.key_b.push(key.1);
        self.key_f.push(key.2);
    }
}
