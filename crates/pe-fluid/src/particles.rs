//! SoA particle state for SPH / PBF.

use pe_math::libm_ops::sqrt;
use pe_math::Vec3;

use crate::Dim;

/// Fluid / boundary particle flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ParticleKind {
    /// Dynamic fluid particle.
    Fluid = 0,
    /// Akinci-style boundary sample (fixed or prescribed motion).
    Boundary = 1,
}

/// Structure-of-arrays particle buffer.
#[derive(Clone, Debug)]
pub struct Particles {
    /// Simulation dimension.
    pub dim: Dim,
    /// Rest density ρ₀ [kg/m³].
    pub rest_density: f64,
    /// Particle mass [kg] (uniform for fluid particles).
    pub mass: f64,
    /// Smoothing length h [m].
    pub h: f64,
    /// Positions [m].
    pub x: Vec<Vec3>,
    /// Velocities [m/s].
    pub v: Vec<Vec3>,
    /// Accelerations [m/s²] (scratch).
    pub a: Vec<Vec3>,
    /// Densities [kg/m³].
    pub rho: Vec<f64>,
    /// Pressures [Pa].
    pub p: Vec<f64>,
    /// Kind flags.
    pub kind: Vec<ParticleKind>,
    /// Akinci boundary volume factor ψ [m³] (0 for fluid).
    pub psi: Vec<f64>,
    /// Outward unit normal for one-sided rigid boundary samples.
    ///
    /// Zero means no half-space filter (tank walls). Non-zero: fluid↔boundary
    /// pairs contribute only when `(x_fluid − x_b) · n_out > 0`.
    pub n_out: Vec<Vec3>,
}

impl Particles {
    /// Empty system.
    pub fn new(dim: Dim, rest_density: f64, mass: f64, h: f64) -> Self {
        Self {
            dim,
            rest_density,
            mass,
            h,
            x: Vec::new(),
            v: Vec::new(),
            a: Vec::new(),
            rho: Vec::new(),
            p: Vec::new(),
            kind: Vec::new(),
            psi: Vec::new(),
            n_out: Vec::new(),
        }
    }

    /// Number of particles (fluid + boundary).
    #[inline]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// True if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Count fluid particles.
    pub fn fluid_count(&self) -> usize {
        self.kind
            .iter()
            .filter(|k| **k == ParticleKind::Fluid)
            .count()
    }

    /// Push a fluid particle.
    pub fn push_fluid(&mut self, x: Vec3, v: Vec3) {
        self.x.push(x);
        self.v.push(v);
        self.a.push(Vec3::zero());
        self.rho.push(self.rest_density);
        self.p.push(0.0);
        self.kind.push(ParticleKind::Fluid);
        self.psi.push(0.0);
        self.n_out.push(Vec3::zero());
    }

    /// Push a fixed boundary particle with Akinci volume `psi` [m³] (2-D: m²).
    pub fn push_boundary(&mut self, x: Vec3, psi: f64) {
        self.push_boundary_oriented(x, psi, Vec3::zero());
    }

    /// Push a boundary sample with an outward normal for one-sided coupling.
    pub fn push_boundary_oriented(&mut self, x: Vec3, psi: f64, n_out: Vec3) {
        self.x.push(x);
        self.v.push(Vec3::zero());
        self.a.push(Vec3::zero());
        self.rho.push(self.rest_density);
        self.p.push(0.0);
        self.kind.push(ParticleKind::Boundary);
        self.psi.push(psi);
        self.n_out.push(n_out);
    }

    /// True if fluid at `x_fluid` may couple to boundary `b`.
    ///
    /// Tank walls (`n_out = 0`) always see fluid. Rigid samples with a
    /// non-zero outward normal reject fluid in the **deep interior**
    /// half-space `(x_f − x_b)·n < −ε` (ε ≈ 0.15 ‖n‖·h), keeping grazing
    /// contacts for impermeability while dropping free-surface hits on dry
    /// faces through the kernel support.
    #[inline]
    pub fn boundary_sees_fluid(&self, b: usize, x_fluid: Vec3) -> bool {
        let n = self.n_out[b];
        let n2 = n.norm_squared();
        if n2 <= 0.0 {
            return true;
        }
        let eps = 0.15 * self.h * sqrt(n2.max(0.0));
        (x_fluid - self.x[b]).dot(n) > -eps
    }

    /// Pack positions as flat `[x0,y0,z0, x1,…]` for hashing / NumPy export.
    pub fn pack_positions(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.len() * 3);
        for p in &self.x {
            out.push(p.x);
            out.push(p.y);
            out.push(p.z);
        }
        out
    }

    /// Pack velocities as flat `[vx,vy,vz,…]`.
    pub fn pack_velocities(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.len() * 3);
        for p in &self.v {
            out.push(p.x);
            out.push(p.y);
            out.push(p.z);
        }
        out
    }

    /// Total fluid mass [kg].
    pub fn total_fluid_mass(&self) -> f64 {
        self.mass * self.fluid_count() as f64
    }
}

/// Estimate particle mass from rest density and lattice spacing `dx` [m].
#[inline]
pub fn mass_from_spacing(rest_density: f64, dx: f64, dim: Dim) -> f64 {
    match dim {
        Dim::D2 => rest_density * dx * dx,
        Dim::D3 => rest_density * dx * dx * dx,
    }
}

/// Calibrate particle mass so a full lattice neighbourhood sums to `rest_density`.
///
/// With coarse `h/dx ≈ 2`, the rectangle-rule sum `Σ V W` is several percent below
/// 1, so `m = ρ₀ dx^d` systematically under-densifies. This helper sets
/// `m = ρ₀ / Σ_j W(|x_j−x_i|, h)` on a reference lattice (self included).
pub fn mass_calibrated_to_rest(
    rest_density: f64,
    dx: f64,
    h: f64,
    dim: Dim,
    kern: crate::kernels::KernelKind,
) -> f64 {
    use crate::kernels::{support_radius, w};
    let support = support_radius(kern, h);
    let n = ((support / dx).ceil() as i32) + 1;
    let mut sum_w = w(kern, 0.0, h, dim);
    for ix in -n..=n {
        for iy in -n..=n {
            let jz_range = match dim {
                Dim::D2 => 0..=0,
                Dim::D3 => -n..=n,
            };
            for iz in jz_range {
                if ix == 0 && iy == 0 && iz == 0 {
                    continue;
                }
                let r = pe_math::libm_ops::sqrt(
                    (ix as f64 * dx) * (ix as f64 * dx)
                        + (iy as f64 * dx) * (iy as f64 * dx)
                        + (iz as f64 * dx) * (iz as f64 * dx),
                );
                if r < support {
                    sum_w += w(kern, r, h, dim);
                }
            }
        }
    }
    if sum_w < 1e-30 {
        mass_from_spacing(rest_density, dx, dim)
    } else {
        rest_density / sum_w
    }
}

/// Fill a rectangular fluid block on a lattice (inclusive-exclusive in index space).
///
/// `origin` is the centre of the first particle; spacing `dx`.
pub fn fill_box(
    parts: &mut Particles,
    origin: Vec3,
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    v0: Vec3,
) {
    let nz = match parts.dim {
        Dim::D2 => 1,
        Dim::D3 => nz,
    };
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let p = Vec3::new(
                    origin.x + ix as f64 * dx,
                    origin.y + iy as f64 * dx,
                    origin.z + iz as f64 * dx,
                );
                parts.push_fluid(p, v0);
            }
        }
    }
}

/// Recompute Akinci boundary volumes `ψ_b = 1 / Σ_{k∈∂} W(|x_b−x_k|, h)`.
///
/// Call after all boundary samples are inserted. The density contribution of a
/// boundary particle is then `ρ₀ ψ_b W`, matching Akinci et al. (2012). The
/// crude `ψ = dx^d` placeholder used at insertion is replaced.
pub fn recomputed_akinci_psi(parts: &mut Particles, kern: crate::kernels::KernelKind) {
    use crate::kernels::{support_radius, w};
    let h = parts.h;
    let dim = parts.dim;
    let support = support_radius(kern, h);
    let n = parts.len();
    // Gather boundary indices (deterministic index order).
    let mut bound = Vec::new();
    for i in 0..n {
        if parts.kind[i] == ParticleKind::Boundary {
            bound.push(i);
        }
    }
    for &bi in &bound {
        let xb = parts.x[bi];
        let mut delta = w(kern, 0.0, h, dim); // self
        for &bj in &bound {
            if bj == bi {
                continue;
            }
            let r = (parts.x[bj] - xb).norm();
            if r < support {
                delta += w(kern, r, h, dim);
            }
        }
        parts.psi[bi] = if delta > 1e-30 { 1.0 / delta } else { 0.0 };
    }
}

/// Sample Akinci boundary particles on an axis-aligned box shell (walls).
///
/// Places particles on the six faces (or four in 2-D) at spacing `dx`, with a
/// provisional `psi = dx^d`. Call [`recomputed_akinci_psi`] afterwards for the
/// paper-accurate volume. Corners/edges may be duplicated slightly.
pub fn fill_box_boundary(parts: &mut Particles, min: Vec3, max: Vec3, dx: f64, layers: usize) {
    let psi = match parts.dim {
        Dim::D2 => dx * dx,
        Dim::D3 => dx * dx * dx,
    };
    let layers = layers.max(1);
    match parts.dim {
        Dim::D2 => {
            // Walls at x=min/max and y=min/max; z fixed at 0.
            let nx = ((max.x - min.x) / dx).round() as usize + 1;
            let ny = ((max.y - min.y) / dx).round() as usize + 1;
            for layer in 0..layers {
                let off = layer as f64 * dx;
                for i in 0..nx {
                    let x = min.x + i as f64 * dx;
                    parts.push_boundary(Vec3::new(x, min.y - off, 0.0), psi);
                    parts.push_boundary(Vec3::new(x, max.y + off, 0.0), psi);
                }
                for j in 1..ny.saturating_sub(1) {
                    let y = min.y + j as f64 * dx;
                    parts.push_boundary(Vec3::new(min.x - off, y, 0.0), psi);
                    parts.push_boundary(Vec3::new(max.x + off, y, 0.0), psi);
                }
            }
        }
        Dim::D3 => {
            let nx = ((max.x - min.x) / dx).round() as usize + 1;
            let ny = ((max.y - min.y) / dx).round() as usize + 1;
            let nz = ((max.z - min.z) / dx).round() as usize + 1;
            for layer in 0..layers {
                let off = layer as f64 * dx;
                // z faces
                for i in 0..nx {
                    for j in 0..ny {
                        let x = min.x + i as f64 * dx;
                        let y = min.y + j as f64 * dx;
                        parts.push_boundary(Vec3::new(x, y, min.z - off), psi);
                        parts.push_boundary(Vec3::new(x, y, max.z + off), psi);
                    }
                }
                // y faces (exclude z edges already covered)
                for i in 0..nx {
                    for k in 1..nz.saturating_sub(1) {
                        let x = min.x + i as f64 * dx;
                        let z = min.z + k as f64 * dx;
                        parts.push_boundary(Vec3::new(x, min.y - off, z), psi);
                        parts.push_boundary(Vec3::new(x, max.y + off, z), psi);
                    }
                }
                // x faces
                for j in 1..ny.saturating_sub(1) {
                    for k in 1..nz.saturating_sub(1) {
                        let y = min.y + j as f64 * dx;
                        let z = min.z + k as f64 * dx;
                        parts.push_boundary(Vec3::new(min.x - off, y, z), psi);
                        parts.push_boundary(Vec3::new(max.x + off, y, z), psi);
                    }
                }
            }
        }
    }
}
