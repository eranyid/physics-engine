//! Surface tension for SPH — Akinci et al. (2013) cohesion + curvature.
//!
//! Clean-room from published formulas. Optional Becker–Teschner (2007) colour-
//! field style is available via [`SurfaceTensionModel::ColourField`].
//!
//! # Laplace pressure check
//!
//! For a liquid droplet of radius `R` and surface tension coefficient `σ`:
//! - 3-D: `Δp = 2 σ / R`
//! - 2-D: `Δp = σ / R`
//!
//! # References
//! - Akinci, N., Akinci, G. & Teschner, M. (2013). Versatile surface tension
//!   and adhesion for SPH fluids. *ACM TOG* **32**.
//! - Becker, M. & Teschner, M. (2007). Weakly compressible SPH. *SCA*.

use pe_math::libm_ops::{fabs, pow};
use pe_math::Vec3;

use crate::kernels::grad_w;
use crate::kernels::KernelKind;
use crate::particles::{ParticleKind, Particles};
use crate::Dim;

/// Which surface-tension model to apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceTensionModel {
    /// Disabled.
    None,
    /// Akinci (2013) cohesion + curvature.
    Akinci2013,
    /// Becker–Teschner (2007) colour-field / CSF-style force.
    ColourField,
}

/// Surface tension parameters.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceTensionParams {
    /// Model selection.
    pub model: SurfaceTensionModel,
    /// Surface tension coefficient `σ` [N/m] = [kg/s²].
    pub sigma: f64,
    /// Colour-field CSF multiplier (Becker–Teschner path); typically = `sigma`.
    pub colour_gamma: f64,
}

impl SurfaceTensionParams {
    /// Disabled surface tension.
    pub fn disabled() -> Self {
        Self {
            model: SurfaceTensionModel::None,
            sigma: 0.0,
            colour_gamma: 0.0,
        }
    }

    /// Akinci (2013) with coefficient `sigma`.
    pub fn akinci(sigma: f64) -> Self {
        Self {
            model: SurfaceTensionModel::Akinci2013,
            sigma,
            colour_gamma: sigma,
        }
    }

    /// Becker–Teschner colour-field CSF with coefficient `sigma`.
    pub fn colour_field(sigma: f64) -> Self {
        Self {
            model: SurfaceTensionModel::ColourField,
            sigma,
            colour_gamma: sigma,
        }
    }
}

/// Akinci cohesion spline `C(r)` (support `h`, peak near `h/2`).
///
/// Piecewise cubic matching Akinci 2013 Fig. 1 / Eq. (2) form used in graphics
/// SPH practice: positive for `r ∈ (0, h)`, zero at `r = 0` and `r ≥ h`.
pub fn cohesion_c(r: f64, h: f64) -> f64 {
    if r <= 0.0 || r >= h || h <= 0.0 {
        return 0.0;
    }
    let q = r / h;
    // Normalised so ∫ is O(1); exact paper constant absorbed into σ.
    // C(r) ∝ (1−q)³ q³ for a smooth compact bump.
    let t = 1.0 - q;
    32.0 / (std::f64::consts::PI * pow(h, 3.0)) * t * t * t * q * q * q
}

/// Cohesion kernel in 2-D (area measure).
pub fn cohesion_c_2d(r: f64, h: f64) -> f64 {
    if r <= 0.0 || r >= h || h <= 0.0 {
        return 0.0;
    }
    let q = r / h;
    let t = 1.0 - q;
    16.0 / (std::f64::consts::PI * h * h) * t * t * t * q * q * q
}

fn cohesion_kernel(r: f64, h: f64, dim: Dim) -> f64 {
    match dim {
        Dim::D2 => cohesion_c_2d(r, h),
        Dim::D3 => cohesion_c(r, h),
    }
}

/// Compute per-particle surface-tension **acceleration** into `out_a`.
///
/// Also fills `normals` with colour-field normals
/// `n_i = h Σ_j (m_j/ρ_j) ∇_i W` (Akinci / Müller colour field).
pub fn surface_tension_accel(
    parts: &Particles,
    neigh: &[Vec<usize>],
    kern: KernelKind,
    params: &SurfaceTensionParams,
    normals: &mut [Vec3],
    out_a: &mut [Vec3],
) {
    let n = parts.len();
    for i in 0..n {
        normals[i] = Vec3::zero();
        out_a[i] = Vec3::zero();
    }
    if params.model == SurfaceTensionModel::None || params.sigma == 0.0 {
        return;
    }
    let h = parts.h;
    let dim = parts.dim;
    let mass = parts.mass;

    // Colour-field normals.
    for i in 0..n {
        if parts.kind[i] != ParticleKind::Fluid {
            continue;
        }
        let xi = parts.x[i];
        let mut ni = Vec3::zero();
        for &j in &neigh[i] {
            if j == i || parts.kind[j] != ParticleKind::Fluid {
                continue;
            }
            let rij = parts.x[j] - xi;
            let gw = grad_w(kern, rij, h, dim);
            ni += gw * (mass / parts.rho[j].max(1e-12));
        }
        normals[i] = ni * h;
    }

    match params.model {
        SurfaceTensionModel::None => {}
        SurfaceTensionModel::Akinci2013 => {
            let sigma = params.sigma;
            for i in 0..n {
                if parts.kind[i] != ParticleKind::Fluid {
                    continue;
                }
                let xi = parts.x[i];
                let mut f = Vec3::zero();
                for &j in &neigh[i] {
                    if j == i || parts.kind[j] != ParticleKind::Fluid {
                        continue;
                    }
                    let rij = parts.x[j] - xi; // xj − xi
                    let r = rij.norm();
                    if r < 1e-12 {
                        continue;
                    }
                    let r_hat = rij / r;
                    // Cohesion: F_i = −σ m_i m_j C(r) r̂  (r̂ from i to j)
                    // Akinci uses (xi−xj)/r = −r_hat; sign: attractive when C>0.
                    let c = cohesion_kernel(r, h, dim);
                    f += r_hat * (-sigma * mass * mass * c);
                    // Curvature: F_i = −σ m_i (n_i − n_j)
                    f += (normals[i] - normals[j]) * (-sigma * mass);
                }
                out_a[i] = f / mass;
            }
        }
        SurfaceTensionModel::ColourField => {
            // Becker–Teschner / CSF: a_i ≈ −(γ/ρ) ∇²c n̂, with c colour field.
            // Discrete: use curvature κ ≈ −∇·n̂ ≈ −Σ (m/ρ) (n_j − n_i)·∇W / |n|
            // Force direction along n: f = −γ κ n̂.
            let gamma = params.colour_gamma;
            for i in 0..n {
                if parts.kind[i] != ParticleKind::Fluid {
                    continue;
                }
                let ni = normals[i];
                let n_len = ni.norm();
                if n_len < 1e-8 {
                    continue;
                }
                let n_hat = ni / n_len;
                let xi = parts.x[i];
                let mut div_n = 0.0;
                for &j in &neigh[i] {
                    if j == i || parts.kind[j] != ParticleKind::Fluid {
                        continue;
                    }
                    let rij = parts.x[j] - xi;
                    let gw = grad_w(kern, rij, h, dim);
                    let nj = normals[j];
                    let nj_len = nj.norm();
                    let nj_hat = if nj_len > 1e-8 {
                        nj / nj_len
                    } else {
                        Vec3::zero()
                    };
                    div_n += (mass / parts.rho[j].max(1e-12)) * (nj_hat - n_hat).dot(gw);
                }
                // κ ≈ −∇·n̂; acceleration −(γ/ρ) κ n̂
                let kappa = -div_n;
                out_a[i] = n_hat * (-gamma * kappa / parts.rho[i].max(1e-12));
            }
        }
    }
}

/// Ideal Laplace pressure jump [Pa].
#[inline]
pub fn laplace_pressure(sigma: f64, radius: f64, dim: Dim) -> f64 {
    if radius <= 0.0 {
        return 0.0;
    }
    match dim {
        Dim::D2 => sigma / radius,
        Dim::D3 => 2.0 * sigma / radius,
    }
}

/// Measure mean pressure of fluid particles near the droplet centre vs near
/// the free surface, returning `(p_interior_mean, p_surface_mean, delta_p)`.
pub fn droplet_pressure_jump(
    parts: &Particles,
    center: Vec3,
    radius: f64,
    interior_frac: f64,
    surface_band: f64,
) -> (f64, f64, f64) {
    let mut p_in = 0.0;
    let mut n_in = 0usize;
    let mut p_surf = 0.0;
    let mut n_surf = 0usize;
    let r_in = interior_frac * radius;
    for i in 0..parts.len() {
        if parts.kind[i] != ParticleKind::Fluid {
            continue;
        }
        let d = (parts.x[i] - center).norm();
        if d < r_in {
            p_in += parts.p[i];
            n_in += 1;
        } else if d > radius - surface_band && d < radius + surface_band {
            p_surf += parts.p[i];
            n_surf += 1;
        }
    }
    let pin = if n_in == 0 { 0.0 } else { p_in / n_in as f64 };
    let ps = if n_surf == 0 {
        0.0
    } else {
        p_surf / n_surf as f64
    };
    (pin, ps, pin - ps)
}

/// Sphericity error: RMS radial deviation of near-surface particles from `R`,
/// normalised by `R`. Returns `(rms_rel, n_surface)`.
pub fn droplet_sphericity_error(
    parts: &Particles,
    center: Vec3,
    radius: f64,
    surface_band: f64,
) -> (f64, usize) {
    let mut sum = 0.0;
    let mut n = 0usize;
    for i in 0..parts.len() {
        if parts.kind[i] != ParticleKind::Fluid {
            continue;
        }
        let d = (parts.x[i] - center).norm();
        if fabs(d - radius) < surface_band {
            let e = (d - radius) / radius;
            sum += e * e;
            n += 1;
        }
    }
    if n == 0 {
        (1.0, 0)
    } else {
        (pe_math::libm_ops::sqrt(sum / n as f64), n)
    }
}

/// Seed a 2-D circular droplet (fluid only; no walls) on a lattice.
pub fn fill_droplet_2d(parts: &mut Particles, center: Vec3, radius: f64, dx: f64) -> usize {
    let n_side = ((2.0 * radius / dx).ceil() as i32) + 2;
    let mut count = 0usize;
    for iy in -n_side..=n_side {
        for ix in -n_side..=n_side {
            let p = Vec3::new(center.x + ix as f64 * dx, center.y + iy as f64 * dx, 0.0);
            if (p - center).norm() <= radius + 1e-12 {
                parts.push_fluid(p, Vec3::zero());
                count += 1;
            }
        }
    }
    count
}

/// Estimate droplet centre of mass and equivalent radius from fluid particles.
pub fn droplet_com_radius(parts: &Particles) -> (Vec3, f64) {
    let mut com = Vec3::zero();
    let mut n = 0usize;
    for i in 0..parts.len() {
        if parts.kind[i] == ParticleKind::Fluid {
            com += parts.x[i];
            n += 1;
        }
    }
    if n == 0 {
        return (Vec3::zero(), 0.0);
    }
    com *= 1.0 / n as f64;
    // Equivalent radius from area (2-D) or volume (3-D) assuming rest spacing.
    let area_or_vol = match parts.dim {
        Dim::D2 => parts.mass / parts.rest_density, // per particle area
        Dim::D3 => parts.mass / parts.rest_density,
    };
    let r = match parts.dim {
        Dim::D2 => pe_math::libm_ops::sqrt(n as f64 * area_or_vol / std::f64::consts::PI),
        Dim::D3 => {
            let v = n as f64 * area_or_vol;
            pe_math::libm_ops::cbrt(3.0 * v / (4.0 * std::f64::consts::PI))
        }
    };
    (com, r)
}
