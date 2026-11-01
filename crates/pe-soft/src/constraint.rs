//! XPBD constraints: distance, bending, volume, tetra FEM, shape matching, pressure.
//!
//! Clean-room from Macklin, Müller, Chentanez (2016) XPBD; Müller et al. (2007) PBD;
//! Macklin (2019) Small Steps; Macklin & Müller (2021) stable neo-Hookean XPBD.
//! Formulas only — no third-party source translated.

use pe_math::libm_ops;
use pe_math::Vec3;

use crate::particle::Particle;

/// Scalar XPBD multiplier accumulator (reset each substep).
#[derive(Clone, Copy, Debug, Default)]
pub struct Multiplier {
    /// Lagrange multiplier λ.
    pub lambda: f64,
}

/// Two-particle distance constraint C = |x_i − x_j| − L₀.
#[derive(Clone, Debug)]
pub struct DistanceConstraint {
    /// Particle i.
    pub i: usize,
    /// Particle j.
    pub j: usize,
    /// Rest length L₀ [m].
    pub rest_length: f64,
    /// Compliance α [m/N] (= 1/k for a linear spring).
    pub compliance: f64,
    /// Multiplier state.
    pub lambda: Multiplier,
    /// `true` after fracture has deactivated this constraint.
    pub broken: bool,
}

/// Distance-based bending: constrain particles across an edge to a rest distance
/// (cheap cloth bend surrogate).
#[derive(Clone, Debug)]
pub struct DistanceBendConstraint {
    /// Particle a (opposite to shared edge).
    pub a: usize,
    /// Particle b (opposite to shared edge).
    pub b: usize,
    /// Rest distance [m].
    pub rest_length: f64,
    /// Compliance α [m/N].
    pub compliance: f64,
    /// Multiplier state.
    pub lambda: Multiplier,
}

/// Dihedral bending: C = arccos(n₁·n₂) − θ₀ for two triangles sharing an edge.
#[derive(Clone, Debug)]
pub struct DihedralBendConstraint {
    /// Triangle 1 opposite vertex.
    pub i0: usize,
    /// Shared edge vertex 0.
    pub i1: usize,
    /// Shared edge vertex 1.
    pub i2: usize,
    /// Triangle 2 opposite vertex.
    pub i3: usize,
    /// Rest dihedral angle [rad].
    pub rest_angle: f64,
    /// Compliance α [1/N] (angle constraint; units consistent with C in rad).
    pub compliance: f64,
    /// Multiplier state.
    pub lambda: Multiplier,
}

/// Tetrahedron volume constraint C = 6(V − V₀).
#[derive(Clone, Debug)]
pub struct VolumeConstraint {
    /// Four corner particle indices.
    pub indices: [usize; 4],
    /// Rest volume V₀ [m³] (signed).
    pub rest_volume: f64,
    /// Compliance α [1/N] for C in m³ · 6.
    pub compliance: f64,
    /// Multiplier state.
    pub lambda: Multiplier,
}

/// Stable neo-Hookean / StVK-style tetrahedral FEM constraint pair
/// (deviatoric + volumetric) after Macklin & Müller (2021).
#[derive(Clone, Debug)]
pub struct TetraFemConstraint {
    /// Four corner particle indices (order defines rest orientation).
    pub indices: [usize; 4],
    /// Rest-shape inverse matrix Dₘ⁻¹ (columns = X₁−X₀, X₂−X₀, X₃−X₀).
    pub dm_inv: [[f64; 3]; 3],
    /// Rest volume V₀ [m³] (unsigned absolute).
    pub rest_volume: f64,
    /// First Lamé λ [Pa].
    pub lame_lambda: f64,
    /// Second Lamé μ [Pa].
    pub lame_mu: f64,
    /// Use stable neo-Hookean (true) or linear StVK-like stretch (false).
    pub neo_hookean: bool,
    /// Deviatoric multiplier.
    pub lambda_dev: Multiplier,
    /// Volumetric multiplier.
    pub lambda_vol: Multiplier,
    /// `true` after fracture has deactivated this element.
    pub broken: bool,
}

/// Shape-matching cluster (Müller et al. 2005 / PBD shape matching).
#[derive(Clone, Debug)]
pub struct ShapeMatchConstraint {
    /// Particle indices in the cluster.
    pub indices: Vec<usize>,
    /// Rest offsets from rest COM [m].
    pub rest_offsets: Vec<Vec3>,
    /// Rest COM [m] (world at rest pose construction).
    pub rest_com: Vec3,
    /// Compliance α (0 = rigid shape match).
    pub compliance: f64,
    /// Multiplier unused (shape match uses direct projection); kept for API symmetry.
    pub lambda: Multiplier,
}

/// Global volume / pressure constraint over a closed surface of triangles
/// C = V − V_target (balloon / soft pressure).
#[derive(Clone, Debug)]
pub struct PressureConstraint {
    /// Triangle indices (each `[i,j,k]` oriented outward).
    pub triangles: Vec<[usize; 3]>,
    /// Target volume [m³].
    pub target_volume: f64,
    /// Compliance α.
    pub compliance: f64,
    /// Multiplier state.
    pub lambda: Multiplier,
}

/// All constraint kinds owned by a soft world (fixed `Vec` order — no `HashMap`).
#[derive(Clone, Debug, Default)]
pub struct ConstraintSet {
    /// Distance / rope / spring.
    pub distances: Vec<DistanceConstraint>,
    /// Distance-based bend.
    pub distance_bends: Vec<DistanceBendConstraint>,
    /// Dihedral bend.
    pub dihedrals: Vec<DihedralBendConstraint>,
    /// Tetra volume.
    pub volumes: Vec<VolumeConstraint>,
    /// Tetra FEM (StVK / neo-Hookean).
    pub tetra_fem: Vec<TetraFemConstraint>,
    /// Optional shape matching.
    pub shape_matches: Vec<ShapeMatchConstraint>,
    /// Balloon / pressure.
    pub pressures: Vec<PressureConstraint>,
}

impl ConstraintSet {
    /// Reset all Lagrange multipliers (call at the start of each substep).
    pub fn reset_lambdas(&mut self) {
        for c in &mut self.distances {
            c.lambda.lambda = 0.0;
        }
        for c in &mut self.distance_bends {
            c.lambda.lambda = 0.0;
        }
        for c in &mut self.dihedrals {
            c.lambda.lambda = 0.0;
        }
        for c in &mut self.volumes {
            c.lambda.lambda = 0.0;
        }
        for c in &mut self.tetra_fem {
            c.lambda_dev.lambda = 0.0;
            c.lambda_vol.lambda = 0.0;
        }
        for c in &mut self.shape_matches {
            c.lambda.lambda = 0.0;
        }
        for c in &mut self.pressures {
            c.lambda.lambda = 0.0;
        }
    }
}

/// Signed tetrahedron volume V = (1/6) det([b−a, c−a, d−a]).
#[inline]
pub fn tetra_volume(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    ab.dot(ac.cross(ad)) / 6.0
}

/// 3×3 determinant.
#[inline]
fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Invert 3×3; returns `None` if singular.
pub fn invert3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let d = det3(m);
    if d.abs() < 1e-30 {
        return None;
    }
    let inv_d = 1.0 / d;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_d,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_d,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_d,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_d,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_d,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_d,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_d,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_d,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_d,
        ],
    ])
}

fn mat_vec(m: &[[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

pub(crate) fn mul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Public 3×3 multiply (fracture / materials helpers).
pub fn mul3_public(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    mul3(a, b)
}

pub(crate) fn transpose3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Public transpose.
pub fn transpose3_public(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    transpose3(m)
}

/// Build rest-shape inverse for a tetrahedron from current particle positions.
pub fn tetra_dm_inv(particles: &[Particle], idx: [usize; 4]) -> Option<([[f64; 3]; 3], f64)> {
    let x0 = particles[idx[0]].x;
    let x1 = particles[idx[1]].x;
    let x2 = particles[idx[2]].x;
    let x3 = particles[idx[3]].x;
    let col0 = x1 - x0;
    let col1 = x2 - x0;
    let col2 = x3 - x0;
    // Columns of Dm
    let dm = [
        [col0.x, col1.x, col2.x],
        [col0.y, col1.y, col2.y],
        [col0.z, col1.z, col2.z],
    ];
    let vol = tetra_volume(x0, x1, x2, x3).abs();
    let inv = invert3(&dm)?;
    Some((inv, vol))
}

/// Apply a generic XPBD scalar constraint with per-particle gradients.
///
/// `grads[k]` corresponds to `indices[k]`. Returns the constraint value C.
pub fn solve_equality(
    particles: &mut [Particle],
    indices: &[usize],
    grads: &[Vec3],
    c_value: f64,
    compliance: f64,
    lambda: &mut Multiplier,
    dt: f64,
) -> f64 {
    debug_assert_eq!(indices.len(), grads.len());
    let alpha_tilde = if dt > 0.0 {
        compliance / (dt * dt)
    } else {
        0.0
    };
    let mut w_sum = 0.0;
    for (k, &i) in indices.iter().enumerate() {
        w_sum += particles[i].inv_mass * grads[k].norm_squared();
    }
    let denom = w_sum + alpha_tilde;
    if denom < 1e-30 {
        return c_value;
    }
    let dl = (-c_value - alpha_tilde * lambda.lambda) / denom;
    lambda.lambda += dl;
    for (k, &i) in indices.iter().enumerate() {
        let w = particles[i].inv_mass;
        if w != 0.0 {
            particles[i].x += grads[k] * (w * dl);
        }
    }
    c_value
}

/// Project a distance constraint once.
pub fn project_distance(particles: &mut [Particle], c: &mut DistanceConstraint, dt: f64) {
    if c.broken {
        return;
    }
    let i = c.i;
    let j = c.j;
    let pi = particles[i].x;
    let pj = particles[j].x;
    let d = pi - pj;
    let len = d.norm();
    if len < 1e-15 {
        return;
    }
    let n = d * (1.0 / len);
    let c_val = len - c.rest_length;
    let grads = [n, n * (-1.0)];
    let idxs = [i, j];
    solve_equality(
        particles,
        &idxs,
        &grads,
        c_val,
        c.compliance,
        &mut c.lambda,
        dt,
    );
}

/// Project a distance-bend constraint (same math as distance).
pub fn project_distance_bend(particles: &mut [Particle], c: &mut DistanceBendConstraint, dt: f64) {
    let mut tmp = DistanceConstraint {
        i: c.a,
        j: c.b,
        rest_length: c.rest_length,
        compliance: c.compliance,
        lambda: c.lambda,
        broken: false,
    };
    project_distance(particles, &mut tmp, dt);
    c.lambda = tmp.lambda;
}

/// Project dihedral bending (XPBD angle constraint).
pub fn project_dihedral(particles: &mut [Particle], c: &mut DihedralBendConstraint, dt: f64) {
    let p0 = particles[c.i0].x;
    let p1 = particles[c.i1].x;
    let p2 = particles[c.i2].x;
    let p3 = particles[c.i3].x;

    let e = p2 - p1;
    let e_len = e.norm();
    if e_len < 1e-12 {
        return;
    }
    let n1 = (p1 - p0).cross(p2 - p0);
    let n2 = (p2 - p3).cross(p1 - p3);
    let n1_len2 = n1.norm_squared();
    let n2_len2 = n2.norm_squared();
    if n1_len2 < 1e-20 || n2_len2 < 1e-20 {
        return;
    }
    let n1u = n1 * (1.0 / libm_ops::sqrt(n1_len2));
    let n2u = n2 * (1.0 / libm_ops::sqrt(n2_len2));
    let d = n1u.dot(n2u).clamp(-1.0, 1.0);
    let phi = libm_ops::acos(d);
    let c_val = phi - c.rest_angle;

    // Gradients from Müller et al. / XPBD cloth notes (stable form).
    let q2 = (p2 - p1) * (1.0 / e_len);
    let _ = q2; // edge direction used implicitly via cross products below

    // dφ/dp ≈ −1/sin(φ) * dn terms; use cross-product form:
    let sin_phi = libm_ops::sqrt((1.0 - d * d).max(0.0));
    if sin_phi < 1e-8 {
        return;
    }
    let grad0 = (p2 - p1).cross(n1u) * (1.0 / libm_ops::sqrt(n1_len2) / sin_phi);
    // Simpler PBD-style gradients on opposite vertices along normals:
    // Use n1u / |e| style from common PBD dihedral:
    let inv_l1 = 1.0 / libm_ops::sqrt(n1_len2);
    let inv_l2 = 1.0 / libm_ops::sqrt(n2_len2);
    let d0 = (p2 - p1).cross(n1u) * inv_l1;
    let d3 = (p1 - p2).cross(n2u) * inv_l2;
    let d1 = (p0 - p2).cross(n1u) * inv_l1 + (p2 - p3).cross(n2u) * inv_l2;
    let d2 = (p1 - p0).cross(n1u) * inv_l1 + (p3 - p1).cross(n2u) * inv_l2;

    // Scale so ∇C ≈ −dφ direction (sign: increase φ when moving along n).
    let scale = -1.0 / sin_phi.max(1e-8);
    let grads = [d0 * scale, d1 * scale, d2 * scale, d3 * scale];
    let idxs = [c.i0, c.i1, c.i2, c.i3];
    let _ = grad0;
    solve_equality(
        particles,
        &idxs,
        &grads,
        c_val,
        c.compliance,
        &mut c.lambda,
        dt,
    );
}

/// Project tetra volume constraint C = 6(V − V₀).
pub fn project_volume(particles: &mut [Particle], c: &mut VolumeConstraint, dt: f64) {
    let i = c.indices;
    let p0 = particles[i[0]].x;
    let p1 = particles[i[1]].x;
    let p2 = particles[i[2]].x;
    let p3 = particles[i[3]].x;
    let v = tetra_volume(p0, p1, p2, p3);
    let c_val = 6.0 * (v - c.rest_volume);
    // ∇_p0 (6V) = (p1−p2)×(p1−p3) etc. — standard:
    let g1 = (p2 - p0).cross(p3 - p0);
    let g2 = (p3 - p0).cross(p1 - p0);
    let g3 = (p1 - p0).cross(p2 - p0);
    let g0 = (p3 - p1).cross(p2 - p1); // = −(g1+g2+g3) for affinity
    let grads = [g0, g1, g2, g3];
    solve_equality(
        particles,
        &i,
        &grads,
        c_val,
        c.compliance,
        &mut c.lambda,
        dt,
    );
}

fn deformation_gradient(
    particles: &[Particle],
    idx: [usize; 4],
    dm_inv: &[[f64; 3]; 3],
) -> [[f64; 3]; 3] {
    let x0 = particles[idx[0]].x;
    let x1 = particles[idx[1]].x;
    let x2 = particles[idx[2]].x;
    let x3 = particles[idx[3]].x;
    let col0 = x1 - x0;
    let col1 = x2 - x0;
    let col2 = x3 - x0;
    let ds = [
        [col0.x, col1.x, col2.x],
        [col0.y, col1.y, col2.y],
        [col0.z, col1.z, col2.z],
    ];
    mul3(&ds, dm_inv)
}

/// Public deformation gradient F = Ds Dm⁻¹ for tet `idx`.
pub fn deformation_gradient_public(
    particles: &[Particle],
    idx: [usize; 4],
    dm_inv: &[[f64; 3]; 3],
) -> [[f64; 3]; 3] {
    deformation_gradient(particles, idx, dm_inv)
}

/// Project tetrahedral FEM via XPBD energy constraints (Macklin & Müller 2021).
///
/// Continuum energy `U = V₀ Ψ(F)`. XPBD match: `C = √(2Ψ)`, `α = 1/V₀` so
/// `C²/(2α) = V₀ Ψ`. Split into μ (deviatoric) and λ (hydrostatic) constraints.
///
/// **Critical:** when `λ ≈ 0` the hydrostatic constraint is **skipped** (not
/// `α = 0`). Treating `α = 0` as rigid `J = 1` was the axial under-stiffness bug:
/// isochoric stretch makes `I_C − 3 = O(ε²)`, killing linear FL/(EA) response.
///
/// - StVK: `Ψ = μ ‖E‖²_F + (λ/2)(tr E)²`, `E = ½(FᵀF − I)`, `P = F(2μ E + λ tr(E) I)`
/// - Stable NH: `Ψ = (μ/2)(I_C − 3) − μ ln J + (λ/2)(ln J)²`,
///   `P = μ F + (λ ln J − μ) F⁻ᵀ` (log form; stress-free at `F = I`)
pub fn project_tetra_fem(particles: &mut [Particle], c: &mut TetraFemConstraint, dt: f64) {
    if c.broken || c.rest_volume <= 1e-30 {
        return;
    }
    let f = deformation_gradient(particles, c.indices, &c.dm_inv);
    let j = det3(&f);
    if j.abs() < 1e-20 {
        return;
    }

    if c.neo_hookean {
        project_tetra_fem_snh(particles, c, &f, j, dt);
    } else {
        project_tetra_fem_stvk(particles, c, &f, dt);
    }
}

/// StVK split: μ-energy then λ-energy (skip λ when `lame_lambda ≈ 0`).
fn project_tetra_fem_stvk(
    particles: &mut [Particle],
    c: &mut TetraFemConstraint,
    f: &[[f64; 3]; 3],
    dt: f64,
) {
    let e = green_strain(f);
    let mut e_frob2 = 0.0;
    for row in &e {
        for &v in row {
            e_frob2 += v * v;
        }
    }

    // --- μ constraint: Ψ_μ = μ ‖E‖²_F, C = √(2 Ψ_μ), α = 1/V₀ ---
    if c.lame_mu > 0.0 {
        let psi_mu = c.lame_mu * e_frob2;
        if psi_mu > 1e-30 {
            // S_μ = 2μ E, P_μ = F S_μ
            let s = scale_mat(&e, 2.0 * c.lame_mu);
            let p = mul3(f, &s);
            project_energy_density(
                particles,
                &c.indices,
                &c.dm_inv,
                c.rest_volume,
                psi_mu,
                &p,
                &mut c.lambda_dev,
                dt,
            );
        }
    }

    // --- λ constraint: Ψ_λ = (λ/2)(tr E)²; skip entirely when λ≈0 (ν=0) ---
    if c.lame_lambda > 1e-30 {
        let f2 = deformation_gradient(particles, c.indices, &c.dm_inv);
        let e2 = green_strain(&f2);
        let tr2 = e2[0][0] + e2[1][1] + e2[2][2];
        let psi_lam = 0.5 * c.lame_lambda * tr2 * tr2;
        if psi_lam > 1e-30 {
            // S_λ = λ tr(E) I, P_λ = F S_λ
            let s = [
                [c.lame_lambda * tr2, 0.0, 0.0],
                [0.0, c.lame_lambda * tr2, 0.0],
                [0.0, 0.0, c.lame_lambda * tr2],
            ];
            let p = mul3(&f2, &s);
            project_energy_density(
                particles,
                &c.indices,
                &c.dm_inv,
                c.rest_volume,
                psi_lam,
                &p,
                &mut c.lambda_vol,
                dt,
            );
        }
    }
}

/// Stable neo-Hookean log form as one energy constraint (μ and λ together).
///
/// `Ψ = (μ/2)(I_C − 3) − μ ln J + (λ/2)(ln J)²`, `P = μ F + (λ ln J − μ) F⁻ᵀ`.
/// Single constraint avoids splitting a μ-term that can be indefinite in isolation.
/// When `λ ≈ 0` the `(ln J)²` term is omitted (not replaced by rigid `J = 1`).
fn project_tetra_fem_snh(
    particles: &mut [Particle],
    c: &mut TetraFemConstraint,
    f: &[[f64; 3]; 3],
    j: f64,
    dt: f64,
) {
    let ft_f = mul3(&transpose3(f), f);
    let i_c = ft_f[0][0] + ft_f[1][1] + ft_f[2][2];
    let ln_j = libm_ops::log(j.abs().max(1e-30));
    let mu = c.lame_mu;
    let lambda = c.lame_lambda.max(0.0);
    if mu <= 0.0 && lambda <= 0.0 {
        return;
    }
    let psi = 0.5 * mu * (i_c - 3.0) - mu * ln_j + 0.5 * lambda * ln_j * ln_j;
    if psi <= 1e-30 {
        return;
    }
    let f_inv_t = match invert3(f) {
        Some(inv) => transpose3(&inv),
        None => return,
    };
    // P = μ F + (λ ln J − μ) F⁻ᵀ
    let p = mat_add(&scale_mat(f, mu), &scale_mat(&f_inv_t, lambda * ln_j - mu));
    project_energy_density(
        particles,
        &c.indices,
        &c.dm_inv,
        c.rest_volume,
        psi,
        &p,
        &mut c.lambda_dev,
        dt,
    );
}

/// XPBD step for energy density `Ψ` with first Piola `P = ∂Ψ/∂F`.
///
/// `C = √(2Ψ)`, `α = 1/V₀`, `∇C = (1/C) ∇Ψ`, `∇ᵢΨ = P Dm⁻ᵀ eᵢ`.
fn project_energy_density(
    particles: &mut [Particle],
    indices: &[usize; 4],
    dm_inv: &[[f64; 3]; 3],
    rest_volume: f64,
    psi: f64,
    piola: &[[f64; 3]; 3],
    lambda: &mut Multiplier,
    dt: f64,
) {
    let c_val = libm_ops::sqrt((2.0 * psi).max(0.0));
    if c_val < 1e-14 || rest_volume <= 1e-30 {
        return;
    }
    let alpha = 1.0 / rest_volume;
    let inv_c = 1.0 / c_val;
    let mut grads = fem_piola_grads(dm_inv, piola);
    for g in &mut grads {
        *g *= inv_c;
    }
    solve_equality(particles, indices, &grads, c_val, alpha, lambda, dt);
}

fn green_strain(f: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c = mul3(&transpose3(f), f);
    [
        [0.5 * (c[0][0] - 1.0), 0.5 * c[0][1], 0.5 * c[0][2]],
        [0.5 * c[1][0], 0.5 * (c[1][1] - 1.0), 0.5 * c[1][2]],
        [0.5 * c[2][0], 0.5 * c[2][1], 0.5 * (c[2][2] - 1.0)],
    ]
}

/// Public Green–Lagrange strain E = ½(FᵀF − I).
pub fn green_strain_public(f: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    green_strain(f)
}

fn scale_mat(m: &[[f64; 3]; 3], s: f64) -> [[f64; 3]; 3] {
    [
        [m[0][0] * s, m[0][1] * s, m[0][2] * s],
        [m[1][0] * s, m[1][1] * s, m[1][2] * s],
        [m[2][0] * s, m[2][1] * s, m[2][2] * s],
    ]
}

fn mat_add(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0] + b[0][0], a[0][1] + b[0][1], a[0][2] + b[0][2]],
        [a[1][0] + b[1][0], a[1][1] + b[1][1], a[1][2] + b[1][2]],
        [a[2][0] + b[2][0], a[2][1] + b[2][1], a[2][2] + b[2][2]],
    ]
}

/// Particle gradients of a scalar that depends on F through first Piola `P`:
/// `∇_{x_{a+1}} = P Dm⁻ᵀ[:, a]`, `∇_{x_0} = −∑_{a=1}^3 ∇_{x_a}`.
fn fem_piola_grads(dm_inv: &[[f64; 3]; 3], piola: &[[f64; 3]; 3]) -> [Vec3; 4] {
    let dmt = transpose3(dm_inv);
    let mut g = [Vec3::zero(); 4];
    for a in 0..3 {
        let col = Vec3::new(dmt[0][a], dmt[1][a], dmt[2][a]);
        g[a + 1] = mat_vec(piola, col);
    }
    g[0] = (g[1] + g[2] + g[3]) * (-1.0);
    g
}

/// Project shape matching (linear polar decomposition of A = Σ m (x−c)(q−q0)ᵀ).
pub fn project_shape_match(particles: &mut [Particle], c: &mut ShapeMatchConstraint, dt: f64) {
    if c.indices.is_empty() {
        return;
    }
    let mut mass_sum = 0.0;
    let mut com = Vec3::zero();
    for &i in &c.indices {
        let m = if particles[i].inv_mass == 0.0 {
            // Treat pinned as unit mass for COM of free particles only — skip.
            continue;
        } else {
            1.0 / particles[i].inv_mass
        };
        mass_sum += m;
        com += particles[i].x * m;
    }
    // Include pinned with their rest weight as infinite → fix COM contribution via rest
    let mut any_dynamic = false;
    for &i in &c.indices {
        if particles[i].inv_mass != 0.0 {
            any_dynamic = true;
            break;
        }
    }
    if !any_dynamic || mass_sum < 1e-30 {
        return;
    }
    com *= 1.0 / mass_sum;

    // A = Σ m (x − com) ⊗ q
    let mut a = [[0.0; 3]; 3];
    for (k, &i) in c.indices.iter().enumerate() {
        let m = if particles[i].inv_mass == 0.0 {
            1.0
        } else {
            1.0 / particles[i].inv_mass
        };
        let p = particles[i].x - com;
        let q = c.rest_offsets[k];
        a[0][0] += m * p.x * q.x;
        a[0][1] += m * p.x * q.y;
        a[0][2] += m * p.x * q.z;
        a[1][0] += m * p.y * q.x;
        a[1][1] += m * p.y * q.y;
        a[1][2] += m * p.y * q.z;
        a[2][0] += m * p.z * q.x;
        a[2][1] += m * p.z * q.y;
        a[2][2] += m * p.z * q.z;
    }
    let r = extract_rotation_polar(&a);
    // Soft compliance: blend toward goal
    let alpha_tilde = if dt > 0.0 {
        c.compliance / (dt * dt)
    } else {
        0.0
    };
    // stiffness blend factor in [0,1]: for α=0 → full projection
    let stiffness = if alpha_tilde < 1e-30 {
        1.0
    } else {
        // heuristic: 1/(1+α̃) style
        1.0 / (1.0 + alpha_tilde)
    };
    for (k, &i) in c.indices.iter().enumerate() {
        if particles[i].inv_mass == 0.0 {
            continue;
        }
        let goal = com + mat_vec(&r, c.rest_offsets[k]);
        let corr = (goal - particles[i].x) * stiffness;
        particles[i].x += corr;
    }
    let _ = c.rest_com;
}

/// Extract a rotation matrix from A via simplified polar iteration (no SVD crate).
fn extract_rotation_polar(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // Start from A and iterate R ← A (Rᵀ A)^{-1/2} approximately via Newton on RᵀR = I
    let r = *a;
    // Normalize columns as a rough orthogonalization (Gram–Schmidt) — stable enough for soft clusters.
    let c0 = Vec3::new(r[0][0], r[1][0], r[2][0]);
    let c1 = Vec3::new(r[0][1], r[1][1], r[2][1]);
    let c2 = Vec3::new(r[0][2], r[1][2], r[2][2]);
    let u0 = if c0.norm_squared() > 1e-20 {
        c0.normalize()
    } else {
        Vec3::unit_x()
    };
    let u1_raw = c1 - u0 * c1.dot(u0);
    let u1 = if u1_raw.norm_squared() > 1e-20 {
        u1_raw.normalize()
    } else {
        let a = if u0.x.abs() < 0.9 {
            Vec3::unit_x()
        } else {
            Vec3::unit_y()
        };
        u0.cross(a).normalize()
    };
    let u2_raw = u0.cross(u1);
    let u2 = if c2.dot(u2_raw) < 0.0 {
        u2_raw * (-1.0)
    } else {
        u2_raw
    };
    // Prefer orientation that keeps det > 0
    let mut out = [[u0.x, u1.x, u2.x], [u0.y, u1.y, u2.y], [u0.z, u1.z, u2.z]];
    if det3(&out) < 0.0 {
        out[0][2] = -out[0][2];
        out[1][2] = -out[1][2];
        out[2][2] = -out[2][2];
    }
    let _ = r;
    out
}

/// Closed-surface volume from oriented triangles (1/6 Σ x_i · (x_j × x_k)).
pub fn mesh_volume(particles: &[Particle], tris: &[[usize; 3]]) -> f64 {
    let mut v = 0.0;
    for t in tris {
        let a = particles[t[0]].x;
        let b = particles[t[1]].x;
        let c = particles[t[2]].x;
        v += a.dot(b.cross(c));
    }
    v / 6.0
}

/// Project global pressure / volume constraint.
pub fn project_pressure(particles: &mut [Particle], c: &mut PressureConstraint, dt: f64) {
    let vol = mesh_volume(particles, &c.triangles);
    let c_val = vol - c.target_volume;
    // Gradient of V w.r.t vertex: accumulate face contributions
    let n = particles.len();
    let mut grads = vec![Vec3::zero(); n];
    for t in &c.triangles {
        let a = particles[t[0]].x;
        let b = particles[t[1]].x;
        let cpos = particles[t[2]].x;
        // ∂/∂a (a·(b×c))/6 = (b×c)/6
        grads[t[0]] += b.cross(cpos) * (1.0 / 6.0);
        grads[t[1]] += cpos.cross(a) * (1.0 / 6.0);
        grads[t[2]] += a.cross(b) * (1.0 / 6.0);
    }
    // Gather only used indices (deterministic: sorted unique)
    let mut used: Vec<usize> = Vec::new();
    for t in &c.triangles {
        used.push(t[0]);
        used.push(t[1]);
        used.push(t[2]);
    }
    used.sort_unstable();
    used.dedup();
    let g_sel: Vec<Vec3> = used.iter().map(|&i| grads[i]).collect();
    solve_equality(
        particles,
        &used,
        &g_sel,
        c_val,
        c.compliance,
        &mut c.lambda,
        dt,
    );
}

/// Project the entire constraint set once (fixed type order).
pub fn project_all(particles: &mut [Particle], constraints: &mut ConstraintSet, dt: f64) {
    for c in &mut constraints.distances {
        project_distance(particles, c, dt);
    }
    for c in &mut constraints.distance_bends {
        project_distance_bend(particles, c, dt);
    }
    for c in &mut constraints.dihedrals {
        project_dihedral(particles, c, dt);
    }
    for c in &mut constraints.volumes {
        project_volume(particles, c, dt);
    }
    for c in &mut constraints.tetra_fem {
        project_tetra_fem(particles, c, dt);
    }
    for c in &mut constraints.shape_matches {
        project_shape_match(particles, c, dt);
    }
    for c in &mut constraints.pressures {
        project_pressure(particles, c, dt);
    }
}
