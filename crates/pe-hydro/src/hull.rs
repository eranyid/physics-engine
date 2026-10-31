//! Hull geometry: box barge, vertical cylinder, sphere, convex mesh.
//!
//! Body frame: origin at geometric centre, +x forward (bow), +y starboard, +z up.
//! Waterplane: world plane `z = water_level`. Upright closed forms when deck normal ≈ +z.

use pe_math::libm_ops;
use pe_math::{Quat, Vec3};

use crate::clip::convex_mesh_submerged;

/// Fresh-water density [kg/m³].
pub const RHO_FRESHWATER: f64 = 1000.0;
/// Typical seawater density [kg/m³].
pub const RHO_SEAWATER: f64 = 1025.0;

/// Result of a submerged-volume query (SI).
#[derive(Clone, Copy, Debug)]
pub struct SubmergedState {
    /// Displaced volume ∇ [m³].
    pub volume: f64,
    /// Centre of buoyancy in world frame [m].
    pub cob_world: Vec3,
    /// Waterplane area [m²].
    pub a_wp: f64,
    /// Waterplane `∫ y² dA` about centroid [m⁴] (roll).
    pub i_xx: f64,
    /// Waterplane `∫ x² dA` about centroid [m⁴] (trim).
    pub i_yy: f64,
    /// COB in the geometric body frame [m].
    pub cob_body: Vec3,
}

impl SubmergedState {
    /// Zero displacement at position `x` [m].
    pub fn empty(x: Vec3) -> Self {
        Self {
            volume: 0.0,
            cob_world: x,
            a_wp: 0.0,
            i_xx: 0.0,
            i_yy: 0.0,
            cob_body: Vec3::zero(),
        }
    }
}

/// Rectangular barge: length `L` (x), beam `B` (y), depth `D` (z); centre at body origin.
#[derive(Clone, Copy, Debug)]
pub struct BoxHull {
    /// Length [m] (x).
    pub length: f64,
    /// Beam [m] (y).
    pub beam: f64,
    /// Depth [m] (z).
    pub depth: f64,
}

impl BoxHull {
    /// Create a box hull; all dimensions [m] must be positive.
    pub fn new(length: f64, beam: f64, depth: f64) -> Self {
        assert!(
            length > 0.0 && beam > 0.0 && depth > 0.0,
            "hull dimensions must be positive"
        );
        Self {
            length,
            beam,
            depth,
        }
    }

    /// Upright draft from keel [m]: `T = m / (ρ L B)`.
    pub fn upright_draft_for_mass(&self, mass: f64, rho: f64) -> f64 {
        mass / (rho * self.length * self.beam)
    }

    /// Keel-to-COB for an upright rectangle [m]: `KB = T/2`.
    pub fn upright_kb(&self, draft: f64) -> f64 {
        0.5 * draft
    }

    /// Roll metacentric radius [m]: `BM = B²/(12 T)`.
    pub fn upright_bm_roll(&self, draft: f64) -> f64 {
        self.beam * self.beam / (12.0 * draft)
    }

    /// Trim metacentric radius [m]: `BM = L²/(12 T)`.
    pub fn upright_bm_trim(&self, draft: f64) -> f64 {
        self.length * self.length / (12.0 * draft)
    }

    /// Metacentric height [m]: `GM = KB + BM − KG`.
    pub fn upright_gm(&self, draft: f64, kg: f64, roll: bool) -> f64 {
        let bm = if roll {
            self.upright_bm_roll(draft)
        } else {
            self.upright_bm_trim(draft)
        };
        self.upright_kb(draft) + bm - kg
    }

    /// Small-angle righting arm [m]: `GZ ≈ GM sin φ` (`φ` [rad]).
    pub fn righting_arm_small(&self, gm: f64, phi: f64) -> f64 {
        gm * libm_ops::sin(phi)
    }

    /// Submerged state at pose `x` [m], `q`, waterplane `z = water_level` [m].
    pub fn submerged(&self, x: Vec3, q: Quat, water_level: f64) -> SubmergedState {
        let r = q.to_matrix();
        if r[2][2] > 0.9998 {
            return self.submerged_upright(x, water_level);
        }
        let (verts, faces) = self.mesh_topology(0.0);
        convex_mesh_submerged(&verts, &faces, x, q, water_level)
    }

    fn submerged_upright(&self, x: Vec3, water_level: f64) -> SubmergedState {
        let (l, b, d) = (self.length, self.beam, self.depth);
        let keel_z = x.z - 0.5 * d;
        let draft = (water_level - keel_z).clamp(0.0, d);
        let a = l * b;
        let vol = a * draft;
        if draft <= 0.0 {
            return SubmergedState::empty(x);
        }
        let cob_z = keel_z + 0.5 * draft;
        SubmergedState {
            volume: vol,
            cob_world: Vec3::new(x.x, x.y, cob_z),
            a_wp: a,
            i_xx: l * b * b * b / 12.0,
            i_yy: b * l * l * l / 12.0,
            cob_body: Vec3::new(0.0, 0.0, -0.5 * d + 0.5 * draft),
        }
    }

    /// Box corner mesh; optional vertical shift of corners [m] (for COM-origin GZ).
    pub fn mesh_topology(&self, z_shift: f64) -> (Vec<Vec3>, Vec<Vec<usize>>) {
        let hx = 0.5 * self.length;
        let hy = 0.5 * self.beam;
        let hz = 0.5 * self.depth;
        let corners = vec![
            Vec3::new(-hx, -hy, -hz + z_shift),
            Vec3::new(-hx, -hy, hz + z_shift),
            Vec3::new(-hx, hy, -hz + z_shift),
            Vec3::new(-hx, hy, hz + z_shift),
            Vec3::new(hx, -hy, -hz + z_shift),
            Vec3::new(hx, -hy, hz + z_shift),
            Vec3::new(hx, hy, -hz + z_shift),
            Vec3::new(hx, hy, hz + z_shift),
        ];
        let faces = vec![
            vec![0, 2, 3, 1],
            vec![4, 5, 7, 6],
            vec![0, 1, 5, 4],
            vec![2, 6, 7, 3],
            vec![0, 4, 6, 2],
            vec![1, 3, 7, 5],
        ];
        (corners, faces)
    }
}

/// Vertical circular cylinder (axis = body +z), centre at body origin.
#[derive(Clone, Copy, Debug)]
pub struct CylinderHull {
    /// Radius [m].
    pub radius: f64,
    /// Height [m].
    pub height: f64,
}

impl CylinderHull {
    /// Create; `radius` and `height` [m] must be positive.
    pub fn new(radius: f64, height: f64) -> Self {
        assert!(radius > 0.0 && height > 0.0);
        Self { radius, height }
    }

    /// Upright draft from keel [m]: `T = m / (ρ π r²)`.
    pub fn upright_draft_for_mass(&self, mass: f64, rho: f64) -> f64 {
        mass / (rho * core::f64::consts::PI * self.radius * self.radius)
    }

    /// `KB = T/2` [m].
    pub fn upright_kb(&self, draft: f64) -> f64 {
        0.5 * draft
    }

    /// `BM = r²/(4 T)` [m].
    pub fn upright_bm_roll(&self, draft: f64) -> f64 {
        self.radius * self.radius / (4.0 * draft)
    }

    /// `GM = KB + BM − KG` [m].
    pub fn upright_gm(&self, draft: f64, kg: f64) -> f64 {
        self.upright_kb(draft) + self.upright_bm_roll(draft) - kg
    }

    /// Submerged state at pose `x` [m], `q`, waterplane [m].
    pub fn submerged(&self, x: Vec3, q: Quat, water_level: f64) -> SubmergedState {
        let r = q.to_matrix();
        if r[2][2] > 0.9998 {
            return self.submerged_upright(x, water_level);
        }
        let n = 32usize;
        let mut verts = Vec::with_capacity(2 * n);
        let h = self.height;
        for i in 0..n {
            let th = 2.0 * core::f64::consts::PI * (i as f64) / (n as f64);
            verts.push(Vec3::new(
                self.radius * libm_ops::cos(th),
                self.radius * libm_ops::sin(th),
                -0.5 * h,
            ));
        }
        for i in 0..n {
            let th = 2.0 * core::f64::consts::PI * (i as f64) / (n as f64);
            verts.push(Vec3::new(
                self.radius * libm_ops::cos(th),
                self.radius * libm_ops::sin(th),
                0.5 * h,
            ));
        }
        let mut faces = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            faces.push(vec![i, j, j + n, i + n]);
        }
        faces.push((0..n).rev().collect());
        faces.push((n..2 * n).collect());
        convex_mesh_submerged(&verts, &faces, x, q, water_level)
    }

    fn submerged_upright(&self, x: Vec3, water_level: f64) -> SubmergedState {
        let h = self.height;
        let keel_z = x.z - 0.5 * h;
        let draft = (water_level - keel_z).clamp(0.0, h);
        let a = core::f64::consts::PI * self.radius * self.radius;
        let vol = a * draft;
        if draft <= 0.0 {
            return SubmergedState::empty(x);
        }
        let i = 0.25 * core::f64::consts::PI * self.radius.powi(4);
        SubmergedState {
            volume: vol,
            cob_world: Vec3::new(x.x, x.y, keel_z + 0.5 * draft),
            a_wp: a,
            i_xx: i,
            i_yy: i,
            cob_body: Vec3::new(0.0, 0.0, -0.5 * h + 0.5 * draft),
        }
    }
}

/// Sphere of radius `R` [m], centre at body origin (mesh-free upright; mesh when heeled).
#[derive(Clone, Copy, Debug)]
pub struct SphereHull {
    /// Radius [m].
    pub radius: f64,
}

impl SphereHull {
    /// Create; `radius` [m] must be positive.
    pub fn new(radius: f64) -> Self {
        assert!(radius > 0.0);
        Self { radius }
    }

    /// Solid volume [m³]: `4/3 π R³`.
    pub fn volume(&self) -> f64 {
        4.0 / 3.0 * core::f64::consts::PI * self.radius * self.radius * self.radius
    }

    /// Upright draft from keel for mass `m` [kg] at density `ρ` [kg/m³].
    ///
    /// Solves the spherical-cap volume equation iteratively.
    pub fn upright_draft_for_mass(&self, mass: f64, rho: f64) -> f64 {
        let target = mass / rho;
        let r = self.radius;
        // Binary search draft ∈ [0, 2R].
        let mut lo = 0.0;
        let mut hi = 2.0 * r;
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if sphere_cap_volume(r, mid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Submerged state; upright uses the spherical-cap closed form.
    pub fn submerged(&self, x: Vec3, q: Quat, water_level: f64) -> SubmergedState {
        let rmat = q.to_matrix();
        if rmat[2][2] > 0.9998 {
            return self.submerged_upright(x, water_level);
        }
        // Low-facet icosphere-like: lat/lon UV sphere (convex).
        let n_lat = 12usize;
        let n_lon = 24usize;
        let mut verts = Vec::new();
        verts.push(Vec3::new(0.0, 0.0, -self.radius));
        for i in 1..n_lat {
            let v = core::f64::consts::PI * (i as f64) / (n_lat as f64);
            let z = -self.radius * libm_ops::cos(v);
            let rr = self.radius * libm_ops::sin(v);
            for j in 0..n_lon {
                let u = 2.0 * core::f64::consts::PI * (j as f64) / (n_lon as f64);
                verts.push(Vec3::new(rr * libm_ops::cos(u), rr * libm_ops::sin(u), z));
            }
        }
        verts.push(Vec3::new(0.0, 0.0, self.radius));
        let mut faces = Vec::new();
        // Bottom fan
        for j in 0..n_lon {
            let a = 1 + j;
            let b = 1 + (j + 1) % n_lon;
            faces.push(vec![0, b, a]);
        }
        // Rings
        for i in 0..n_lat - 2 {
            let row0 = 1 + i * n_lon;
            let row1 = 1 + (i + 1) * n_lon;
            for j in 0..n_lon {
                let j1 = (j + 1) % n_lon;
                faces.push(vec![row0 + j, row0 + j1, row1 + j1, row1 + j]);
            }
        }
        // Top fan
        let top = verts.len() - 1;
        let last_row = 1 + (n_lat - 2) * n_lon;
        for j in 0..n_lon {
            let a = last_row + j;
            let b = last_row + (j + 1) % n_lon;
            faces.push(vec![top, a, b]);
        }
        convex_mesh_submerged(&verts, &faces, x, q, water_level)
    }

    fn submerged_upright(&self, x: Vec3, water_level: f64) -> SubmergedState {
        let r = self.radius;
        let keel_z = x.z - r;
        let draft = (water_level - keel_z).clamp(0.0, 2.0 * r);
        if draft <= 0.0 {
            return SubmergedState::empty(x);
        }
        let vol = sphere_cap_volume(r, draft);
        // COB of a spherical segment from the bottom: standard formula.
        // Height h = draft; distance of COB from sphere centre along +z (keel→deck):
        // z_cob_from_center = -R + (3 (2R - h)^2) / (4 (3R - h)) wait — use known:
        // For cap of height h measured from the *bottom*: centre of volume of the
        // submerged portion relative to sphere centre is
        // z = −(3/4) (2R − h)² / (3R − h)   when h is from bottom...
        // Cap from bottom: equivalent to sphere minus upper empty cap of height 2R-h.
        // Submerged centroid from centre:  (3 (h − R)² − 3 R²) / (4 (3R − (2R− wait)))
        // Standard: for submerged height h from keel, COB from centre:
        //   z_c = − (3 (2R − h)²) / (4 (3R − h))   ? Let's use:
        // Spherical segment height h from bottom; centroid from the flat face is
        //   3(2R−h)²/(4(3R−h)) toward the sphere centre from the waterplane...
        // Waterplane at keel_z + draft. Sphere centre at x.z.
        // Distance of COB below waterplane (into water) for a bottom cap of height h:
        //   from sphere centre: z = (3 (h − R)^2 − 3 R^2? )
        // Verified formula (Wikipedia spherical cap, centroid from sphere centre):
        // Cap of height h at the *top*: z = 3(2R−h)² / (4(3R−h)) above centre if cap is upper.
        // Our submerged region is a bottom cap of height h=draft.
        // Bottom-cap centroid relative to centre: z = − 3(2R−h)² / (4(3R−h)).
        let h = draft;
        let z_from_center = -3.0 * (2.0 * r - h) * (2.0 * r - h) / (4.0 * (3.0 * r - h));
        let cob_z = x.z + z_from_center;
        // Waterplane area of sphere: π a² where a² = h (2R − h)
        let a_wp = core::f64::consts::PI * h * (2.0 * r - h);
        // Second moments about waterplane centre ≈ those of a circle of that radius.
        let a2 = h * (2.0 * r - h);
        let i = 0.25 * core::f64::consts::PI * a2 * a2;
        SubmergedState {
            volume: vol,
            cob_world: Vec3::new(x.x, x.y, cob_z),
            a_wp,
            i_xx: i,
            i_yy: i,
            cob_body: Vec3::new(0.0, 0.0, z_from_center),
        }
    }
}

/// Volume [m³] of a spherical segment of height `h` [m] from the bottom of a sphere radius `r` [m].
pub fn sphere_cap_volume(r: f64, h: f64) -> f64 {
    let h = h.clamp(0.0, 2.0 * r);
    // Cap formula V = π h² (3R − h) / 3 works for a cap measured from either pole.
    core::f64::consts::PI * h * h * (3.0 * r - h) / 3.0
}

/// Convex mesh: body-frame vertices and faces (CCW outward index loops).
#[derive(Clone, Debug)]
pub struct ConvexMeshHull {
    /// Vertices in the geometric body frame [m].
    pub vertices: Vec<Vec3>,
    /// Faces as index loops (outward CCW).
    pub faces: Vec<Vec<usize>>,
}

impl ConvexMeshHull {
    /// Create from vertices [m] and faces.
    pub fn new(vertices: Vec<Vec3>, faces: Vec<Vec<usize>>) -> Self {
        Self { vertices, faces }
    }

    /// Submerged state at pose `x` [m], `q`, waterplane [m].
    pub fn submerged(&self, x: Vec3, q: Quat, water_level: f64) -> SubmergedState {
        convex_mesh_submerged(&self.vertices, &self.faces, x, q, water_level)
    }
}

/// Hull kind used by [`crate::force::Buoyancy`].
#[derive(Clone, Debug)]
pub enum Hull {
    /// Rectangular barge.
    Box(BoxHull),
    /// Vertical cylinder.
    Cylinder(CylinderHull),
    /// Sphere.
    Sphere(SphereHull),
    /// Convex polyhedron.
    Mesh(ConvexMeshHull),
}

impl Hull {
    /// Submerged state at pose [m] / waterplane [m].
    pub fn submerged(&self, x: Vec3, q: Quat, water_level: f64) -> SubmergedState {
        match self {
            Hull::Box(h) => h.submerged(x, q, water_level),
            Hull::Cylinder(h) => h.submerged(x, q, water_level),
            Hull::Sphere(h) => h.submerged(x, q, water_level),
            Hull::Mesh(h) => h.submerged(x, q, water_level),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    #[test]
    fn barge_kb_bm_gm_published() {
        let hull = BoxHull::new(30.0, 8.0, 4.0);
        let draft = 2.0;
        let kg = 1.5;
        let kb = hull.upright_kb(draft);
        let bm = hull.upright_bm_roll(draft);
        let gm = hull.upright_gm(draft, kg, true);
        assert_abs_diff_eq!(kb, draft / 2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(bm, 8.0_f64.powi(2) / (12.0 * draft), epsilon = 1e-15);
        assert_abs_diff_eq!(gm, kb + bm - kg, epsilon = 1e-15);
    }

    #[test]
    fn sphere_half_density_half_submerged() {
        let r = 1.0;
        let hull = SphereHull::new(r);
        let rho_w = RHO_FRESHWATER;
        let mass = (rho_w / 2.0) * hull.volume();
        let draft = hull.upright_draft_for_mass(mass, rho_w);
        assert_relative_eq!(draft, r, max_relative = 1e-6);
        let x = Vec3::new(0.0, 0.0, 0.0); // centre at waterplane ⇒ half submerged
        let sub = hull.submerged(x, Quat::identity(), 0.0);
        assert_relative_eq!(sub.volume, 0.5 * hull.volume(), max_relative = 1e-9);
    }
}
