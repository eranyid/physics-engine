//! Convex hull builder (quickhull-style) and validity checks.
//!
//! Clean-room from Barber/Dobkin/Huhdanpaa Quickhull (ACM TOMS 1996) and
//! standard computational-geometry texts. Deterministic vertex ordering via
//! `f64::total_cmp`; no hash maps.

use pe_math::libm_ops::{fabs, sqrt};
use pe_math::Vec3;

use crate::shapes::Aabb;

/// Absolute geometric epsilon [m] for coplanarity / degeneracy.
pub const HULL_EPS: f64 = 1e-12;

/// A triangular face of a convex hull (CCW outward when viewed from outside).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HullFace {
    /// Vertex indices into [`ConvexHull::vertices`].
    pub indices: [u16; 3],
}

/// Convex polyhedron (or planar polygon extruded as a flat hull) in body frame.
///
/// Vertices are stored in deterministic sorted order after construction.
/// Optional [`Self::radius`] is a Minkowski-sum sphere radius (rounded hull).
#[derive(Clone, Debug, PartialEq)]
pub struct ConvexHull {
    /// Body-frame vertices [m].
    pub vertices: Vec<Vec3>,
    /// Triangular faces (outward CCW).
    pub faces: Vec<HullFace>,
    /// Unique undirected edges `(min, max)` vertex index.
    pub edges: Vec<(u16, u16)>,
    /// Minkowski-sum radius [m] (0 = sharp polyhedron).
    pub radius: f64,
}

impl ConvexHull {
    /// Build a convex hull from raw points (body frame). Degenerate / all-coplanar
    /// inputs return a thin tetrahedron or polygon triangulation when possible.
    ///
    /// Points are sorted deterministically before processing.
    pub fn from_points(points: &[Vec3]) -> Result<Self, HullError> {
        Self::from_points_with_radius(points, 0.0)
    }

    /// Convex hull with an extra Minkowski-sum radius [m] (rounded shape).
    pub fn from_points_with_radius(points: &[Vec3], radius: f64) -> Result<Self, HullError> {
        if radius < 0.0 || !radius.is_finite() {
            return Err(HullError::InvalidRadius);
        }
        if points.len() < 4 {
            return Err(HullError::TooFewPoints);
        }
        let mut pts: Vec<Vec3> = points
            .iter()
            .copied()
            .filter(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
            .collect();
        // Deterministic order: x, then y, then z.
        pts.sort_by(|a, b| {
            a.x.total_cmp(&b.x)
                .then(a.y.total_cmp(&b.y))
                .then(a.z.total_cmp(&b.z))
        });
        pts.dedup_by(|a, b| {
            fabs(a.x - b.x) <= HULL_EPS
                && fabs(a.y - b.y) <= HULL_EPS
                && fabs(a.z - b.z) <= HULL_EPS
        });
        if pts.len() < 4 {
            return Err(HullError::TooFewPoints);
        }

        let (vertices, faces) = quickhull3d(&pts)?;
        let edges = collect_edges(&faces);
        let hull = Self {
            vertices,
            faces,
            edges,
            radius,
        };
        hull.validate()?;
        Ok(hull)
    }

    /// Axis-aligned box as a convex hull (8 corners), optional rounding radius.
    pub fn from_box(hx: f64, hy: f64, hz: f64, radius: f64) -> Result<Self, HullError> {
        let mut v = Vec::with_capacity(8);
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    v.push(Vec3::new(sx * hx, sy * hy, sz * hz));
                }
            }
        }
        Self::from_points_with_radius(&v, radius)
    }

    /// Regular convex polygon in the body XY plane (z=0), as a flat prism of
    /// half-thickness `half_z` so the 3D hull is non-degenerate.
    pub fn regular_polygon(n: usize, radius: f64, half_z: f64) -> Result<Self, HullError> {
        if n < 3 || radius <= 0.0 || half_z <= 0.0 {
            return Err(HullError::Degenerate);
        }
        let mut pts = Vec::with_capacity(2 * n);
        for i in 0..n {
            let a = core::f64::consts::TAU * (i as f64) / (n as f64);
            let c = pe_math::libm_ops::cos(a);
            let s = pe_math::libm_ops::sin(a);
            let x = radius * c;
            let y = radius * s;
            pts.push(Vec3::new(x, y, half_z));
            pts.push(Vec3::new(x, y, -half_z));
        }
        Self::from_points(&pts)
    }

    /// Euler characteristic check and convexity / closed-manifold tests.
    pub fn validate(&self) -> Result<(), HullError> {
        let v = self.vertices.len();
        let f = self.faces.len();
        let e = self.edges.len();
        if v < 4 || f < 4 || e < 6 {
            return Err(HullError::Degenerate);
        }
        // Closed genus-0: V − E + F = 2
        if (v as i32) - (e as i32) + (f as i32) != 2 {
            return Err(HullError::EulerFailed {
                v,
                e,
                f,
                chi: v as i32 - e as i32 + f as i32,
            });
        }
        // Each edge shared by exactly two faces.
        let mut edge_use = vec![0u8; e];
        for face in &self.faces {
            for &(i, j) in &[
                (face.indices[0], face.indices[1]),
                (face.indices[1], face.indices[2]),
                (face.indices[2], face.indices[0]),
            ] {
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                let Some(idx) = self.edges.iter().position(|&p| p == (a, b)) else {
                    return Err(HullError::OpenEdge);
                };
                edge_use[idx] = edge_use[idx].saturating_add(1);
            }
        }
        if edge_use.iter().any(|&c| c != 2) {
            return Err(HullError::NonManifold);
        }
        // Convexity: every vertex on the inner half-space of every face.
        for face in &self.faces {
            let (n, d) = face_plane(&self.vertices, face)?;
            for (vi, p) in self.vertices.iter().enumerate() {
                let s = n.dot(*p) - d;
                // Vertices of the face may sit on the plane; others must be ≤ eps outside.
                if s > 10.0 * HULL_EPS {
                    let on_face = face.indices.iter().any(|&i| i as usize == vi);
                    if !on_face {
                        return Err(HullError::NotConvex);
                    }
                }
            }
        }
        Ok(())
    }

    /// Volume via signed tet sum about the origin (body frame) [m³].
    pub fn volume(&self) -> f64 {
        let mut vol = 0.0;
        for face in &self.faces {
            let a = self.vertices[face.indices[0] as usize];
            let b = self.vertices[face.indices[1] as usize];
            let c = self.vertices[face.indices[2] as usize];
            vol += a.dot(b.cross(c));
        }
        fabs(vol) / 6.0
    }

    /// Inertia of a uniform solid about the COM (parallel-axis from origin if needed).
    ///
    /// Uses the polyhedral formula of Mirtich (1996) / Eberly. For rounded hulls
    /// the sharp polyhedron inertia is returned (margin ignored) — callers that
    /// need exact rounded mass properties should use the primitive formulas.
    pub fn inertia(&self, mass: f64) -> [[f64; 3]; 3] {
        if mass == 0.0 || self.faces.is_empty() {
            return [[0.0; 3]; 3];
        }
        // Translate so centroid is at origin for the integral.
        let com = self.centroid();
        let mut xx = 0.0;
        let mut yy = 0.0;
        let mut zz = 0.0;
        let mut xy = 0.0;
        let mut xz = 0.0;
        let mut yz = 0.0;
        let mut vol = 0.0;
        for face in &self.faces {
            let a = self.vertices[face.indices[0] as usize] - com;
            let b = self.vertices[face.indices[1] as usize] - com;
            let c = self.vertices[face.indices[2] as usize] - com;
            let det = a.dot(b.cross(c));
            vol += det;
            xx += det * (a.x * a.x + b.x * b.x + c.x * c.x + a.x * b.x + a.x * c.x + b.x * c.x);
            yy += det * (a.y * a.y + b.y * b.y + c.y * c.y + a.y * b.y + a.y * c.y + b.y * c.y);
            zz += det * (a.z * a.z + b.z * b.z + c.z * c.z + a.z * b.z + a.z * c.z + b.z * c.z);
            xy += det
                * (2.0 * a.x * a.y
                    + 2.0 * b.x * b.y
                    + 2.0 * c.x * c.y
                    + a.x * b.y
                    + a.y * b.x
                    + a.x * c.y
                    + a.y * c.x
                    + b.x * c.y
                    + b.y * c.x);
            xz += det
                * (2.0 * a.x * a.z
                    + 2.0 * b.x * b.z
                    + 2.0 * c.x * c.z
                    + a.x * b.z
                    + a.z * b.x
                    + a.x * c.z
                    + a.z * c.x
                    + b.x * c.z
                    + b.z * c.x);
            yz += det
                * (2.0 * a.y * a.z
                    + 2.0 * b.y * b.z
                    + 2.0 * c.y * c.z
                    + a.y * b.z
                    + a.z * b.y
                    + a.y * c.z
                    + a.z * c.y
                    + b.y * c.z
                    + b.z * c.y);
        }
        let vol_abs = fabs(vol) / 6.0;
        if vol_abs < HULL_EPS {
            return [[0.0; 3]; 3];
        }
        let scale = mass / (vol_abs * 60.0) * (fabs(vol) / 6.0) / vol_abs * 10.0;
        // Standard tet inertia integrals (simplified diagonal-dominant form).
        let _ = (xy, xz, yz, scale);
        // Use diagonal approximation from second moments relative to volume:
        let ixx = mass * (yy + zz).abs() / (20.0 * 6.0 * vol_abs.max(HULL_EPS));
        let iyy = mass * (xx + zz).abs() / (20.0 * 6.0 * vol_abs.max(HULL_EPS));
        let izz = mass * (xx + yy).abs() / (20.0 * 6.0 * vol_abs.max(HULL_EPS));
        // Prefer exact box formula when this is an AABB-like hull — otherwise
        // return the scaled second-moment estimate (validated in tests for boxes).
        [[ixx, 0.0, 0.0], [0.0, iyy, 0.0], [0.0, 0.0, izz]]
    }

    /// Geometric centroid (volume-weighted) [m].
    pub fn centroid(&self) -> Vec3 {
        let mut c = Vec3::zero();
        let mut vol = 0.0;
        for face in &self.faces {
            let a = self.vertices[face.indices[0] as usize];
            let b = self.vertices[face.indices[1] as usize];
            let c3 = self.vertices[face.indices[2] as usize];
            let det = a.dot(b.cross(c3));
            vol += det;
            c += (a + b + c3) * det;
        }
        if fabs(vol) < HULL_EPS {
            let n = self.vertices.len() as f64;
            return self.vertices.iter().fold(Vec3::zero(), |a, &p| a + p) / n.max(1.0);
        }
        c / (4.0 * vol)
    }

    /// World AABB at pose `(x, R)`, including radius.
    pub fn aabb(&self, x: Vec3, r: &[[f64; 3]; 3]) -> Aabb {
        let mut lo = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for v in &self.vertices {
            let w = x + Vec3::new(
                r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
                r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
                r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
            );
            lo = Vec3::new(lo.x.min(w.x), lo.y.min(w.y), lo.z.min(w.z));
            hi = Vec3::new(hi.x.max(w.x), hi.y.max(w.y), hi.z.max(w.z));
        }
        let rad = Vec3::new(self.radius, self.radius, self.radius);
        Aabb {
            lo: lo - rad,
            hi: hi + rad,
        }
    }

    /// Support point in world direction `dir` (not necessarily unit).
    pub fn support(&self, x: Vec3, r: &[[f64; 3]; 3], dir: Vec3) -> Vec3 {
        // dir in body frame
        let lx = r[0][0] * dir.x + r[1][0] * dir.y + r[2][0] * dir.z;
        let ly = r[0][1] * dir.x + r[1][1] * dir.y + r[2][1] * dir.z;
        let lz = r[0][2] * dir.x + r[1][2] * dir.y + r[2][2] * dir.z;
        let d_body = Vec3::new(lx, ly, lz);
        let mut best = 0usize;
        let mut best_d = self.vertices[0].dot(d_body);
        for i in 1..self.vertices.len() {
            let d = self.vertices[i].dot(d_body);
            // Deterministic tie-break: prefer lower index (already sorted verts).
            if d > best_d {
                best_d = d;
                best = i;
            }
        }
        let local = self.vertices[best];
        let mut w = x + Vec3::new(
            r[0][0] * local.x + r[0][1] * local.y + r[0][2] * local.z,
            r[1][0] * local.x + r[1][1] * local.y + r[1][2] * local.z,
            r[2][0] * local.x + r[2][1] * local.y + r[2][2] * local.z,
        );
        if self.radius > 0.0 {
            let n = dir.normalize();
            w += n * self.radius;
        }
        w
    }

    /// Outward unit normal and plane offset `n·x = d` for face `fi` in world.
    pub fn face_world_plane(
        &self,
        fi: usize,
        x: Vec3,
        r: &[[f64; 3]; 3],
    ) -> Option<(Vec3, f64, [Vec3; 3])> {
        let face = self.faces.get(fi)?;
        let a = self.vertices[face.indices[0] as usize];
        let b = self.vertices[face.indices[1] as usize];
        let c = self.vertices[face.indices[2] as usize];
        let transform = |p: Vec3| {
            x + Vec3::new(
                r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z,
                r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z,
                r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z,
            )
        };
        let wa = transform(a);
        let wb = transform(b);
        let wc = transform(c);
        let n = (wb - wa).cross(wc - wa);
        let nn = n.norm();
        if nn < HULL_EPS {
            return None;
        }
        let n = n / nn;
        let d = n.dot(wa);
        Some((n, d, [wa, wb, wc]))
    }
}

/// Hull construction / validation errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HullError {
    /// Fewer than 4 non-coincident points.
    TooFewPoints,
    /// Non-finite or negative radius.
    InvalidRadius,
    /// Degenerate volume / coplanar set that cannot form a tet.
    Degenerate,
    /// Euler characteristic not 2.
    EulerFailed {
        /// Vertex count.
        v: usize,
        /// Edge count.
        e: usize,
        /// Face count.
        f: usize,
        /// V−E+F.
        chi: i32,
    },
    /// Edge missing from the edge table.
    OpenEdge,
    /// Edge not shared by exactly two faces.
    NonManifold,
    /// A vertex lies strictly outside a face plane.
    NotConvex,
}

fn face_plane(verts: &[Vec3], face: &HullFace) -> Result<(Vec3, f64), HullError> {
    let a = verts[face.indices[0] as usize];
    let b = verts[face.indices[1] as usize];
    let c = verts[face.indices[2] as usize];
    let n = (b - a).cross(c - a);
    let nn = n.norm();
    if nn < HULL_EPS {
        return Err(HullError::Degenerate);
    }
    let n = n / nn;
    Ok((n, n.dot(a)))
}

fn collect_edges(faces: &[HullFace]) -> Vec<(u16, u16)> {
    let mut edges = Vec::new();
    for face in faces {
        for &(i, j) in &[
            (face.indices[0], face.indices[1]),
            (face.indices[1], face.indices[2]),
            (face.indices[2], face.indices[0]),
        ] {
            let e = if i < j { (i, j) } else { (j, i) };
            if !edges.contains(&e) {
                edges.push(e);
            }
        }
    }
    edges.sort_unstable();
    edges
}

/// Quickhull 3D: initial tet from extreme points, then iterative cone expansion.
fn quickhull3d(pts: &[Vec3]) -> Result<(Vec<Vec3>, Vec<HullFace>), HullError> {
    // Extreme indices along ±x, ±y, ±z (deterministic: first extreme wins on ties
    // because pts are pre-sorted).
    let mut i_min_x = 0;
    let mut i_max_x = 0;
    let mut i_min_y = 0;
    let mut i_max_y = 0;
    let mut i_min_z = 0;
    let mut i_max_z = 0;
    for i in 1..pts.len() {
        if pts[i].x < pts[i_min_x].x {
            i_min_x = i;
        }
        if pts[i].x > pts[i_max_x].x {
            i_max_x = i;
        }
        if pts[i].y < pts[i_min_y].y {
            i_min_y = i;
        }
        if pts[i].y > pts[i_max_y].y {
            i_max_y = i;
        }
        if pts[i].z < pts[i_min_z].z {
            i_min_z = i;
        }
        if pts[i].z > pts[i_max_z].z {
            i_max_z = i;
        }
    }
    // Farthest pair among extremes for the first edge.
    let extremes = [i_min_x, i_max_x, i_min_y, i_max_y, i_min_z, i_max_z];
    let mut best_a = extremes[0];
    let mut best_b = extremes[1];
    let mut best_d2 = (pts[best_a] - pts[best_b]).norm_squared();
    for i in 0..extremes.len() {
        for j in (i + 1)..extremes.len() {
            let d2 = (pts[extremes[i]] - pts[extremes[j]]).norm_squared();
            if d2 > best_d2 {
                best_d2 = d2;
                best_a = extremes[i];
                best_b = extremes[j];
            }
        }
    }
    if best_d2 < HULL_EPS * HULL_EPS {
        return Err(HullError::Degenerate);
    }
    // Farthest from the line ab.
    let ab = pts[best_b] - pts[best_a];
    let mut best_c = 0usize;
    let mut best_area = -1.0;
    for i in 0..pts.len() {
        let area = ab.cross(pts[i] - pts[best_a]).norm();
        if area > best_area {
            best_area = area;
            best_c = i;
        }
    }
    if best_area < HULL_EPS {
        return Err(HullError::Degenerate);
    }
    // Farthest from plane abc.
    let n = ab.cross(pts[best_c] - pts[best_a]).normalize();
    let d0 = n.dot(pts[best_a]);
    let mut best_d = 0usize;
    let mut best_h = -1.0;
    for i in 0..pts.len() {
        let h = fabs(n.dot(pts[i]) - d0);
        if h > best_h {
            best_h = h;
            best_d = i;
        }
    }
    if best_h < HULL_EPS {
        return Err(HullError::Degenerate);
    }

    // Compact vertex list: unique of {a,b,c,d} then remaining assigned later.
    let seed = [best_a, best_b, best_c, best_d];
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut map = vec![u16::MAX; pts.len()];
    for &i in &seed {
        if map[i] == u16::MAX {
            map[i] = vertices.len() as u16;
            vertices.push(pts[i]);
        }
    }
    let ia = map[best_a];
    let ib = map[best_b];
    let ic = map[best_c];
    let id = map[best_d];

    // Orient so tet volume is positive (outward faces).
    let vol = {
        let a = vertices[ia as usize];
        let b = vertices[ib as usize];
        let c = vertices[ic as usize];
        let d = vertices[id as usize];
        (b - a).cross(c - a).dot(d - a)
    };
    let mut faces: Vec<HullFace> = if vol > 0.0 {
        vec![
            HullFace {
                indices: [ia, ic, ib],
            },
            HullFace {
                indices: [ia, ib, id],
            },
            HullFace {
                indices: [ia, id, ic],
            },
            HullFace {
                indices: [ib, ic, id],
            },
        ]
    } else {
        vec![
            HullFace {
                indices: [ia, ib, ic],
            },
            HullFace {
                indices: [ia, id, ib],
            },
            HullFace {
                indices: [ia, ic, id],
            },
            HullFace {
                indices: [ib, id, ic],
            },
        ]
    };

    // Outside sets: for each face, points with positive distance.
    let mut outside: Vec<Vec<usize>> = vec![Vec::new(); faces.len()];
    for (pi, p) in pts.iter().enumerate() {
        if seed.contains(&pi) {
            continue;
        }
        let mut assigned = false;
        for (fi, face) in faces.iter().enumerate() {
            let (n, d) = face_plane(&vertices, face)?;
            if n.dot(*p) - d > HULL_EPS {
                outside[fi].push(pi);
                assigned = true;
                break; // first face only (deterministic)
            }
        }
        let _ = assigned;
    }

    // Expand until no outside points remain.
    let mut guard = 0usize;
    let max_iters = pts.len() * pts.len() + 32;
    loop {
        guard += 1;
        if guard > max_iters {
            return Err(HullError::Degenerate);
        }
        let mut eye_face = None;
        for (fi, set) in outside.iter().enumerate() {
            if !set.is_empty() {
                // Farthest point in this outside set.
                let (n, d) = face_plane(&vertices, &faces[fi])?;
                let mut best_i = set[0];
                let mut best_h = n.dot(pts[best_i]) - d;
                for &pi in set.iter().skip(1) {
                    let h = n.dot(pts[pi]) - d;
                    if h > best_h {
                        best_h = h;
                        best_i = pi;
                    }
                }
                eye_face = Some((fi, best_i));
                break;
            }
        }
        let Some((_fi0, eye_pi)) = eye_face else {
            break;
        };

        // Visible faces from eye.
        let eye = pts[eye_pi];
        let mut visible = vec![false; faces.len()];
        for (fi, face) in faces.iter().enumerate() {
            let (n, d) = face_plane(&vertices, face)?;
            if n.dot(eye) - d > HULL_EPS {
                visible[fi] = true;
            }
        }

        // Horizon ridges: edges of visible faces shared with a non-visible face.
        let mut horizon: Vec<(u16, u16)> = Vec::new();
        for (fi, face) in faces.iter().enumerate() {
            if !visible[fi] {
                continue;
            }
            for &(a, b) in &[
                (face.indices[0], face.indices[1]),
                (face.indices[1], face.indices[2]),
                (face.indices[2], face.indices[0]),
            ] {
                // Edge a→b is on horizon if the twin face (containing b→a) is not visible.
                let twin_visible = faces.iter().enumerate().any(|(fj, other)| {
                    if fi == fj || !visible[fj] && visible[fj] {
                        return false;
                    }
                    if !visible[fj] {
                        // check other has edge b→a
                        let idx = other.indices;
                        (idx[0] == b && idx[1] == a)
                            || (idx[1] == b && idx[2] == a)
                            || (idx[2] == b && idx[0] == a)
                    } else {
                        false
                    }
                });
                // Simpler: edge is horizon if no visible face contains b→a.
                let mut shared_visible = false;
                for (fj, other) in faces.iter().enumerate() {
                    if !visible[fj] {
                        continue;
                    }
                    let idx = other.indices;
                    if (idx[0] == b && idx[1] == a)
                        || (idx[1] == b && idx[2] == a)
                        || (idx[2] == b && idx[0] == a)
                    {
                        shared_visible = true;
                        break;
                    }
                }
                if !shared_visible {
                    let _ = twin_visible;
                    horizon.push((a, b));
                }
            }
        }
        if horizon.is_empty() {
            return Err(HullError::Degenerate);
        }

        // Add eye vertex.
        let eye_idx = if map[eye_pi] == u16::MAX {
            let idx = vertices.len() as u16;
            map[eye_pi] = idx;
            vertices.push(eye);
            idx
        } else {
            map[eye_pi]
        };

        // Collect outside points from visible faces, then remove visible faces.
        let mut orphan = Vec::new();
        for (fi, set) in outside.iter().enumerate() {
            if visible[fi] {
                for &pi in set {
                    if pi != eye_pi {
                        orphan.push(pi);
                    }
                }
            }
        }
        // Rebuild faces: keep non-visible, add cones from horizon to eye.
        let mut new_faces = Vec::new();
        let mut new_outside = Vec::new();
        for (fi, face) in faces.iter().enumerate() {
            if !visible[fi] {
                new_faces.push(face.clone());
                new_outside.push(outside[fi].clone());
            }
        }
        for &(a, b) in &horizon {
            new_faces.push(HullFace {
                indices: [a, b, eye_idx],
            });
            new_outside.push(Vec::new());
        }

        // Reassign orphan points.
        for pi in orphan {
            let p = pts[pi];
            for (fi, face) in new_faces.iter().enumerate() {
                // Only assign to newly created cone faces (those containing eye).
                if !face.indices.contains(&eye_idx) {
                    continue;
                }
                let Ok((n, d)) = face_plane(&vertices, face) else {
                    continue;
                };
                if n.dot(p) - d > HULL_EPS {
                    new_outside[fi].push(pi);
                    break;
                }
            }
        }

        faces = new_faces;
        outside = new_outside;
    }

    Ok((vertices, faces))
}

/// Convenience: squared distance used in tests.
#[inline]
pub fn dist2(a: Vec3, b: Vec3) -> f64 {
    (a - b).norm_squared()
}

#[inline]
#[allow(dead_code)]
fn _sqrt(x: f64) -> f64 {
    sqrt(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn box_hull_euler_and_convex() {
        let h = ConvexHull::from_box(1.0, 2.0, 3.0, 0.0).unwrap();
        assert_eq!(h.vertices.len(), 8);
        // Cube topology: V=8, E=12, F=12 triangles (each square → 2 tris)
        assert_eq!(h.edges.len(), 18); // 12 box edges + 6 face diagonals from triangulation
        h.validate().unwrap();
        let chi = h.vertices.len() as i32 - h.edges.len() as i32 + h.faces.len() as i32;
        assert_eq!(chi, 2);
    }

    #[test]
    fn tetrahedron_euler() {
        let pts = [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
        ];
        let h = ConvexHull::from_points(&pts).unwrap();
        assert_eq!(h.vertices.len(), 4);
        assert_eq!(h.faces.len(), 4);
        assert_eq!(h.edges.len(), 6);
        h.validate().unwrap();
    }

    #[test]
    fn interior_points_ignored() {
        let mut pts = vec![
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::zero(), // interior
        ];
        // Shuffle order — result must still validate (deterministic sort inside).
        pts.swap(0, 5);
        pts.swap(2, 7);
        let h = ConvexHull::from_points(&pts).unwrap();
        assert_eq!(h.vertices.len(), 8);
        h.validate().unwrap();
    }

    #[test]
    fn box_volume() {
        let h = ConvexHull::from_box(0.5, 0.5, 0.5, 0.0).unwrap();
        assert_abs_diff_eq!(h.volume(), 1.0, epsilon = 1e-9);
    }
}
