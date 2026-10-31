//! Convex-mesh waterplane clipping (Sutherland–Hodgman on `z ≤ water_level`).
//!
//! Deterministic: portable `libm` for `atan2` / `sqrt`; fixed face order; stable
//! waterplane polygon sort by angle about the centroid.

use pe_math::libm_ops;
use pe_math::{Quat, Vec3};

use crate::hull::SubmergedState;

/// Clip a convex polyhedron (body-frame verts + CCW-outward faces) to `z ≤ water_level`.
pub fn convex_mesh_submerged(
    verts_body: &[Vec3],
    faces: &[Vec<usize>],
    x: Vec3,
    q: Quat,
    water_level: f64,
) -> SubmergedState {
    let mut verts_w = Vec::with_capacity(verts_body.len());
    for v in verts_body {
        verts_w.push(q.rotate(*v) + x);
    }

    let mut clipped_faces: Vec<Vec<Vec3>> = Vec::new();
    let mut wp_edges: Vec<(Vec3, Vec3)> = Vec::new();
    for face in faces {
        let poly: Vec<Vec3> = face.iter().map(|&i| verts_w[i]).collect();
        let (clipped, edge) = clip_polygon_z(&poly, water_level);
        if clipped.len() >= 3 {
            clipped_faces.push(clipped);
        }
        if let Some(e) = edge {
            wp_edges.push(e);
        }
    }

    if clipped_faces.is_empty() {
        return SubmergedState::empty(x);
    }

    let mut a_wp = 0.0;
    let mut i_xx = 0.0;
    let mut i_yy = 0.0;
    if !wp_edges.is_empty() {
        let wp_poly = order_waterplane_polygon(&wp_edges);
        if wp_poly.len() >= 3 {
            let (a, _, _, ixx, iyy) = polygon_area_moments(&wp_poly);
            a_wp = a;
            i_xx = ixx;
            i_yy = iyy;
            clipped_faces.push(wp_poly);
        }
    }

    let (vol, cob) = polyhedron_volume_centroid(&clipped_faces);
    if vol <= 1e-18 {
        return SubmergedState::empty(x);
    }
    let cob_body = q.inverse_rotate(cob - x);
    SubmergedState {
        volume: vol,
        cob_world: cob,
        a_wp,
        i_xx,
        i_yy,
        cob_body,
    }
}

fn clip_polygon_z(poly: &[Vec3], z0: f64) -> (Vec<Vec3>, Option<(Vec3, Vec3)>) {
    let mut out = Vec::new();
    let mut wp = Vec::new();
    let n = poly.len();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let a_in = a.z <= z0 + 1e-12;
        let b_in = b.z <= z0 + 1e-12;
        if a_in && b_in {
            out.push(b);
        } else if a_in && !b_in {
            let p = intersect_z(a, b, z0);
            out.push(p);
            wp.push(p);
        } else if !a_in && b_in {
            let p = intersect_z(a, b, z0);
            out.push(p);
            out.push(b);
            wp.push(p);
        }
    }
    let edge = if wp.len() == 2 {
        Some((wp[0], wp[1]))
    } else {
        None
    };
    (out, edge)
}

fn intersect_z(a: Vec3, b: Vec3, z0: f64) -> Vec3 {
    let dz = b.z - a.z;
    let mut t = if libm_ops::fabs(dz) < 1e-15 {
        0.0
    } else {
        (z0 - a.z) / dz
    };
    t = t.clamp(0.0, 1.0);
    a + (b - a) * t
}

fn polyhedron_volume_centroid(faces_pts: &[Vec<Vec3>]) -> (f64, Vec3) {
    let mut vol = 0.0;
    let mut c = Vec3::zero();
    for poly in faces_pts {
        let p0 = poly[0];
        for i in 1..poly.len().saturating_sub(1) {
            let p1 = poly[i];
            let p2 = poly[i + 1];
            let cross = (p1 - p0).cross(p2 - p0);
            let v = p0.dot(cross) / 6.0;
            let cent = (p0 + p1 + p2) / 4.0;
            vol += v;
            c += cent * v;
        }
    }
    if libm_ops::fabs(vol) < 1e-18 {
        return (0.0, Vec3::zero());
    }
    if vol < 0.0 {
        vol = -vol;
        c = -c;
    }
    (vol, c / vol)
}

fn order_waterplane_polygon(edges: &[(Vec3, Vec3)]) -> Vec<Vec3> {
    let mut pts = Vec::new();
    for &(a, b) in edges {
        pts.push(a);
        pts.push(b);
    }
    let mut uniq: Vec<Vec3> = Vec::new();
    for p in pts {
        if !uniq.iter().any(|u| (p - *u).norm() < 1e-9) {
            uniq.push(p);
        }
    }
    if uniq.len() < 3 {
        return uniq;
    }
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for p in &uniq {
        cx += p.x;
        cy += p.y;
        cz += p.z;
    }
    let n = uniq.len() as f64;
    let c = Vec3::new(cx / n, cy / n, cz / n);
    uniq.sort_by(|a, b| {
        let aa = libm_ops::atan2(a.y - c.y, a.x - c.x);
        let bb = libm_ops::atan2(b.y - c.y, b.x - c.x);
        aa.total_cmp(&bb)
    });
    uniq
}

/// Area [m²], centroid x/y [m], and `I_xx=∫y²`, `I_yy=∫x²` [m⁴] about the centroid.
pub fn polygon_area_moments(pts_in: &[Vec3]) -> (f64, f64, f64, f64, f64) {
    let mut pts: Vec<Vec3> = pts_in.to_vec();
    if pts.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    if (pts[0] - pts[pts.len() - 1]).norm() > 1e-15 {
        pts.push(pts[0]);
    }
    let mut a = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut ixx_o = 0.0;
    let mut iyy_o = 0.0;
    for i in 0..pts.len() - 1 {
        let (x0, y0) = (pts[i].x, pts[i].y);
        let (x1, y1) = (pts[i + 1].x, pts[i + 1].y);
        let cross = x0 * y1 - x1 * y0;
        a += cross;
        cx += (x0 + x1) * cross;
        cy += (y0 + y1) * cross;
        ixx_o += (y0 * y0 + y0 * y1 + y1 * y1) * cross;
        iyy_o += (x0 * x0 + x0 * x1 + x1 * x1) * cross;
    }
    a *= 0.5;
    if libm_ops::fabs(a) < 1e-18 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    cx /= 6.0 * a;
    cy /= 6.0 * a;
    ixx_o /= 12.0;
    iyy_o /= 12.0;
    let a_abs = libm_ops::fabs(a);
    let sign = if a > 0.0 { 1.0 } else { -1.0 };
    let ixx = (sign * ixx_o - a_abs * cy * cy).max(0.0);
    let iyy = (sign * iyy_o - a_abs * cx * cx).max(0.0);
    (a_abs, cx, cy, ixx, iyy)
}
