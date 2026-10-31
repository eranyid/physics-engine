"""Narrow phase. Every routine returns a list of Contact with the normal pointing from shape A
to shape B and depth > 0 meaning penetration [m]."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .shapes import Sphere, Box, Capsule, Plane


@dataclass
class Contact:
    point: np.ndarray   # world point [m]
    normal: np.ndarray  # unit, A -> B
    depth: float        # penetration [m]


def _c(p, n, d):
    return Contact(np.asarray(p, float), np.asarray(n, float), float(d))


def sphere_sphere(xa, ra, xb, rb):
    d = xb - xa
    dist = np.linalg.norm(d)
    if dist >= ra + rb:
        return []
    n = d / dist if dist > 1e-12 else np.array([0.0, 0.0, 1.0])
    depth = ra + rb - dist
    return [_c(xa + n * (ra - 0.5 * depth), n, depth)]


def sphere_plane(xa, r, plane: Plane):
    n = plane.n
    dist = n @ xa - plane.offset
    if dist >= r:
        return []
    depth = r - dist
    return [_c(xa - n * (r - 0.5 * depth), -n, depth)]


def box_plane(xa, Ra, box: Box, plane: Plane):
    n = plane.n
    out = []
    for v in box.vertices(xa, Ra):
        dist = n @ v - plane.offset
        if dist < 0:
            out.append(_c(v - 0.5 * dist * n, -n, -dist))
    return out


def sphere_box(xa, r, xb, Rb, box: Box):
    h = box.h
    p = Rb.T @ (xa - xb)
    q = np.clip(p, -h, h)
    d = p - q
    dist = np.linalg.norm(d)
    if dist > 1e-12:
        if dist >= r:
            return []
        n_local = -d / dist          # sphere -> box
        depth = r - dist
        return [_c(xb + Rb @ q + Rb @ n_local * (0.5 * depth), Rb @ n_local, depth)]
    # centre inside the box: exit through the nearest face
    k = int(np.argmin(h - np.abs(p)))
    s = 1.0 if p[k] >= 0 else -1.0
    n_local = np.zeros(3)
    n_local[k] = -s
    depth = r + h[k] - abs(p[k])
    return [_c(xa, Rb @ n_local, depth)]


def _closest_seg_seg(p1, q1, p2, q2):
    """Closest points between segments [p1,q1] and [p2,q2] (Ericson, RTCD 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-14
    if a <= eps and e <= eps:
        return p1, p2
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > eps else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return p1 + d1 * s, p2 + d2 * t


def _closest_on_segment(p, q, x):
    d = q - p
    t = np.clip((x - p) @ d / (d @ d), 0, 1) if d @ d > 0 else 0.0
    return p + t * d


def capsule_plane(xa, Ra, cap: Capsule, plane: Plane):
    out = []
    for e in cap.segment(xa, Ra):
        out += sphere_plane(e, cap.radius, plane)
    return out


def capsule_sphere(xa, Ra, cap: Capsule, xb, rb):
    c = _closest_on_segment(*cap.segment(xa, Ra), xb)
    return sphere_sphere(c, cap.radius, xb, rb)


def capsule_capsule(xa, Ra, ca: Capsule, xb, Rb, cb: Capsule):
    pa, pb = _closest_seg_seg(*ca.segment(xa, Ra), *cb.segment(xb, Rb))
    return sphere_sphere(pa, ca.radius, pb, cb.radius)


def _reduce(points, depths, nmax=4):
    if len(points) <= nmax:
        return list(range(len(points)))
    P = np.asarray(points)
    i0 = int(np.argmax(depths))
    i1 = int(np.argmax(np.linalg.norm(P - P[i0], axis=1)))
    ax = P[i1] - P[i0]
    cr = np.cross(ax, P - P[i0])
    ref = cr[int(np.argmax(np.linalg.norm(cr, axis=1)))]
    s = cr @ ref
    return list(dict.fromkeys([i0, i1, int(np.argmax(s)), int(np.argmin(s))]))


def box_box(xa, Ra, A: Box, xb, Rb, B: Box):
    """Separating-axis test over 15 axes; face contacts produce a clipped manifold (<= 4 points),
    edge-edge contacts a single point."""
    ha, hb = A.h, B.h
    t = xb - xa
    best = (np.inf, None, None)   # (overlap, axis, kind)
    best_face = (np.inf, None, None)
    for kind, axes in (("A", [(i, Ra[:, i]) for i in range(3)]),
                       ("B", [(j, Rb[:, j]) for j in range(3)])):
        for idx, L in axes:
            d = t @ L
            ov = ha @ np.abs(Ra.T @ L) + hb @ np.abs(Rb.T @ L) - abs(d)
            if ov < 0:
                return []
            if ov < best_face[0]:
                best_face = (ov, L * (1 if d >= 0 else -1), (kind, idx))
    best_edge = (np.inf, None, None)
    for i in range(3):
        for j in range(3):
            L = np.cross(Ra[:, i], Rb[:, j])
            nl = np.linalg.norm(L)
            if nl < 1e-6:
                continue
            L = L / nl
            d = t @ L
            ov = ha @ np.abs(Ra.T @ L) + hb @ np.abs(Rb.T @ L) - abs(d)
            if ov < 0:
                return []
            if ov < best_edge[0]:
                best_edge = (ov, L * (1 if d >= 0 else -1), ("E", (i, j)))
    best = best_face
    if best_edge[0] < 0.95 * best_face[0] - 1e-5:   # prefer faces (stable manifolds)
        best = best_edge
    ov, n, (kind, idx) = best
    if kind == "E":
        i, j = idx
        pa = xa + sum(ha[k] * np.sign(Ra[:, k] @ n) * Ra[:, k] for k in range(3) if k != i)
        pb = xb - sum(hb[k] * np.sign(Rb[:, k] @ n) * Rb[:, k] for k in range(3) if k != j)
        ca, cb = _closest_seg_seg(pa - ha[i] * Ra[:, i], pa + ha[i] * Ra[:, i],
                                  pb - hb[j] * Rb[:, j], pb + hb[j] * Rb[:, j])
        return [_c(0.5 * (ca + cb), n, ov)]
    if kind == "A":
        xr, Rr, hr, xi, Ri, hi, nref = xa, Ra, ha, xb, Rb, hb, n
    else:
        xr, Rr, hr, xi, Ri, hi, nref = xb, Rb, hb, xa, Ra, ha, -n
    k = idx
    # incident face: the face of the other box most anti-parallel to nref
    dots = Ri.T @ nref
    m = int(np.argmax(np.abs(dots)))
    s = -np.sign(dots[m]) or -1.0
    fc = xi + s * hi[m] * Ri[:, m]
    u_, v_ = [a for a in range(3) if a != m]
    U, V = Ri[:, u_] * hi[u_], Ri[:, v_] * hi[v_]
    poly = [fc + U + V, fc - U + V, fc - U - V, fc + U - V]
    # clip against the 4 side planes of the reference face
    for a in [a for a in range(3) if a != k]:
        for sign in (1.0, -1.0):
            nrm, off = sign * Rr[:, a], sign * (Rr[:, a] @ xr) + hr[a]
            out = []
            for p, q in zip(poly, poly[1:] + poly[:1]):
                dp, dq = nrm @ p - off, nrm @ q - off
                if dp <= 0:
                    out.append(p)
                if dp * dq < 0:
                    out.append(p + (q - p) * (dp / (dp - dq)))
            poly = out
            if not poly:
                return []
    pts, deps = [], []
    face_off = nref @ xr + hr[k]
    for p in poly:
        dep = face_off - nref @ p
        if dep >= 0:
            pts.append(p + 0.5 * dep * nref)
            deps.append(dep)
    keep = _reduce(pts, np.asarray(deps))
    return [_c(pts[i], n, deps[i]) for i in keep]


def collide(sa, xa, Ra, sb, xb, Rb):
    """Dispatch on shape types; normal always A -> B."""
    ta, tb = type(sa), type(sb)
    order = [Sphere, Capsule, Box, Plane]
    if order.index(ta) > order.index(tb):
        cs = collide(sb, xb, Rb, sa, xa, Ra)
        return [Contact(c.point, -c.normal, c.depth) for c in cs]
    if ta is Sphere:
        if tb is Sphere:
            return sphere_sphere(xa, sa.radius, xb, sb.radius)
        if tb is Capsule:
            return [Contact(c.point, -c.normal, c.depth) for c in capsule_sphere(xb, Rb, sb, xa, sa.radius)]
        if tb is Box:
            return sphere_box(xa, sa.radius, xb, Rb, sb)
        return sphere_plane(xa, sa.radius, sb)
    if ta is Capsule:
        if tb is Capsule:
            return capsule_capsule(xa, Ra, sa, xb, Rb, sb)
        if tb is Plane:
            return capsule_plane(xa, Ra, sa, sb)
        raise NotImplementedError("capsule-box collision is not implemented")
    if ta is Box:
        if tb is Box:
            return box_box(xa, Ra, sa, xb, Rb, sb)
        return box_plane(xa, Ra, sa, sb)
    return []  # plane-plane
