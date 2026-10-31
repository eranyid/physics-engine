"""Ship hydrostatics and simple hull resistance (SI).

Hulls live in a body frame with origin at the geometric centre, +x forward (bow),
+y starboard, +z up (deck). Pass `com_body` (COM in that frame) into Buoyancy when
the rigid-body origin (COM) is offset from the geometric centre.

Waterplane: world plane z = water_level. Buoyancy = ρ g ∇ through the centre of
buoyancy. KB, BM, GM, GZ helpers are pure functions for analytic checks.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from . import quaternion as quat
from .units import STANDARD_GRAVITY

RHO_FRESHWATER = 1000.0
RHO_SEAWATER = 1025.0


@dataclass(frozen=True)
class SubmergedState:
    volume: float                 # ∇ [m^3]
    cob_world: np.ndarray         # centre of buoyancy [m]
    A_wp: float                   # waterplane area [m^2]
    I_xx: float                   # ∫ y^2 dA about waterplane centroid [m^4] (roll)
    I_yy: float                   # ∫ x^2 dA about waterplane centroid [m^4] (trim)
    cob_body: np.ndarray          # COB in geometric body frame [m]


class Hull(Protocol):
    def submerged(self, x: np.ndarray, q: np.ndarray, water_level: float) -> SubmergedState: ...


@dataclass
class BoxHull:
    """Rectangular barge: length L (x), beam B (y), depth D (z); centre at body origin."""
    length: float
    beam: float
    depth: float

    def __post_init__(self):
        if min(self.length, self.beam, self.depth) <= 0:
            raise ValueError("hull dimensions must be positive")

    def upright_draft_for_mass(self, mass: float, rho: float) -> float:
        """Draft from keel [m]: T = m / (ρ L B)."""
        return mass / (rho * self.length * self.beam)

    def upright_KB(self, draft: float) -> float:
        return 0.5 * draft

    def upright_BM_roll(self, draft: float) -> float:
        """BM = I/∇ with I = L B^3/12, ∇ = L B T → B²/(12 T)."""
        return self.beam ** 2 / (12.0 * draft)

    def upright_BM_trim(self, draft: float) -> float:
        return self.length ** 2 / (12.0 * draft)

    def upright_GM(self, draft: float, KG: float, roll: bool = True) -> float:
        BM = self.upright_BM_roll(draft) if roll else self.upright_BM_trim(draft)
        return self.upright_KB(draft) + BM - KG

    def righting_arm_small(self, GM: float, phi: float) -> float:
        return GM * np.sin(phi)

    def submerged(self, x: np.ndarray, q: np.ndarray, water_level: float = 0.0) -> SubmergedState:
        R = quat.to_matrix(q)
        # Analytic upright path when deck normal ≈ world +z.
        if float(R[2, 2]) > 0.9998:
            return self._submerged_upright(x, water_level)
        L, B, D = self.length, self.beam, self.depth
        hx, hy, hz = 0.5 * L, 0.5 * B, 0.5 * D
        # Vertex order chosen so listed faces are outward CCW.
        # Indices: (−x−y−z)=0 (−x−y+z)=1 (−x+y−z)=2 (−x+y+z)=3
        #          (+x−y−z)=4 (+x−y+z)=5 (+x+y−z)=6 (+x+y+z)=7
        corners = np.array([
            [-hx, -hy, -hz], [-hx, -hy,  hz], [-hx,  hy, -hz], [-hx,  hy,  hz],
            [ hx, -hy, -hz], [ hx, -hy,  hz], [ hx,  hy, -hz], [ hx,  hy,  hz],
        ], float)
        faces = [
            [0, 2, 3, 1],  # −x
            [4, 5, 7, 6],  # +x
            [0, 1, 5, 4],  # −y
            [2, 6, 7, 3],  # +y
            [0, 4, 6, 2],  # −z
            [1, 3, 7, 5],  # +z
        ]
        return _convex_mesh_submerged(corners, faces, x, q, water_level)

    def _submerged_upright(self, x: np.ndarray, water_level: float) -> SubmergedState:
        L, B, D = self.length, self.beam, self.depth
        keel_z = x[2] - 0.5 * D
        draft = float(np.clip(water_level - keel_z, 0.0, D))
        A = L * B
        Vol = A * draft
        if draft <= 0:
            return SubmergedState(0.0, np.asarray(x, float).copy(), 0.0, 0.0, 0.0, np.zeros(3))
        cob_z = keel_z + 0.5 * draft
        cob_world = np.array([x[0], x[1], cob_z])
        cob_body = np.array([0.0, 0.0, -0.5 * D + 0.5 * draft])
        I_xx = L * B ** 3 / 12.0
        I_yy = B * L ** 3 / 12.0
        return SubmergedState(Vol, cob_world, A, I_xx, I_yy, cob_body)


@dataclass
class CylinderHull:
    """Vertical circular cylinder (axis = body +z), centre at body origin."""
    radius: float
    height: float

    def __post_init__(self):
        if self.radius <= 0 or self.height <= 0:
            raise ValueError("radius and height must be positive")

    def upright_draft_for_mass(self, mass: float, rho: float) -> float:
        return mass / (rho * np.pi * self.radius ** 2)

    def upright_KB(self, draft: float) -> float:
        return 0.5 * draft

    def upright_BM_roll(self, draft: float) -> float:
        return self.radius ** 2 / (4.0 * draft)

    def upright_GM(self, draft: float, KG: float) -> float:
        return self.upright_KB(draft) + self.upright_BM_roll(draft) - KG

    def submerged(self, x: np.ndarray, q: np.ndarray, water_level: float = 0.0) -> SubmergedState:
        R = quat.to_matrix(q)
        if float(R[2, 2]) > 0.9998:
            return self._submerged_upright(x, water_level)
        n = 32
        th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        r, h = self.radius, self.height
        bottom = np.column_stack([r * np.cos(th), r * np.sin(th), np.full(n, -0.5 * h)])
        top = np.column_stack([r * np.cos(th), r * np.sin(th), np.full(n, 0.5 * h)])
        verts = np.vstack([bottom, top])
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, j + n, i + n])          # outward side
        faces.append(list(range(n - 1, -1, -1)))         # bottom (−z)
        faces.append(list(range(n, 2 * n)))               # top (+z)
        return _convex_mesh_submerged(verts, faces, x, q, water_level)

    def _submerged_upright(self, x: np.ndarray, water_level: float) -> SubmergedState:
        h = self.height
        keel_z = x[2] - 0.5 * h
        draft = float(np.clip(water_level - keel_z, 0.0, h))
        A = np.pi * self.radius ** 2
        Vol = A * draft
        if draft <= 0:
            return SubmergedState(0.0, np.asarray(x, float).copy(), 0.0, 0.0, 0.0, np.zeros(3))
        cob_world = np.array([x[0], x[1], keel_z + 0.5 * draft])
        cob_body = np.array([0.0, 0.0, -0.5 * h + 0.5 * draft])
        I = 0.25 * np.pi * self.radius ** 4
        return SubmergedState(Vol, cob_world, A, I, I, cob_body)


@dataclass
class ConvexMeshHull:
    """Convex mesh: body-frame vertices (N,3) and faces (CCW outward index loops)."""
    vertices: np.ndarray
    faces: list

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, float)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be (N, 3)")

    def submerged(self, x: np.ndarray, q: np.ndarray, water_level: float = 0.0) -> SubmergedState:
        return _convex_mesh_submerged(self.vertices, self.faces, x, q, water_level)


def _convex_mesh_submerged(verts_body, faces, x, q, water_level: float) -> SubmergedState:
    """Clip convex polyhedron to z <= water_level; close with waterplane face."""
    R = quat.to_matrix(q)
    x = np.asarray(x, float)
    verts_w = (R @ np.asarray(verts_body, float).T).T + x

    clipped_faces = []
    wp_edges = []
    for face in faces:
        poly = [verts_w[i] for i in face]
        clipped, edge = _clip_polygon_z(poly, water_level)
        if len(clipped) >= 3:
            clipped_faces.append(clipped)
        if edge is not None:
            wp_edges.append(edge)

    if not clipped_faces:
        return SubmergedState(0.0, x.copy(), 0.0, 0.0, 0.0, np.zeros(3))

    A_wp = I_xx = I_yy = 0.0
    wp_poly = None
    if wp_edges:
        wp_poly = _order_waterplane_polygon(wp_edges)
        if len(wp_poly) >= 3:
            A_wp, _, _, I_xx, I_yy = _polygon_area_moments(wp_poly)
            # Close the submerged solid with the waterplane face (normal +z ⇒ CCW in xy).
            # Outward normal for the submerged body points into the air (+z), so CCW when
            # viewed from above: already the order from atan2 sort if area > 0.
            clipped_faces.append(list(wp_poly))

    Vol, cob = _polyhedron_volume_centroid(clipped_faces)
    if Vol <= 1e-18:
        return SubmergedState(0.0, x.copy(), 0.0, 0.0, 0.0, np.zeros(3))

    cob_body = R.T @ (cob - x)
    return SubmergedState(Vol, cob, A_wp, I_xx, I_yy, cob_body)


def _clip_polygon_z(poly, z0: float):
    """Clip polygon to z <= z0. Returns (clipped_pts, waterplane_edge or None)."""
    out = []
    wp = []
    n = len(poly)
    for i in range(n):
        a = np.asarray(poly[i], float)
        b = np.asarray(poly[(i + 1) % n], float)
        a_in = a[2] <= z0 + 1e-12
        b_in = b[2] <= z0 + 1e-12
        if a_in and b_in:
            out.append(b)
        elif a_in and not b_in:
            p = _intersect_z(a, b, z0)
            out.append(p)
            wp.append(p)
        elif (not a_in) and b_in:
            p = _intersect_z(a, b, z0)
            out.append(p)
            out.append(b)
            wp.append(p)
    edge = (wp[0], wp[1]) if len(wp) == 2 else None
    return out, edge


def _intersect_z(a, b, z0):
    dz = b[2] - a[2]
    t = 0.0 if abs(dz) < 1e-15 else (z0 - a[2]) / dz
    t = float(np.clip(t, 0.0, 1.0))
    return a + t * (b - a)


def _polyhedron_volume_centroid(faces_pts):
    Vol = 0.0
    C = np.zeros(3)
    for poly in faces_pts:
        p0 = np.asarray(poly[0], float)
        for i in range(1, len(poly) - 1):
            p1 = np.asarray(poly[i], float)
            p2 = np.asarray(poly[i + 1], float)
            cross = np.cross(p1 - p0, p2 - p0)
            v = float(np.dot(p0, cross)) / 6.0
            c = (p0 + p1 + p2) / 4.0
            Vol += v
            C += v * c
    if abs(Vol) < 1e-18:
        return 0.0, np.zeros(3)
    if Vol < 0:
        Vol = -Vol
        C = -C
    return Vol, C / Vol


def _order_waterplane_polygon(edges):
    pts = []
    for a, b in edges:
        pts.append(np.asarray(a, float))
        pts.append(np.asarray(b, float))
    uniq = []
    for p in pts:
        if not any(np.linalg.norm(p - u) < 1e-9 for u in uniq):
            uniq.append(p)
    if len(uniq) < 3:
        return uniq
    c = np.mean(uniq, axis=0)
    uniq.sort(key=lambda p: np.arctan2(p[1] - c[1], p[0] - c[0]))
    return uniq


def _polygon_area_moments(pts):
    """Area, centroid x/y, and Ixx=∫y², Iyy=∫x² about the centroid for a z=const polygon."""
    pts = [np.asarray(p, float) for p in pts]
    if not np.allclose(pts[0], pts[-1]):
        pts = pts + [pts[0]]
    A = 0.0
    Cx = Cy = 0.0
    Ixx_o = Iyy_o = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i][0], pts[i][1]
        x1, y1 = pts[i + 1][0], pts[i + 1][1]
        cross = x0 * y1 - x1 * y0
        A += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross
        Ixx_o += (y0 * y0 + y0 * y1 + y1 * y1) * cross
        Iyy_o += (x0 * x0 + x0 * x1 + x1 * x1) * cross
    A *= 0.5
    if abs(A) < 1e-18:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    Cx /= (6.0 * A)
    Cy /= (6.0 * A)
    Ixx_o /= 12.0
    Iyy_o /= 12.0
    A_abs = abs(A)
    # Parallel-axis: I_centroid = I_origin - A * c^2 (using signed→abs consistently).
    sign = 1.0 if A > 0 else -1.0
    Ixx = sign * Ixx_o - A_abs * Cy * Cy
    Iyy = sign * Iyy_o - A_abs * Cx * Cx
    return A_abs, Cx, Cy, max(0.0, Ixx), max(0.0, Iyy)


# ----- Analytic helpers -------------------------------------------------------

def metacentric_height(KB: float, BM: float, KG: float) -> float:
    return KB + BM - KG


def BM_from_inertia(I_wp: float, volume: float) -> float:
    if volume <= 0:
        raise ValueError("volume must be positive")
    return I_wp / volume


def righting_arm(GM: float, phi: float) -> float:
    return GM * np.sin(phi)


def righting_moment(rho: float, volume: float, GZ: float, g: float = STANDARD_GRAVITY) -> float:
    return rho * g * volume * GZ


def heave_period(mass: float, rho: float, A_wp: float, g: float = STANDARD_GRAVITY) -> float:
    """T = 2π √(m / (ρ g A_wp))."""
    return 2.0 * np.pi * np.sqrt(mass / (rho * g * A_wp))


def roll_period(k: float, GM: float, g: float = STANDARD_GRAVITY) -> float:
    """T = 2π k / √(g GM)."""
    if GM <= 0:
        raise ValueError("GM must be positive for a stable roll period")
    return 2.0 * np.pi * k / np.sqrt(g * GM)


def box_righting_arm_curve(hull: BoxHull, draft: float, KG: float, phis) -> np.ndarray:
    """GZ(φ) for a rectangular barge; body origin = COM. Positive GZ is restoring."""
    D = hull.depth
    z_com_geom = -0.5 * D + KG  # COM in geometric frame
    L, B = hull.length, hull.beam
    hx, hy, hz = 0.5 * L, 0.5 * B, 0.5 * D
    # Same topology as BoxHull; shift so body origin is the COM.
    corners = np.array([
        [-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, -hz], [-hx, hy, hz],
        [hx, -hy, -hz], [hx, -hy, hz], [hx, hy, -hz], [hx, hy, hz],
    ], float)
    corners[:, 2] -= z_com_geom
    faces = [
        [0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5],
    ]
    phis = np.asarray(phis, float)
    gz = np.empty(len(phis))
    for i, phi in enumerate(phis):
        q = quat.from_axis_angle([1.0, 0.0, 0.0], float(phi))
        # COM world height = keel + KG = -draft + KG when upright (water at 0).
        x = np.array([0.0, 0.0, -draft + KG])
        sub = _convex_mesh_submerged(corners, faces, x, q, 0.0)
        if sub.volume <= 0:
            gz[i] = 0.0
            continue
        # Gravity direction in body (COM) frame; GZ > 0 means restoring for heel +φ.
        g_body = quat.to_matrix(q).T @ np.array([0.0, 0.0, -1.0])
        cob_com = sub.cob_body
        # Moment about +x per (ρ g V) is (cob × g)_x; for φ>0 a restoring moment is −x,
        # so the conventional positive GZ equals (cob × g)_x  (with our heel sign).
        gz[i] = float(np.cross(cob_com, g_body)[0])
    return gz


def angle_of_vanishing_stability(hull: BoxHull, draft: float, KG: float,
                                 phi_max: float = np.deg2rad(85.0), n: int = 120) -> float:
    """Smallest φ > 0 at which GZ crosses from positive to non-positive."""
    phis = np.linspace(0.0, phi_max, n)
    gz = box_righting_arm_curve(hull, draft, KG, phis)
    for i in range(1, len(phis)):
        if gz[i - 1] > 0.0 and gz[i] <= 0.0:
            t = gz[i - 1] / (gz[i - 1] - gz[i] + 1e-30)
            return float(phis[i - 1] + t * (phis[i] - phis[i - 1]))
    return float(phi_max)


# ----- Force generators -------------------------------------------------------

class Buoyancy:
    """Archimedes force ρ g ∇ +z and torque (r_COB − r_COM) × F on one body."""

    def __init__(
        self,
        body: int,
        hull: Hull,
        rho: float = RHO_FRESHWATER,
        g: float = STANDARD_GRAVITY,
        water_level: float = 0.0,
        com_body=(0.0, 0.0, 0.0),
    ):
        self.body = int(body)
        self.hull = hull
        self.rho = float(rho)
        self.g = float(g)
        self.water_level = float(water_level)
        self.com_body = np.asarray(com_body, float)

    def submerged(self, world, s) -> SubmergedState:
        i = self.body
        R = quat.to_matrix(s.q[i])
        x_geom = s.x[i] - R @ self.com_body
        return self.hull.submerged(x_geom, s.q[i], self.water_level)

    def apply(self, world, s, t, F, T):
        i = self.body
        sub = self.submerged(world, s)
        if sub.volume <= 0.0:
            return
        Fb = np.array([0.0, 0.0, self.rho * self.g * sub.volume])
        F[i] += Fb
        T[i] += np.cross(sub.cob_world - s.x[i], Fb)


class HullDrag:
    """Quadratic resistance R = ½ ρ V² S (C_f + C_w) opposing velocity.

    C_w may be a float or callable C_w(Fr) with Fr = V / √(g L).
    """

    def __init__(
        self,
        body: int,
        S: float,
        L: float,
        C_f: float = 0.004,
        C_w=0.0,
        rho: float = RHO_FRESHWATER,
        g: float = STANDARD_GRAVITY,
        wind=(0.0, 0.0, 0.0),
        include_heave: bool = False,
    ):
        self.body = int(body)
        self.S = float(S)
        self.L = float(L)
        self.C_f = float(C_f)
        self.C_w = C_w
        self.rho = float(rho)
        self.g = float(g)
        self.wind = np.asarray(wind, float)
        self.include_heave = bool(include_heave)

    def apply(self, world, s, t, F, T):
        i = self.body
        u = np.asarray(s.v[i] - self.wind, float).copy()
        if not self.include_heave:
            u[2] = 0.0
        V = float(np.linalg.norm(u))
        if V < 1e-12:
            return
        Fr = V / np.sqrt(self.g * self.L)
        Cw = float(self.C_w(Fr) if callable(self.C_w) else self.C_w)
        F[i] -= 0.5 * self.rho * (self.C_f + Cw) * self.S * V * u


def box_added_mass_heave(rho: float, A_wp: float, draft: float) -> float:
    """Optional heave added mass ≈ ½ ρ A_wp min(draft, √A_wp); bake into RigidBody.mass."""
    return 0.5 * rho * A_wp * min(draft, np.sqrt(max(A_wp, 0.0)))


def box_added_mass_surge(rho: float, volume: float, Cx: float = 0.5) -> float:
    """Optional surge added mass ≈ Cx ρ ∇."""
    return Cx * rho * volume
