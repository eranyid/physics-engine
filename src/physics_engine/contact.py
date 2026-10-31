"""Contact dynamics: sequential impulses (projected Gauss-Seidel) with restitution and Coulomb
friction, embedded in a semi-implicit Euler step, plus non-energetic positional correction.

Step (dt):
  1. kick:   v += dt F/m,  L += dt T              (forces from the World's generators)
  2. detect: broad phase (sweep-and-prune) + narrow phase at the current poses
  3. solve:  sequential impulses on (v, L); impulse P at lever r changes v by P/m and L by r x P
  4. drift:  x += dt v, q advanced with the implicit-midpoint exponential map (stage 1)
  5. correct: re-detect, translate bodies apart by beta*(depth - slop) per pair, velocities untouched
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from . import quaternion as quat
from .shapes import Material, Plane
from .broadphase import sweep_and_prune
from .narrowphase import collide
from .integrators import _drift_rotation


@dataclass
class StaticCollider:
    shape: object
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=quat.identity)
    material: Material = field(default_factory=Material)


@dataclass
class ContactPoint:
    a: int
    b: int
    point: np.ndarray
    normal: np.ndarray
    depth: float


def _basis(n):
    t = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    t /= np.linalg.norm(t)
    return t, np.cross(n, t)


class ContactWorld:
    """Wraps a stage-1 World; body i gets shapes[i] and materials[i]. Statics have infinite mass."""

    def __init__(self, world, shapes, statics=(), materials=None, iterations: int = 20,
                 restitution_threshold: float = 0.2, slop: float = 1e-4, beta: float = 0.2,
                 warm_start: bool = True):
        self.world = world
        self.shapes = list(shapes)
        assert len(self.shapes) == world.n
        self.statics = list(statics)
        self.materials = list(materials) if materials is not None else [Material()] * world.n
        self.iterations = iterations
        self.restitution_threshold = restitution_threshold  # m/s, below this e is ignored
        self.slop, self.beta = slop, beta
        self.warm_start = warm_start
        self.contacts: list[ContactPoint] = []
        self._cache: dict = {}   # (a, b, k) -> (normal impulse, tangent impulse vector, point)
        n, s = world.n, len(self.statics)
        self._static = np.array([False] * n + [True] * s)
        self._sx = np.array([c.position for c in self.statics], float).reshape(s, 3)
        self._sR = np.array([quat.to_matrix(np.asarray(c.orientation, float)) for c in self.statics]).reshape(s, 3, 3)

    @property
    def t(self):
        return self.world.t

    # --- detection -----------------------------------------------------------
    def _poses(self, st):
        R = quat.to_matrix(st.q)
        return np.concatenate([st.x, self._sx]), np.concatenate([R, self._sR])

    def detect(self, st=None) -> list[ContactPoint]:
        st = st if st is not None else self.world.state
        X, R = self._poses(st)
        allshapes = self.shapes + [c.shape for c in self.statics]
        lo, hi = zip(*(s.aabb(X[i], R[i]) for i, s in enumerate(allshapes)))
        pairs = sorted(sweep_and_prune(np.array(lo), np.array(hi), skip=self._static))
        out = []
        for i, j in pairs:
            for c in collide(allshapes[i], X[i], R[i], allshapes[j], X[j], R[j]):
                out.append(ContactPoint(i, j, c.point, c.normal, c.depth))
        return out

    def _mat(self, i):
        n = self.world.n
        return self.materials[i] if i < n else self.statics[i - n].material

    # --- velocity solve ------------------------------------------------------
    def solve_velocities(self, st, contacts):
        w, n = self.world, self.world.n
        N = n + len(self.statics)
        im = np.zeros(N); im[:n] = 1.0 / w.mass
        Iinv = np.zeros((N, 3, 3)); Iinv[:n] = w.inv_inertia_world(st.q)
        X = np.concatenate([st.x, self._sx])
        V = np.zeros((N, 3)); V[:n] = st.v
        L = np.zeros((N, 3)); L[:n] = st.L
        rows = []
        cache, self._cache = self._cache, {}
        counter: dict = {}
        for c in contacts:
            k = counter.get((c.a, c.b), 0)
            counter[(c.a, c.b)] = k + 1
            ra, rb = c.point - X[c.a], c.point - X[c.b]
            ma, mb = self._mat(c.a), self._mat(c.b)
            mu = np.sqrt(ma.friction * mb.friction)
            e = max(ma.restitution, mb.restitution)

            def kmass(d):
                ka = np.cross(Iinv[c.a] @ np.cross(ra, d), ra)
                kb = np.cross(Iinv[c.b] @ np.cross(rb, d), rb)
                return 1.0 / (im[c.a] + im[c.b] + d @ (ka + kb))
            t1, t2 = _basis(c.normal)
            vrel = (V[c.b] + np.cross(Iinv[c.b] @ L[c.b], rb)) - (V[c.a] + np.cross(Iinv[c.a] @ L[c.a], ra))
            vn0 = vrel @ c.normal
            bias = -e * vn0 if vn0 < -self.restitution_threshold else 0.0
            rows.append([c, ra, rb, mu, kmass(c.normal), kmass(t1), kmass(t2), t1, t2, bias, 0.0, np.zeros(2), (c.a, c.b, k)])

        def apply(a, b, ra, rb, P):
            V[a] -= im[a] * P; L[a] -= np.cross(ra, P) if a < n else 0.0
            V[b] += im[b] * P; L[b] += np.cross(rb, P) if b < n else 0.0

        if self.warm_start:   # re-apply last step's impulses for persistent contact features
            for r in rows:
                old = cache.get(r[12])
                if old is None or np.linalg.norm(old[2] - r[0].point) > 0.01:
                    continue
                c = r[0]
                r[10] = old[0]
                r[11] = np.array([old[1] @ r[7], old[1] @ r[8]])
                apply(c.a, c.b, r[1], r[2], r[10] * c.normal + r[11][0] * r[7] + r[11][1] * r[8])
        for _ in range(self.iterations):
            for r in rows:
                c, ra, rb, mu, kn, k1, k2, t1, t2, bias = r[:10]
                a, b = c.a, c.b
                vrel = (V[b] + np.cross(Iinv[b] @ L[b], rb)) - (V[a] + np.cross(Iinv[a] @ L[a], ra))
                # friction (uses current normal impulse as the Coulomb bound; disk projection)
                old = r[11].copy()
                jt = old - np.array([k1 * (vrel @ t1), k2 * (vrel @ t2)])
                lim = mu * r[10]
                nj = np.linalg.norm(jt)
                if nj > lim:
                    jt *= lim / nj
                d = jt - old
                r[11] = jt
                apply(a, b, ra, rb, d[0] * t1 + d[1] * t2)
                vrel = (V[b] + np.cross(Iinv[b] @ L[b], rb)) - (V[a] + np.cross(Iinv[a] @ L[a], ra))
                # normal
                lam = max(r[10] + kn * (bias - vrel @ c.normal), 0.0)
                apply(a, b, ra, rb, (lam - r[10]) * c.normal)
                r[10] = lam
        for r in rows:
            self._cache[r[12]] = (r[10], r[11][0] * r[7] + r[11][1] * r[8], r[0].point)
        st.v[:] = V[:n]
        st.L[:] = L[:n]
        return rows

    # --- position correction (no velocity change) --------------------------
    def correct_positions(self, st):
        im = np.concatenate([1.0 / self.world.mass, np.zeros(len(self.statics))])
        worst = {}
        for c in self.detect(st):
            if c.depth > worst.get((c.a, c.b), (0.0, None))[0]:
                worst[(c.a, c.b)] = (c.depth, c.normal)
        for (a, b), (d, nrm) in worst.items():
            C = self.beta * max(d - self.slop, 0.0)
            s = im[a] + im[b]
            if C <= 0 or s == 0:
                continue
            if a < self.world.n:
                st.x[a] -= nrm * C * im[a] / s
            if b < self.world.n:
                st.x[b] += nrm * C * im[b] / s

    # --- step ----------------------------------------------------------------
    def step(self, dt: float) -> None:
        w = self.world
        st = w.state.copy()
        F, T = w.forces(st, w.t)
        st.v += dt * F / w.mass[:, None]
        st.L += dt * T
        self.contacts = self.detect(st)
        self.solve_velocities(st, self.contacts)
        st.x += dt * st.v
        st.q = quat.normalize(_drift_rotation(w, st.q, st.L, dt))
        self.correct_positions(st)
        w.state = st
        w.t += dt

    def run(self, t_end: float, dt: float, record=None):
        out = []
        while self.world.t < t_end - 1e-12 * max(1.0, abs(t_end)):
            self.step(min(dt, t_end - self.world.t))
            if record is not None:
                out.append(record(self))
        return out
