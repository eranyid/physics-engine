//! 3-D compressible Euler finite-volume solver on a uniform voxel grid.
//!
//! # Scheme
//! - Unsplit finite volume, conserved `U = (ρ, ρu, ρv, ρw, E)`,
//!   `E = p/(γ−1) + ½ρ|u|²`.
//! - Face fluxes: 3-D **HLLC** (Toro 2009, §10.4) in face-normal coordinates,
//!   tangential momenta advected through the star states.
//! - MUSCL / minmod reconstruction of primitives, falling back to first order
//!   at strong pressure jumps (same rule as [`crate::euler::Euler1d`]).
//! - SSP-RK2 (Heun) in time; CFL on `Σ_d (|u_d| + a) / dx`.
//!
//! # Geometry
//! - `solid` voxels are rigid reflective walls (mirror ghost: normal velocity
//!   flipped). Doors / windows are simply fluid voxels.
//! - Each domain face is [`Boundary::Outflow`] (zero-gradient) or
//!   [`Boundary::Wall`] (reflective; also a symmetry plane).
//!
//! # Determinism
//! Face fluxes are computed per face, then gathered per cell in a fixed
//! order, so every cell update is independent. With the `parallel` feature
//! (rayon over z-slabs) results are bit-identical to the serial path.
//! Portable `libm`, no FMA.
//!
//! SI: `ρ` [kg/m³], velocity [m/s], `p` [Pa], lengths [m], `t` [s], `E` [J/m³].

use pe_math::libm_ops;
use pe_math::Vec3;

/// Number of conserved variables.
pub const NVAR: usize = 5;

/// Conserved state of one cell.
pub type Cons = [f64; NVAR];

/// Primitive state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Prim3 {
    /// Density [kg/m³].
    pub rho: f64,
    /// Velocity [m/s].
    pub u: [f64; 3],
    /// Pressure [Pa].
    pub p: f64,
}

impl Prim3 {
    /// At rest.
    pub fn at_rest(rho: f64, p: f64) -> Self {
        Self {
            rho,
            u: [0.0; 3],
            p,
        }
    }

    /// Conserved variables.
    pub fn to_cons(self, gamma: f64) -> Cons {
        let ke = 0.5
            * self.rho
            * (self.u[0] * self.u[0] + self.u[1] * self.u[1] + self.u[2] * self.u[2]);
        [
            self.rho,
            self.rho * self.u[0],
            self.rho * self.u[1],
            self.rho * self.u[2],
            self.p / (gamma - 1.0) + ke,
        ]
    }

    /// From conserved, with density / pressure floors.
    pub fn from_cons(c: &Cons, gamma: f64, rho_floor: f64, p_floor: f64) -> Self {
        let rho = c[0].max(rho_floor);
        let u = [c[1] / rho, c[2] / rho, c[3] / rho];
        let ke = 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        let p = ((gamma - 1.0) * (c[4] - ke)).max(p_floor);
        Self { rho, u, p }
    }

    /// Sound speed [m/s].
    pub fn sound_speed(self, gamma: f64) -> f64 {
        libm_ops::sqrt(gamma * self.p / self.rho)
    }

    /// Mirror across a wall normal to axis `d` (normal velocity flipped).
    pub fn mirrored(mut self, d: usize) -> Self {
        self.u[d] = -self.u[d];
        self
    }
}

/// Domain-face boundary condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// Zero-gradient outflow (waves leave the domain).
    Outflow,
    /// Rigid reflective wall (also an exact symmetry plane).
    Wall,
}

/// Physical flux of `w` along axis `d`.
fn physical_flux(w: Prim3, d: usize, gamma: f64) -> Cons {
    let c = w.to_cons(gamma);
    let un = w.u[d];
    let mut f = [
        c[0] * un,
        c[1] * un,
        c[2] * un,
        c[3] * un,
        (c[4] + w.p) * un,
    ];
    f[1 + d] += w.p;
    f
}

/// 3-D HLLC flux along axis `d` (Toro 2009, eqs. 10.38–10.40, Davis speeds).
pub fn hllc3(wl: Prim3, wr: Prim3, d: usize, gamma: f64) -> Cons {
    let al = wl.sound_speed(gamma);
    let ar = wr.sound_speed(gamma);
    let ul = wl.u[d];
    let ur = wr.u[d];
    let sl = (ul - al).min(ur - ar);
    let sr = (ul + al).max(ur + ar);
    if sl >= 0.0 {
        return physical_flux(wl, d, gamma);
    }
    if sr <= 0.0 {
        return physical_flux(wr, d, gamma);
    }
    let num = wr.p - wl.p + wl.rho * ul * (sl - ul) - wr.rho * ur * (sr - ur);
    let mut den = wl.rho * (sl - ul) - wr.rho * (sr - ur);
    if libm_ops::fabs(den) < 1e-30 {
        den = libm_ops::copysign(1e-30, den);
    }
    let sm = num / den;
    let star = |w: Prim3, s: f64, un: f64| -> (Cons, Cons) {
        let c = w.to_cons(gamma);
        let mut dsm = s - sm;
        if libm_ops::fabs(dsm) < 1e-30 {
            dsm = libm_ops::copysign(1e-30, dsm);
        }
        let factor = w.rho * (s - un) / dsm;
        let e_star = c[4] / w.rho + (sm - un) * (sm + w.p / (w.rho * (s - un)));
        let mut us = [
            factor,
            factor * w.u[0],
            factor * w.u[1],
            factor * w.u[2],
            factor * e_star,
        ];
        us[1 + d] = factor * sm;
        (us, c)
    };
    let (w, s, un) = if sm >= 0.0 {
        (wl, sl, ul)
    } else {
        (wr, sr, ur)
    };
    let (us, c) = star(w, s, un);
    let f = physical_flux(w, d, gamma);
    let mut out = [0.0; NVAR];
    for k in 0..NVAR {
        out[k] = f[k] + s * (us[k] - c[k]);
    }
    out
}

/// TVD slope limiter for MUSCL reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limiter {
    /// Most diffusive, most robust.
    Minmod,
    /// Van Leer (harmonic).
    VanLeer,
    /// Monotonized central — sharpest of the three (default).
    MonotonizedCentral,
}

impl Limiter {
    /// Limited slope from backward difference `a` and forward difference `b`.
    #[inline]
    pub fn slope(self, a: f64, b: f64) -> f64 {
        if a * b <= 0.0 {
            return 0.0;
        }
        match self {
            Limiter::Minmod => {
                if libm_ops::fabs(a) < libm_ops::fabs(b) {
                    a
                } else {
                    b
                }
            }
            Limiter::VanLeer => 2.0 * a * b / (a + b),
            Limiter::MonotonizedCentral => {
                let m = (2.0 * libm_ops::fabs(a))
                    .min(2.0 * libm_ops::fabs(b))
                    .min(0.5 * libm_ops::fabs(a + b));
                libm_ops::copysign(m, a)
            }
        }
    }
}

/// MUSCL face value of `c` toward `+d` (`sign = +0.5`) or `−d` (`−0.5`).
/// TVD limiters keep the face value between the neighbors, so density and
/// pressure stay positive.
fn muscl(lim: Limiter, m: Prim3, c: Prim3, p: Prim3, sign: f64) -> Prim3 {
    let mut u = c.u;
    for k in 0..3 {
        u[k] += sign * lim.slope(c.u[k] - m.u[k], p.u[k] - c.u[k]);
    }
    Prim3 {
        rho: (c.rho + sign * lim.slope(c.rho - m.rho, p.rho - c.rho)).max(1e-14),
        u,
        p: (c.p + sign * lim.slope(c.p - m.p, p.p - c.p)).max(1e-14),
    }
}

/// 3-D Euler grid.
#[derive(Clone, Debug)]
pub struct Grid3d {
    /// Cells along x.
    pub nx: usize,
    /// Cells along y.
    pub ny: usize,
    /// Cells along z.
    pub nz: usize,
    /// Cell size [m].
    pub dx: f64,
    /// Minimum corner [m].
    pub origin: Vec3,
    /// Ratio of specific heats.
    pub gamma: f64,
    /// Conserved state per cell (index `i + nx (j + ny k)`).
    pub u: Vec<Cons>,
    /// Solid (wall) voxels.
    pub solid: Vec<bool>,
    /// Boundary at `[x−, x+, y−, y+, z−, z+]`.
    pub boundary: [Boundary; 6],
    /// Ambient state (initial condition and solid-cell filler).
    pub ambient: Prim3,
    /// Density floor [kg/m³].
    pub rho_floor: f64,
    /// Pressure floor [Pa].
    pub p_floor: f64,
    /// Relative pressure jump `|Δp| / Σp` above which a face falls back to
    /// first-order (piecewise-constant) states. `f64::INFINITY` = always MUSCL
    /// (minmod keeps reconstructions bounded, so this stays positive).
    pub first_order_jump: f64,
    /// MUSCL slope limiter.
    pub limiter: Limiter,
    /// Simulation time [s].
    pub t: f64,
    /// Steps taken.
    pub steps: u64,
}

/// Ambient sea-level air density [kg/m³].
pub const RHO_AIR: f64 = 1.225;
/// Ambient sea-level pressure [Pa].
pub const P_AIR: f64 = 101_325.0;

impl Grid3d {
    /// Grid of `nx × ny × nz` cells of size `dx` from `origin`, filled with
    /// `ambient`, all boundaries outflow.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        origin: Vec3,
        ambient: Prim3,
        gamma: f64,
    ) -> Self {
        let n = nx.max(1) * ny.max(1) * nz.max(1);
        Self {
            nx: nx.max(1),
            ny: ny.max(1),
            nz: nz.max(1),
            dx,
            origin,
            gamma,
            u: vec![ambient.to_cons(gamma); n],
            solid: vec![false; n],
            boundary: [Boundary::Outflow; 6],
            ambient,
            rho_floor: 1e-6 * ambient.rho,
            p_floor: 1e-6 * ambient.p,
            first_order_jump: f64::INFINITY,
            limiter: Limiter::MonotonizedCentral,
            t: 0.0,
            steps: 0,
        }
    }

    /// Sea-level air grid (γ = 1.4).
    pub fn air(nx: usize, ny: usize, nz: usize, dx: f64, origin: Vec3) -> Self {
        Self::new(nx, ny, nz, dx, origin, Prim3::at_rest(RHO_AIR, P_AIR), 1.4)
    }

    /// Cell count.
    pub fn len(&self) -> usize {
        self.u.len()
    }

    /// `true` if the grid has no cells (never, by construction).
    pub fn is_empty(&self) -> bool {
        self.u.is_empty()
    }

    /// Linear index of `(i, j, k)`.
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    /// Cell-center position [m].
    pub fn center(&self, i: usize, j: usize, k: usize) -> Vec3 {
        self.origin
            + Vec3::new(
                (i as f64 + 0.5) * self.dx,
                (j as f64 + 0.5) * self.dx,
                (k as f64 + 0.5) * self.dx,
            )
    }

    /// Cell containing `p`, if inside the domain.
    pub fn cell_of(&self, p: Vec3) -> Option<(usize, usize, usize)> {
        let r = (p - self.origin) / self.dx;
        if r.x < 0.0 || r.y < 0.0 || r.z < 0.0 {
            return None;
        }
        let (i, j, k) = (
            libm_ops::floor(r.x) as usize,
            libm_ops::floor(r.y) as usize,
            libm_ops::floor(r.z) as usize,
        );
        (i < self.nx && j < self.ny && k < self.nz).then_some((i, j, k))
    }

    /// Cell volume [m³].
    pub fn cell_volume(&self) -> f64 {
        self.dx * self.dx * self.dx
    }

    /// Primitive state of cell `c`.
    pub fn prim(&self, c: usize) -> Prim3 {
        Prim3::from_cons(&self.u[c], self.gamma, self.rho_floor, self.p_floor)
    }

    /// Overpressure `p − p_ambient` [Pa] of cell `c` (0 for solid cells).
    pub fn overpressure(&self, c: usize) -> f64 {
        if self.solid[c] {
            0.0
        } else {
            self.prim(c).p - self.ambient.p
        }
    }

    /// Mark a cell solid (resets its state to ambient).
    pub fn set_solid(&mut self, c: usize, solid: bool) {
        self.solid[c] = solid;
        self.u[c] = self.ambient.to_cons(self.gamma);
    }

    /// Total mass [kg] and total energy [J] in fluid cells.
    pub fn totals(&self) -> (f64, f64) {
        let v = self.cell_volume();
        let mut m = 0.0;
        let mut e = 0.0;
        for (c, s) in self.u.iter().zip(&self.solid) {
            if !*s {
                m += c[0] * v;
                e += c[4] * v;
            }
        }
        (m, e)
    }

    /// Largest `Σ_d (|u_d| + a)` over fluid cells [m/s]. (A max reduction is
    /// order-independent, so the parallel path is bit-identical.)
    pub fn max_signal_sum(&self) -> f64 {
        let per_cell = map_cells(self.len(), self.nx * self.ny, |c| {
            if self.solid[c] {
                return 0.0;
            }
            let w = self.prim(c);
            let a = w.sound_speed(self.gamma);
            libm_ops::fabs(w.u[0]) + libm_ops::fabs(w.u[1]) + libm_ops::fabs(w.u[2]) + 3.0 * a
        });
        per_cell.into_iter().fold(0.0, f64::max)
    }

    /// Stable time step for CFL number `cfl` (≈ 0.4) [s].
    pub fn stable_dt(&self, cfl: f64) -> f64 {
        cfl * self.dx / self.max_signal_sum().max(1e-30)
    }

    fn dims(&self) -> [usize; 3] {
        [self.nx, self.ny, self.nz]
    }

    /// Neighbor primitive of cell `(ijk)` offset by `off` (±1, ±2) along `d`,
    /// using ghost rules: solid or wall → mirror of the nearest fluid cell
    /// on our side, outflow → copy of the boundary cell.
    fn neighbor(&self, prim: &[Prim3], ijk: [usize; 3], d: usize, off: isize) -> Prim3 {
        let n = self.dims()[d] as isize;
        let step = off.signum();
        let mut last = ijk;
        let mut cur = ijk;
        for _ in 0..off.unsigned_abs() {
            let next = cur[d] as isize + step;
            if next < 0 || next >= n {
                let face = if step < 0 { 2 * d } else { 2 * d + 1 };
                let w = prim[self.idx(last[0], last[1], last[2])];
                return match self.boundary[face] {
                    Boundary::Outflow => w,
                    Boundary::Wall => w.mirrored(d),
                };
            }
            let mut cand = cur;
            cand[d] = next as usize;
            if self.solid[self.idx(cand[0], cand[1], cand[2])] {
                return prim[self.idx(last[0], last[1], last[2])].mirrored(d);
            }
            last = cand;
            cur = cand;
        }
        prim[self.idx(cur[0], cur[1], cur[2])]
    }

    /// Flux through the `−d` face of fluid cell `ijk`.
    fn low_face_flux(&self, prim: &[Prim3], ijk: [usize; 3], d: usize) -> Cons {
        let g = self.gamma;
        let c = prim[self.idx(ijk[0], ijk[1], ijk[2])];
        let m1 = self.neighbor(prim, ijk, d, -1);
        // Left neighbor is a ghost (wall / solid / outflow)?
        let left_is_real = {
            let i = ijk[d] as isize - 1;
            if i < 0 {
                false
            } else {
                let mut cand = ijk;
                cand[d] = i as usize;
                !self.solid[self.idx(cand[0], cand[1], cand[2])]
            }
        };
        if !left_is_real {
            return hllc3(m1, c, d, g);
        }
        let mut lijk = ijk;
        lijk[d] -= 1;
        let m2 = self.neighbor(prim, lijk, d, -1);
        let p1 = self.neighbor(prim, ijk, d, 1);
        let jump = libm_ops::fabs(c.p - m1.p) / (c.p + m1.p);
        let (wl, wr) = if jump > self.first_order_jump {
            (m1, c)
        } else {
            (
                muscl(self.limiter, m2, m1, c, 0.5),
                muscl(self.limiter, m1, c, p1, -0.5),
            )
        };
        hllc3(wl, wr, d, g)
    }

    /// Flux through the `+d` face of fluid cell `ijk`, only used when the
    /// `+d` neighbor is a ghost (otherwise it is that neighbor's low face).
    fn high_ghost_flux(&self, prim: &[Prim3], ijk: [usize; 3], d: usize) -> Cons {
        let c = prim[self.idx(ijk[0], ijk[1], ijk[2])];
        let p1 = self.neighbor(prim, ijk, d, 1);
        hllc3(c, p1, d, self.gamma)
    }

    fn high_is_ghost(&self, ijk: [usize; 3], d: usize) -> bool {
        let i = ijk[d] + 1;
        if i >= self.dims()[d] {
            return true;
        }
        let mut cand = ijk;
        cand[d] = i;
        self.solid[self.idx(cand[0], cand[1], cand[2])]
    }

    /// Right-hand side `dU/dt` for state `u` (solid cells get zero).
    fn rhs(&self, u: &[Cons]) -> Vec<Cons> {
        let g = self.gamma;
        let n = self.len();
        let plane = self.nx * self.ny;
        let prim: Vec<Prim3> = map_cells(n, plane, |c| {
            Prim3::from_cons(&u[c], g, self.rho_floor, self.p_floor)
        });
        // Low-face fluxes for every fluid cell and direction.
        let low = |c: usize| -> [Cons; 3] {
            if self.solid[c] {
                return [[0.0; NVAR]; 3];
            }
            let ijk = [c % self.nx, (c / self.nx) % self.ny, c / plane];
            [
                self.low_face_flux(&prim, ijk, 0),
                self.low_face_flux(&prim, ijk, 1),
                self.low_face_flux(&prim, ijk, 2),
            ]
        };
        let lows: Vec<[Cons; 3]> = map_cells(n, plane, low);
        let inv_dx = 1.0 / self.dx;
        let gather = |c: usize| -> Cons {
            if self.solid[c] {
                return [0.0; NVAR];
            }
            let ijk = [c % self.nx, (c / self.nx) % self.ny, c / plane];
            let strides = [1, self.nx, plane];
            let mut out = [0.0; NVAR];
            for d in 0..3 {
                let hi = if self.high_is_ghost(ijk, d) {
                    self.high_ghost_flux(&prim, ijk, d)
                } else {
                    lows[c + strides[d]][d]
                };
                let lo = lows[c][d];
                for k in 0..NVAR {
                    out[k] -= (hi[k] - lo[k]) * inv_dx;
                }
            }
            out
        };
        map_cells(n, plane, gather)
    }

    fn floor_cell(&self, c: &mut Cons) {
        let w = Prim3::from_cons(c, self.gamma, self.rho_floor, self.p_floor);
        if c[0] < self.rho_floor || !c[0].is_finite() {
            *c = w.to_cons(self.gamma);
            return;
        }
        let ke = 0.5 * (c[1] * c[1] + c[2] * c[2] + c[3] * c[3]) / c[0];
        if !((self.gamma - 1.0) * (c[4] - ke) >= self.p_floor) {
            c[4] = self.p_floor / (self.gamma - 1.0) + ke;
        }
    }

    /// One SSP-RK2 step of size `dt` [s].
    pub fn step(&mut self, dt: f64) {
        let n = self.len();
        let plane = self.nx * self.ny;
        let k1 = self.rhs(&self.u);
        let u1: Vec<Cons> = map_cells(n, plane, |c| {
            let u0 = self.u[c];
            if self.solid[c] {
                return u0;
            }
            let mut v = [0.0; NVAR];
            for k in 0..NVAR {
                v[k] = u0[k] + dt * k1[c][k];
            }
            self.floor_cell(&mut v);
            v
        });
        drop(k1);
        let k2 = self.rhs(&u1);
        let u2: Vec<Cons> = map_cells(n, plane, |c| {
            let u0 = self.u[c];
            if self.solid[c] {
                return u0;
            }
            let mut v = [0.0; NVAR];
            for k in 0..NVAR {
                v[k] = 0.5 * u0[k] + 0.5 * (u1[c][k] + dt * k2[c][k]);
            }
            self.floor_cell(&mut v);
            v
        });
        self.u = u2;
        self.t += dt;
        self.steps += 1;
    }

    /// Step with the CFL time step; returns `dt` used.
    pub fn step_cfl(&mut self, cfl: f64) -> f64 {
        let dt = self.stable_dt(cfl);
        self.step(dt);
        dt
    }

    /// Advance to `t_end` [s] (last step clipped), at most `max_steps` steps.
    /// Calls `each(&grid, dt)` after every step. Returns steps taken.
    pub fn advance_to<F: FnMut(&Grid3d, f64)>(
        &mut self,
        t_end: f64,
        cfl: f64,
        max_steps: usize,
        mut each: F,
    ) -> usize {
        let mut n = 0;
        while self.t < t_end && n < max_steps {
            let mut dt = self.stable_dt(cfl);
            if !(dt.is_finite() && dt > 0.0) {
                break;
            }
            if self.t + dt > t_end {
                dt = t_end - self.t;
            }
            self.step(dt);
            each(self, dt);
            n += 1;
        }
        n
    }

    /// Trilinear overpressure [Pa] at `p`, using only fluid cells
    /// (weights renormalized; 0 if no fluid cell around).
    pub fn sample_overpressure(&self, p: Vec3) -> f64 {
        self.sample_with(p, |c| self.overpressure(c))
    }

    /// Trilinear primitive-derived sample of `f(cell)` at `p` over fluid cells.
    pub fn sample_with<F: Fn(usize) -> f64>(&self, p: Vec3, f: F) -> f64 {
        let r = (p - self.origin) / self.dx - Vec3::new(0.5, 0.5, 0.5);
        let fl = [
            libm_ops::floor(r.x),
            libm_ops::floor(r.y),
            libm_ops::floor(r.z),
        ];
        let fr = [r.x - fl[0], r.y - fl[1], r.z - fl[2]];
        let dims = self.dims();
        let mut acc = 0.0;
        let mut wsum = 0.0;
        for corner in 0..8usize {
            let mut ijk = [0usize; 3];
            let mut w = 1.0;
            for d in 0..3 {
                let bit = (corner >> d) & 1;
                let i = fl[d] as isize + bit as isize;
                let i = i.clamp(0, dims[d] as isize - 1);
                ijk[d] = i as usize;
                w *= if bit == 1 { fr[d] } else { 1.0 - fr[d] };
            }
            let c = self.idx(ijk[0], ijk[1], ijk[2]);
            if self.solid[c] || w <= 0.0 {
                continue;
            }
            acc += w * f(c);
            wsum += w;
        }
        if wsum > 0.0 {
            acc / wsum
        } else {
            0.0
        }
    }
}

/// Apply `f` to every cell index, serial or (feature `parallel`) over z-slabs.
/// Output order is the cell order either way, so results are bit-identical.
#[cfg(not(feature = "parallel"))]
fn map_cells<T, F: Fn(usize) -> T + Sync>(n: usize, _plane: usize, f: F) -> Vec<T> {
    (0..n).map(f).collect()
}

#[cfg(feature = "parallel")]
fn map_cells<T: Send, F: Fn(usize) -> T + Sync>(n: usize, plane: usize, f: F) -> Vec<T> {
    use rayon::prelude::*;
    let plane = plane.max(1);
    let slabs: Vec<Vec<T>> = (0..n.div_ceil(plane))
        .into_par_iter()
        .map(|k| (k * plane..((k + 1) * plane).min(n)).map(&f).collect())
        .collect();
    slabs.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hllc3_reduces_to_physical_flux_for_equal_states() {
        let w = Prim3 {
            rho: 1.2,
            u: [30.0, -5.0, 2.0],
            p: 1.1e5,
        };
        for d in 0..3 {
            let f = hllc3(w, w, d, 1.4);
            let e = physical_flux(w, d, 1.4);
            for k in 0..NVAR {
                assert!(
                    (f[k] - e[k]).abs() <= 1e-9 * e[k].abs().max(1.0),
                    "d={d} k={k}"
                );
            }
        }
    }

    #[test]
    fn wall_flux_is_pure_pressure() {
        let w = Prim3 {
            rho: 1.2,
            u: [50.0, 3.0, -1.0],
            p: 2e5,
        };
        let f = hllc3(w, w.mirrored(0), 0, 1.4);
        assert!(f[0].abs() < 1e-9 && f[2].abs() < 1e-9 && f[3].abs() < 1e-9 && f[4].abs() < 1e-6);
        assert!(f[1] > w.p);
    }

    #[test]
    fn quiescent_grid_stays_quiescent() {
        let mut g = Grid3d::air(6, 5, 4, 0.1, Vec3::zero());
        g.boundary = [Boundary::Wall; 6];
        let c = g.idx(2, 2, 1);
        g.set_solid(c, true);
        let before = g.u.clone();
        for _ in 0..5 {
            g.step_cfl(0.4);
        }
        for (a, b) in g.u.iter().zip(&before) {
            for k in 0..NVAR {
                assert!((a[k] - b[k]).abs() <= 1e-9 * b[k].abs().max(1.0));
            }
        }
    }
}
