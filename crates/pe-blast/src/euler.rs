//! 1-D compressible Euler finite-volume solver (HLLC + MUSCL + SSP-RK2).
//!
//! # References
//! - Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid
//!   Dynamics*, 3rd ed. Springer — HLLC flux, exact Riemann solver, MUSCL.
//! - Gottlieb & Shu (1998). Total variation diminishing Runge–Kutta schemes.
//!   SSP-RK2 (Heun).
//! - Sod, G. A. (1978). A survey of several finite difference methods for
//!   systems of nonlinear hyperbolic conservation laws. *J. Comput. Phys.*
//!
//! Geometry: planar (`ν=1`) or spherical (`ν=3`) source terms. SI: `ρ` [kg/m³],
//! `u` [m/s], `p` [Pa], `x`/`r` [m], `t` [s]. Deterministic: portable `libm`,
//! no FMA, fixed cell order.

use pe_math::libm_ops;

/// Ideal-gas γ (default air).
pub const GAMMA_DEFAULT: f64 = 1.4;

/// Conserved primitive cell state.
#[derive(Clone, Copy, Debug, Default)]
pub struct Prim {
    /// Density [kg/m³].
    pub rho: f64,
    /// Velocity [m/s] (radial for spherical).
    pub u: f64,
    /// Pressure [Pa].
    pub p: f64,
}

impl Prim {
    /// Conserved `(ρ, ρu, E)` with `E = p/(γ−1) + ½ ρ u²` [J/m³].
    pub fn to_cons(self, gamma: f64) -> [f64; 3] {
        let e = self.p / (gamma - 1.0) + 0.5 * self.rho * self.u * self.u;
        [self.rho, self.rho * self.u, e]
    }

    /// From conserved; returns `None` if non-physical.
    pub fn from_cons(u: [f64; 3], gamma: f64) -> Option<Self> {
        let rho = u[0];
        if rho <= 0.0 {
            return None;
        }
        let v = u[1] / rho;
        let kin = 0.5 * rho * v * v;
        let p = (gamma - 1.0) * (u[2] - kin);
        if p <= 0.0 {
            return None;
        }
        Some(Self { rho, u: v, p })
    }

    /// Sound speed [m/s].
    pub fn sound_speed(self, gamma: f64) -> f64 {
        libm_ops::sqrt(gamma * self.p / self.rho.max(1e-30))
    }
}

/// Mesh geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Geometry1d {
    /// Planar slab.
    Planar,
    /// Spherical (radial), geometric source terms.
    Spherical,
}

/// 1-D Euler mesh / solver state.
#[derive(Clone, Debug)]
pub struct Euler1d {
    /// Cell-center coordinates [m].
    pub x: Vec<f64>,
    /// Cell widths [m].
    pub dx: Vec<f64>,
    /// Conserved variables per cell.
    pub u: Vec<[f64; 3]>,
    /// γ.
    pub gamma: f64,
    /// Geometry.
    pub geometry: Geometry1d,
    /// Simulation time [s].
    pub t: f64,
}

impl Euler1d {
    /// Uniform mesh on `[x0, x1]` with `n` cells, constant initial state.
    pub fn uniform(
        x0: f64,
        x1: f64,
        n: usize,
        init: Prim,
        gamma: f64,
        geometry: Geometry1d,
    ) -> Self {
        let n = n.max(1);
        let dx = (x1 - x0) / n as f64;
        let mut x = Vec::with_capacity(n);
        let mut dxs = Vec::with_capacity(n);
        let mut u = Vec::with_capacity(n);
        let cons = init.to_cons(gamma);
        for i in 0..n {
            x.push(x0 + (i as f64 + 0.5) * dx);
            dxs.push(dx);
            u.push(cons);
        }
        Self {
            x,
            dx: dxs,
            u,
            gamma,
            geometry,
            t: 0.0,
        }
    }

    /// Sod shock-tube IC on `[0,1]`: left `(1,0,1)`, right `(0.125,0,0.1)`.
    pub fn sod(n: usize, gamma: f64) -> Self {
        let mut s = Self::uniform(
            0.0,
            1.0,
            n,
            Prim {
                rho: 1.0,
                u: 0.0,
                p: 1.0,
            },
            gamma,
            Geometry1d::Planar,
        );
        for i in 0..n {
            if s.x[i] >= 0.5 {
                s.u[i] = Prim {
                    rho: 0.125,
                    u: 0.0,
                    p: 0.1,
                }
                .to_cons(gamma);
            }
        }
        s
    }

    /// Number of cells.
    pub fn n(&self) -> usize {
        self.u.len()
    }

    /// Maximum wave speed for CFL [m/s].
    pub fn max_wave_speed(&self) -> f64 {
        let mut amax: f64 = 0.0;
        for ui in &self.u {
            if let Some(p) = Prim::from_cons(*ui, self.gamma) {
                let s = libm_ops::fabs(p.u) + p.sound_speed(self.gamma);
                if s > amax {
                    amax = s;
                }
            }
        }
        amax
    }

    /// Advance with SSP-RK2 under CFL `cfl` (dimensionless).
    pub fn step_cfl(&mut self, cfl: f64) {
        let a = self.max_wave_speed().max(1e-30);
        let dx_min = self.dx.iter().cloned().fold(f64::INFINITY, f64::min);
        let dt = cfl * dx_min / a;
        self.step(dt);
    }

    /// One SSP-RK2 step of size `dt` [s].
    pub fn step(&mut self, dt: f64) {
        let n = self.n();
        let u0 = self.u.clone();
        let k1 = self.rhs(&u0);
        let mut u1 = vec![[0.0; 3]; n];
        for i in 0..n {
            for k in 0..3 {
                u1[i][k] = u0[i][k] + dt * k1[i][k];
            }
            floor_cons(&mut u1[i], self.gamma);
        }
        let k2 = self.rhs(&u1);
        for i in 0..n {
            for k in 0..3 {
                self.u[i][k] = 0.5 * u0[i][k] + 0.5 * (u1[i][k] + dt * k2[i][k]);
            }
            floor_cons(&mut self.u[i], self.gamma);
        }
        self.t += dt;
    }

    /// Run until `t_end` [s] with fixed CFL (caps at 1_000_000 substeps).
    pub fn run_cfl(&mut self, t_end: f64, cfl: f64) {
        let mut steps = 0usize;
        while self.t < t_end - 1e-15 {
            let a = self.max_wave_speed().max(1e-30);
            let dx_min = self.dx.iter().cloned().fold(f64::INFINITY, f64::min);
            let mut dt = cfl * dx_min / a;
            if !(dt.is_finite() && dt > 0.0) {
                break;
            }
            if self.t + dt > t_end {
                dt = t_end - self.t;
            }
            self.step(dt);
            steps += 1;
            if steps > 1_000_000 {
                break;
            }
        }
    }

    fn rhs(&self, u: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let n = u.len();
        let g = self.gamma;
        // Reconstruct interface primitives with MUSCL + minmod.
        let mut prim = vec![Prim::default(); n];
        for i in 0..n {
            prim[i] = Prim::from_cons(u[i], g).unwrap_or(Prim {
                rho: 1e-12,
                u: 0.0,
                p: 1e-12,
            });
        }
        let mut flux = vec![[0.0; 3]; n + 1];
        for i in 0..=n {
            let (pl, pr) = if i == 0 {
                (prim[0], prim[0])
            } else if i == n {
                (prim[n - 1], prim[n - 1])
            } else {
                // First-order Godunov faces + optional MUSCL. Use MUSCL away from
                // large jumps; fall back to piecewise constant near shocks for
                // robustness (still uses minmod MUSCL when slopes are mild).
                let left_m = muscl_interface(&prim, i - 1, true);
                let right_m = muscl_interface(&prim, i, false);
                let jump =
                    libm_ops::fabs(prim[i].p - prim[i - 1].p) / (prim[i].p + prim[i - 1].p + 1e-30);
                if jump > 0.2 {
                    (prim[i - 1], prim[i])
                } else {
                    (left_m, right_m)
                }
            };
            flux[i] = hllc(pl, pr, g);
            for f in &mut flux[i] {
                if !f.is_finite() {
                    *f = 0.0;
                }
            }
        }
        let mut out = vec![[0.0; 3]; n];
        for i in 0..n {
            let dx = self.dx[i];
            for k in 0..3 {
                out[i][k] = -(flux[i + 1][k] - flux[i][k]) / dx;
            }
            // Spherical geometric source: −(ν−1)/r · F_geom
            if self.geometry == Geometry1d::Spherical {
                let r = self.x[i].max(1e-12);
                let p = prim[i];
                let e = p.p / (g - 1.0) + 0.5 * p.rho * p.u * p.u;
                // ν−1 = 2: sources on (ρu, E) from divergence in spherical coords
                out[i][0] -= 2.0 / r * (p.rho * p.u);
                out[i][1] -= 2.0 / r * (p.rho * p.u * p.u);
                out[i][2] -= 2.0 / r * (p.u * (e + p.p));
            }
        }
        out
    }
}

fn floor_cons(u: &mut [f64; 3], gamma: f64) {
    if !u[0].is_finite() || u[0] < 1e-12 {
        u[0] = 1e-12;
    }
    let v = u[1] / u[0];
    let mut p = (gamma - 1.0) * (u[2] - 0.5 * u[0] * v * v);
    if !p.is_finite() || p < 1e-12 {
        p = 1e-12;
        u[2] = p / (gamma - 1.0) + 0.5 * u[0] * v * v;
    }
}

fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else if libm_ops::fabs(a) < libm_ops::fabs(b) {
        a
    } else {
        b
    }
}

fn muscl_interface(prim: &[Prim], i: usize, to_right: bool) -> Prim {
    let n = prim.len();
    let im = if i == 0 { 0 } else { i - 1 };
    let ip = if i + 1 >= n { n - 1 } else { i + 1 };
    let c = prim[i];
    let slope = |vl: f64, v: f64, vr: f64| minmod(v - vl, vr - v);
    let dr = slope(prim[im].rho, c.rho, prim[ip].rho);
    let du = slope(prim[im].u, c.u, prim[ip].u);
    let dp = slope(prim[im].p, c.p, prim[ip].p);
    let s = if to_right { 0.5 } else { -0.5 };
    Prim {
        rho: (c.rho + s * dr).max(1e-14),
        u: c.u + s * du,
        p: (c.p + s * dp).max(1e-14),
    }
}

/// HLLC Riemann flux (Toro).
pub fn hllc(wl: Prim, wr: Prim, gamma: f64) -> [f64; 3] {
    let al = wl.sound_speed(gamma);
    let ar = wr.sound_speed(gamma);
    let sl = (wl.u - al).min(wr.u - ar);
    let sr = (wl.u + al).max(wr.u + ar);
    let ul = wl.to_cons(gamma);
    let ur = wr.to_cons(gamma);
    let fl = [
        wl.rho * wl.u,
        wl.rho * wl.u * wl.u + wl.p,
        wl.u * (ul[2] + wl.p),
    ];
    let fr = [
        wr.rho * wr.u,
        wr.rho * wr.u * wr.u + wr.p,
        wr.u * (ur[2] + wr.p),
    ];
    if sl >= 0.0 {
        return fl;
    }
    if sr <= 0.0 {
        return fr;
    }
    // Star speed
    let num = wr.p - wl.p + wl.rho * wl.u * (sl - wl.u) - wr.rho * wr.u * (sr - wr.u);
    let den = wl.rho * (sl - wl.u) - wr.rho * (sr - wr.u);
    let den = if libm_ops::fabs(den) < 1e-30 {
        libm_ops::copysign(1e-30, den)
    } else {
        den
    };
    let sm = num / den;
    let star = |w: Prim, s: f64, uu: [f64; 3]| -> [f64; 3] {
        let mut dsm = s - sm;
        if libm_ops::fabs(dsm) < 1e-30 {
            dsm = libm_ops::copysign(1e-30, dsm);
        }
        let mut dsu = s - w.u;
        if libm_ops::fabs(dsu) < 1e-30 {
            dsu = libm_ops::copysign(1e-30, dsu);
        }
        let factor = w.rho * (s - w.u) / dsm;
        let e_star = uu[2] / w.rho + (sm - w.u) * (sm + w.p / (w.rho * dsu));
        [factor, factor * sm, factor * e_star]
    };
    let uls = star(wl, sl, ul);
    let urs = star(wr, sr, ur);
    if sm >= 0.0 {
        [
            fl[0] + sl * (uls[0] - ul[0]),
            fl[1] + sl * (uls[1] - ul[1]),
            fl[2] + sl * (uls[2] - ul[2]),
        ]
    } else {
        [
            fr[0] + sr * (urs[0] - ur[0]),
            fr[1] + sr * (urs[1] - ur[1]),
            fr[2] + sr * (urs[2] - ur[2]),
        ]
    }
}

/// Exact Riemann solution for ideal gas (Toro), for Sod validation.
#[derive(Clone, Copy, Debug)]
pub struct ExactRiemann {
    /// Left primitive.
    pub left: Prim,
    /// Right primitive.
    pub right: Prim,
    /// γ.
    pub gamma: f64,
}

impl ExactRiemann {
    /// Sample exact solution at `x` [m] and `t` [s] with discontinuity at `x0`.
    pub fn sample(&self, x: f64, t: f64, x0: f64) -> Prim {
        let g = self.gamma;
        let wl = self.left;
        let wr = self.right;
        let al = wl.sound_speed(g);
        let ar = wr.sound_speed(g);
        // Star pressure via Toro adaptive Newton (2 rarefactions guess then iterate).
        let mut p_star = 0.5 * (wl.p + wr.p);
        p_star = p_star.max(1e-12);
        for _ in 0..40 {
            let (fl, fpl) = pressure_function(p_star, wl, al, g);
            let (fr, fpr) = pressure_function(p_star, wr, ar, g);
            let f = fl + fr + wr.u - wl.u;
            let fp = fpl + fpr;
            let dp = -f / fp.max(1e-30);
            p_star = (p_star + dp).max(1e-12);
            if libm_ops::fabs(dp) < 1e-12 * p_star {
                break;
            }
        }
        let (_, _) = pressure_function(p_star, wl, al, g);
        let ul_star_du = {
            let (fl, _) = pressure_function(p_star, wl, al, g);
            let (fr, _) = pressure_function(p_star, wr, ar, g);
            0.5 * (wl.u + wr.u) + 0.5 * (fr - fl)
        };
        let u_star = ul_star_du;
        let s = if t <= 0.0 { 0.0 } else { (x - x0) / t };
        if s <= u_star {
            // Left wave
            sample_side(s, wl, al, p_star, u_star, g, true)
        } else {
            sample_side(s, wr, ar, p_star, u_star, g, false)
        }
    }
}

fn pressure_function(p: f64, w: Prim, a: f64, g: f64) -> (f64, f64) {
    let gm1 = g - 1.0;
    let gp1 = g + 1.0;
    if p <= w.p {
        // Rarefaction
        let f = 2.0 * a / gm1 * (libm_ops::pow(p / w.p, gm1 / (2.0 * g)) - 1.0);
        // d/dp of 2a/(γ−1) [(p/p0)^((γ−1)/2γ) − 1]
        let fp = (1.0 / w.p) * libm_ops::pow(p / w.p, -0.5 * (g + 1.0) / g) * a / g;
        (f, fp)
    } else {
        // Shock
        let a_p = 2.0 / (gp1 * w.rho);
        let b_p = gm1 / gp1 * w.p;
        let f = (p - w.p) * libm_ops::sqrt(a_p / (p + b_p));
        let fp = libm_ops::sqrt(a_p / (p + b_p)) * (1.0 - 0.5 * (p - w.p) / (p + b_p));
        (f, fp)
    }
}

fn sample_side(s: f64, w: Prim, a: f64, p_star: f64, u_star: f64, g: f64, left: bool) -> Prim {
    let gm1 = g - 1.0;
    let gp1 = g + 1.0;
    if p_star <= w.p {
        // Rarefaction
        let a_star = a * libm_ops::pow(p_star / w.p, gm1 / (2.0 * g));
        if left {
            let sh_head = w.u - a;
            let sh_tail = u_star - a_star;
            if s <= sh_head {
                return w;
            }
            if s >= sh_tail {
                return Prim {
                    rho: w.rho * libm_ops::pow(p_star / w.p, 1.0 / g),
                    u: u_star,
                    p: p_star,
                };
            }
            // Fan
            let u = (2.0 * a + gm1 * w.u + 2.0 * s) / (gp1);
            let a_fan = (2.0 * a + gm1 * (w.u - s)) / (gp1);
            let rho = w.rho * libm_ops::pow(a_fan / a, 2.0 / gm1);
            let p = w.p * libm_ops::pow(a_fan / a, 2.0 * g / gm1);
            Prim { rho, u, p }
        } else {
            let sh_head = w.u + a;
            let sh_tail = u_star + a_star;
            if s >= sh_head {
                return w;
            }
            if s <= sh_tail {
                return Prim {
                    rho: w.rho * libm_ops::pow(p_star / w.p, 1.0 / g),
                    u: u_star,
                    p: p_star,
                };
            }
            let u = (-2.0 * a + gm1 * w.u + 2.0 * s) / (gp1);
            let a_fan = (2.0 * a - gm1 * (w.u - s)) / (gp1);
            let rho = w.rho * libm_ops::pow(a_fan / a, 2.0 / gm1);
            let p = w.p * libm_ops::pow(a_fan / a, 2.0 * g / gm1);
            Prim { rho, u, p }
        }
    } else {
        // Shock
        let ratio = p_star / w.p;
        let rho_star = w.rho * (ratio * gp1 / gm1 + 1.0) / (ratio + gp1 / gm1);
        if left {
            let s_shock = w.u - a * libm_ops::sqrt((gp1 / (2.0 * g)) * ratio + gm1 / (2.0 * g));
            if s <= s_shock {
                w
            } else {
                Prim {
                    rho: rho_star,
                    u: u_star,
                    p: p_star,
                }
            }
        } else {
            let s_shock = w.u + a * libm_ops::sqrt((gp1 / (2.0 * g)) * ratio + gm1 / (2.0 * g));
            if s >= s_shock {
                w
            } else {
                Prim {
                    rho: rho_star,
                    u: u_star,
                    p: p_star,
                }
            }
        }
    }
}

/// L1 density error vs exact Riemann on Sod at time `t`.
pub fn sod_l1_density_error(solver: &Euler1d, t: f64) -> f64 {
    let exact = ExactRiemann {
        left: Prim {
            rho: 1.0,
            u: 0.0,
            p: 1.0,
        },
        right: Prim {
            rho: 0.125,
            u: 0.0,
            p: 0.1,
        },
        gamma: solver.gamma,
    };
    let mut err = 0.0;
    let mut vol = 0.0;
    for i in 0..solver.n() {
        let p = Prim::from_cons(solver.u[i], solver.gamma).unwrap_or_default();
        let e = exact.sample(solver.x[i], t, 0.5);
        err += libm_ops::fabs(p.rho - e.rho) * solver.dx[i];
        vol += solver.dx[i];
    }
    err / vol.max(1e-30)
}
