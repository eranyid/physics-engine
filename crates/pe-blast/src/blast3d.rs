//! Scene-level 3-D blast simulation: voxel walls, detonation, pressure probes,
//! rigid-body loads and image output, on top of [`crate::euler3d::Grid3d`].
//!
//! ```text
//! let mut scene = BlastScene::new(min, max, 0.1);
//! scene.fill_ground(0.0);
//! scene.fill_box(wall_center, wall_half, IDENTITY);
//! scene.detonate(origin, 1.0 /* kg TNT-eq */);
//! let gauge = scene.add_probe(point);
//! scene.run_until(0.02, |s, dt| s.apply_to_world(&mut world, &faces, dt));
//! let rec = &scene.probes[gauge];   // peak, impulse, arrival
//! ```
//!
//! The charge is described only by its **TNT-equivalent mass**; the energy
//! released is `W_eq · 4.184 MJ/kg` (the standard TNT-equivalence
//! convention) deposited as a high-pressure ball. Within ~3 ball radii the
//! near field is not resolved; use the analytic models there.

use pe_core::World;
use pe_math::libm_ops;
use pe_math::Vec3;

use crate::euler::{Euler1d, Geometry1d, Prim};
use crate::euler3d::{Boundary, Grid3d, Prim3};
use crate::loading::{BlastSource, ExposedFace};

/// TNT-equivalence energy [J/kg].
pub const TNT_ENERGY_J_PER_KG: f64 = 4.184e6;

/// Default CFL number.
pub const DEFAULT_CFL: f64 = 0.4;

/// TNT density [kg/m³] used to size the source sphere.
pub const TNT_DENSITY: f64 = 1600.0;

/// Detonation options for [`BlastScene::detonate_with`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detonation {
    /// Burst point [m].
    pub origin: Vec3,
    /// TNT-equivalent mass [kg].
    pub w_eq_kg: f64,
    /// Charge lies on a flat rigid ground plane through `origin` (hemispherical
    /// burst): the 1-D run uses `2 W` (mirror image) and ground below is ignored
    /// for the clearance.
    pub surface_burst: bool,
    /// Remap radius [m]; `None` = clearance − 2 cells. `Some(0.0)` forces a
    /// direct 3-D deposit.
    pub remap_radius_m: Option<f64>,
    /// Ball radius [m] for a direct deposit (`None` = `0.5 W^{1/3}`).
    pub direct_radius_m: Option<f64>,
    /// 1-D cells per charge radius.
    pub cells_per_charge_radius: f64,
    /// Cap on 1-D cells (coarsens `dx₁` for very large remap radii).
    pub max_1d_cells: usize,
}

impl Detonation {
    /// Free-air burst with automatic remap.
    pub fn new(origin: Vec3, w_eq_kg: f64) -> Self {
        Self {
            origin,
            w_eq_kg,
            surface_burst: false,
            remap_radius_m: None,
            direct_radius_m: None,
            cells_per_charge_radius: 8.0,
            max_1d_cells: 20_000,
        }
    }

    /// Mark as a surface burst.
    pub fn surface(mut self) -> Self {
        self.surface_burst = true;
        self
    }

    /// Fix the remap radius [m].
    pub fn remap_at(mut self, r: f64) -> Self {
        self.remap_radius_m = Some(r);
        self
    }
}

/// What [`BlastScene::detonate_with`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DetonationReport {
    /// `true` if the 1-D → 3-D remap was used.
    pub remapped: bool,
    /// Remap radius [m] (0 if direct).
    pub remap_radius_m: f64,
    /// Time since detonation at the remap [s].
    pub remap_time_s: f64,
    /// 3-D cells written.
    pub cells: usize,
    /// Energy added to the 3-D grid [J] (above ambient).
    pub energy_mapped_j: f64,
}

/// Anything that can report overpressure at a point and time.
pub trait PressureField {
    /// Overpressure `p − p0` [Pa] at world point `x` and world time `t` [s].
    fn overpressure_at(&self, x: Vec3, t: f64) -> f64;
}

impl PressureField for BlastSource {
    fn overpressure_at(&self, x: Vec3, t: f64) -> f64 {
        let face = ExposedFace {
            body: 0,
            center: x,
            normal: (self.origin - x).normalize(),
            area: 0.0,
        };
        crate::loading::face_pressure(self, &face, t)
    }
}

/// Pressure gauge record.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeRecord {
    /// Position [m].
    pub position: Vec3,
    /// Peak overpressure [Pa].
    pub peak_overpressure: f64,
    /// Time of peak [s].
    pub peak_time: f64,
    /// Positive-phase impulse `∫ max(p − p0, 0) dt` [Pa·s] (trapezoid).
    pub positive_impulse: f64,
    /// Negative-phase impulse `∫ min(p − p0, 0) dt` [Pa·s] (≤ 0).
    pub negative_impulse: f64,
    /// First time overpressure exceeded the arrival threshold [s].
    pub arrival_time: Option<f64>,
    /// `(t, overpressure)` history, if enabled.
    pub history: Vec<(f64, f64)>,
    last: f64,
}

impl ProbeRecord {
    fn record(&mut self, t: f64, dt: f64, p: f64, threshold: f64, history: bool) {
        self.positive_impulse += 0.5 * (self.last.max(0.0) + p.max(0.0)) * dt;
        self.negative_impulse += 0.5 * (self.last.min(0.0) + p.min(0.0)) * dt;
        if p > self.peak_overpressure {
            self.peak_overpressure = p;
            self.peak_time = t;
        }
        if self.arrival_time.is_none() && p >= threshold {
            self.arrival_time = Some(t);
        }
        if history {
            self.history.push((t, p));
        }
        self.last = p;
    }

    fn new(position: Vec3) -> Self {
        Self {
            position,
            peak_overpressure: 0.0,
            peak_time: 0.0,
            positive_impulse: 0.0,
            negative_impulse: 0.0,
            arrival_time: None,
            history: Vec::new(),
            last: 0.0,
        }
    }
}

/// A 3-D blast scene.
#[derive(Clone, Debug)]
pub struct BlastScene {
    /// The flow solver.
    pub grid: Grid3d,
    /// Pressure gauges.
    pub probes: Vec<ProbeRecord>,
    /// Record full probe histories.
    pub record_history: bool,
    /// Overpressure that counts as shock arrival [Pa].
    pub arrival_threshold_pa: f64,
    /// CFL number.
    pub cfl: f64,
    /// Energy deposited so far [J].
    pub energy_deposited_j: f64,
}

impl BlastScene {
    /// Air-filled box `[min, max]` with cubic cells of size `dx` [m]; outflow on
    /// every side (call [`fill_ground`](Self::fill_ground) or set
    /// `grid.boundary` for walls).
    pub fn new(min: Vec3, max: Vec3, dx: f64) -> Self {
        let ext = max - min;
        let n = |e: f64| (libm_ops::floor(e / dx + 0.5) as usize).max(1);
        Self {
            grid: Grid3d::air(n(ext.x), n(ext.y), n(ext.z), dx, min),
            probes: Vec::new(),
            record_history: false,
            arrival_threshold_pa: 1000.0,
            cfl: DEFAULT_CFL,
            energy_deposited_j: 0.0,
        }
    }

    /// Wrap an existing grid.
    pub fn from_grid(grid: Grid3d) -> Self {
        Self {
            grid,
            probes: Vec::new(),
            record_history: false,
            arrival_threshold_pa: 1000.0,
            cfl: DEFAULT_CFL,
            energy_deposited_j: 0.0,
        }
    }

    /// Simulation time [s].
    pub fn t(&self) -> f64 {
        self.grid.t
    }

    fn fill_where<F: Fn(Vec3) -> bool>(&mut self, inside: F) -> usize {
        let g = &self.grid;
        let mut cells = Vec::new();
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..g.nx {
                    if inside(g.center(i, j, k)) {
                        cells.push(g.idx(i, j, k));
                    }
                }
            }
        }
        for &c in &cells {
            self.grid.set_solid(c, true);
        }
        cells.len()
    }

    /// Make every cell whose center is below `z` solid, and the bottom
    /// boundary a wall. Returns cells filled.
    pub fn fill_ground(&mut self, z: f64) -> usize {
        self.grid.boundary[4] = Boundary::Wall;
        self.fill_where(|c| c.z < z)
    }

    /// Solid oriented box (center, half extents, rotation row-major).
    pub fn fill_box(&mut self, center: Vec3, half: Vec3, rot: [[f64; 3]; 3]) -> usize {
        self.fill_where(|c| {
            let d = c - center;
            // Local = Rᵀ d.
            let l = Vec3::new(
                rot[0][0] * d.x + rot[1][0] * d.y + rot[2][0] * d.z,
                rot[0][1] * d.x + rot[1][1] * d.y + rot[2][1] * d.z,
                rot[0][2] * d.x + rot[1][2] * d.y + rot[2][2] * d.z,
            );
            libm_ops::fabs(l.x) <= half.x
                && libm_ops::fabs(l.y) <= half.y
                && libm_ops::fabs(l.z) <= half.z
        })
    }

    /// Solid axis-aligned box from corners.
    pub fn fill_aabb(&mut self, min: Vec3, max: Vec3) -> usize {
        self.fill_where(|c| {
            c.x >= min.x
                && c.x <= max.x
                && c.y >= min.y
                && c.y <= max.y
                && c.z >= min.z
                && c.z <= max.z
        })
    }

    /// Solid sphere.
    pub fn fill_sphere(&mut self, center: Vec3, radius: f64) -> usize {
        self.fill_where(|c| (c - center).norm_squared() <= radius * radius)
    }

    /// Clear (make fluid) an axis-aligned box — doors, windows, breaches.
    pub fn clear_aabb(&mut self, min: Vec3, max: Vec3) -> usize {
        let g = &self.grid;
        let mut cells = Vec::new();
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let c = g.center(i, j, k);
                    if c.x >= min.x
                        && c.x <= max.x
                        && c.y >= min.y
                        && c.y <= max.y
                        && c.z >= min.z
                        && c.z <= max.z
                    {
                        cells.push(g.idx(i, j, k));
                    }
                }
            }
        }
        for &c in &cells {
            self.grid.set_solid(c, false);
        }
        cells.len()
    }

    /// Deposit `energy_j` [J] as a uniform overpressure ball of radius
    /// `radius_m` (raised to at least 1.5 cells). The discrete deposited energy
    /// equals `energy_j` exactly. Returns the number of cells used.
    pub fn deposit_energy(&mut self, center: Vec3, energy_j: f64, radius_m: f64) -> usize {
        let g = &self.grid;
        let r = radius_m.max(1.5 * g.dx);
        let mut cells = Vec::new();
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let c = g.idx(i, j, k);
                    if !g.solid[c] && (g.center(i, j, k) - center).norm_squared() <= r * r {
                        cells.push(c);
                    }
                }
            }
        }
        if cells.is_empty() {
            // Fall back to the containing (or nearest fluid) cell.
            if let Some((i, j, k)) = g.cell_of(center) {
                let c = g.idx(i, j, k);
                if !g.solid[c] {
                    cells.push(c);
                }
            }
        }
        if cells.is_empty() || energy_j.is_nan() || energy_j <= 0.0 {
            return 0;
        }
        let de = energy_j / (cells.len() as f64 * g.cell_volume());
        for &c in &cells {
            self.grid.u[c][4] += de;
        }
        self.energy_deposited_j += energy_j;
        cells.len()
    }

    /// Detonate `w_eq_kg` kg TNT-equivalent at `origin` with default options
    /// (free-air burst, automatic 1-D → 3-D remap). Returns the report.
    pub fn detonate(&mut self, origin: Vec3, w_eq_kg: f64) -> DetonationReport {
        self.detonate_with(&Detonation::new(origin, w_eq_kg))
    }

    /// Detonate with explicit options.
    ///
    /// **Remap** (default): the early, free-field expansion is solved on a fine
    /// 1-D spherical grid ([`crate::euler::Euler1d`], ~8 cells per charge
    /// radius) until the shock reaches the remap radius, then the radial
    /// solution is written into the 3-D grid (`u = u_r r̂`). This keeps the
    /// near-field peak that a coarse 3-D grid cannot resolve. The remap radius
    /// defaults to the clearance to the nearest solid voxel or wall boundary
    /// minus two cells. Probes inside the remap radius record the 1-D history.
    ///
    /// The source is a sphere of TNT density (1600 kg/m³) holding the
    /// TNT-equivalent energy (γ = 1.4 throughout — an engineering source, not
    /// detonation-product chemistry).
    ///
    /// When the clearance is too small (< 3 cells) the energy is deposited
    /// directly into the 3-D grid ([`deposit_energy`](Self::deposit_energy)).
    pub fn detonate_with(&mut self, d: &Detonation) -> DetonationReport {
        let w = d.w_eq_kg.max(0.0);
        let w_eff = if d.surface_burst { 2.0 * w } else { w };
        let energy = w_eff * TNT_ENERGY_J_PER_KG;
        let r_charge = libm_ops::cbrt(3.0 * w_eff / (4.0 * core::f64::consts::PI * TNT_DENSITY));
        let dx = self.grid.dx;
        let r_map = match d.remap_radius_m {
            Some(r) => r,
            None => self.clearance(d.origin, d.surface_burst) - 2.0 * dx,
        };
        if w <= 0.0 {
            return DetonationReport::default();
        }
        if r_map < 3.0 * dx || r_map <= 2.0 * r_charge {
            let r_ball = d.direct_radius_m.unwrap_or(0.5 * libm_ops::cbrt(w));
            let cells = self.deposit_energy(d.origin, w * TNT_ENERGY_J_PER_KG, r_ball);
            return DetonationReport {
                remapped: false,
                remap_radius_m: 0.0,
                remap_time_s: 0.0,
                cells,
                energy_mapped_j: w * TNT_ENERGY_J_PER_KG,
            };
        }
        // --- 1-D spherical run --------------------------------------------
        let amb = self.grid.ambient;
        let g = self.grid.gamma;
        let mut dx1 = r_charge / d.cells_per_charge_radius.max(2.0);
        let mut n1 = libm_ops::floor((r_map + 6.0 * dx1) / dx1) as usize + 1;
        if n1 > d.max_1d_cells {
            dx1 = (r_map + 6.0 * dx1) / d.max_1d_cells as f64;
            n1 = d.max_1d_cells;
        }
        let r_end = n1 as f64 * dx1;
        let mut s1 = Euler1d::uniform(
            0.0,
            r_end,
            n1,
            Prim {
                rho: amb.rho,
                u: 0.0,
                p: amb.p,
            },
            g,
            Geometry1d::Spherical,
        );
        let mut vol = 0.0;
        let mut src = Vec::new();
        for i in 0..n1 {
            if s1.x[i] < r_charge.max(dx1) {
                let (a, b) = (s1.x[i] - 0.5 * s1.dx[i], s1.x[i] + 0.5 * s1.dx[i]);
                vol += 4.0 / 3.0 * core::f64::consts::PI * (b * b * b - a * a * a);
                src.push(i);
            }
        }
        for &i in &src {
            s1.u[i] = Prim {
                rho: TNT_DENSITY,
                u: 0.0,
                p: amb.p,
            }
            .to_cons(g);
            s1.u[i][2] += energy / vol;
        }
        let t0 = self.grid.t;
        let thr = self.arrival_threshold_pa;
        let inner: Vec<usize> = (0..self.probes.len())
            .filter(|&k| {
                let r = (self.probes[k].position - d.origin).norm();
                r < r_map
            })
            .collect();
        let sample1 = |s1: &Euler1d, r: f64| -> Prim {
            let x = (r / dx1 - 0.5).max(0.0);
            let i = (libm_ops::floor(x) as usize).min(n1 - 1);
            let j = (i + 1).min(n1 - 1);
            let f = (x - i as f64).clamp(0.0, 1.0);
            let pi = Prim::from_cons(s1.u[i], g).unwrap_or(Prim {
                rho: amb.rho,
                u: 0.0,
                p: amb.p,
            });
            let pj = Prim::from_cons(s1.u[j], g).unwrap_or(pi);
            Prim {
                rho: pi.rho + f * (pj.rho - pi.rho),
                u: pi.u + f * (pj.u - pi.u),
                p: pi.p + f * (pj.p - pi.p),
            }
        };
        let mut steps = 0usize;
        loop {
            // Shock front: outermost cell above the threshold.
            let mut front = 0.0;
            for i in (0..n1).rev() {
                let p = Prim::from_cons(s1.u[i], g).map(|w| w.p).unwrap_or(amb.p);
                if p - amb.p > thr {
                    front = s1.x[i];
                    break;
                }
            }
            if front >= r_map || steps >= 2_000_000 {
                break;
            }
            let t_before = s1.t;
            s1.step_cfl(self.cfl);
            steps += 1;
            let dt = s1.t - t_before;
            for &k in &inner {
                let r = (self.probes[k].position - d.origin).norm();
                let p = sample1(&s1, r).p - amb.p;
                self.probes[k].record(t0 + s1.t, dt, p, thr, self.record_history);
            }
        }
        // --- map onto the 3-D grid (conservative 4×4×4 sub-sampling) -------
        // Each 3-D cell gets the average of the 1-D conserved state over 64
        // sub-points; the overpressure energy is then rescaled so the grid
        // holds exactly the 1-D energy of the mapped solid angle.
        const NS: usize = 4;
        let r_fill = r_map + 2.0 * dx;
        let vcell = self.grid.cell_volume();
        let amb_e = amb.to_cons(g)[4];
        let mut mapped: Vec<(usize, [f64; 5])> = Vec::new();
        let mut e_sub = 0.0; // Σ (E − E_amb) V over mapped cells
        let mut e_exact = 0.0; // same, from the 1-D shell integral per sub-volume
        for k in 0..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..self.grid.nx {
                    let c = self.grid.idx(i, j, k);
                    if self.grid.solid[c] {
                        continue;
                    }
                    let cc = self.grid.center(i, j, k) - d.origin;
                    if cc.norm() > r_fill + dx {
                        continue;
                    }
                    let mut acc = [0.0; 5];
                    for a in 0..NS {
                        for b in 0..NS {
                            for e in 0..NS {
                                let off = Vec3::new(
                                    (a as f64 + 0.5) / NS as f64 - 0.5,
                                    (b as f64 + 0.5) / NS as f64 - 0.5,
                                    (e as f64 + 0.5) / NS as f64 - 0.5,
                                ) * dx;
                                let rv = cc + off;
                                let r = rv.norm();
                                let w1 = if r < r_end {
                                    sample1(&s1, r)
                                } else {
                                    Prim {
                                        rho: amb.rho,
                                        u: 0.0,
                                        p: amb.p,
                                    }
                                };
                                let dir = if r > 1e-12 { rv / r } else { Vec3::zero() };
                                let cons = Prim3 {
                                    rho: w1.rho,
                                    u: [w1.u * dir.x, w1.u * dir.y, w1.u * dir.z],
                                    p: w1.p,
                                }
                                .to_cons(g);
                                for q in 0..5 {
                                    acc[q] += cons[q];
                                }
                            }
                        }
                    }
                    let inv = 1.0 / (NS * NS * NS) as f64;
                    for q in &mut acc {
                        *q *= inv;
                    }
                    if (acc[4] - amb_e).abs() <= 1e-12 * amb_e && acc[0] == amb.rho {
                        continue;
                    }
                    e_sub += (acc[4] - amb_e) * vcell;
                    mapped.push((c, acc));
                }
            }
        }
        // Target: the exact charge energy times the covered solid angle. (The
        // 1-D spherical solver's geometric source terms are not exactly
        // conservative, so its own energy integral is not used.)
        let covered = if mapped.is_empty() {
            0.0
        } else {
            // Fraction of the r_fill sphere volume that is fluid and in-domain.
            let full = 4.0 / 3.0 * core::f64::consts::PI * r_fill * r_fill * r_fill;
            let mut inside = 0.0;
            for k in 0..self.grid.nz {
                for j in 0..self.grid.ny {
                    for i in 0..self.grid.nx {
                        let c = self.grid.idx(i, j, k);
                        if !self.grid.solid[c]
                            && (self.grid.center(i, j, k) - d.origin).norm() <= r_fill
                        {
                            inside += vcell;
                        }
                    }
                }
            }
            (inside / full).min(1.0)
        };
        e_exact += energy * covered;
        let scale = if e_sub > 0.0 { e_exact / e_sub } else { 1.0 };
        let mut e_mapped = 0.0;
        for (c, mut st) in mapped.iter().copied() {
            st[4] = amb_e + (st[4] - amb_e) * scale;
            // Keep pressure physical after the rescale.
            let ke = 0.5 * (st[1] * st[1] + st[2] * st[2] + st[3] * st[3]) / st[0];
            let p_min = self.grid.p_floor / (g - 1.0) + ke;
            if st[4] < p_min {
                st[4] = p_min;
            }
            e_mapped += (st[4] - amb_e) * vcell;
            self.grid.u[c] = st;
        }
        let cells = mapped.len();
        self.grid.t = t0 + s1.t;
        self.energy_deposited_j += e_mapped;
        DetonationReport {
            remapped: true,
            remap_radius_m: r_map,
            remap_time_s: s1.t,
            cells,
            energy_mapped_j: e_mapped,
        }
    }

    /// Distance [m] from `p` to the nearest solid voxel center or wall
    /// boundary plane (outflow faces ignored). With `ignore_below`, solids and
    /// the z− wall below `p.z` are ignored (surface bursts).
    pub fn clearance(&self, p: Vec3, ignore_below: bool) -> f64 {
        let g = &self.grid;
        let mut best = f64::INFINITY;
        let lo = g.origin;
        let hi = g.origin + Vec3::new(g.nx as f64 * g.dx, g.ny as f64 * g.dx, g.nz as f64 * g.dx);
        let planes = [
            (Boundary::Wall == g.boundary[0], p.x - lo.x),
            (Boundary::Wall == g.boundary[1], hi.x - p.x),
            (Boundary::Wall == g.boundary[2], p.y - lo.y),
            (Boundary::Wall == g.boundary[3], hi.y - p.y),
            (Boundary::Wall == g.boundary[4] && !ignore_below, p.z - lo.z),
            (Boundary::Wall == g.boundary[5], hi.z - p.z),
        ];
        for (wall, d) in planes {
            if wall {
                best = best.min(d);
            }
        }
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..g.nx {
                    if !g.solid[g.idx(i, j, k)] {
                        continue;
                    }
                    let c = g.center(i, j, k);
                    if ignore_below && c.z < p.z {
                        continue;
                    }
                    best = best.min((c - p).norm() - 0.5 * g.dx);
                }
            }
        }
        best.max(0.0)
    }

    /// Add a pressure gauge; returns its index.
    pub fn add_probe(&mut self, position: Vec3) -> usize {
        self.probes.push(ProbeRecord::new(position));
        self.probes.len() - 1
    }

    fn update_probes(&mut self, dt: f64) {
        let t = self.grid.t;
        let (thr, hist) = (self.arrival_threshold_pa, self.record_history);
        for k in 0..self.probes.len() {
            let p = self.grid.sample_overpressure(self.probes[k].position);
            self.probes[k].record(t, dt, p, thr, hist);
        }
    }

    /// One CFL step; updates probes and returns `dt`.
    pub fn step(&mut self) -> f64 {
        let dt = self.grid.step_cfl(self.cfl);
        self.update_probes(dt);
        dt
    }

    /// Run until `t_end` [s] (or `max_steps`), calling `each(scene, dt)` after
    /// every step. Returns steps taken.
    pub fn run_until_with<F: FnMut(&mut BlastScene, f64)>(
        &mut self,
        t_end: f64,
        max_steps: usize,
        mut each: F,
    ) -> usize {
        let mut n = 0;
        while self.grid.t < t_end && n < max_steps {
            let mut dt = self.grid.stable_dt(self.cfl);
            if !(dt.is_finite() && dt > 0.0) {
                break;
            }
            if self.grid.t + dt > t_end {
                dt = t_end - self.grid.t;
            }
            self.grid.step(dt);
            self.update_probes(dt);
            each(self, dt);
            n += 1;
        }
        n
    }

    /// Run until `t_end` [s] (or `max_steps`). Returns steps taken.
    pub fn run_until(&mut self, t_end: f64, max_steps: usize) -> usize {
        self.run_until_with(t_end, max_steps, |_, _| {})
    }

    /// Overpressure [Pa] acting on `face`, sampled one cell outside it.
    pub fn face_pressure(&self, face: &ExposedFace) -> f64 {
        let n = face.normal.normalize();
        self.grid
            .sample_overpressure(face.center + n * self.grid.dx)
    }

    /// Apply `p · A · dt` impulses (inward along `−normal`) on `faces`.
    /// Returns total impulse applied [N·s].
    pub fn apply_to_world(&self, world: &mut World, faces: &[ExposedFace], dt: f64) -> Vec3 {
        let mut total = Vec3::zero();
        for f in faces {
            let p = self.face_pressure(f);
            if p == 0.0 || f.area <= 0.0 {
                continue;
            }
            let j = f.normal.normalize() * (-p * f.area * dt);
            world.apply_impulse(f.body, j, f.center);
            total += j;
        }
        total
    }

    /// Write a horizontal (`axis = 2`) or vertical (`0`, `1`) slice of
    /// overpressure as a binary PPM image. Colour: blue (suction) → black
    /// (ambient) → red/yellow (overpressure, log scale up to `p_max`);
    /// solids are grey.
    pub fn slice_ppm(&self, axis: usize, index: usize, p_max: f64) -> Vec<u8> {
        let g = &self.grid;
        let (w, h) = match axis {
            0 => (g.ny, g.nz),
            1 => (g.nx, g.nz),
            _ => (g.nx, g.ny),
        };
        let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
        let lp = libm_ops::log(1.0 + p_max.max(1.0));
        for row in (0..h).rev() {
            for col in 0..w {
                let (i, j, k) = match axis {
                    0 => (index.min(g.nx - 1), col, row),
                    1 => (col, index.min(g.ny - 1), row),
                    _ => (col, row, index.min(g.nz - 1)),
                };
                let c = g.idx(i, j, k);
                let rgb = if g.solid[c] {
                    [110, 110, 110]
                } else {
                    let p = g.overpressure(c);
                    let s = (libm_ops::log(1.0 + libm_ops::fabs(p)) / lp).clamp(0.0, 1.0);
                    if p >= 0.0 {
                        [
                            (255.0 * (2.0 * s).min(1.0)) as u8,
                            (255.0 * (2.0 * s - 1.0).max(0.0)) as u8,
                            0,
                        ]
                    } else {
                        [0, (90.0 * s) as u8, (255.0 * s) as u8]
                    }
                };
                out.extend_from_slice(&rgb);
            }
        }
        out
    }
}

/// FNV-1a over every cell's conserved state (cell order).
pub fn grid_hash(g: &Grid3d) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for c in &g.u {
        for x in c {
            for b in x.to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    h
}
