//! Uniform-grid neighbour search with **deterministic** sorted cell lists.
//!
//! No `HashMap` iteration on the step path: particles are sorted by cell key,
//! and cell ranges are stored in dense arrays indexed by a compact remap of
//! occupied cells (themselves sorted by key).

use pe_math::Vec3;

use crate::Dim;

/// Flat cell key for a 3-D (or 2-D with `kz=0`) integer lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellKey {
    /// Cell index x.
    pub ix: i32,
    /// Cell index y.
    pub iy: i32,
    /// Cell index z (0 in 2-D mode).
    pub iz: i32,
}

impl CellKey {
    /// Construct.
    #[inline]
    pub const fn new(ix: i32, iy: i32, iz: i32) -> Self {
        Self { ix, iy, iz }
    }
}

/// Uniform spatial hash rebuilt every step (sort + scan).
#[derive(Clone, Debug)]
pub struct UniformGrid {
    /// Cell size [m] (usually = kernel support).
    pub cell_size: f64,
    /// Dimension (controls neighbour stencil: 9 vs 27).
    pub dim: Dim,
    /// Particle index order sorted by [`CellKey`].
    pub order: Vec<usize>,
    /// Cell key per particle (original index).
    pub keys: Vec<CellKey>,
    /// Sorted unique occupied cell keys.
    pub occupied: Vec<CellKey>,
    /// For each occupied cell: `(start, end)` into [`Self::order`].
    pub ranges: Vec<(usize, usize)>,
}

impl UniformGrid {
    /// Create an empty grid with the given cell size and dimension.
    pub fn new(cell_size: f64, dim: Dim) -> Self {
        Self {
            cell_size,
            dim,
            order: Vec::new(),
            keys: Vec::new(),
            occupied: Vec::new(),
            ranges: Vec::new(),
        }
    }

    /// Map a world position to a cell key.
    #[inline]
    pub fn key_of(&self, p: Vec3) -> CellKey {
        let inv = 1.0 / self.cell_size;
        let ix = floor_i32(p.x * inv);
        let iy = floor_i32(p.y * inv);
        let iz = match self.dim {
            Dim::D2 => 0,
            Dim::D3 => floor_i32(p.z * inv),
        };
        CellKey::new(ix, iy, iz)
    }

    /// Rebuild from particle positions (`pos[i] = (x,y,z)`).
    ///
    /// Deterministic: sort by `(ix, iy, iz, particle_index)`.
    pub fn rebuild(&mut self, pos: &[Vec3]) {
        let n = pos.len();
        self.keys.resize(n, CellKey::new(0, 0, 0));
        self.order.clear();
        self.order.reserve(n);
        for i in 0..n {
            self.keys[i] = self.key_of(pos[i]);
            self.order.push(i);
        }
        let keys = &self.keys;
        self.order
            .sort_by(|&a, &b| keys[a].cmp(&keys[b]).then_with(|| a.cmp(&b)));

        self.occupied.clear();
        self.ranges.clear();
        if n == 0 {
            return;
        }
        let mut start = 0usize;
        let mut cur = keys[self.order[0]];
        for s in 1..n {
            let k = keys[self.order[s]];
            if k != cur {
                self.occupied.push(cur);
                self.ranges.push((start, s));
                start = s;
                cur = k;
            }
        }
        self.occupied.push(cur);
        self.ranges.push((start, n));
    }

    /// Binary-search the range of an occupied cell; `None` if empty.
    #[inline]
    pub fn cell_range(&self, key: CellKey) -> Option<(usize, usize)> {
        let idx = self.occupied.binary_search(&key).ok()?;
        Some(self.ranges[idx])
    }

    /// Collect neighbour particle indices for particle `i` into `out`
    /// (does **not** clear `out`; caller clears). Includes self.
    ///
    /// Neighbour cells are visited in lexicographic `(dx,dy,dz)` order;
    /// within each cell, particles appear in ascending original index order
    /// (because the cell list was sorted by key then index).
    pub fn neighbours(&self, i: usize, out: &mut Vec<usize>) {
        let key = self.keys[i];
        let dz_range = match self.dim {
            Dim::D2 => 0..=0,
            Dim::D3 => -1..=1,
        };
        for dz in dz_range {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nk = CellKey::new(key.ix + dx, key.iy + dy, key.iz + dz);
                    if let Some((a, b)) = self.cell_range(nk) {
                        for s in a..b {
                            out.push(self.order[s]);
                        }
                    }
                }
            }
        }
    }
}

#[inline]
fn floor_i32(x: f64) -> i32 {
    // Portable floor toward −∞. Clamp to i32 range to avoid cast/overflow panics
    // if a particle tunnels far outside the domain.
    if !x.is_finite() {
        return 0;
    }
    let clamped = x.clamp((i32::MIN as f64) + 1.0, (i32::MAX as f64) - 1.0);
    let t = clamped as i32;
    if (t as f64) > clamped {
        t.saturating_sub(1)
    } else {
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorted_order_is_deterministic() {
        let mut g = UniformGrid::new(1.0, Dim::D3);
        let pos = vec![
            Vec3::new(0.1, 0.1, 0.1),
            Vec3::new(1.1, 0.1, 0.1),
            Vec3::new(0.2, 0.1, 0.1),
            Vec3::new(-0.5, 0.0, 0.0),
        ];
        g.rebuild(&pos);
        // Particles 0 and 2 share a cell; within-cell order by index: 0 then 2.
        let mut n = Vec::new();
        g.neighbours(0, &mut n);
        assert!(n.contains(&0) && n.contains(&2) && n.contains(&1));
        let mut g2 = UniformGrid::new(1.0, Dim::D3);
        g2.rebuild(&pos);
        assert_eq!(g.order, g2.order);
        assert_eq!(g.occupied, g2.occupied);
    }
}
