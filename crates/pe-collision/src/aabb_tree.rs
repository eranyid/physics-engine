//! Dynamic AABB tree broad phase (SAH-style insertion, fat AABBs, rotations).
//!
//! Clean-room implementation inspired by the published description of Box2D's
//! dynamic tree (Erin Catto / Box2D docs) and standard SAH BVH insertion costs.
//! No third-party source was copied — see `THIRD_PARTY.md`.
//!
//! Pair output is always sorted by `(min id, max id)` for determinism.

use pe_math::Vec3;

use crate::broadphase::BroadPhase;
use crate::shapes::Aabb;

/// Default fat-AABB expansion [m] added to each extent.
pub const DEFAULT_FAT_MARGIN: f64 = 0.05;

const NULL: i32 = -1;

#[derive(Clone, Copy, Debug)]
struct Node {
    aabb: Aabb,
    /// Parent index, or `NULL`.
    parent: i32,
    /// Child 0 / next free; for leaves, unused as child.
    child1: i32,
    /// Child 1; for leaves, unused.
    child2: i32,
    /// Proxy user id (broad-phase index) when `height == 0`.
    proxy_id: i32,
    /// Leaf height is 0; free nodes use −1.
    height: i32,
}

impl Node {
    fn free() -> Self {
        Self {
            aabb: Aabb {
                lo: Vec3::zero(),
                hi: Vec3::zero(),
            },
            parent: NULL,
            child1: NULL,
            child2: NULL,
            proxy_id: NULL,
            height: -1,
        }
    }

    fn is_leaf(&self) -> bool {
        self.child1 == NULL
    }
}

/// Dynamic AABB tree implementing [`BroadPhase`].
///
/// Each call to [`pairs`](BroadPhase::pairs) rebuilds the tree by inserting
/// fat AABBs in ascending proxy-id order (deterministic), then queries
/// overlapping leaf pairs.
#[derive(Clone, Debug)]
pub struct DynamicAabbTree {
    /// Fat-AABB margin [m] expanded on each side.
    pub fat_margin: f64,
    nodes: Vec<Node>,
    root: i32,
    free_list: i32,
    proxy_leaf: Vec<i32>,
}

impl Default for DynamicAabbTree {
    fn default() -> Self {
        Self::new(DEFAULT_FAT_MARGIN)
    }
}

impl DynamicAabbTree {
    /// Construct with a fat-AABB margin [m].
    pub fn new(fat_margin: f64) -> Self {
        Self {
            fat_margin,
            nodes: Vec::new(),
            root: NULL,
            free_list: NULL,
            proxy_leaf: Vec::new(),
        }
    }

    fn fatten(&self, a: &Aabb) -> Aabb {
        let m = Vec3::new(self.fat_margin, self.fat_margin, self.fat_margin);
        // Infinite AABBs (planes) stay infinite.
        if !a.lo.x.is_finite() || !a.hi.x.is_finite() {
            return *a;
        }
        Aabb {
            lo: a.lo - m,
            hi: a.hi + m,
        }
    }

    fn allocate_node(&mut self) -> i32 {
        if self.free_list != NULL {
            let id = self.free_list;
            self.free_list = self.nodes[id as usize].child1;
            self.nodes[id as usize] = Node::free();
            return id;
        }
        let id = self.nodes.len() as i32;
        self.nodes.push(Node::free());
        id
    }

    fn surface_area(a: &Aabb) -> f64 {
        let e = a.hi - a.lo;
        if !e.x.is_finite() {
            return f64::INFINITY;
        }
        2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
    }

    fn union(a: &Aabb, b: &Aabb) -> Aabb {
        Aabb {
            lo: Vec3::new(a.lo.x.min(b.lo.x), a.lo.y.min(b.lo.y), a.lo.z.min(b.lo.z)),
            hi: Vec3::new(a.hi.x.max(b.hi.x), a.hi.y.max(b.hi.y), a.hi.z.max(b.hi.z)),
        }
    }

    fn contains(outer: &Aabb, inner: &Aabb) -> bool {
        outer.lo.x <= inner.lo.x
            && outer.lo.y <= inner.lo.y
            && outer.lo.z <= inner.lo.z
            && outer.hi.x >= inner.hi.x
            && outer.hi.y >= inner.hi.y
            && outer.hi.z >= inner.hi.z
    }

    fn overlaps(a: &Aabb, b: &Aabb) -> bool {
        a.lo.x <= b.hi.x
            && a.lo.y <= b.hi.y
            && a.lo.z <= b.hi.z
            && b.lo.x <= a.hi.x
            && b.lo.y <= a.hi.y
            && b.lo.z <= a.hi.z
    }

    /// Clear and rebuild from fattened AABBs (insertion order = proxy id).
    pub fn rebuild(&mut self, aabbs: &[Aabb]) {
        self.nodes.clear();
        self.free_list = NULL;
        self.root = NULL;
        self.proxy_leaf.clear();
        self.proxy_leaf.resize(aabbs.len(), NULL);
        for (id, aabb) in aabbs.iter().enumerate() {
            self.insert_leaf(id as i32, self.fatten(aabb));
        }
    }

    fn insert_leaf(&mut self, proxy_id: i32, aabb: Aabb) {
        let leaf = self.allocate_node();
        self.nodes[leaf as usize].aabb = aabb;
        self.nodes[leaf as usize].proxy_id = proxy_id;
        self.nodes[leaf as usize].height = 0;
        self.proxy_leaf[proxy_id as usize] = leaf;

        if self.root == NULL {
            self.root = leaf;
            self.nodes[leaf as usize].parent = NULL;
            return;
        }

        // SAH: walk from root choosing the child that yields lower combined area cost.
        let mut index = self.root;
        while !self.nodes[index as usize].is_leaf() {
            let child1 = self.nodes[index as usize].child1;
            let child2 = self.nodes[index as usize].child2;
            let area = Self::surface_area(&self.nodes[index as usize].aabb);
            let combined = Self::union(&self.nodes[index as usize].aabb, &aabb);
            let combined_area = Self::surface_area(&combined);
            let cost = 2.0 * combined_area;
            let inheritance = 2.0 * (combined_area - area);

            let cost1 = {
                let u = Self::union(&self.nodes[child1 as usize].aabb, &aabb);
                if self.nodes[child1 as usize].is_leaf() {
                    Self::surface_area(&u) + inheritance
                } else {
                    let old = Self::surface_area(&self.nodes[child1 as usize].aabb);
                    Self::surface_area(&u) - old + inheritance
                }
            };
            let cost2 = {
                let u = Self::union(&self.nodes[child2 as usize].aabb, &aabb);
                if self.nodes[child2 as usize].is_leaf() {
                    Self::surface_area(&u) + inheritance
                } else {
                    let old = Self::surface_area(&self.nodes[child2 as usize].aabb);
                    Self::surface_area(&u) - old + inheritance
                }
            };

            if cost < cost1 && cost < cost2 {
                break;
            }
            index = if cost1 < cost2 { child1 } else { child2 };
        }

        let sibling = index;
        let old_parent = self.nodes[sibling as usize].parent;
        let new_parent = self.allocate_node();
        self.nodes[new_parent as usize].parent = old_parent;
        self.nodes[new_parent as usize].aabb =
            Self::union(&aabb, &self.nodes[sibling as usize].aabb);
        self.nodes[new_parent as usize].height = self.nodes[sibling as usize].height + 1;
        self.nodes[new_parent as usize].child1 = sibling;
        self.nodes[new_parent as usize].child2 = leaf;
        self.nodes[sibling as usize].parent = new_parent;
        self.nodes[leaf as usize].parent = new_parent;

        if old_parent != NULL {
            if self.nodes[old_parent as usize].child1 == sibling {
                self.nodes[old_parent as usize].child1 = new_parent;
            } else {
                self.nodes[old_parent as usize].child2 = new_parent;
            }
        } else {
            self.root = new_parent;
        }

        self.refit_and_rotate(self.nodes[leaf as usize].parent);
    }

    fn refit_and_rotate(&mut self, mut index: i32) {
        while index != NULL {
            index = self.balance(index);
            let child1 = self.nodes[index as usize].child1;
            let child2 = self.nodes[index as usize].child2;
            debug_assert!(child1 != NULL && child2 != NULL);
            self.nodes[index as usize].height = 1 + self.nodes[child1 as usize]
                .height
                .max(self.nodes[child2 as usize].height);
            self.nodes[index as usize].aabb = Self::union(
                &self.nodes[child1 as usize].aabb,
                &self.nodes[child2 as usize].aabb,
            );
            index = self.nodes[index as usize].parent;
        }
    }

    /// Tree rotation (AVL-style) to reduce imbalance — Box2D dynamic-tree idea.
    fn balance(&mut self, i_a: i32) -> i32 {
        if self.nodes[i_a as usize].is_leaf() || self.nodes[i_a as usize].height < 2 {
            return i_a;
        }
        let i_b = self.nodes[i_a as usize].child1;
        let i_c = self.nodes[i_a as usize].child2;
        let balance = self.nodes[i_c as usize].height - self.nodes[i_b as usize].height;
        if balance > 1 {
            // Rotate C up.
            let i_f = self.nodes[i_c as usize].child1;
            let i_g = self.nodes[i_c as usize].child2;
            self.nodes[i_c as usize].child1 = i_a;
            self.nodes[i_c as usize].parent = self.nodes[i_a as usize].parent;
            self.nodes[i_a as usize].parent = i_c;

            if self.nodes[i_c as usize].parent != NULL {
                let p = self.nodes[i_c as usize].parent;
                if self.nodes[p as usize].child1 == i_a {
                    self.nodes[p as usize].child1 = i_c;
                } else {
                    self.nodes[p as usize].child2 = i_c;
                }
            } else {
                self.root = i_c;
            }

            if self.nodes[i_f as usize].height > self.nodes[i_g as usize].height {
                self.nodes[i_c as usize].child2 = i_f;
                self.nodes[i_a as usize].child2 = i_g;
                self.nodes[i_g as usize].parent = i_a;
                self.nodes[i_a as usize].aabb = Self::union(
                    &self.nodes[i_b as usize].aabb,
                    &self.nodes[i_g as usize].aabb,
                );
                self.nodes[i_c as usize].aabb = Self::union(
                    &self.nodes[i_a as usize].aabb,
                    &self.nodes[i_f as usize].aabb,
                );
                self.nodes[i_a as usize].height = 1 + self.nodes[i_b as usize]
                    .height
                    .max(self.nodes[i_g as usize].height);
                self.nodes[i_c as usize].height = 1 + self.nodes[i_a as usize]
                    .height
                    .max(self.nodes[i_f as usize].height);
            } else {
                self.nodes[i_c as usize].child2 = i_g;
                self.nodes[i_a as usize].child2 = i_f;
                self.nodes[i_f as usize].parent = i_a;
                self.nodes[i_a as usize].aabb = Self::union(
                    &self.nodes[i_b as usize].aabb,
                    &self.nodes[i_f as usize].aabb,
                );
                self.nodes[i_c as usize].aabb = Self::union(
                    &self.nodes[i_a as usize].aabb,
                    &self.nodes[i_g as usize].aabb,
                );
                self.nodes[i_a as usize].height = 1 + self.nodes[i_b as usize]
                    .height
                    .max(self.nodes[i_f as usize].height);
                self.nodes[i_c as usize].height = 1 + self.nodes[i_a as usize]
                    .height
                    .max(self.nodes[i_g as usize].height);
            }
            return i_c;
        }
        if balance < -1 {
            let i_d = self.nodes[i_b as usize].child1;
            let i_e = self.nodes[i_b as usize].child2;
            self.nodes[i_b as usize].child1 = i_a;
            self.nodes[i_b as usize].parent = self.nodes[i_a as usize].parent;
            self.nodes[i_a as usize].parent = i_b;

            if self.nodes[i_b as usize].parent != NULL {
                let p = self.nodes[i_b as usize].parent;
                if self.nodes[p as usize].child1 == i_a {
                    self.nodes[p as usize].child1 = i_b;
                } else {
                    self.nodes[p as usize].child2 = i_b;
                }
            } else {
                self.root = i_b;
            }

            if self.nodes[i_d as usize].height > self.nodes[i_e as usize].height {
                self.nodes[i_b as usize].child2 = i_d;
                self.nodes[i_a as usize].child1 = i_e;
                self.nodes[i_e as usize].parent = i_a;
                self.nodes[i_a as usize].aabb = Self::union(
                    &self.nodes[i_c as usize].aabb,
                    &self.nodes[i_e as usize].aabb,
                );
                self.nodes[i_b as usize].aabb = Self::union(
                    &self.nodes[i_a as usize].aabb,
                    &self.nodes[i_d as usize].aabb,
                );
                self.nodes[i_a as usize].height = 1 + self.nodes[i_c as usize]
                    .height
                    .max(self.nodes[i_e as usize].height);
                self.nodes[i_b as usize].height = 1 + self.nodes[i_a as usize]
                    .height
                    .max(self.nodes[i_d as usize].height);
            } else {
                self.nodes[i_b as usize].child2 = i_e;
                self.nodes[i_a as usize].child1 = i_d;
                self.nodes[i_d as usize].parent = i_a;
                self.nodes[i_a as usize].aabb = Self::union(
                    &self.nodes[i_c as usize].aabb,
                    &self.nodes[i_d as usize].aabb,
                );
                self.nodes[i_b as usize].aabb = Self::union(
                    &self.nodes[i_a as usize].aabb,
                    &self.nodes[i_e as usize].aabb,
                );
                self.nodes[i_a as usize].height = 1 + self.nodes[i_c as usize]
                    .height
                    .max(self.nodes[i_d as usize].height);
                self.nodes[i_b as usize].height = 1 + self.nodes[i_a as usize]
                    .height
                    .max(self.nodes[i_e as usize].height);
            }
            return i_b;
        }
        i_a
    }

    /// Collect overlapping proxy pairs from the current tree.
    pub fn query_pairs(&self, skip_static: Option<&[bool]>) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        if self.root == NULL {
            return pairs;
        }
        // Stack of (node_a, node_b) with node_a <= node_b for internal traversal.
        let mut stack: Vec<(i32, i32)> = Vec::new();
        // Self-overlap query starting from root children.
        let root = self.root;
        if self.nodes[root as usize].is_leaf() {
            return pairs;
        }
        stack.push((
            self.nodes[root as usize].child1,
            self.nodes[root as usize].child2,
        ));
        // Also need all sibling pairs under each internal node — classic tree pair walk:
        // push (child1, child2) and recurse into each child with itself via a second stack.
        let mut self_stack = vec![root];
        while let Some(node) = self_stack.pop() {
            if self.nodes[node as usize].is_leaf() {
                continue;
            }
            let c1 = self.nodes[node as usize].child1;
            let c2 = self.nodes[node as usize].child2;
            stack.push((c1, c2));
            self_stack.push(c1);
            self_stack.push(c2);
        }

        while let Some((a, b)) = stack.pop() {
            let na = &self.nodes[a as usize];
            let nb = &self.nodes[b as usize];
            if !Self::overlaps(&na.aabb, &nb.aabb) {
                continue;
            }
            if na.is_leaf() && nb.is_leaf() {
                let mut i = na.proxy_id as usize;
                let mut j = nb.proxy_id as usize;
                if i > j {
                    core::mem::swap(&mut i, &mut j);
                }
                if let Some(skip) = skip_static {
                    if skip[i] && skip[j] {
                        continue;
                    }
                }
                // Exact (non-fat) overlap is filtered by the caller via original AABBs;
                // fat AABBs may over-report — ContactWorld re-checks via narrow phase.
                // For BroadPhase equivalence tests we filter with the input AABBs outside.
                pairs.push((i, j));
            } else if na.is_leaf() {
                stack.push((a, nb.child1));
                stack.push((a, nb.child2));
            } else if nb.is_leaf() {
                stack.push((na.child1, b));
                stack.push((na.child2, b));
            } else {
                stack.push((na.child1, nb.child1));
                stack.push((na.child1, nb.child2));
                stack.push((na.child2, nb.child1));
                stack.push((na.child2, nb.child2));
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// Validate: every leaf AABB is contained by all ancestors.
    pub fn validate_invariants(&self) -> bool {
        if self.root == NULL {
            return true;
        }
        for (leaf_idx, &leaf) in self.proxy_leaf.iter().enumerate() {
            if leaf == NULL {
                continue;
            }
            let leaf_aabb = self.nodes[leaf as usize].aabb;
            let mut p = self.nodes[leaf as usize].parent;
            while p != NULL {
                if !Self::contains(&self.nodes[p as usize].aabb, &leaf_aabb) {
                    let _ = leaf_idx;
                    return false;
                }
                p = self.nodes[p as usize].parent;
            }
        }
        // Internal nodes equal the union of children.
        for n in &self.nodes {
            if n.height < 0 || n.is_leaf() {
                continue;
            }
            let u = Self::union(
                &self.nodes[n.child1 as usize].aabb,
                &self.nodes[n.child2 as usize].aabb,
            );
            if n.aabb != u {
                return false;
            }
        }
        true
    }
}

impl BroadPhase for DynamicAabbTree {
    fn pairs(&self, aabbs: &[Aabb], skip_static: &[bool]) -> Vec<(usize, usize)> {
        // Rebuild on a clone so `&self` stays unchanged for the trait signature.
        let mut tree = self.clone();
        tree.rebuild(aabbs);
        let fat_pairs = tree.query_pairs(Some(skip_static));
        // Filter to true (tight) AABB overlaps so we match SAP/brute.
        let mut out = Vec::with_capacity(fat_pairs.len());
        for (i, j) in fat_pairs {
            if Self::overlaps(&aabbs[i], &aabbs[j]) {
                out.push((i, j));
            }
        }
        out
    }
}

/// Convenience: tree pairs with optional skip flags (rebuilds internally).
pub fn dynamic_tree_pairs(aabbs: &[Aabb], skip_static: Option<&[bool]>) -> Vec<(usize, usize)> {
    let skip = skip_static.unwrap_or(&[]);
    let skip_owned: Vec<bool>;
    let skip_ref: &[bool] = if skip.is_empty() {
        skip_owned = vec![false; aabbs.len()];
        &skip_owned
    } else {
        skip
    };
    DynamicAabbTree::default().pairs(aabbs, skip_ref)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broadphase::{brute_force_pairs, sweep_and_prune};

    #[test]
    fn tree_equals_brute_and_sap() {
        for seed in 0..10u64 {
            let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let mut next = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                state
            };
            let uni = |next: &mut dyn FnMut() -> u64, a: f64, b: f64| -> f64 {
                let u = (next() >> 11) as f64 / ((1u64 << 53) as f64);
                a + (b - a) * u
            };
            let n = 120;
            let mut aabbs = Vec::with_capacity(n);
            let mut skip = vec![false; n];
            for i in 0..n {
                let cx = uni(&mut next, 0.0, 8.0);
                let cy = uni(&mut next, 0.0, 8.0);
                let cz = uni(&mut next, 0.0, 8.0);
                let ex = uni(&mut next, 0.05, 0.6);
                let ey = uni(&mut next, 0.05, 0.6);
                let ez = uni(&mut next, 0.05, 0.6);
                aabbs.push(Aabb {
                    lo: Vec3::new(cx - ex, cy - ey, cz - ez),
                    hi: Vec3::new(cx + ex, cy + ey, cz + ez),
                });
                skip[i] = uni(&mut next, 0.0, 1.0) < 0.15;
            }
            let expect = brute_force_pairs(&aabbs, Some(&skip));
            assert_eq!(
                sweep_and_prune(&aabbs, Some(&skip)),
                expect,
                "sap seed {seed}"
            );
            assert_eq!(
                dynamic_tree_pairs(&aabbs, Some(&skip)),
                expect,
                "tree seed {seed}"
            );
            let mut tree = DynamicAabbTree::default();
            tree.rebuild(&aabbs);
            assert!(tree.validate_invariants(), "invariants seed {seed}");
        }
    }
}
