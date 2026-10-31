//! Persistent contact islands (union-find) and sleeping.
//!
//! Islands are rebuilt each step from the current touching pair list in
//! deterministic `(min id, max id)` order. Sleeping follows Box2D-style
//! thresholds (linear, angular, time-to-sleep) without copying Box2D source.
//!
//! Defaults (SI): `|v| < 0.05 m/s`, `|ω| < 2°/s`, `time_to_sleep = 0.5 s`.
//! A sleeping island wakes on a new contact involving an awake body/static, or
//! when a non-zero force/torque is applied to a sleeping body.

use pe_math::libm_ops::sqrt;
use pe_math::Vec3;

/// Default linear sleep tolerance [m/s].
pub const DEFAULT_LINEAR_SLEEP: f64 = 0.05;
/// Default angular sleep tolerance [rad/s] (= 2°/s).
pub const DEFAULT_ANGULAR_SLEEP: f64 = 2.0 * core::f64::consts::PI / 180.0;
/// Default time a body must stay under thresholds before sleeping [s].
pub const DEFAULT_TIME_TO_SLEEP: f64 = 0.5;

/// Sleep / island bookkeeping for dynamic bodies.
#[derive(Clone, Debug)]
pub struct SleepState {
    /// Enable island sleeping.
    pub enabled: bool,
    /// Linear speed threshold [m/s].
    pub linear_tolerance: f64,
    /// Angular speed threshold [rad/s].
    pub angular_tolerance: f64,
    /// Required quiet time [s].
    pub time_to_sleep: f64,
    /// Per-dynamic-body awake flag.
    pub awake: Vec<bool>,
    /// Accumulated quiet time [s] while under thresholds.
    pub sleep_time: Vec<f64>,
    /// Island id per dynamic body (−1 if alone / unset); rebuilt each step.
    pub island: Vec<i32>,
    /// Number of islands after last rebuild.
    pub island_count: usize,
    /// Whether each island is sleeping (length `island_count`).
    pub island_sleeping: Vec<bool>,
}

impl SleepState {
    /// Allocate sleep state for `n` dynamic bodies (all awake).
    ///
    /// Sleeping is **off** by default so the PGS oracle path stays bit-identical
    /// to M3; enable via [`ContactWorld::set_allow_sleep`](crate::ContactWorld::set_allow_sleep).
    pub fn new(n: usize) -> Self {
        Self {
            enabled: false,
            linear_tolerance: DEFAULT_LINEAR_SLEEP,
            angular_tolerance: DEFAULT_ANGULAR_SLEEP,
            time_to_sleep: DEFAULT_TIME_TO_SLEEP,
            awake: vec![true; n],
            sleep_time: vec![0.0; n],
            island: vec![-1; n],
            island_count: 0,
            island_sleeping: Vec::new(),
        }
    }

    /// Resize when the body count changes.
    pub fn resize(&mut self, n: usize) {
        self.awake.resize(n, true);
        self.sleep_time.resize(n, 0.0);
        self.island.resize(n, -1);
    }

    /// Wake a single dynamic body and clear its sleep timer.
    pub fn wake_body(&mut self, i: usize) {
        if i < self.awake.len() {
            self.awake[i] = true;
            self.sleep_time[i] = 0.0;
        }
    }

    /// Wake every dynamic body in the same island as `i`.
    pub fn wake_island_of(&mut self, i: usize) {
        if i >= self.island.len() {
            return;
        }
        let id = self.island[i];
        if id < 0 {
            self.wake_body(i);
            return;
        }
        for j in 0..self.island.len() {
            if self.island[j] == id {
                self.wake_body(j);
            }
        }
    }

    /// Wake body `i` if `|f|` or `|τ|` is non-negligible.
    pub fn wake_from_force(&mut self, i: usize, f: Vec3, tau: Vec3) {
        const EPS: f64 = 1e-12;
        if f.norm_squared() > EPS || tau.norm_squared() > EPS {
            self.wake_island_of(i);
        }
    }

    /// Rebuild islands from sorted touching pairs among indices `[0, n)`.
    ///
    /// Static partners (`>= n`) do not merge distinct dynamic components; they
    /// only ensure the dynamic endpoint is present in some island.
    pub fn rebuild_islands(&mut self, n: usize, pairs: &[(usize, usize)]) {
        self.resize(n);
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank = vec![0u8; n];

        let find = |parent: &mut [usize], mut x: usize| -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        };
        let union = |parent: &mut [usize], rank: &mut [u8], a: usize, b: usize| {
            let mut ra = find(parent, a);
            let mut rb = find(parent, b);
            if ra == rb {
                return;
            }
            if rank[ra] < rank[rb] {
                core::mem::swap(&mut ra, &mut rb);
            }
            parent[rb] = ra;
            if rank[ra] == rank[rb] {
                rank[ra] = rank[ra].saturating_add(1);
            }
        };

        for &(a, b) in pairs {
            if a < n && b < n {
                union(&mut parent, &mut rank, a, b);
            }
        }

        // Assign dense island ids in ascending root order for determinism.
        let mut root_of = vec![0usize; n];
        let mut roots: Vec<usize> = Vec::new();
        for i in 0..n {
            let r = find(&mut parent, i);
            root_of[i] = r;
            if r == i {
                roots.push(i);
            }
        }
        roots.sort_unstable();
        let mut id_of = vec![-1i32; n];
        for (k, &r) in roots.iter().enumerate() {
            id_of[r] = k as i32;
        }
        for i in 0..n {
            self.island[i] = id_of[root_of[i]];
        }
        self.island_count = roots.len();
        self.island_sleeping = vec![false; self.island_count];
    }

    /// After solving: update sleep timers for awake bodies; put quiet islands to sleep.
    pub fn update_sleep(
        &mut self,
        dt: f64,
        lin_speed: &[f64],
        ang_speed: &[f64],
        has_contact: &[bool],
    ) {
        if !self.enabled {
            return;
        }
        let n = self.awake.len();
        debug_assert_eq!(lin_speed.len(), n);
        debug_assert_eq!(ang_speed.len(), n);

        for i in 0..n {
            if !self.awake[i] {
                continue;
            }
            let quiet = lin_speed[i] < self.linear_tolerance
                && ang_speed[i] < self.angular_tolerance
                && has_contact[i];
            if quiet {
                self.sleep_time[i] += dt;
            } else {
                self.sleep_time[i] = 0.0;
            }
        }

        // Island sleeps only if every member is quiet long enough.
        for island in 0..self.island_count {
            let mut members = 0usize;
            let mut ready = 0usize;
            for i in 0..n {
                if self.island[i] == island as i32 {
                    members += 1;
                    if self.awake[i] && self.sleep_time[i] >= self.time_to_sleep {
                        ready += 1;
                    } else if !self.awake[i] {
                        ready += 1; // already asleep member counts as ready
                        members += 0;
                    }
                }
            }
            // Recount carefully:
            let mut all_ready = true;
            let mut any = false;
            for i in 0..n {
                if self.island[i] != island as i32 {
                    continue;
                }
                any = true;
                if self.awake[i] && self.sleep_time[i] < self.time_to_sleep {
                    all_ready = false;
                    break;
                }
            }
            let _ = (members, ready);
            if any && all_ready {
                self.island_sleeping[island] = true;
                for i in 0..n {
                    if self.island[i] == island as i32 {
                        self.awake[i] = false;
                    }
                }
            }
        }
    }

    /// Wake islands that gained a contact with an awake partner or a static.
    ///
    /// `prev_pairs` / `curr_pairs` are sorted. A pair is "new" if absent previously.
    pub fn wake_from_new_contacts(
        &mut self,
        n: usize,
        prev_pairs: &[(usize, usize)],
        curr_pairs: &[(usize, usize)],
    ) {
        if !self.enabled {
            return;
        }
        let mut i = 0usize;
        let mut j = 0usize;
        while j < curr_pairs.len() {
            let p = curr_pairs[j];
            while i < prev_pairs.len() && prev_pairs[i] < p {
                i += 1;
            }
            let is_new = i >= prev_pairs.len() || prev_pairs[i] != p;
            if is_new {
                let (a, b) = p;
                if a < n {
                    // Contact with static or with another body: wake a's island.
                    if b >= n || (b < n && self.awake[b]) {
                        self.wake_island_of(a);
                    } else if b < n && self.awake[a] {
                        self.wake_island_of(b);
                    } else {
                        // both sleeping + new contact between them — wake both
                        self.wake_island_of(a);
                        if b < n {
                            self.wake_island_of(b);
                        }
                    }
                }
            }
            j += 1;
        }
    }

    /// True if dynamic body `i` is awake (or sleep disabled).
    #[inline]
    pub fn is_awake(&self, i: usize) -> bool {
        !self.enabled || self.awake.get(i).copied().unwrap_or(true)
    }
}

/// Angular speed `|ω|` from world angular momentum and inverse inertia.
#[inline]
pub fn angular_speed(l: Vec3, iinv: &[[f64; 3]; 3]) -> f64 {
    let wx = iinv[0][0] * l.x + iinv[0][1] * l.y + iinv[0][2] * l.z;
    let wy = iinv[1][0] * l.x + iinv[1][1] * l.y + iinv[1][2] * l.z;
    let wz = iinv[2][0] * l.x + iinv[2][1] * l.y + iinv[2][2] * l.z;
    sqrt(wx * wx + wy * wy + wz * wz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_find_merges_chain_deterministically() {
        let mut s = SleepState::new(4);
        // pairs must be sorted
        let pairs = vec![(0, 1), (1, 2)];
        s.rebuild_islands(4, &pairs);
        assert_eq!(s.island[0], s.island[1]);
        assert_eq!(s.island[1], s.island[2]);
        assert_ne!(s.island[0], s.island[3]);
        assert_eq!(s.island_count, 2);
    }

    #[test]
    fn quiet_island_falls_asleep() {
        let mut s = SleepState::new(2);
        s.enabled = true;
        s.time_to_sleep = 0.1;
        s.rebuild_islands(2, &[(0, 1)]);
        let lin = [0.01, 0.01];
        let ang = [0.0, 0.0];
        let contacts = [true, true];
        s.update_sleep(0.05, &lin, &ang, &contacts);
        assert!(s.awake[0] && s.awake[1]);
        s.update_sleep(0.05, &lin, &ang, &contacts);
        s.update_sleep(0.05, &lin, &ang, &contacts);
        assert!(!s.awake[0] && !s.awake[1]);
    }

    #[test]
    fn new_contact_wakes_sleeping_island() {
        let mut s = SleepState::new(2);
        s.enabled = true;
        s.rebuild_islands(2, &[]);
        s.awake = vec![false, true];
        s.island = vec![0, 1];
        s.island_count = 2;
        s.island_sleeping = vec![true, false];
        s.wake_from_new_contacts(2, &[], &[(0, 1)]);
        assert!(s.awake[0]);
    }
}
