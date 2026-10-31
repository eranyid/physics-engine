//! Deterministic seeded PRNG for turbulence / spectral waves.
//!
//! Clean-room implementation of **xoshiro256\*\*** (Blackman & Vigna).
//! No external RNG crates — identical across OS/ISA when the seed is fixed.
//!
//! # References
//! - Blackman, D. & Vigna, S. (2018/2021). Scrambled linear pseudorandom number
//!   generators. *ACM Trans. Math. Softw.* 47. Algorithm at <https://prng.di.unimi.it/>.
//! - SplitMix64 seeding: Steele, Lea & Flood (2014); used only to expand a `u64`
//!   seed into the 256-bit xoshiro state (same reference page).

/// xoshiro256** generator (deterministic, portable).
#[derive(Clone, Debug)]
pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

#[inline]
fn rotl(x: u64, k: u32) -> u64 {
    x.rotate_left(k)
}

/// SplitMix64 step used only for seeding (Steele et al. / Vigna reference).
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl Xoshiro256StarStar {
    /// Seed from a single `u64` via SplitMix64 (never all-zero state).
    pub fn seed(seed: u64) -> Self {
        let mut sm = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            *slot = splitmix64(&mut sm);
        }
        // Guard against the forbidden all-zero state.
        if s.iter().all(|&v| v == 0) {
            s[0] = 0x9e37_79b9_7f4a_7c15;
        }
        Self { s }
    }

    /// Next raw `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = rotl(self.s[1].wrapping_mul(5), 7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = rotl(self.s[3], 45);
        result
    }

    /// Uniform in `[0, 1)` from the high 53 bits (IEEE double mantissa).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[lo, hi)`.
    #[inline]
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Standard normal via Box–Muller (uses portable `libm`).
    pub fn normal(&mut self) -> f64 {
        // Reject u1 = 0 to avoid log(0).
        let mut u1 = self.next_f64();
        while u1 <= 1e-300 {
            u1 = self.next_f64();
        }
        let u2 = self.next_f64();
        let r = crate::libm_ops::sqrt(-2.0 * crate::libm_ops::log(u1));
        let theta = 2.0 * core::f64::consts::PI * u2;
        r * crate::libm_ops::cos(theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_is_deterministic() {
        let mut a = Xoshiro256StarStar::seed(42);
        let mut b = Xoshiro256StarStar::seed(42);
        for _ in 0..32 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Xoshiro256StarStar::seed(1);
        let mut b = Xoshiro256StarStar::seed(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn f64_in_unit_interval() {
        let mut r = Xoshiro256StarStar::seed(7);
        for _ in 0..1000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }
}
