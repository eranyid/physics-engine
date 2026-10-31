//! Compensated (Kahan / Neumaier) summation and origin rebasing helpers.
//!
//! Long orbital integrations accumulate round-off in position updates. Compensated
//! summation keeps a running error term; origin rebasing recentres the system when
//! coordinates grow large relative to inter-body separations.
//!
//! References (algorithms, not code):
//! - Kahan, W. (1965). Further remarks on reducing truncation errors.
//! - Neumaier, A. (1974). Rundungsfehleranalyse einiger Verfahren zur Summation
//!   endlicher Summen. *ZAMM*.

use pe_math::Vec3;

/// Neumaier compensated sum of a slice of `f64` (more robust than plain Kahan when
/// terms change sign / magnitude).
pub fn neumaier_sum(xs: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for &x in xs {
        let t = sum + x;
        if pe_math::libm_ops::fabs(sum) >= pe_math::libm_ops::fabs(x) {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }
        sum = t;
    }
    sum + c
}

/// Classic Kahan summation of a slice.
pub fn kahan_sum(xs: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for &x in xs {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Compensated 3-vector accumulator (independent Neumaier channels per component).
#[derive(Clone, Debug, Default)]
pub struct CompensatedVec3 {
    /// Current value [same units as accumulated increments].
    pub value: Vec3,
    /// Running compensation [same units].
    compensation: Vec3,
}

impl CompensatedVec3 {
    /// Start at `v` with zero compensation.
    pub fn new(v: Vec3) -> Self {
        Self {
            value: v,
            compensation: Vec3::zero(),
        }
    }

    /// Add `delta` with Neumaier compensation (component-wise).
    pub fn add(&mut self, delta: Vec3) {
        self.value.x = Self::neumaier_step(self.value.x, delta.x, &mut self.compensation.x);
        self.value.y = Self::neumaier_step(self.value.y, delta.y, &mut self.compensation.y);
        self.value.z = Self::neumaier_step(self.value.z, delta.z, &mut self.compensation.z);
    }

    fn neumaier_step(sum: f64, x: f64, c: &mut f64) -> f64 {
        let t = sum + x;
        if pe_math::libm_ops::fabs(sum) >= pe_math::libm_ops::fabs(x) {
            *c += (sum - t) + x;
        } else {
            *c += (x - t) + sum;
        }
        t
    }

    /// Flush compensation into the value and zero the residual.
    pub fn flush(&mut self) {
        self.value += self.compensation;
        self.compensation = Vec3::zero();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn neumaier_beats_naive_on_cancellation() {
        // Classic example: many tiny terms + a large cancellation.
        let mut xs = vec![1.0, 1e-16, 1e-16, 1e-16, 1e-16];
        xs.push(-1.0);
        let naive: f64 = xs.iter().sum();
        let neu = neumaier_sum(&xs);
        // Naive often loses the 4e-16; Neumaier retains it.
        assert!(neu > naive - 1e-30);
        assert_abs_diff_eq!(neu, 4e-16, epsilon = 1e-30);
    }
}
