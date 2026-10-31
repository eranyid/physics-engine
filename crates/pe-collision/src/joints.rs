//! Maximal-coordinate joints solved with soft PGS / soft-step (M5b).
//!
//! Clean-room from published Catto / Box2D v3 joint descriptions (algorithms only —
//! no source copied). Softness uses the same Hertz / ζ → (`bias_rate`, `mass_scale`,
//! `impulse_scale`) map as contacts ([`crate::softness::Softness`]).
//!
//! Supported kinds: distance, revolute (limits / motor / spring), prismatic
//! (limits / motor), weld (soft), wheel / suspension, mouse / target.
//!
//! Unverified / deferred (stated in `docs/JOINTS.md`): full 3-D gear / pulley,
//! Baumgarte-free hard XPBD joints, and Rapier-style reduced-coordinate solvers.

use pe_core::body::mat_vec;
use pe_math::libm_ops::{atan2, fabs, sin, sqrt};
use pe_math::{Quat, Vec3};

use crate::softness::Softness;

/// Sentinel: joint body index meaning “world / infinite-mass ground”.
pub const WORLD_BODY: usize = usize::MAX;

/// Softness defaults for joints (Box2D v3–style SI lift; Hertz before Nyquist cap).
pub const DEFAULT_JOINT_HERTZ: f64 = 60.0;
/// Default joint damping ratio ζ (critically-to-over damped).
pub const DEFAULT_JOINT_ZETA: f64 = 0.7;

/// Per-joint spring softness (Hertz + ζ). `None` → hard (`Softness::HARD`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointSoftParams {
    /// Spring frequency [Hz].
    pub hertz: f64,
    /// Damping ratio ζ (dimensionless).
    pub zeta: f64,
}

impl Default for JointSoftParams {
    fn default() -> Self {
        Self {
            hertz: DEFAULT_JOINT_HERTZ,
            zeta: DEFAULT_JOINT_ZETA,
        }
    }
}

impl JointSoftParams {
    /// Softness for sub-step `h`, capped at `0.25 / h` (same cap as contacts).
    pub fn softness(self, h: f64) -> Softness {
        if self.hertz <= 0.0 || h <= 0.0 {
            return Softness::HARD;
        }
        let cap = 0.25 / h;
        let hz = if self.hertz < cap { self.hertz } else { cap };
        Softness::from_hertz_zeta(hz, self.zeta, h)
    }
}

/// Motor / limit configuration shared by revolute and prismatic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointMotor {
    /// Enable velocity motor.
    pub enabled: bool,
    /// Target speed [rad/s] (revolute) or [m/s] (prismatic).
    pub target_speed: f64,
    /// Max motor force [N] or torque [N·m].
    pub max_force: f64,
}

impl Default for JointMotor {
    fn default() -> Self {
        Self {
            enabled: false,
            target_speed: 0.0,
            max_force: 0.0,
        }
    }
}

/// Angular or linear limit band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointLimit {
    /// Enable limit.
    pub enabled: bool,
    /// Lower bound [rad] or [m].
    pub lower: f64,
    /// Upper bound [rad] or [m].
    pub upper: f64,
}

impl Default for JointLimit {
    fn default() -> Self {
        Self {
            enabled: false,
            lower: 0.0,
            upper: 0.0,
        }
    }
}

/// Break thresholds. Non-positive ⇒ never break on that channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointBreak {
    /// Max |reaction force| [N] before break. `≤ 0` disables.
    pub force: f64,
    /// Max |reaction torque| [N·m] before break. `≤ 0` disables.
    pub torque: f64,
}

impl Default for JointBreak {
    fn default() -> Self {
        Self {
            force: 0.0,
            torque: 0.0,
        }
    }
}

/// Joint kind with type-specific geometry.
#[derive(Clone, Debug, PartialEq)]
pub enum JointKind {
    /// Keep distance between anchors equal to `length` [m].
    Distance {
        /// Rest length [m].
        length: f64,
    },
    /// Hinge: shared point + free rotation about `local_axis_a` (body A).
    Revolute {
        /// Hinge axis in body A [unit].
        local_axis_a: Vec3,
        /// Reference axis in body B for angle measure (body frame, unit).
        local_axis_b: Vec3,
        /// Reference zero-angle direction in A (⊥ axis), body frame.
        local_ref_a: Vec3,
        /// Matching reference in B.
        local_ref_b: Vec3,
        /// Soft torsional spring about the hinge (optional).
        spring_hertz: f64,
        /// Spring damping ζ.
        spring_zeta: f64,
        /// Rest angle [rad] for the spring.
        spring_angle: f64,
    },
    /// Slider along `local_axis_a` (body A); rotation locked about non-axis directions.
    Prismatic {
        /// Slide axis in body A [unit].
        local_axis_a: Vec3,
        /// Matching axis in body B [unit].
        local_axis_b: Vec3,
        /// Zero-offset reference (translation along axis at construction).
        reference_translation: f64,
    },
    /// Soft weld: lock relative pose (3 linear + 3 angular).
    Weld {
        /// Relative orientation bodyB→bodyA at construction (stored as quaternion).
        q_rel: Quat,
    },
    /// Wheel / suspension: point on axis + spring along axis; free spin about axis.
    Wheel {
        /// Suspension axis in body A [unit] (typically “up”).
        local_axis_a: Vec3,
        /// Matching axis in body B.
        local_axis_b: Vec3,
        /// Suspension spring Hertz.
        spring_hertz: f64,
        /// Suspension ζ.
        spring_zeta: f64,
        /// Rest length along axis [m] (signed offset of anchors projected on axis).
        rest_length: f64,
    },
    /// Soft mouse / target: pull body A anchor toward a world-space target.
    Mouse {
        /// World-space target [m].
        target: Vec3,
        /// Max force [N] (Coulomb-like clamp on soft impulse rate). `≤ 0` ⇒ uncapped.
        max_force: f64,
    },
}

/// A joint between two bodies (or body ↔ world).
#[derive(Clone, Debug)]
pub struct Joint {
    /// Body A index (`WORLD_BODY` allowed but unusual).
    pub body_a: usize,
    /// Body B index, or [`WORLD_BODY`] for ground.
    pub body_b: usize,
    /// Anchor on A in body A frame [m].
    pub local_anchor_a: Vec3,
    /// Anchor on B in body B frame [m] (ignored for mouse; world for `WORLD_BODY`).
    pub local_anchor_b: Vec3,
    /// Kind-specific data.
    pub kind: JointKind,
    /// Softness for bilateral rows (distance / point / angular).
    pub soft: JointSoftParams,
    /// Motor (revolute / prismatic).
    pub motor: JointMotor,
    /// Limit (revolute / prismatic).
    pub limit: JointLimit,
    /// Break thresholds.
    pub break_threshold: JointBreak,
    /// Accumulated reaction force [N] (world), last step — for queries.
    pub reaction_force: Vec3,
    /// Accumulated reaction torque on A [N·m] (world), last step.
    pub reaction_torque: Vec3,
    /// Broken joints are skipped by the solver.
    pub broken: bool,
    /// Warm-start scalar impulses (packed; length depends on kind).
    pub warm: Vec<f64>,
}

impl Joint {
    /// Distance joint of rest length `length` [m].
    pub fn distance(a: usize, b: usize, local_a: Vec3, local_b: Vec3, length: f64) -> Self {
        Self::base(a, b, local_a, local_b, JointKind::Distance { length })
    }

    /// Revolute (hinge) about `axis_a` (body A). Axes / refs should be unit and
    /// mutually consistent at construction (angle ≈ 0).
    #[allow(clippy::too_many_arguments)]
    pub fn revolute(
        a: usize,
        b: usize,
        local_a: Vec3,
        local_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        ref_a: Vec3,
        ref_b: Vec3,
    ) -> Self {
        Self::base(
            a,
            b,
            local_a,
            local_b,
            JointKind::Revolute {
                local_axis_a: axis_a.normalize(),
                local_axis_b: axis_b.normalize(),
                local_ref_a: ref_a.normalize(),
                local_ref_b: ref_b.normalize(),
                spring_hertz: 0.0,
                spring_zeta: 0.0,
                spring_angle: 0.0,
            },
        )
    }

    /// Prismatic (slider) along `axis_a`.
    pub fn prismatic(
        a: usize,
        b: usize,
        local_a: Vec3,
        local_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        reference_translation: f64,
    ) -> Self {
        Self::base(
            a,
            b,
            local_a,
            local_b,
            JointKind::Prismatic {
                local_axis_a: axis_a.normalize(),
                local_axis_b: axis_b.normalize(),
                reference_translation,
            },
        )
    }

    /// Soft weld locking relative pose `q_rel` = `q_a⁻¹ ⊗ q_b` at construction.
    pub fn weld(a: usize, b: usize, local_a: Vec3, local_b: Vec3, q_rel: Quat) -> Self {
        Self::base(a, b, local_a, local_b, JointKind::Weld { q_rel })
    }

    /// Wheel / suspension along `axis_a`.
    #[allow(clippy::too_many_arguments)]
    pub fn wheel(
        a: usize,
        b: usize,
        local_a: Vec3,
        local_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        spring_hertz: f64,
        spring_zeta: f64,
        rest_length: f64,
    ) -> Self {
        Self::base(
            a,
            b,
            local_a,
            local_b,
            JointKind::Wheel {
                local_axis_a: axis_a.normalize(),
                local_axis_b: axis_b.normalize(),
                spring_hertz,
                spring_zeta,
                rest_length,
            },
        )
    }

    /// Mouse / target joint pulling anchor A toward `target` (world).
    pub fn mouse(a: usize, local_a: Vec3, target: Vec3, max_force: f64) -> Self {
        let mut j = Self::base(
            a,
            WORLD_BODY,
            local_a,
            Vec3::zero(),
            JointKind::Mouse { target, max_force },
        );
        j.soft = JointSoftParams {
            hertz: 30.0,
            zeta: 0.7,
        };
        j
    }

    fn base(a: usize, b: usize, local_a: Vec3, local_b: Vec3, kind: JointKind) -> Self {
        Self {
            body_a: a,
            body_b: b,
            local_anchor_a: local_a,
            local_anchor_b: local_b,
            kind,
            soft: JointSoftParams::default(),
            motor: JointMotor::default(),
            limit: JointLimit::default(),
            break_threshold: JointBreak::default(),
            reaction_force: Vec3::zero(),
            reaction_torque: Vec3::zero(),
            broken: false,
            warm: Vec::new(),
        }
    }

    /// Builder: softness.
    pub fn with_soft(mut self, soft: JointSoftParams) -> Self {
        self.soft = soft;
        self
    }

    /// Builder: motor.
    pub fn with_motor(mut self, motor: JointMotor) -> Self {
        self.motor = motor;
        self
    }

    /// Builder: limit.
    pub fn with_limit(mut self, limit: JointLimit) -> Self {
        self.limit = limit;
        self
    }

    /// Builder: break thresholds.
    pub fn with_break(mut self, br: JointBreak) -> Self {
        self.break_threshold = br;
        self
    }

    /// Set mouse target (no-op for other kinds).
    pub fn set_target(&mut self, target: Vec3) {
        if let JointKind::Mouse {
            target: t,
            max_force: _,
        } = &mut self.kind
        {
            *t = target;
        }
    }
}

/// One scalar bilateral (or unilateral limit) soft constraint row.
#[derive(Clone, Debug)]
pub struct JointRow {
    /// Owning joint index into the joints slice.
    pub joint: usize,
    /// Body A index (or [`WORLD_BODY`]).
    pub a: usize,
    /// Body B index (or [`WORLD_BODY`]).
    pub b: usize,
    /// Impulse direction at A (world); B gets −dir at its lever.
    pub dir: Vec3,
    /// Lever arm from A COM to anchor [m].
    pub ra: Vec3,
    /// Lever arm from B COM to anchor [m].
    pub rb: Vec3,
    /// Effective mass scalar.
    pub k: f64,
    /// Softness for this row.
    pub soft: Softness,
    /// Position error C (constraint value).
    pub c: f64,
    /// Velocity bias target (motor / restitution-like); added to Ċ.
    pub bias_vel: f64,
    /// Accumulated impulse.
    pub lam: f64,
    /// Lower clamp on λ (for limits / motors). `-inf` as very large negative.
    pub lam_min: f64,
    /// Upper clamp on λ.
    pub lam_max: f64,
    /// If true, impulse contributes to reaction force (linear); else torque-only angular.
    pub is_linear: bool,
    /// Angular-only: apply ±dir as torque (no linear force). When false and
    /// `is_linear`, apply force at anchors.
    pub angular_only: bool,
}

const INF: f64 = 1e300;

fn rotate(r: [[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
        r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
        r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
    )
}

fn body_pose(
    i: usize,
    n: usize,
    x: &[Vec3],
    rmat: &[[[f64; 3]; 3]],
    static_x: &[Vec3],
    static_r: &[[[f64; 3]; 3]],
) -> (Vec3, [[f64; 3]; 3]) {
    if i == WORLD_BODY {
        return (Vec3::zero(), Quat::identity().to_matrix());
    }
    if i < n {
        (x[i], rmat[i])
    } else {
        let s = i - n;
        (static_x[s], static_r[s])
    }
}

#[allow(clippy::too_many_arguments)]
fn eff_mass_linear(
    dir: Vec3,
    ra: Vec3,
    rb: Vec3,
    a: usize,
    b: usize,
    n: usize,
    im: &[f64],
    iinv: &[[[f64; 3]; 3]],
) -> f64 {
    let mut den = 0.0;
    if a < n {
        den += im[a] + dir.dot(mat_vec(&iinv[a], ra.cross(dir)).cross(ra));
    }
    if b < n {
        den += im[b] + dir.dot(mat_vec(&iinv[b], rb.cross(dir)).cross(rb));
    }
    if den > 0.0 {
        1.0 / den
    } else {
        0.0
    }
}

fn eff_mass_angular(axis: Vec3, a: usize, b: usize, n: usize, iinv: &[[[f64; 3]; 3]]) -> f64 {
    let mut den = 0.0;
    if a < n {
        den += axis.dot(mat_vec(&iinv[a], axis));
    }
    if b < n {
        den += axis.dot(mat_vec(&iinv[b], axis));
    }
    if den > 0.0 {
        1.0 / den
    } else {
        0.0
    }
}

fn orthonormal_basis(axis: Vec3) -> (Vec3, Vec3) {
    let a = axis.normalize();
    let t = if fabs(a.x) < 0.9 {
        a.cross(Vec3::unit_x())
    } else {
        a.cross(Vec3::unit_y())
    };
    let t1 = t.normalize();
    let t2 = a.cross(t1);
    (t1, t2)
}

fn hinge_angle(axis: Vec3, ref_a: Vec3, ref_b: Vec3) -> f64 {
    // Project refs into plane ⊥ axis; atan2 of oriented angle from A to B.
    let xa = (ref_a - axis * axis.dot(ref_a)).normalize();
    let xb = (ref_b - axis * axis.dot(ref_b)).normalize();
    let s = axis.dot(xa.cross(xb));
    let c = xa.dot(xb);
    atan2(s, c)
}

/// Build joint constraint rows for the current poses / velocities.
#[allow(clippy::too_many_arguments)]
pub fn build_joint_rows(
    joints: &[Joint],
    n: usize,
    x: &[Vec3],
    rmat: &[[[f64; 3]; 3]],
    static_x: &[Vec3],
    static_r: &[[[f64; 3]; 3]],
    im: &[f64],
    iinv: &[[[f64; 3]; 3]],
    soft_h: f64,
    use_soft: bool,
) -> Vec<JointRow> {
    let mut rows = Vec::new();
    for (ji, j) in joints.iter().enumerate() {
        if j.broken {
            continue;
        }
        let soft = if use_soft {
            j.soft.softness(soft_h)
        } else {
            Softness::HARD
        };
        let (xa, ra_m) = body_pose(j.body_a, n, x, rmat, static_x, static_r);
        let (xb, rb_m) = body_pose(j.body_b, n, x, rmat, static_x, static_r);
        let ra = if j.body_a == WORLD_BODY {
            j.local_anchor_a
        } else {
            rotate(ra_m, j.local_anchor_a)
        };
        let rb = if j.body_b == WORLD_BODY {
            j.local_anchor_b
        } else {
            rotate(rb_m, j.local_anchor_b)
        };
        let pa = xa + ra;
        let pb = xb + rb;

        match &j.kind {
            JointKind::Distance { length } => {
                let d = pb - pa;
                let dist = d.norm();
                let dir = if dist > 1e-12 {
                    d / dist
                } else {
                    Vec3::unit_x()
                };
                let c = dist - *length;
                rows.push(JointRow {
                    joint: ji,
                    a: j.body_a,
                    b: j.body_b,
                    dir,
                    ra,
                    rb,
                    k: eff_mass_linear(dir, ra, rb, j.body_a, j.body_b, n, im, iinv),
                    soft,
                    c,
                    bias_vel: 0.0,
                    lam: 0.0,
                    lam_min: -INF,
                    lam_max: INF,
                    is_linear: true,
                    angular_only: false,
                });
            }
            JointKind::Revolute {
                local_axis_a,
                local_axis_b,
                local_ref_a,
                local_ref_b,
                spring_hertz,
                spring_zeta,
                spring_angle,
            } => {
                // Point constraint (3 axes).
                let delta = pb - pa;
                for axis in [Vec3::unit_x(), Vec3::unit_y(), Vec3::unit_z()] {
                    rows.push(JointRow {
                        joint: ji,
                        a: j.body_a,
                        b: j.body_b,
                        dir: axis,
                        ra,
                        rb,
                        k: eff_mass_linear(axis, ra, rb, j.body_a, j.body_b, n, im, iinv),
                        soft,
                        c: delta.dot(axis),
                        bias_vel: 0.0,
                        lam: 0.0,
                        lam_min: -INF,
                        lam_max: INF,
                        is_linear: true,
                        angular_only: false,
                    });
                }
                let ax_a = rotate(ra_m, *local_axis_a);
                let ax_b = rotate(rb_m, *local_axis_b);
                // Align hinge axes: two constraints ax_a × ax_b ≈ 0 (⊥ components).
                let (t1, t2) = orthonormal_basis(ax_a);
                for t in [t1, t2] {
                    let c = t.dot(ax_b); // want ax_b ⊥ t ⇒ c=0 when ax_b ‖ ax_a
                                         // Angular Jacobian: for small error, torque about t×ax_a direction.
                                         // Use axis = t × ax_a as the angular impulse direction (standard).
                    let axis = t.cross(ax_a);
                    let an = axis.norm();
                    if an < 1e-12 {
                        continue;
                    }
                    let axis = axis / an;
                    rows.push(JointRow {
                        joint: ji,
                        a: j.body_a,
                        b: j.body_b,
                        dir: axis,
                        ra: Vec3::zero(),
                        rb: Vec3::zero(),
                        k: eff_mass_angular(axis, j.body_a, j.body_b, n, iinv),
                        soft,
                        // C ≈ angle between axes projected; use ax_a·(ax_b×t) style:
                        c: ax_a.cross(ax_b).dot(t),
                        bias_vel: 0.0,
                        lam: 0.0,
                        lam_min: -INF,
                        lam_max: INF,
                        is_linear: false,
                        angular_only: true,
                    });
                    let _ = c;
                }
                let ref_aw = rotate(ra_m, *local_ref_a);
                let ref_bw = rotate(rb_m, *local_ref_b);
                let angle = hinge_angle(ax_a, ref_aw, ref_bw);

                if *spring_hertz > 0.0 {
                    let soft_s = Softness::from_hertz_zeta(
                        {
                            let cap = 0.25 / soft_h.max(1e-12);
                            if *spring_hertz < cap {
                                *spring_hertz
                            } else {
                                cap
                            }
                        },
                        *spring_zeta,
                        soft_h,
                    );
                    rows.push(JointRow {
                        joint: ji,
                        a: j.body_a,
                        b: j.body_b,
                        dir: ax_a,
                        ra: Vec3::zero(),
                        rb: Vec3::zero(),
                        k: eff_mass_angular(ax_a, j.body_a, j.body_b, n, iinv),
                        soft: soft_s,
                        c: angle - *spring_angle,
                        bias_vel: 0.0,
                        lam: 0.0,
                        lam_min: -INF,
                        lam_max: INF,
                        is_linear: false,
                        angular_only: true,
                    });
                }
                if j.limit.enabled {
                    let lo = j.limit.lower;
                    let hi = j.limit.upper;
                    if angle <= lo {
                        rows.push(JointRow {
                            joint: ji,
                            a: j.body_a,
                            b: j.body_b,
                            dir: ax_a,
                            ra: Vec3::zero(),
                            rb: Vec3::zero(),
                            k: eff_mass_angular(ax_a, j.body_a, j.body_b, n, iinv),
                            soft,
                            c: angle - lo,
                            bias_vel: 0.0,
                            lam: 0.0,
                            // Unilateral: push angle up ⇒ λ ≥ 0 on this sign convention
                            lam_min: 0.0,
                            lam_max: INF,
                            is_linear: false,
                            angular_only: true,
                        });
                    } else if angle >= hi {
                        rows.push(JointRow {
                            joint: ji,
                            a: j.body_a,
                            b: j.body_b,
                            dir: -ax_a,
                            ra: Vec3::zero(),
                            rb: Vec3::zero(),
                            k: eff_mass_angular(ax_a, j.body_a, j.body_b, n, iinv),
                            soft,
                            c: hi - angle,
                            bias_vel: 0.0,
                            lam: 0.0,
                            lam_min: 0.0,
                            lam_max: INF,
                            is_linear: false,
                            angular_only: true,
                        });
                    }
                }
                if j.motor.enabled {
                    let max_tau = j.motor.max_force;
                    // Motor: Ċ = (ω_b − ω_a)·axis; want ω_a·axis → target with b fixed
                    // ⇒ (−ω_a)·axis + bias = 0 ⇒ bias = target.
                    rows.push(JointRow {
                        joint: ji,
                        a: j.body_a,
                        b: j.body_b,
                        dir: ax_a,
                        ra: Vec3::zero(),
                        rb: Vec3::zero(),
                        k: eff_mass_angular(ax_a, j.body_a, j.body_b, n, iinv),
                        soft: Softness::HARD,
                        c: 0.0,
                        bias_vel: j.motor.target_speed,
                        lam: 0.0,
                        lam_min: -max_tau * soft_h.max(1e-12),
                        lam_max: max_tau * soft_h.max(1e-12),
                        is_linear: false,
                        angular_only: true,
                    });
                }
            }
            JointKind::Prismatic {
                local_axis_a,
                local_axis_b: _,
                reference_translation,
            } => {
                let axis = rotate(ra_m, *local_axis_a);
                let (t1, t2) = orthonormal_basis(axis);
                let delta = pb - pa;
                // Lock perpendicular translation.
                for t in [t1, t2] {
                    rows.push(JointRow {
                        joint: ji,
                        a: j.body_a,
                        b: j.body_b,
                        dir: t,
                        ra,
                        rb,
                        k: eff_mass_linear(t, ra, rb, j.body_a, j.body_b, n, im, iinv),
                        soft,
                        c: delta.dot(t),
                        bias_vel: 0.0,
                        lam: 0.0,
                        lam_min: -INF,
                        lam_max: INF,
                        is_linear: true,
                        angular_only: false,
                    });
                }
                // Lock angular DOFs except about axis (two constraints).
                for t in [t1, t2] {
                    rows.push(JointRow {
                        joint: ji,
                        a: j.body_a,
                        b: j.body_b,
                        dir: t,
                        ra: Vec3::zero(),
                        rb: Vec3::zero(),
                        k: eff_mass_angular(t, j.body_a, j.body_b, n, iinv),
                        soft,
                        c: 0.0, // velocity-level angular lock; position via soft bias of relative ω
                        bias_vel: 0.0,
                        lam: 0.0,
                        lam_min: -INF,
                        lam_max: INF,
                        is_linear: false,
                        angular_only: true,
                    });
                }
                let trans = delta.dot(axis) - *reference_translation;
                if j.limit.enabled {
                    let lo = j.limit.lower;
                    let hi = j.limit.upper;
                    if trans <= lo {
                        rows.push(JointRow {
                            joint: ji,
                            a: j.body_a,
                            b: j.body_b,
                            dir: axis,
                            ra,
                            rb,
                            k: eff_mass_linear(axis, ra, rb, j.body_a, j.body_b, n, im, iinv),
                            soft,
                            c: trans - lo,
                            bias_vel: 0.0,
                            lam: 0.0,
                            lam_min: 0.0,
                            lam_max: INF,
                            is_linear: true,
                            angular_only: false,
                        });
                    } else if trans >= hi {
                        rows.push(JointRow {
                            joint: ji,
                            a: j.body_a,
                            b: j.body_b,
                            dir: -axis,
                            ra,
                            rb,
                            k: eff_mass_linear(axis, ra, rb, j.body_a, j.body_b, n, im, iinv),
                            soft,
                            c: hi - trans,
                            bias_vel: 0.0,
                            lam: 0.0,
                            lam_min: 0.0,
                            lam_max: INF,
                            is_linear: true,
                            angular_only: false,
                        });
                    }
                }
                if j.motor.enabled {
                    let max_f = j.motor.max_force;
                    // Ċ = (v_b − v_a)·axis; with b fixed want −v_a·axis + bias = 0
                    // at v_a·axis = target ⇒ bias = target.
                    rows.push(JointRow {
                        joint: ji,
                        a: j.body_a,
                        b: j.body_b,
                        dir: axis,
                        ra,
                        rb,
                        k: eff_mass_linear(axis, ra, rb, j.body_a, j.body_b, n, im, iinv),
                        soft: Softness::HARD,
                        c: 0.0,
                        bias_vel: j.motor.target_speed,
                        lam: 0.0,
                        lam_min: -max_f * soft_h.max(1e-12),
                        lam_max: max_f * soft_h.max(1e-12),
                        is_linear: true,
                        angular_only: false,
                    });
                }
            }
            JointKind::Weld { q_rel } => {
                let delta = pb - pa;
                for axis in [Vec3::unit_x(), Vec3::unit_y(), Vec3::unit_z()] {
                    rows.push(JointRow {
                        joint: ji,
                        a: j.body_a,
                        b: j.body_b,
                        dir: axis,
                        ra,
                        rb,
                        k: eff_mass_linear(axis, ra, rb, j.body_a, j.body_b, n, im, iinv),
                        soft,
                        c: delta.dot(axis),
                        bias_vel: 0.0,
                        lam: 0.0,
                        lam_min: -INF,
                        lam_max: INF,
                        is_linear: true,
                        angular_only: false,
                    });
                }
                // Angular: q_err = q_a ⊗ q_rel ⊗ q_b⁻¹ ≈ identity; use small-angle vector.
                let qa = if j.body_a < n {
                    // Reconstruct from matrix is heavy; use stored rmat via quat from callers.
                    // Approximate angular error from relative rotation matrix.
                    let _ = q_rel;
                    angular_error_from_rel(ra_m, rb_m, *q_rel)
                } else {
                    Vec3::zero()
                };
                for (k, axis) in [Vec3::unit_x(), Vec3::unit_y(), Vec3::unit_z()]
                    .into_iter()
                    .enumerate()
                {
                    let c = match k {
                        0 => qa.x,
                        1 => qa.y,
                        _ => qa.z,
                    };
                    rows.push(JointRow {
                        joint: ji,
                        a: j.body_a,
                        b: j.body_b,
                        dir: axis,
                        ra: Vec3::zero(),
                        rb: Vec3::zero(),
                        k: eff_mass_angular(axis, j.body_a, j.body_b, n, iinv),
                        soft,
                        c,
                        bias_vel: 0.0,
                        lam: 0.0,
                        lam_min: -INF,
                        lam_max: INF,
                        is_linear: false,
                        angular_only: true,
                    });
                }
            }
            JointKind::Wheel {
                local_axis_a,
                local_axis_b: _,
                spring_hertz,
                spring_zeta,
                rest_length,
            } => {
                let axis = rotate(ra_m, *local_axis_a);
                let (t1, t2) = orthonormal_basis(axis);
                let delta = pb - pa;
                for t in [t1, t2] {
                    rows.push(JointRow {
                        joint: ji,
                        a: j.body_a,
                        b: j.body_b,
                        dir: t,
                        ra,
                        rb,
                        k: eff_mass_linear(t, ra, rb, j.body_a, j.body_b, n, im, iinv),
                        soft,
                        c: delta.dot(t),
                        bias_vel: 0.0,
                        lam: 0.0,
                        lam_min: -INF,
                        lam_max: INF,
                        is_linear: true,
                        angular_only: false,
                    });
                }
                let soft_s = Softness::from_hertz_zeta(
                    {
                        let cap = 0.25 / soft_h.max(1e-12);
                        if *spring_hertz < cap {
                            *spring_hertz
                        } else {
                            cap
                        }
                    },
                    *spring_zeta,
                    soft_h,
                );
                let sep = delta.dot(axis) - *rest_length;
                rows.push(JointRow {
                    joint: ji,
                    a: j.body_a,
                    b: j.body_b,
                    dir: axis,
                    ra,
                    rb,
                    k: eff_mass_linear(axis, ra, rb, j.body_a, j.body_b, n, im, iinv),
                    soft: soft_s,
                    c: sep,
                    bias_vel: 0.0,
                    lam: 0.0,
                    lam_min: -INF,
                    lam_max: INF,
                    is_linear: true,
                    angular_only: false,
                });
            }
            JointKind::Mouse { target, max_force } => {
                let delta = *target - pa;
                for axis in [Vec3::unit_x(), Vec3::unit_y(), Vec3::unit_z()] {
                    let mut lam_min = -INF;
                    let mut lam_max = INF;
                    if *max_force > 0.0 {
                        let lim = *max_force * soft_h.max(1e-12);
                        lam_min = -lim;
                        lam_max = lim;
                    }
                    rows.push(JointRow {
                        joint: ji,
                        a: j.body_a,
                        b: WORLD_BODY,
                        dir: axis,
                        ra,
                        rb: Vec3::zero(),
                        k: eff_mass_linear(
                            axis,
                            ra,
                            Vec3::zero(),
                            j.body_a,
                            WORLD_BODY,
                            n,
                            im,
                            iinv,
                        ),
                        soft,
                        c: -delta.dot(axis), // C = pa − target; drive to 0
                        bias_vel: 0.0,
                        lam: 0.0,
                        lam_min,
                        lam_max,
                        is_linear: true,
                        angular_only: false,
                    });
                }
            }
        }
    }
    rows
}

fn angular_error_from_rel(ra: [[f64; 3]; 3], rb: [[f64; 3]; 3], q_rel: Quat) -> Vec3 {
    // Desired: R_a * R(q_rel) = R_b  ⇒  R_err = R_a * R(q_rel) * R_bᵀ ≈ I
    let r_rel = q_rel.to_matrix();
    let mut rab = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rab[i][j] = ra[i][0] * r_rel[0][j] + ra[i][1] * r_rel[1][j] + ra[i][2] * r_rel[2][j];
        }
    }
    // R_err = rab * rbᵀ
    let mut err = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            err[i][j] = rab[i][0] * rb[j][0] + rab[i][1] * rb[j][1] + rab[i][2] * rb[j][2];
        }
    }
    // Small-angle: θ_x ≈ (R32 − R23)/2, …
    Vec3::new(
        0.5 * (err[2][1] - err[1][2]),
        0.5 * (err[0][2] - err[2][0]),
        0.5 * (err[1][0] - err[0][1]),
    )
}

fn omega_at(i: usize, n: usize, l: &[Vec3], iinv: &[[[f64; 3]; 3]]) -> Vec3 {
    if i < n {
        mat_vec(&iinv[i], l[i])
    } else {
        Vec3::zero()
    }
}

fn vel_at(i: usize, n: usize, v: &[Vec3], l: &[Vec3], iinv: &[[[f64; 3]; 3]], r: Vec3) -> Vec3 {
    if i < n {
        v[i] + omega_at(i, n, l, iinv).cross(r)
    } else {
        Vec3::zero()
    }
}

/// Apply joint impulse (force at anchors or pure torque).
#[allow(clippy::too_many_arguments)]
fn apply_joint_impulse(
    r: &JointRow,
    n: usize,
    awake: Option<&[bool]>,
    p_or_tau: f64,
    v: &mut [Vec3],
    l: &mut [Vec3],
    im: &[f64],
) {
    let awake_ok = |i: usize| {
        if i >= n {
            true
        } else {
            match awake {
                Some(a) => a[i],
                None => true,
            }
        }
    };
    if r.angular_only {
        let tau = r.dir * p_or_tau;
        if r.a < n && awake_ok(r.a) {
            l[r.a] -= tau;
        }
        if r.b < n && awake_ok(r.b) {
            l[r.b] += tau;
        }
    } else {
        let p = r.dir * p_or_tau;
        if r.a < n && awake_ok(r.a) {
            v[r.a] -= p * im[r.a];
            l[r.a] -= r.ra.cross(p);
        }
        if r.b < n && awake_ok(r.b) {
            v[r.b] += p * im[r.b];
            l[r.b] += r.rb.cross(p);
        }
    }
}

/// Warm-start joint rows from stored impulses, applying them to velocities.
pub fn warm_start_joints(
    rows: &mut [JointRow],
    joints: &[Joint],
    n: usize,
    awake: Option<&[bool]>,
    v: &mut [Vec3],
    l: &mut [Vec3],
    im: &[f64],
) {
    // Count rows per joint to index into warm vectors.
    let mut cursor = vec![0usize; joints.len()];
    for r in rows.iter_mut() {
        let ji = r.joint;
        let idx = cursor[ji];
        cursor[ji] = idx + 1;
        if idx < joints[ji].warm.len() {
            r.lam = joints[ji].warm[idx];
            if r.lam != 0.0 {
                apply_joint_impulse(r, n, awake, r.lam, v, l, im);
            }
        }
    }
}

/// One PGS pass over joint rows (soft or hard).
pub fn solve_joint_rows(
    rows: &mut [JointRow],
    n: usize,
    awake: Option<&[bool]>,
    v: &mut [Vec3],
    l: &mut [Vec3],
    iinv: &[[[f64; 3]; 3]],
    im: &[f64],
) {
    for r in rows.iter_mut() {
        let cdot = if r.angular_only {
            let wa = omega_at(r.a, n, l, iinv);
            let wb = omega_at(r.b, n, l, iinv);
            (wb - wa).dot(r.dir)
        } else {
            let va = vel_at(r.a, n, v, l, iinv, r.ra);
            let vb = vel_at(r.b, n, v, l, iinv, r.rb);
            (vb - va).dot(r.dir)
        };
        let bias = r.soft.bias_rate * r.c + r.bias_vel;
        let incremental = -r.soft.mass_scale * r.k * (cdot + bias) - r.soft.impulse_scale * r.lam;
        let new_lam = (r.lam + incremental).clamp(r.lam_min, r.lam_max);
        let applied = new_lam - r.lam;
        r.lam = new_lam;
        if applied != 0.0 {
            apply_joint_impulse(r, n, awake, applied, v, l, im);
        }
    }
}

/// Write warm impulses back and accumulate reaction force/torque; mark breaks.
pub fn finalize_joints(joints: &mut [Joint], rows: &[JointRow], h: f64) {
    for j in joints.iter_mut() {
        j.warm.clear();
        j.reaction_force = Vec3::zero();
        j.reaction_torque = Vec3::zero();
    }
    for r in rows {
        let j = &mut joints[r.joint];
        j.warm.push(r.lam);
        // Impulse → average force over sub-step h.
        let inv_h = if h > 0.0 { 1.0 / h } else { 0.0 };
        if r.angular_only {
            j.reaction_torque += r.dir * (r.lam * inv_h);
        } else if r.is_linear {
            j.reaction_force += r.dir * (r.lam * inv_h);
            j.reaction_torque += r.ra.cross(r.dir * (r.lam * inv_h));
        }
    }
    for j in joints.iter_mut() {
        if j.broken {
            continue;
        }
        let bf = j.break_threshold.force;
        let bt = j.break_threshold.torque;
        if bf > 0.0 && j.reaction_force.norm() > bf {
            j.broken = true;
        }
        if bt > 0.0 && j.reaction_torque.norm() > bt {
            j.broken = true;
        }
    }
}

/// Collect sorted unique dynamic–dynamic pairs connected by active joints.
pub fn joint_island_pairs(joints: &[Joint], n: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for j in joints {
        if j.broken {
            continue;
        }
        let a = j.body_a;
        let b = j.body_b;
        if a < n && b < n {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            pairs.push((lo, hi));
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Merge contact pairs with joint pairs (sorted).
pub fn merge_island_pairs(
    contact_pairs: &[(usize, usize)],
    joint_pairs: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(contact_pairs.len() + joint_pairs.len());
    out.extend_from_slice(contact_pairs);
    out.extend_from_slice(joint_pairs);
    out.sort_unstable();
    out.dedup();
    out
}

/// Relative hinge angle [rad] for a revolute joint (0 at construction refs).
pub fn revolute_angle(
    joint: &Joint,
    n: usize,
    x: &[Vec3],
    q: &[Quat],
    static_x: &[Vec3],
    static_q: &[Quat],
) -> Option<f64> {
    let JointKind::Revolute {
        local_axis_a,
        local_ref_a,
        local_ref_b,
        ..
    } = &joint.kind
    else {
        return None;
    };
    let qa = if joint.body_a == WORLD_BODY {
        Quat::identity()
    } else if joint.body_a < n {
        q[joint.body_a]
    } else {
        static_q[joint.body_a - n]
    };
    let qb = if joint.body_b == WORLD_BODY {
        Quat::identity()
    } else if joint.body_b < n {
        q[joint.body_b]
    } else {
        static_q[joint.body_b - n]
    };
    let _ = (x, static_x);
    let ax = qa.rotate(*local_axis_a);
    let ra = qa.rotate(*local_ref_a);
    let rb = qb.rotate(*local_ref_b);
    Some(hinge_angle(ax, ra, rb))
}

/// Analytic small-angle pendulum period `2π √(L/g)`.
pub fn pendulum_period_small(length: f64, g: f64) -> f64 {
    2.0 * core::f64::consts::PI * sqrt(length / g)
}

/// Complete elliptic integral of the first kind K(k) via AGM (Abramowitz & Stegun).
pub fn elliptic_k(k: f64) -> f64 {
    // K(k) = π/2 / AGM(1, √(1−k²))
    let mut a = 1.0;
    let mut b = sqrt((1.0 - k * k).max(0.0));
    for _ in 0..32 {
        let an = 0.5 * (a + b);
        let bn = sqrt(a * b);
        a = an;
        b = bn;
        if fabs(a - b) < 1e-15 {
            break;
        }
    }
    core::f64::consts::PI / (2.0 * a)
}

/// Large-amplitude simple-pendulum period `T = 4 √(L/g) K(sin(θ₀/2))`.
pub fn pendulum_period_large(length: f64, g: f64, theta0: f64) -> f64 {
    let k = sin(0.5 * theta0);
    4.0 * sqrt(length / g) * elliptic_k(k)
}

/// Spring–mass period `2π √(m/k)` (undamped).
pub fn spring_mass_period(mass: f64, stiffness: f64) -> f64 {
    2.0 * core::f64::consts::PI * sqrt(mass / stiffness)
}

/// Map Hertz to undamped stiffness for effective-mass `m`: `k = m (2π f)²`.
pub fn hertz_to_stiffness(mass: f64, hertz: f64) -> f64 {
    let w = 2.0 * core::f64::consts::PI * hertz;
    mass * w * w
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn small_pendulum_period_formula() {
        let t = pendulum_period_small(1.0, 9.80665);
        assert_relative_eq!(t, 2.006409, epsilon = 1e-6);
    }

    #[test]
    fn large_amplitude_longer_than_small() {
        let t0 = pendulum_period_small(1.0, 9.80665);
        let t1 = pendulum_period_large(1.0, 9.80665, 1.0);
        assert!(t1 > t0);
    }

    #[test]
    fn elliptic_k_zero_is_pi_over_2() {
        assert_relative_eq!(
            elliptic_k(0.0),
            core::f64::consts::PI / 2.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn joint_pairs_sorted_unique() {
        let j = [
            Joint::distance(1, 0, Vec3::zero(), Vec3::zero(), 1.0),
            Joint::distance(0, 1, Vec3::zero(), Vec3::zero(), 1.0),
            Joint::distance(2, WORLD_BODY, Vec3::zero(), Vec3::zero(), 1.0),
        ];
        let p = joint_island_pairs(&j, 3);
        assert_eq!(p, vec![(0, 1)]);
    }
}
