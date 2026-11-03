//! Simulation baking: run a scene offline and export per-body transform curves.
//!
//! # Interpolation (exact sample times between physics steps)
//!
//! Physics advances at fixed `physics_dt`. Output samples are requested at
//! `t_i = i / sample_fps` (plus a final sample at `duration` when that time is
//! not already on the grid). Between physics knots `(t₀, t₁)` with
//! `s = (t − t₀) / physics_dt ∈ [0, 1]`:
//!
//! * **Position** — cubic Hermite using endpoint positions and linear velocities:
//!   `p(s) = h₀₀ p₀ + h₁₀ Δt v₀ + h₀₁ p₁ + h₁₁ Δt v₁` with the standard Hermite
//!   basis `h₀₀ = 2s³−3s²+1`, `h₁₀ = s³−2s²+s`, `h₀₁ = −2s³+3s²`, `h₁₁ = s³−s²`.
//! * **Rotation** — unit-quaternion [`Quat::slerp`] (short arc).
//! * **Velocities** (optional) — linear lerp of `v` and world-frame `ω`.
//!
//! When `t` lands exactly on a physics knot, the stored sample is copied (no
//! interpolation). Resamples at different `sample_fps` therefore agree
//! **bit-exactly** at coincident times (same physics trajectory + same
//! Hermite/slerp).
//!
//! # Keyframe reduction
//!
//! Optional Douglas–Peucker-style reduction over the dense output samples,
//! measuring deviation against Hermite (position) and slerp (rotation) between
//! candidate endpoints. Every discarded sample reconstructs within the stated
//! `max_position_error` [m] and `max_rotation_error` [rad] bounds.
//!
//! Schema and Swift usage: [`docs/BAKE.md`](../../../docs/BAKE.md).

use pe_math::{libm_ops, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::document::SceneDocument;
use crate::error::SceneError;

/// Bake document schema version written by this crate.
pub const BAKE_SCHEMA_VERSION: u32 = 1;

/// Options for [`bake_scene`].
#[derive(Clone, Debug, PartialEq)]
pub struct BakeOptions {
    /// Simulation duration [s] (`> 0`).
    pub duration: f64,
    /// Fixed physics step [s] (`> 0`).
    pub physics_dt: f64,
    /// Output sample rate [Hz] (`> 0`).
    pub sample_fps: f64,
    /// Include linear and angular velocity curves.
    pub include_velocities: bool,
    /// Optional keyframe reduction error bound.
    pub reduce: Option<BakeReduceParams>,
}

impl Default for BakeOptions {
    fn default() -> Self {
        Self {
            duration: 1.0,
            physics_dt: 1.0 / 240.0,
            sample_fps: 24.0,
            include_velocities: true,
            reduce: None,
        }
    }
}

/// Max reconstruction error for keyframe reduction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BakeReduceParams {
    /// Max Euclidean position error [m] vs the dense Hermite/slerp curve.
    pub max_position_error: f64,
    /// Max short-arc rotation error [rad] vs the dense slerp curve.
    pub max_rotation_error: f64,
}

/// Interpolation method labels recorded in the bake document.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BakeInterpolation {
    /// Position interpolant (`"hermite"`).
    pub position: String,
    /// Rotation interpolant (`"slerp"`).
    pub rotation: String,
}

impl Default for BakeInterpolation {
    fn default() -> Self {
        Self {
            position: "hermite".into(),
            rotation: "slerp".into(),
        }
    }
}

/// One body's transform curve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BakeBodyCurve {
    /// Body index in the source scene.
    pub index: usize,
    /// Optional debug name from the scene.
    #[serde(default)]
    pub name: String,
    /// Sample times [s] (strictly increasing after reduction; uniform before).
    pub times: Vec<f64>,
    /// Positions `[x,y,z]` [m] at each time.
    pub position: Vec<[f64; 3]>,
    /// Orientations `[w,x,y,z]` at each time.
    pub rotation: Vec<[f64; 4]>,
    /// Linear velocities `[vx,vy,vz]` [m/s] when requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_velocity: Option<Vec<[f64; 3]>>,
    /// World-frame angular velocities `[ωx,ωy,ωz]` [rad/s] when requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angular_velocity: Option<Vec<[f64; 3]>>,
}

/// Versioned bake document (JSON schema v1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BakeDocument {
    /// Schema version (currently 1).
    pub schema_version: u32,
    /// Physics step used during the bake [s].
    pub physics_dt: f64,
    /// Requested duration [s].
    pub duration: f64,
    /// Requested output rate [Hz] (before reduction).
    pub sample_fps: f64,
    /// Whether velocity curves are present.
    pub include_velocities: bool,
    /// Interpolation method used to build samples from physics knots.
    pub interpolation: BakeInterpolation,
    /// Reduction parameters applied, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reduction: Option<BakeReduceParams>,
    /// Per-body curves (order = body index).
    pub bodies: Vec<BakeBodyCurve>,
}

impl BakeDocument {
    /// Compact JSON (deterministic field order from serde derive).
    pub fn to_json(&self) -> Result<String, SceneError> {
        serde_json::to_string(self).map_err(|e| SceneError::Parse {
            format: "json",
            message: e.to_string(),
        })
    }

    /// Pretty JSON.
    pub fn to_json_pretty(&self) -> Result<String, SceneError> {
        serde_json::to_string_pretty(self).map_err(|e| SceneError::Parse {
            format: "json",
            message: e.to_string(),
        })
    }

    /// Parse bake JSON (never panics).
    pub fn from_json(s: &str) -> Result<Self, SceneError> {
        let doc: Self = serde_json::from_str(s)
            .map_err(|e| crate::error::map_format_error("json", e.to_string()))?;
        if doc.schema_version != BAKE_SCHEMA_VERSION {
            return Err(SceneError::UnsupportedVersion {
                found: doc.schema_version,
                max_supported: BAKE_SCHEMA_VERSION,
            });
        }
        doc.validate()?;
        Ok(doc)
    }

    /// Structural validation.
    pub fn validate(&self) -> Result<(), SceneError> {
        if self.schema_version != BAKE_SCHEMA_VERSION {
            return Err(SceneError::UnsupportedVersion {
                found: self.schema_version,
                max_supported: BAKE_SCHEMA_VERSION,
            });
        }
        for (bi, b) in self.bodies.iter().enumerate() {
            let n = b.times.len();
            if b.position.len() != n || b.rotation.len() != n {
                return Err(SceneError::Invalid {
                    detail: format!("bake body[{bi}] curve length mismatch"),
                });
            }
            if let Some(v) = &b.linear_velocity {
                if v.len() != n {
                    return Err(SceneError::Invalid {
                        detail: format!("bake body[{bi}] linear_velocity length mismatch"),
                    });
                }
            }
            if let Some(w) = &b.angular_velocity {
                if w.len() != n {
                    return Err(SceneError::Invalid {
                        detail: format!("bake body[{bi}] angular_velocity length mismatch"),
                    });
                }
            }
            for i in 1..n {
                if b.times[i] <= b.times[i - 1] {
                    return Err(SceneError::Invalid {
                        detail: format!("bake body[{bi}] times not strictly increasing"),
                    });
                }
            }
        }
        Ok(())
    }

    /// Evaluate body `bi` at time `t` by Hermite/slerp between stored keys.
    ///
    /// `t` is clamped to the curve's time range. Used by reduction verification
    /// and host tools that want continuous playback of a (possibly reduced) bake.
    pub fn sample_body_at(&self, bi: usize, t: f64) -> Result<(Vec3, Quat), SceneError> {
        let b = self.bodies.get(bi).ok_or_else(|| SceneError::Invalid {
            detail: format!("bake body index {bi} out of range"),
        })?;
        if b.times.is_empty() {
            return Err(SceneError::Invalid {
                detail: "empty bake curve".into(),
            });
        }
        let (p, q, _, _) = sample_curve_at(b, t, false);
        Ok((p, q))
    }
}

/// Cubic Hermite position between knots separated by `dt`, parameter `s ∈ [0,1]`.
#[inline]
pub fn hermite_position(p0: Vec3, v0: Vec3, p1: Vec3, v1: Vec3, dt: f64, s: f64) -> Vec3 {
    let s2 = s * s;
    let s3 = s2 * s;
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;
    p0 * h00 + v0 * (h10 * dt) + p1 * h01 + v1 * (h11 * dt)
}

#[inline]
fn lerp_vec3(a: Vec3, b: Vec3, s: f64) -> Vec3 {
    a * (1.0 - s) + b * s
}

#[derive(Clone, Copy)]
struct Knot {
    t: f64,
    x: Vec3,
    q: Quat,
    v: Vec3,
    w: Vec3,
}

fn validate_options(opt: &BakeOptions) -> Result<(), SceneError> {
    let checks = [
        ("duration", opt.duration),
        ("physics_dt", opt.physics_dt),
        ("sample_fps", opt.sample_fps),
    ];
    for (name, v) in checks {
        if !v.is_finite() || v <= 0.0 {
            return Err(SceneError::Invalid {
                detail: format!("bake {name} must be finite and > 0"),
            });
        }
    }
    if let Some(r) = opt.reduce {
        if !r.max_position_error.is_finite()
            || r.max_position_error < 0.0
            || !r.max_rotation_error.is_finite()
            || r.max_rotation_error < 0.0
        {
            return Err(SceneError::Invalid {
                detail: "bake reduce errors must be finite and ≥ 0".into(),
            });
        }
    }
    Ok(())
}

fn output_times(duration: f64, sample_fps: f64) -> Vec<f64> {
    // Inclusive grid: 0, 1/fps, …, floor(duration*fps)/fps, and duration if needed.
    let n_steps = libm_ops::floor(duration * sample_fps + 1e-12) as usize;
    let mut times = Vec::with_capacity(n_steps + 2);
    for i in 0..=n_steps {
        times.push((i as f64) / sample_fps);
    }
    let last = *times.last().unwrap();
    if duration > last + 1e-15 {
        times.push(duration);
    } else if (duration - last).abs() > 0.0 && duration >= last {
        // Ensure the final stamp equals `duration` exactly when on-grid.
        let last_i = times.len() - 1;
        times[last_i] = duration;
    }
    times
}

fn interpolate_knot(a: &Knot, b: &Knot, t: f64) -> Knot {
    if (t - a.t).abs() <= 0.0 || b.t <= a.t {
        return *a;
    }
    if (t - b.t).abs() <= 0.0 {
        return *b;
    }
    let dt = b.t - a.t;
    let s = (t - a.t) / dt;
    Knot {
        t,
        x: hermite_position(a.x, a.v, b.x, b.v, dt, s),
        q: a.q.slerp(b.q, s),
        v: lerp_vec3(a.v, b.v, s),
        w: lerp_vec3(a.w, b.w, s),
    }
}

fn sample_curve_at(
    curve: &BakeBodyCurve,
    t: f64,
    want_vel: bool,
) -> (Vec3, Quat, Option<Vec3>, Option<Vec3>) {
    let n = curve.times.len();
    debug_assert!(n > 0);
    if t <= curve.times[0] {
        let v = curve
            .linear_velocity
            .as_ref()
            .map(|vs| Vec3::from_slice(&vs[0]));
        let w = curve
            .angular_velocity
            .as_ref()
            .map(|ws| Vec3::from_slice(&ws[0]));
        return (
            Vec3::from_slice(&curve.position[0]),
            Quat::from_slice(&curve.rotation[0]),
            if want_vel { v } else { None },
            if want_vel { w } else { None },
        );
    }
    if t >= curve.times[n - 1] {
        let v = curve
            .linear_velocity
            .as_ref()
            .map(|vs| Vec3::from_slice(&vs[n - 1]));
        let w = curve
            .angular_velocity
            .as_ref()
            .map(|ws| Vec3::from_slice(&ws[n - 1]));
        return (
            Vec3::from_slice(&curve.position[n - 1]),
            Quat::from_slice(&curve.rotation[n - 1]),
            if want_vel { v } else { None },
            if want_vel { w } else { None },
        );
    }
    let mut i = 0usize;
    while i + 1 < n && curve.times[i + 1] < t {
        i += 1;
    }
    let t0 = curve.times[i];
    let t1 = curve.times[i + 1];
    let a = Knot {
        t: t0,
        x: Vec3::from_slice(&curve.position[i]),
        q: Quat::from_slice(&curve.rotation[i]),
        v: curve
            .linear_velocity
            .as_ref()
            .map(|vs| Vec3::from_slice(&vs[i]))
            .unwrap_or(Vec3::zero()),
        w: curve
            .angular_velocity
            .as_ref()
            .map(|ws| Vec3::from_slice(&ws[i]))
            .unwrap_or(Vec3::zero()),
    };
    let b = Knot {
        t: t1,
        x: Vec3::from_slice(&curve.position[i + 1]),
        q: Quat::from_slice(&curve.rotation[i + 1]),
        v: curve
            .linear_velocity
            .as_ref()
            .map(|vs| Vec3::from_slice(&vs[i + 1]))
            .unwrap_or(Vec3::zero()),
        w: curve
            .angular_velocity
            .as_ref()
            .map(|ws| Vec3::from_slice(&ws[i + 1]))
            .unwrap_or(Vec3::zero()),
    };
    // Exact knot hit: copy (bit-identical across fps at coincident times).
    if (t - t0).abs() == 0.0 {
        return (
            a.x,
            a.q,
            if want_vel { Some(a.v) } else { None },
            if want_vel { Some(a.w) } else { None },
        );
    }
    if (t - t1).abs() == 0.0 {
        return (
            b.x,
            b.q,
            if want_vel { Some(b.v) } else { None },
            if want_vel { Some(b.w) } else { None },
        );
    }
    let k = interpolate_knot(&a, &b, t);
    (
        k.x,
        k.q,
        if want_vel { Some(k.v) } else { None },
        if want_vel { Some(k.w) } else { None },
    )
}

fn reduce_curve(curve: &BakeBodyCurve, params: BakeReduceParams) -> BakeBodyCurve {
    let n = curve.times.len();
    if n <= 2 {
        return curve.clone();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    reduce_segment(curve, 0, n - 1, params, &mut keep);

    let mut times = Vec::new();
    let mut position = Vec::new();
    let mut rotation = Vec::new();
    let mut lin = curve.linear_velocity.as_ref().map(|_| Vec::new());
    let mut ang = curve.angular_velocity.as_ref().map(|_| Vec::new());
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        times.push(curve.times[i]);
        position.push(curve.position[i]);
        rotation.push(curve.rotation[i]);
        if let (Some(out), Some(src)) = (lin.as_mut(), curve.linear_velocity.as_ref()) {
            out.push(src[i]);
        }
        if let (Some(out), Some(src)) = (ang.as_mut(), curve.angular_velocity.as_ref()) {
            out.push(src[i]);
        }
    }
    BakeBodyCurve {
        index: curve.index,
        name: curve.name.clone(),
        times,
        position,
        rotation,
        linear_velocity: lin,
        angular_velocity: ang,
    }
}

fn reduce_segment(
    curve: &BakeBodyCurve,
    i0: usize,
    i1: usize,
    params: BakeReduceParams,
    keep: &mut [bool],
) {
    if i1 <= i0 + 1 {
        return;
    }
    let a = Knot {
        t: curve.times[i0],
        x: Vec3::from_slice(&curve.position[i0]),
        q: Quat::from_slice(&curve.rotation[i0]),
        v: curve
            .linear_velocity
            .as_ref()
            .map(|vs| Vec3::from_slice(&vs[i0]))
            .unwrap_or(Vec3::zero()),
        w: Vec3::zero(),
    };
    let b = Knot {
        t: curve.times[i1],
        x: Vec3::from_slice(&curve.position[i1]),
        q: Quat::from_slice(&curve.rotation[i1]),
        v: curve
            .linear_velocity
            .as_ref()
            .map(|vs| Vec3::from_slice(&vs[i1]))
            .unwrap_or(Vec3::zero()),
        w: Vec3::zero(),
    };
    let mut worst_i = i0;
    let mut worst_pos = 0.0_f64;
    let mut worst_rot = 0.0_f64;
    for i in (i0 + 1)..i1 {
        let k = interpolate_knot(&a, &b, curve.times[i]);
        let p = Vec3::from_slice(&curve.position[i]);
        let q = Quat::from_slice(&curve.rotation[i]);
        let dp = (p - k.x).norm();
        let dr = q.angle_between(k.q);
        if dp > worst_pos || (dp == worst_pos && dr > worst_rot) {
            worst_pos = dp;
            worst_rot = dr;
            worst_i = i;
        } else if dr > worst_rot {
            worst_rot = dr;
            worst_i = i;
        }
    }
    if worst_pos > params.max_position_error || worst_rot > params.max_rotation_error {
        keep[worst_i] = true;
        reduce_segment(curve, i0, worst_i, params, keep);
        reduce_segment(curve, worst_i, i1, params, keep);
    }
}

/// Run `scene` for `opt.duration` and build a [`BakeDocument`].
pub fn bake_scene(scene: &SceneDocument, opt: &BakeOptions) -> Result<BakeDocument, SceneError> {
    validate_options(opt)?;
    let mut cw = scene.instantiate()?;
    cw.set_allow_sleep(false);

    let n_bodies = cw.world.n;
    let times_out = output_times(opt.duration, opt.sample_fps);
    if times_out.is_empty() {
        return Err(SceneError::Invalid {
            detail: "bake produced no sample times".into(),
        });
    }

    // Physics knots: sample at t=0, then after each step.
    let n_phys = libm_ops::floor(opt.duration / opt.physics_dt + 1e-12) as usize;
    let mut knots: Vec<Vec<Knot>> = (0..n_bodies)
        .map(|_| Vec::with_capacity(n_phys + 2))
        .collect();

    let capture_knot = |cw: &pe_collision::ContactWorld| -> Vec<Knot> {
        let t = cw.world.t;
        (0..cw.world.n)
            .map(|i| Knot {
                t,
                x: cw.world.state.x[i],
                q: cw.world.state.q[i],
                v: cw.world.state.v[i],
                w: cw.world.omega_i(&cw.world.state, i),
            })
            .collect()
    };

    {
        let k0 = capture_knot(&cw);
        for (i, k) in k0.into_iter().enumerate() {
            knots[i].push(k);
        }
    }

    for step in 0..n_phys {
        cw.step(opt.physics_dt);
        // Force exact knot time to avoid float drift in t.
        let t_exact = (step + 1) as f64 * opt.physics_dt;
        cw.world.t = t_exact;
        let k = capture_knot(&cw);
        for (i, ki) in k.into_iter().enumerate() {
            knots[i].push(Knot { t: t_exact, ..ki });
        }
    }
    // If duration is past the last full step, take one more partial step… we do
    // **not** — baking uses fixed dt only. The last output sample at `duration`
    // interpolates within the last full interval, or copies the final knot when
    // `duration` is an integer multiple of `physics_dt`.
    if (opt.duration - (n_phys as f64) * opt.physics_dt).abs() > 1e-15
        && opt.duration > (n_phys as f64) * opt.physics_dt
    {
        // Extend with one more full step so Hermite can reach `duration` inside
        // an interval (documented: duration should preferably be n·dt).
        cw.step(opt.physics_dt);
        let t_exact = (n_phys + 1) as f64 * opt.physics_dt;
        cw.world.t = t_exact;
        let k = capture_knot(&cw);
        for (i, ki) in k.into_iter().enumerate() {
            knots[i].push(Knot { t: t_exact, ..ki });
        }
    }

    let mut bodies = Vec::with_capacity(n_bodies);
    for (bi, ks) in knots.iter().enumerate() {
        let mut curve = BakeBodyCurve {
            index: bi,
            name: scene
                .bodies
                .get(bi)
                .map(|b| b.name.clone())
                .unwrap_or_default(),
            times: Vec::with_capacity(times_out.len()),
            position: Vec::with_capacity(times_out.len()),
            rotation: Vec::with_capacity(times_out.len()),
            linear_velocity: if opt.include_velocities {
                Some(Vec::with_capacity(times_out.len()))
            } else {
                None
            },
            angular_velocity: if opt.include_velocities {
                Some(Vec::with_capacity(times_out.len()))
            } else {
                None
            },
        };
        let mut j = 0usize;
        for &t in &times_out {
            while j + 1 < ks.len() && ks[j + 1].t < t {
                j += 1;
            }
            let sample = if j + 1 < ks.len() {
                // Exact knot: prefer bit-identical copy.
                if (t - ks[j].t).abs() == 0.0 {
                    ks[j]
                } else if (t - ks[j + 1].t).abs() == 0.0 {
                    ks[j + 1]
                } else {
                    interpolate_knot(&ks[j], &ks[j + 1], t)
                }
            } else {
                ks[j]
            };
            curve.times.push(t);
            curve.position.push(sample.x.to_array());
            curve.rotation.push(sample.q.to_array());
            if let Some(ref mut lv) = curve.linear_velocity {
                lv.push(sample.v.to_array());
            }
            if let Some(ref mut av) = curve.angular_velocity {
                av.push(sample.w.to_array());
            }
        }
        if let Some(r) = opt.reduce {
            curve = reduce_curve(&curve, r);
        }
        bodies.push(curve);
    }

    Ok(BakeDocument {
        schema_version: BAKE_SCHEMA_VERSION,
        physics_dt: opt.physics_dt,
        duration: opt.duration,
        sample_fps: opt.sample_fps,
        include_velocities: opt.include_velocities,
        interpolation: BakeInterpolation::default(),
        reduction: opt.reduce,
        bodies,
    })
}

/// FNV-1a 64-bit hash of compact bake JSON bytes (cross-OS lock).
pub fn bake_json_hash64(doc: &BakeDocument) -> Result<u64, SceneError> {
    let json = doc.to_json()?;
    Ok(fnv1a64_bytes(json.as_bytes()))
}

fn fnv1a64_bytes(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Canonical free-fall / projectile bake used for the locked cross-OS hash.
pub fn canonical_bake_document() -> Result<BakeDocument, SceneError> {
    let scene = canonical_bake_scene();
    let opt = BakeOptions {
        duration: 1.0,
        physics_dt: 1.0 / 240.0,
        sample_fps: 24.0,
        include_velocities: true,
        reduce: None,
    };
    bake_scene(&scene, &opt)
}

/// Scene for the locked bake hash: unit mass projectile under Earth gravity.
pub fn canonical_bake_scene() -> SceneDocument {
    use crate::types::{BodyDesc, ForceDesc, MaterialDesc, ShapeDesc, SolverSettings};
    use pe_math::STANDARD_GRAVITY;

    SceneDocument {
        schema_version: crate::types::SCENE_SCHEMA_VERSION,
        bodies: vec![BodyDesc {
            mass: 1.0,
            inertia_body: [[0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]],
            position: [0.0, 0.0, 10.0],
            velocity: [3.0, 0.0, 4.0],
            orientation: [1.0, 0.0, 0.0, 0.0],
            angular_velocity: [0.0, 0.0, 0.0],
            name: "projectile".into(),
            bullet: false,
        }],
        shapes: vec![ShapeDesc::Sphere { radius: 0.1 }],
        materials: vec![MaterialDesc {
            restitution: 0.0,
            friction: 0.0,
        }],
        statics: Vec::new(),
        joints: Vec::new(),
        forces: vec![ForceDesc::UniformGravity {
            g: [0.0, 0.0, -STANDARD_GRAVITY],
        }],
        solver: SolverSettings {
            allow_sleep: false,
            ..SolverSettings::default()
        },
        state: None,
        runtime: None,
    }
}

/// Locked FNV-1a of compact JSON for [`canonical_bake_document`].
///
/// Updated only when the bake schema or canonical scenario intentionally changes.
pub const CANONICAL_BAKE_JSON_HASH: u64 = 0x3e9a_97a8_5da0_cbc6;

/// Compute the canonical bake JSON hash.
pub fn canonical_bake_json_hash() -> u64 {
    let doc = canonical_bake_document().expect("canonical bake");
    bake_json_hash64(&doc).expect("bake json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use pe_math::STANDARD_GRAVITY;

    #[test]
    fn hermite_recovers_quadratic_free_fall_midpoint() {
        // Exact free-fall knots: p = p0 + v0 t + ½ a t², v = v0 + a t.
        let dt = 0.1;
        let a = Vec3::new(0.0, 0.0, -STANDARD_GRAVITY);
        let p0 = Vec3::new(0.0, 0.0, 10.0);
        let v0 = Vec3::new(1.0, 0.0, 2.0);
        let p1 = p0 + v0 * dt + a * (0.5 * dt * dt);
        let v1 = v0 + a * dt;
        let s = 0.5;
        let p = hermite_position(p0, v0, p1, v1, dt, s);
        let t = s * dt;
        let analytic = p0 + v0 * t + a * (0.5 * t * t);
        assert_abs_diff_eq!(p.x, analytic.x, epsilon = 1e-14);
        assert_abs_diff_eq!(p.y, analytic.y, epsilon = 1e-14);
        assert_abs_diff_eq!(p.z, analytic.z, epsilon = 1e-14);
    }

    #[test]
    fn output_times_include_endpoints() {
        let t = output_times(1.0, 24.0);
        assert_eq!(t.first().copied(), Some(0.0));
        assert_eq!(t.last().copied(), Some(1.0));
        assert_eq!(t.len(), 25); // 0..=24
    }

    #[test]
    fn bake_rejects_non_positive_dt() {
        let scene = canonical_bake_scene();
        let opt_zero = BakeOptions {
            physics_dt: 0.0,
            ..BakeOptions::default()
        };
        assert!(bake_scene(&scene, &opt_zero).is_err());
        let opt_nan = BakeOptions {
            physics_dt: f64::NAN,
            ..BakeOptions::default()
        };
        assert!(bake_scene(&scene, &opt_nan).is_err());
    }
}
