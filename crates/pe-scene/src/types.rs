//! Serde-friendly scene descriptors (arrays instead of pe-math types).

use serde::{Deserialize, Serialize};

/// Current scene schema version written by this crate.
pub const SCENE_SCHEMA_VERSION: u32 = 1;

/// Binary magic `PESC` (Physics Engine SCene).
pub const SCENE_BINARY_MAGIC: [u8; 4] = *b"PESC";

/// Rigid body description (SI).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BodyDesc {
    /// Mass [kg].
    pub mass: f64,
    /// Body-frame inertia about COM [kg m²], row-major 3×3.
    pub inertia_body: [[f64; 3]; 3],
    /// COM position [m].
    pub position: [f64; 3],
    /// Linear velocity [m/s].
    pub velocity: [f64; 3],
    /// Orientation quaternion `[w,x,y,z]`.
    pub orientation: [f64; 4],
    /// Angular velocity [rad/s] world frame (used when packing initial L).
    pub angular_velocity: [f64; 3],
    /// Optional debug name.
    #[serde(default)]
    pub name: String,
}

impl Default for BodyDesc {
    fn default() -> Self {
        Self {
            mass: 1.0,
            inertia_body: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            position: [0.0; 3],
            velocity: [0.0; 3],
            orientation: [1.0, 0.0, 0.0, 0.0],
            angular_velocity: [0.0; 3],
            name: String::new(),
        }
    }
}

/// Collision shape descriptor.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ShapeDesc {
    /// Sphere radius [m].
    Sphere {
        /// Radius [m].
        radius: f64,
    },
    /// Box half-extents [m].
    Box {
        /// Half-extents `[hx,hy,hz]` [m].
        half_extents: [f64; 3],
    },
    /// Capsule.
    Capsule {
        /// Sweep radius [m].
        radius: f64,
        /// Half segment length [m].
        half_height: f64,
    },
    /// Infinite plane.
    Plane {
        /// Outward normal (not necessarily unit).
        normal: [f64; 3],
        /// Plane offset [m].
        offset: f64,
    },
}

/// Contact material.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MaterialDesc {
    /// Restitution `e`.
    pub restitution: f64,
    /// Coulomb friction `μ`.
    pub friction: f64,
}

impl Default for MaterialDesc {
    fn default() -> Self {
        Self {
            restitution: 0.0,
            friction: 0.5,
        }
    }
}

/// Static collider.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StaticDesc {
    /// Shape.
    pub shape: ShapeDesc,
    /// World position [m].
    #[serde(default)]
    pub position: [f64; 3],
    /// Orientation `[w,x,y,z]`.
    #[serde(default = "identity_quat")]
    pub orientation: [f64; 4],
    /// Material.
    #[serde(default)]
    pub material: MaterialDesc,
}

fn identity_quat() -> [f64; 4] {
    [1.0, 0.0, 0.0, 0.0]
}

/// Force generator descriptor (stage-1 set).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ForceDesc {
    /// Uniform gravity `F = m g`.
    UniformGravity {
        /// `g` [m/s²].
        g: [f64; 3],
    },
    /// Newtonian pairwise gravity.
    NewtonianGravity {
        /// Gravitational constant.
        g_const: f64,
        /// Softening length [m].
        softening: f64,
    },
    /// Linear drag.
    LinearDrag {
        /// Coefficient [kg/s] (broadcast).
        b: f64,
        /// Wind [m/s].
        #[serde(default)]
        wind: [f64; 3],
    },
    /// Quadratic drag.
    QuadraticDrag {
        /// Coefficient [kg/m] (broadcast).
        k: f64,
        /// Wind [m/s].
        #[serde(default)]
        wind: [f64; 3],
    },
    /// Hookean spring to a fixed anchor.
    Spring {
        /// Body index.
        body: usize,
        /// Anchor [m].
        anchor: [f64; 3],
        /// Stiffness [N/m].
        k: f64,
    },
    /// Constant torque.
    ConstantTorque {
        /// Body index.
        body: usize,
        /// Torque [N·m].
        tau: [f64; 3],
    },
}

/// Joint kind payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JointKindDesc {
    /// Distance joint.
    Distance {
        /// Rest length [m].
        length: f64,
    },
    /// Revolute hinge.
    Revolute {
        /// Axis in A.
        local_axis_a: [f64; 3],
        /// Axis in B.
        local_axis_b: [f64; 3],
        /// Ref in A.
        local_ref_a: [f64; 3],
        /// Ref in B.
        local_ref_b: [f64; 3],
        /// Spring Hertz (0 = off).
        #[serde(default)]
        spring_hertz: f64,
        /// Spring ζ.
        #[serde(default)]
        spring_zeta: f64,
        /// Rest angle [rad].
        #[serde(default)]
        spring_angle: f64,
    },
    /// Prismatic slider.
    Prismatic {
        /// Axis in A.
        local_axis_a: [f64; 3],
        /// Axis in B.
        local_axis_b: [f64; 3],
        /// Reference translation [m].
        reference_translation: f64,
    },
    /// Soft weld.
    Weld {
        /// Relative quaternion `[w,x,y,z]`.
        q_rel: [f64; 4],
    },
    /// Wheel / suspension.
    Wheel {
        /// Axis in A.
        local_axis_a: [f64; 3],
        /// Axis in B.
        local_axis_b: [f64; 3],
        /// Spring Hertz.
        spring_hertz: f64,
        /// Spring ζ.
        spring_zeta: f64,
        /// Rest length [m].
        rest_length: f64,
    },
    /// Mouse / target.
    Mouse {
        /// World target [m].
        target: [f64; 3],
        /// Max force [N].
        max_force: f64,
    },
}

/// Joint descriptor.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct JointDesc {
    /// Body A (`usize::MAX` = world).
    pub body_a: u64,
    /// Body B (`usize::MAX` as u64 = world).
    pub body_b: u64,
    /// Anchor A [m].
    pub local_anchor_a: [f64; 3],
    /// Anchor B [m].
    pub local_anchor_b: [f64; 3],
    /// Kind.
    pub kind: JointKindDesc,
    /// Soft Hertz.
    #[serde(default = "default_joint_hertz")]
    pub soft_hertz: f64,
    /// Soft ζ.
    #[serde(default = "default_joint_zeta")]
    pub soft_zeta: f64,
    /// Motor enabled.
    #[serde(default)]
    pub motor_enabled: bool,
    /// Motor target speed.
    #[serde(default)]
    pub motor_target_speed: f64,
    /// Motor max force.
    #[serde(default)]
    pub motor_max_force: f64,
    /// Limit enabled.
    #[serde(default)]
    pub limit_enabled: bool,
    /// Limit lower.
    #[serde(default)]
    pub limit_lower: f64,
    /// Limit upper.
    #[serde(default)]
    pub limit_upper: f64,
    /// Break force (≤0 disables).
    #[serde(default)]
    pub break_force: f64,
    /// Break torque (≤0 disables).
    #[serde(default)]
    pub break_torque: f64,
    /// Broken flag.
    #[serde(default)]
    pub broken: bool,
    /// Warm-start impulses as `f64` bit patterns.
    #[serde(default)]
    pub warm_bits: Vec<u64>,
}

fn default_joint_hertz() -> f64 {
    60.0
}
fn default_joint_zeta() -> f64 {
    0.7
}

/// Contact solver settings.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SolverSettings {
    /// `"pgs"` or `"soft_step"`.
    #[serde(default = "default_solver_mode")]
    pub mode: String,
    /// PGS iterations.
    #[serde(default = "default_iterations")]
    pub iterations: usize,
    /// Restitution threshold [m/s] (PGS).
    #[serde(default = "default_rest_thresh")]
    pub restitution_threshold: f64,
    /// Positional slop [m].
    #[serde(default = "default_slop")]
    pub slop: f64,
    /// Positional β.
    #[serde(default = "default_beta")]
    pub beta: f64,
    /// Warm start contacts.
    #[serde(default = "default_true")]
    pub warm_start: bool,
    /// Broad phase: `sap` / `brute` / `dynamic_tree`.
    #[serde(default = "default_broad")]
    pub broad_phase: String,
    /// Soft-step substeps.
    #[serde(default = "default_substeps")]
    pub soft_substeps: usize,
    /// Soft contact Hertz.
    #[serde(default = "default_contact_hertz")]
    pub soft_contact_hertz: f64,
    /// Soft contact ζ.
    #[serde(default = "default_contact_zeta")]
    pub soft_contact_zeta: f64,
    /// Allow island sleeping.
    #[serde(default)]
    pub allow_sleep: bool,
    /// CCD speculative.
    #[serde(default)]
    pub ccd_speculative: bool,
    /// CCD TOI.
    #[serde(default)]
    pub ccd_toi: bool,
}

fn default_solver_mode() -> String {
    "pgs".into()
}
fn default_iterations() -> usize {
    20
}
fn default_rest_thresh() -> f64 {
    0.2
}
fn default_slop() -> f64 {
    1e-4
}
fn default_beta() -> f64 {
    0.2
}
fn default_true() -> bool {
    true
}
fn default_broad() -> String {
    "sap".into()
}
fn default_substeps() -> usize {
    4
}
fn default_contact_hertz() -> f64 {
    30.0
}
fn default_contact_zeta() -> f64 {
    10.0
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            mode: default_solver_mode(),
            iterations: default_iterations(),
            restitution_threshold: default_rest_thresh(),
            slop: default_slop(),
            beta: default_beta(),
            warm_start: true,
            broad_phase: default_broad(),
            soft_substeps: default_substeps(),
            soft_contact_hertz: default_contact_hertz(),
            soft_contact_zeta: default_contact_zeta(),
            allow_sleep: false,
            ccd_speculative: false,
            ccd_toi: false,
        }
    }
}

/// Packed dynamic state snapshot (SoA).
///
/// Floats are stored as IEEE-754 bit patterns (`u64`) so JSON/RON/binary all
/// round-trip **bit-exactly** (JSON decimal floats alone cannot).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StateSnapshot {
    /// Simulation time [s] as `f64::to_bits`.
    pub t_bits: u64,
    /// Positions `[x0,y0,z0,…]` as `f64` bit patterns.
    pub x_bits: Vec<u64>,
    /// Quaternions `[w0,x0,y0,z0,…]` as bit patterns.
    pub q_bits: Vec<u64>,
    /// Velocities as bit patterns.
    pub v_bits: Vec<u64>,
    /// Angular momenta L as bit patterns.
    pub l_bits: Vec<u64>,
}

impl StateSnapshot {
    /// Simulation time [s].
    pub fn t(&self) -> f64 {
        f64::from_bits(self.t_bits)
    }

    /// Decode position components to `f64`.
    pub fn x_f64(&self) -> Vec<f64> {
        self.x_bits.iter().copied().map(f64::from_bits).collect()
    }

    /// Decode quaternion components to `f64`.
    pub fn q_f64(&self) -> Vec<f64> {
        self.q_bits.iter().copied().map(f64::from_bits).collect()
    }

    /// Decode velocity components to `f64`.
    pub fn v_f64(&self) -> Vec<f64> {
        self.v_bits.iter().copied().map(f64::from_bits).collect()
    }

    /// Decode angular-momentum components to `f64`.
    pub fn l_f64(&self) -> Vec<f64> {
        self.l_bits.iter().copied().map(f64::from_bits).collect()
    }
}

/// One warm-start cache row with bit-exact floats.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WarmCacheEntry {
    /// Body A.
    pub a: usize,
    /// Body B.
    pub b: usize,
    /// Feature id.
    pub feature: u32,
    /// Normal impulse bits.
    pub normal_bits: u64,
    /// Tangent impulse bits `[3]`.
    pub tangent_bits: [u64; 3],
    /// Contact point bits `[3]`.
    pub point_bits: [u64; 3],
}

/// Soft-step warm row.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SoftWarmEntry {
    /// Body A.
    pub a: usize,
    /// Body B.
    pub b: usize,
    /// Normal impulse bits.
    pub normal_bits: u64,
    /// Tangent impulse bits.
    pub tangent_bits: [u64; 3],
}

/// Solver runtime needed for bit-exact continue after restore.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct RuntimeSnapshot {
    /// Warm-start cache.
    #[serde(default)]
    pub warm_cache: Vec<WarmCacheEntry>,
    /// Soft warm rows.
    #[serde(default)]
    pub soft_warm: Vec<SoftWarmEntry>,
    /// Previous broad-phase pairs.
    #[serde(default)]
    pub prev_pairs: Vec<(usize, usize)>,
    /// Per-body awake flags.
    #[serde(default)]
    pub awake: Vec<bool>,
    /// Per-body sleep timers as `f64` bit patterns.
    #[serde(default)]
    pub sleep_time_bits: Vec<u64>,
    /// Per-body bullet flags.
    #[serde(default)]
    pub bullet: Vec<bool>,
}
