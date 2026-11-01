//! Versioned scene document: serialize, validate, instantiate, capture.

use pe_collision::ContactWorld;
use pe_core::diagnostics::state_hash64;
use pe_core::{RigidBody, World};
use pe_math::STANDARD_GRAVITY;
use serde::{Deserialize, Serialize};

use crate::convert::{
    apply_solver, apply_state_snapshot, body_from_desc, body_to_desc_with_bullet, force_to_box,
    joint_from_desc, joint_to_desc, material_from_desc, material_to_desc, shape_from_desc,
    shape_to_desc, solver_from_world, state_to_snapshot, static_from_desc, static_to_desc, v3,
};
use crate::error::{map_format_error, SceneError};
use crate::migrate::migrate_document;
use crate::types::{
    BodyDesc, ForceDesc, JointDesc, MaterialDesc, RuntimeSnapshot, ShapeDesc, SolverSettings,
    StateSnapshot, StaticDesc, SCENE_SCHEMA_VERSION,
};

/// Full scene: bodies, shapes, materials, joints, forces, solver, optional state.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SceneDocument {
    /// Schema version (must be migrated to current before instantiate).
    pub schema_version: u32,
    /// Dynamic bodies (order = body index).
    pub bodies: Vec<BodyDesc>,
    /// One shape per dynamic body.
    pub shapes: Vec<ShapeDesc>,
    /// One material per dynamic body.
    pub materials: Vec<MaterialDesc>,
    /// Static colliders.
    #[serde(default)]
    pub statics: Vec<StaticDesc>,
    /// Joints.
    #[serde(default)]
    pub joints: Vec<JointDesc>,
    /// Force generators in fixed apply order.
    #[serde(default)]
    pub forces: Vec<ForceDesc>,
    /// Solver / gravity-adjacent settings.
    #[serde(default)]
    pub solver: SolverSettings,
    /// Optional packed state (overrides body initial poses when present).
    #[serde(default)]
    pub state: Option<StateSnapshot>,
    /// Optional solver runtime for bit-exact continue.
    #[serde(default)]
    pub runtime: Option<RuntimeSnapshot>,
}

impl Default for SceneDocument {
    fn default() -> Self {
        Self {
            schema_version: SCENE_SCHEMA_VERSION,
            bodies: Vec::new(),
            shapes: Vec::new(),
            materials: Vec::new(),
            statics: Vec::new(),
            joints: Vec::new(),
            forces: Vec::new(),
            solver: SolverSettings::default(),
            state: None,
            runtime: None,
        }
    }
}

impl SceneDocument {
    /// Validate lengths and finite fields without constructing a world.
    pub fn validate(&self) -> Result<(), SceneError> {
        if self.schema_version != SCENE_SCHEMA_VERSION {
            return Err(SceneError::UnsupportedVersion {
                found: self.schema_version,
                max_supported: SCENE_SCHEMA_VERSION,
            });
        }
        let n = self.bodies.len();
        if self.shapes.len() != n {
            return Err(SceneError::Invalid {
                detail: format!("shapes.len()={} != bodies.len()={n}", self.shapes.len()),
            });
        }
        if self.materials.len() != n {
            return Err(SceneError::Invalid {
                detail: format!(
                    "materials.len()={} != bodies.len()={n}",
                    self.materials.len()
                ),
            });
        }
        for (i, b) in self.bodies.iter().enumerate() {
            if !b.mass.is_finite() || b.mass <= 0.0 {
                return Err(SceneError::Invalid {
                    detail: format!("body[{i}] mass invalid"),
                });
            }
        }
        if let Some(st) = &self.state {
            if st.x_bits.len() != n * 3
                || st.q_bits.len() != n * 4
                || st.v_bits.len() != n * 3
                || st.l_bits.len() != n * 3
            {
                return Err(SceneError::Invalid {
                    detail: "state snapshot length mismatch".into(),
                });
            }
        }
        Ok(())
    }

    /// Serialize to pretty JSON.
    pub fn to_json_pretty(&self) -> Result<String, SceneError> {
        serde_json::to_string_pretty(self).map_err(|e| map_format_error("json", e.to_string()))
    }

    /// Serialize to compact JSON.
    pub fn to_json(&self) -> Result<String, SceneError> {
        serde_json::to_string(self).map_err(|e| map_format_error("json", e.to_string()))
    }

    /// Parse JSON (never panics). Runs schema migration to current.
    pub fn from_json(s: &str) -> Result<Self, SceneError> {
        let doc: Self =
            serde_json::from_str(s).map_err(|e| map_format_error("json", e.to_string()))?;
        migrate_document(doc)
    }

    /// Serialize to RON (pretty).
    pub fn to_ron_pretty(&self) -> Result<String, SceneError> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::new())
            .map_err(|e| map_format_error("ron", e.to_string()))
    }

    /// Parse RON (never panics). Runs schema migration to current.
    pub fn from_ron(s: &str) -> Result<Self, SceneError> {
        let doc: Self = ron::from_str(s).map_err(|e| map_format_error("ron", e.to_string()))?;
        migrate_document(doc)
    }

    /// Compact binary: magic + version + UTF-8 JSON payload (deterministic).
    pub fn to_binary(&self) -> Result<Vec<u8>, SceneError> {
        crate::binary::encode(self)
    }

    /// Decode binary (never panics).
    pub fn from_binary(bytes: &[u8]) -> Result<Self, SceneError> {
        crate::binary::decode(bytes)
    }

    /// Build a [`ContactWorld`] from this document, applying state + runtime if present.
    pub fn instantiate(&self) -> Result<ContactWorld, SceneError> {
        self.validate()?;
        let bodies: Result<Vec<RigidBody>, _> = self.bodies.iter().map(body_from_desc).collect();
        let bodies = bodies?;
        let shapes: Result<Vec<_>, _> = self.shapes.iter().map(shape_from_desc).collect();
        let shapes = shapes?;
        let mats: Vec<_> = self.materials.iter().map(material_from_desc).collect();
        let statics: Result<Vec<_>, _> = self.statics.iter().map(static_from_desc).collect();
        let statics = statics?;
        let gens: Result<Vec<_>, _> = self.forces.iter().map(force_to_box).collect();
        let gens = gens?;
        let world = World::new(bodies, gens);
        let mut cw = ContactWorld::new(world, shapes, statics, Some(mats));
        apply_solver(&self.solver, &mut cw);
        for (i, b) in self.bodies.iter().enumerate() {
            if b.bullet {
                cw.set_bullet(i, true);
            }
        }
        for j in &self.joints {
            let joint = joint_from_desc(j)?;
            cw.add_joint(joint);
        }
        // Overwrite joint warms after add (add may wake but preserves joints vec).
        for (i, jdesc) in self.joints.iter().enumerate() {
            if i < cw.joints.len() {
                cw.joints[i].warm = jdesc
                    .warm_bits
                    .iter()
                    .copied()
                    .map(f64::from_bits)
                    .collect();
                cw.joints[i].broken = jdesc.broken;
            }
        }
        if let Some(snap) = &self.state {
            apply_state_snapshot(&mut cw.world.state, snap)?;
            cw.world.t = snap.t();
        }
        if let Some(rt) = &self.runtime {
            use crate::convert::from_bits3;
            use crate::types::WarmCacheEntry;
            let entries: Vec<pe_collision::WarmCacheRow> = rt
                .warm_cache
                .iter()
                .map(|e: &WarmCacheEntry| pe_collision::WarmCacheRow {
                    a: e.a,
                    b: e.b,
                    feature: e.feature,
                    normal_impulse: f64::from_bits(e.normal_bits),
                    tangent_impulse: from_bits3(e.tangent_bits),
                    point: from_bits3(e.point_bits),
                })
                .collect();
            cw.set_warm_cache_entries(&entries);
            let soft: Vec<_> = rt
                .soft_warm
                .iter()
                .map(|e| {
                    (
                        e.a,
                        e.b,
                        f64::from_bits(e.normal_bits),
                        v3(from_bits3(e.tangent_bits)),
                    )
                })
                .collect();
            cw.set_soft_warm_entries(soft);
            cw.set_prev_pairs(rt.prev_pairs.clone());
            if rt.awake.len() == cw.world.n {
                cw.sleep.awake = rt.awake.clone();
            }
            if rt.sleep_time_bits.len() == cw.world.n {
                cw.sleep.sleep_time = rt
                    .sleep_time_bits
                    .iter()
                    .copied()
                    .map(f64::from_bits)
                    .collect();
            }
            if rt.bullet.len() == cw.world.n {
                cw.bullet = rt.bullet.clone();
            }
        }
        Ok(cw)
    }

    /// Capture a live [`ContactWorld`] into a document.
    ///
    /// Force generators cannot be recovered from trait objects — pass the same
    /// `forces` list used to build the world (order matters for determinism).
    pub fn capture(cw: &ContactWorld, forces: Vec<ForceDesc>) -> Result<Self, SceneError> {
        let n = cw.world.n;
        let mut bodies = Vec::with_capacity(n);
        for i in 0..n {
            let omega = cw.world.omega_i(&cw.world.state, i);
            let b = RigidBody {
                mass: cw.world.mass[i],
                inertia_body: cw.world.inertia_body[i],
                position: cw.world.state.x[i],
                velocity: cw.world.state.v[i],
                orientation: cw.world.state.q[i],
                angular_velocity: omega,
                name: String::new(),
            };
            let bullet = cw.bullet.get(i).copied().unwrap_or(false);
            bodies.push(body_to_desc_with_bullet(&b, bullet));
        }
        let shapes: Vec<ShapeDesc> = cw
            .shapes
            .iter()
            .map(shape_to_desc)
            .collect::<Result<_, _>>()?;
        let materials: Vec<_> = cw.materials.iter().copied().map(material_to_desc).collect();
        let statics: Vec<_> = cw
            .statics
            .iter()
            .map(static_to_desc)
            .collect::<Result<_, _>>()?;
        let joints: Vec<_> = cw.joints.iter().map(joint_to_desc).collect();
        use crate::convert::bits3;
        use crate::types::{SoftWarmEntry, WarmCacheEntry};
        let warm_cache: Vec<WarmCacheEntry> = cw
            .warm_cache_entries()
            .into_iter()
            .map(|e| WarmCacheEntry {
                a: e.a,
                b: e.b,
                feature: e.feature,
                normal_bits: e.normal_impulse.to_bits(),
                tangent_bits: bits3(e.tangent_impulse),
                point_bits: bits3(e.point),
            })
            .collect();
        let soft_warm: Vec<SoftWarmEntry> = cw
            .soft_warm_entries()
            .iter()
            .map(|&(a, b, n, t)| SoftWarmEntry {
                a,
                b,
                normal_bits: n.to_bits(),
                tangent_bits: bits3(t.to_array()),
            })
            .collect();
        Ok(Self {
            schema_version: SCENE_SCHEMA_VERSION,
            bodies,
            shapes,
            materials,
            statics,
            joints,
            forces,
            solver: solver_from_world(cw),
            state: Some(state_to_snapshot(&cw.world.state, cw.world.t)),
            runtime: Some(RuntimeSnapshot {
                warm_cache,
                soft_warm,
                prev_pairs: cw.prev_pairs().to_vec(),
                awake: cw.sleep.awake.clone(),
                sleep_time_bits: cw.sleep.sleep_time.iter().map(|x| x.to_bits()).collect(),
                bullet: cw.bullet.clone(),
            }),
        })
    }

    /// Canonical 10-box pyramid scene (matches `pe_collision::canonical_collision_world`).
    pub fn canonical_pyramid() -> Self {
        let h = 0.5;
        let mut bodies = Vec::new();
        let mut shapes = Vec::new();
        let mut mats = Vec::new();
        let mut row = 0usize;
        let mut placed = 0usize;
        while placed < 10 {
            let n = row + 1;
            let x0 = -(n as f64 - 1.0) * h;
            for i in 0..n {
                if placed >= 10 {
                    break;
                }
                let x = x0 + 2.0 * h * (i as f64);
                let z = h + 2.0 * h * (row as f64);
                let inertia = pe_collision::shapes::BoxShape::new(h, h, h).inertia(1.0);
                bodies.push(BodyDesc {
                    mass: 1.0,
                    inertia_body: inertia,
                    position: [x, 0.0, z],
                    velocity: [0.0; 3],
                    orientation: [1.0, 0.0, 0.0, 0.0],
                    angular_velocity: [0.0; 3],
                    name: format!("box{placed}"),
                    bullet: false,
                });
                shapes.push(ShapeDesc::Box {
                    half_extents: [h, h, h],
                });
                mats.push(MaterialDesc {
                    restitution: 0.0,
                    friction: 0.5,
                });
                placed += 1;
            }
            row += 1;
        }
        Self {
            schema_version: SCENE_SCHEMA_VERSION,
            bodies,
            shapes,
            materials: mats,
            statics: vec![StaticDesc {
                shape: ShapeDesc::Plane {
                    normal: [0.0, 0.0, 1.0],
                    offset: 0.0,
                },
                position: [0.0; 3],
                orientation: [1.0, 0.0, 0.0, 0.0],
                material: MaterialDesc {
                    restitution: 0.0,
                    friction: 1.0,
                },
            }],
            joints: Vec::new(),
            forces: vec![ForceDesc::UniformGravity {
                g: [0.0, 0.0, -STANDARD_GRAVITY],
            }],
            solver: SolverSettings {
                iterations: 20,
                ..SolverSettings::default()
            },
            state: None,
            runtime: None,
        }
    }
}

/// Run pyramid 300 steps, save/load, 300 more; compare to uninterrupted 600.
///
/// Returns `(interrupted_hash, uninterrupted_hash)`.
pub fn round_trip_continue_hashes() -> Result<(u64, u64), SceneError> {
    let scene = SceneDocument::canonical_pyramid();
    let dt = 1.0 / 240.0;

    let mut full = scene.instantiate()?;
    for _ in 0..600 {
        full.step(dt);
    }
    let h_full = state_hash64(&full.world.state);

    let mut a = scene.instantiate()?;
    for _ in 0..300 {
        a.step(dt);
    }
    let mid = SceneDocument::capture(&a, scene.forces.clone())?;
    let bytes = mid.to_binary()?;
    let mid2 = SceneDocument::from_binary(&bytes)?;
    let mut b = mid2.instantiate()?;
    for _ in 0..300 {
        b.step(dt);
    }
    let h_rt = state_hash64(&b.world.state);
    Ok((h_rt, h_full))
}

/// Locked cross-OS scene round-trip continue hash (must match uninterrupted pyramid).
///
/// Equals [`pe_collision::CANONICAL_COLLISION_HASH`] when restore is bit-exact.
/// Unchanged by schema v2 when the scene uses only v1 primitives.
pub const CANONICAL_SCENE_ROUNDTRIP_HASH: u64 = pe_collision::CANONICAL_COLLISION_HASH;

/// Compute the scene continue hash (save after 300, load, 300 more).
pub fn canonical_scene_roundtrip_hash() -> u64 {
    let (h_rt, h_full) = round_trip_continue_hashes().expect("canonical scene round-trip");
    assert_eq!(
        h_rt, h_full,
        "save/load/continue diverged from uninterrupted run"
    );
    h_rt
}

impl SceneDocument {
    /// Canonical M6 convex scene (hull + capsule + rounded box) as a schema-v2 document.
    pub fn canonical_convex_document() -> Self {
        // Capture from the locked pe-collision scenario so hashes stay identical.
        let cw = pe_collision::canonical_convex_world();
        let forces = vec![ForceDesc::UniformGravity {
            g: [0.0, 0.0, -STANDARD_GRAVITY],
        }];
        let mut doc = SceneDocument::capture(&cw, forces).expect("capture convex world");
        // Drop runtime/state so instantiate rebuilds from body poses (matches hash helper).
        doc.state = None;
        doc.runtime = None;
        doc
    }
}

/// Save/load/continue hashes for the M6 convex scene (300 + 300 vs 600).
pub fn convex_round_trip_continue_hashes() -> Result<(u64, u64), SceneError> {
    let scene = SceneDocument::canonical_convex_document();
    let dt = 1.0 / 240.0;

    let mut full = scene.instantiate()?;
    for _ in 0..600 {
        full.step(dt);
    }
    let h_full = state_hash64(&full.world.state);

    let mut a = scene.instantiate()?;
    for _ in 0..300 {
        a.step(dt);
    }
    let mid = SceneDocument::capture(&a, scene.forces.clone())?;
    let bytes = mid.to_binary()?;
    let mid2 = SceneDocument::from_binary(&bytes)?;
    let mut b = mid2.instantiate()?;
    for _ in 0..300 {
        b.step(dt);
    }
    let h_rt = state_hash64(&b.world.state);
    Ok((h_rt, h_full))
}

/// Locked convex scene round-trip hash (= [`pe_collision::CANONICAL_CONVEX_HASH`]).
pub const CANONICAL_SCENE_CONVEX_ROUNDTRIP_HASH: u64 = pe_collision::CANONICAL_CONVEX_HASH;

/// Convex scene continue hash (must match uninterrupted M6 convex run).
pub fn canonical_scene_convex_roundtrip_hash() -> u64 {
    let (h_rt, h_full) = convex_round_trip_continue_hashes().expect("convex scene round-trip");
    assert_eq!(
        h_rt, h_full,
        "convex save/load/continue diverged from uninterrupted run"
    );
    h_rt
}

/// Godot CI drop scenario: box on ground plane, Y-up gravity (Godot convention).
///
/// Half-extent 0.25 m, start at `y = 2`, `dt = 1/60`, 180 steps. Returns
/// `(final_y, state_hash64)`.
pub fn godot_drop_box_result() -> (f64, u64) {
    let h = 0.25;
    let inertia = pe_collision::shapes::BoxShape::new(h, h, h).inertia(1.0);
    let doc = SceneDocument {
        schema_version: SCENE_SCHEMA_VERSION,
        bodies: vec![BodyDesc {
            mass: 1.0,
            inertia_body: inertia,
            position: [0.0, 2.0, 0.0],
            velocity: [0.0; 3],
            orientation: [1.0, 0.0, 0.0, 0.0],
            angular_velocity: [0.0; 3],
            name: "drop_box".into(),
            bullet: false,
        }],
        shapes: vec![ShapeDesc::Box {
            half_extents: [h, h, h],
        }],
        materials: vec![MaterialDesc {
            restitution: 0.0,
            friction: 0.5,
        }],
        statics: vec![StaticDesc {
            shape: ShapeDesc::Plane {
                normal: [0.0, 1.0, 0.0],
                offset: 0.0,
            },
            position: [0.0; 3],
            orientation: [1.0, 0.0, 0.0, 0.0],
            material: MaterialDesc {
                restitution: 0.0,
                friction: 1.0,
            },
        }],
        joints: Vec::new(),
        forces: vec![ForceDesc::UniformGravity {
            g: [0.0, -STANDARD_GRAVITY, 0.0],
        }],
        solver: SolverSettings {
            iterations: 20,
            ..SolverSettings::default()
        },
        state: None,
        runtime: None,
    };
    let mut cw = doc.instantiate().expect("godot drop scene");
    let dt = 1.0 / 60.0;
    for _ in 0..180 {
        cw.step(dt);
    }
    let y = cw.world.state.x[0].y;
    let hash = state_hash64(&cw.world.state);
    (y, hash)
}
