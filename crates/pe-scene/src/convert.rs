//! Convert between scene descriptors and pe-core / pe-collision types.

use pe_collision::joints::{
    Joint, JointBreak, JointKind, JointLimit, JointMotor, JointSoftParams, WORLD_BODY,
};
use pe_collision::shapes::{BoxShape, Capsule, Material, Plane, Shape, Sphere};
use pe_collision::{
    BroadPhaseKind, CcdParams, ConvexHull, RoundedShape, SoftStepParams, SolverMode, StaticCollider,
};
use pe_core::forces::{
    ConstantTorque, ForceGenerator, LinearDrag, NewtonianGravity, QuadraticDrag, Spring,
    UniformGravity,
};
use pe_core::{RigidBody, State};
use pe_math::{Quat, Vec3};

use crate::error::SceneError;
use crate::types::{
    BodyDesc, ForceDesc, JointDesc, JointKindDesc, MaterialDesc, ShapeDesc, SolverSettings,
    StateSnapshot, StaticDesc,
};

#[inline]
pub(crate) fn v3(a: [f64; 3]) -> Vec3 {
    Vec3::from_slice(&a)
}

#[inline]
pub(crate) fn q4(a: [f64; 4]) -> Quat {
    Quat::from_slice(&a).normalize()
}

pub(crate) fn body_from_desc(d: &BodyDesc) -> Result<RigidBody, SceneError> {
    if !(d.mass.is_finite() && d.mass > 0.0) {
        return Err(SceneError::Invalid {
            detail: "body mass must be finite and > 0".into(),
        });
    }
    if d.orientation.iter().any(|x| !x.is_finite()) {
        return Err(SceneError::Invalid {
            detail: "body orientation has non-finite component".into(),
        });
    }
    Ok(RigidBody {
        mass: d.mass,
        inertia_body: d.inertia_body,
        position: v3(d.position),
        velocity: v3(d.velocity),
        orientation: q4(d.orientation),
        angular_velocity: v3(d.angular_velocity),
        name: d.name.clone(),
    })
}

pub(crate) fn body_to_desc_with_bullet(b: &RigidBody, bullet: bool) -> BodyDesc {
    BodyDesc {
        mass: b.mass,
        inertia_body: b.inertia_body,
        position: b.position.to_array(),
        velocity: b.velocity.to_array(),
        orientation: b.orientation.to_array(),
        angular_velocity: b.angular_velocity.to_array(),
        name: b.name.clone(),
        bullet,
    }
}

pub(crate) fn shape_from_desc(d: &ShapeDesc) -> Result<Shape, SceneError> {
    Ok(match d {
        ShapeDesc::Sphere { radius } => {
            if !(radius.is_finite() && *radius >= 0.0) {
                return Err(SceneError::Invalid {
                    detail: "sphere radius must be finite and ≥ 0".into(),
                });
            }
            Shape::Sphere(Sphere::new(*radius))
        }
        ShapeDesc::Box { half_extents } => {
            if half_extents.iter().any(|x| !x.is_finite() || *x < 0.0) {
                return Err(SceneError::Invalid {
                    detail: "box half_extents must be finite and ≥ 0".into(),
                });
            }
            Shape::Box(BoxShape::new(
                half_extents[0],
                half_extents[1],
                half_extents[2],
            ))
        }
        ShapeDesc::Capsule {
            radius,
            half_height,
        } => {
            if !(radius.is_finite()
                && *radius >= 0.0
                && half_height.is_finite()
                && *half_height >= 0.0)
            {
                return Err(SceneError::Invalid {
                    detail: "capsule radius/half_height must be finite and ≥ 0".into(),
                });
            }
            Shape::Capsule(Capsule::new(*radius, *half_height))
        }
        ShapeDesc::Plane { normal, offset } => {
            if normal.iter().any(|x| !x.is_finite()) || !offset.is_finite() {
                return Err(SceneError::Invalid {
                    detail: "plane normal/offset must be finite".into(),
                });
            }
            Shape::Plane(Plane::new(v3(*normal), *offset))
        }
        ShapeDesc::Convex { vertices, radius } => {
            if !(radius.is_finite() && *radius >= 0.0) {
                return Err(SceneError::Invalid {
                    detail: "convex radius must be finite and ≥ 0".into(),
                });
            }
            if vertices.len() < 4 {
                return Err(SceneError::Invalid {
                    detail: format!("convex hull needs ≥ 4 vertices, got {}", vertices.len()),
                });
            }
            if vertices.iter().any(|p| p.iter().any(|c| !c.is_finite())) {
                return Err(SceneError::Invalid {
                    detail: "convex vertices must be finite".into(),
                });
            }
            let pts: Vec<_> = vertices.iter().map(|a| v3(*a)).collect();
            let hull = ConvexHull::from_points_with_radius(&pts, *radius).map_err(|e| {
                SceneError::Invalid {
                    detail: format!("convex hull build failed: {e:?}"),
                }
            })?;
            Shape::Convex(hull)
        }
        ShapeDesc::Rounded { inner, radius } => {
            if !(radius.is_finite() && *radius >= 0.0) {
                return Err(SceneError::Invalid {
                    detail: "rounded radius must be finite and ≥ 0".into(),
                });
            }
            match inner.as_ref() {
                ShapeDesc::Plane { .. } => {
                    return Err(SceneError::Invalid {
                        detail: "rounded inner cannot be a plane".into(),
                    });
                }
                ShapeDesc::Rounded { .. } => {
                    return Err(SceneError::Invalid {
                        detail: "nested rounded shapes are not allowed; flatten radii".into(),
                    });
                }
                _ => {}
            }
            let core = shape_from_desc(inner)?;
            if matches!(core, Shape::Plane(_)) {
                return Err(SceneError::Invalid {
                    detail: "rounded inner cannot be a plane".into(),
                });
            }
            Shape::Rounded(RoundedShape::new(core, *radius))
        }
    })
}

pub(crate) fn shape_to_desc(s: &Shape) -> Result<ShapeDesc, SceneError> {
    Ok(match s {
        Shape::Sphere(sp) => ShapeDesc::Sphere { radius: sp.radius },
        Shape::Box(b) => ShapeDesc::Box {
            half_extents: b.half_extents.to_array(),
        },
        Shape::Capsule(c) => ShapeDesc::Capsule {
            radius: c.radius,
            half_height: c.half_height,
        },
        Shape::Plane(p) => ShapeDesc::Plane {
            normal: p.normal.to_array(),
            offset: p.offset,
        },
        Shape::Convex(h) => ShapeDesc::Convex {
            vertices: h.vertices.iter().map(|v| v.to_array()).collect(),
            radius: h.radius,
        },
        Shape::Rounded(rd) => ShapeDesc::Rounded {
            inner: Box::new(shape_to_desc(&rd.inner)?),
            radius: rd.radius,
        },
    })
}

pub(crate) fn material_from_desc(m: &MaterialDesc) -> Material {
    Material::new(m.restitution, m.friction)
}

pub(crate) fn material_to_desc(m: Material) -> MaterialDesc {
    MaterialDesc {
        restitution: m.restitution,
        friction: m.friction,
    }
}

pub(crate) fn static_from_desc(d: &StaticDesc) -> Result<StaticCollider, SceneError> {
    Ok(StaticCollider {
        shape: shape_from_desc(&d.shape)?,
        position: v3(d.position),
        orientation: q4(d.orientation),
        material: material_from_desc(&d.material),
    })
}

pub(crate) fn static_to_desc(s: &StaticCollider) -> Result<StaticDesc, SceneError> {
    Ok(StaticDesc {
        shape: shape_to_desc(&s.shape)?,
        position: s.position.to_array(),
        orientation: s.orientation.to_array(),
        material: material_to_desc(s.material),
    })
}

/// Build a boxed force generator from a serializable descriptor.
pub fn force_to_box(f: &ForceDesc) -> Result<Box<dyn ForceGenerator>, SceneError> {
    Ok(match f {
        ForceDesc::UniformGravity { g } => Box::new(UniformGravity::new(v3(*g))),
        ForceDesc::NewtonianGravity { g_const, softening } => {
            Box::new(NewtonianGravity::with_params(*g_const, *softening))
        }
        ForceDesc::LinearDrag { b, wind } => Box::new(LinearDrag::new(*b, v3(*wind))),
        ForceDesc::QuadraticDrag { k, wind } => Box::new(QuadraticDrag::new(*k, v3(*wind))),
        ForceDesc::Spring { body, anchor, k } => Box::new(Spring::new(*body, v3(*anchor), *k)),
        ForceDesc::ConstantTorque { body, tau } => Box::new(ConstantTorque::new(*body, v3(*tau))),
    })
}

fn body_id(u: u64) -> usize {
    if u == u64::MAX {
        WORLD_BODY
    } else {
        u as usize
    }
}

fn body_id_u64(i: usize) -> u64 {
    if i == WORLD_BODY {
        u64::MAX
    } else {
        i as u64
    }
}

pub(crate) fn joint_from_desc(d: &JointDesc) -> Result<Joint, SceneError> {
    let kind = match &d.kind {
        JointKindDesc::Distance { length } => JointKind::Distance { length: *length },
        JointKindDesc::Revolute {
            local_axis_a,
            local_axis_b,
            local_ref_a,
            local_ref_b,
            spring_hertz,
            spring_zeta,
            spring_angle,
        } => JointKind::Revolute {
            local_axis_a: v3(*local_axis_a),
            local_axis_b: v3(*local_axis_b),
            local_ref_a: v3(*local_ref_a),
            local_ref_b: v3(*local_ref_b),
            spring_hertz: *spring_hertz,
            spring_zeta: *spring_zeta,
            spring_angle: *spring_angle,
        },
        JointKindDesc::Prismatic {
            local_axis_a,
            local_axis_b,
            reference_translation,
        } => JointKind::Prismatic {
            local_axis_a: v3(*local_axis_a),
            local_axis_b: v3(*local_axis_b),
            reference_translation: *reference_translation,
        },
        JointKindDesc::Weld { q_rel } => JointKind::Weld { q_rel: q4(*q_rel) },
        JointKindDesc::Wheel {
            local_axis_a,
            local_axis_b,
            spring_hertz,
            spring_zeta,
            rest_length,
        } => JointKind::Wheel {
            local_axis_a: v3(*local_axis_a),
            local_axis_b: v3(*local_axis_b),
            spring_hertz: *spring_hertz,
            spring_zeta: *spring_zeta,
            rest_length: *rest_length,
        },
        JointKindDesc::Mouse { target, max_force } => JointKind::Mouse {
            target: v3(*target),
            max_force: *max_force,
        },
    };
    let j = Joint {
        body_a: body_id(d.body_a),
        body_b: body_id(d.body_b),
        local_anchor_a: v3(d.local_anchor_a),
        local_anchor_b: v3(d.local_anchor_b),
        kind,
        soft: JointSoftParams {
            hertz: d.soft_hertz,
            zeta: d.soft_zeta,
        },
        motor: JointMotor {
            enabled: d.motor_enabled,
            target_speed: d.motor_target_speed,
            max_force: d.motor_max_force,
        },
        limit: JointLimit {
            enabled: d.limit_enabled,
            lower: d.limit_lower,
            upper: d.limit_upper,
        },
        break_threshold: JointBreak {
            force: d.break_force,
            torque: d.break_torque,
        },
        reaction_force: Vec3::zero(),
        reaction_torque: Vec3::zero(),
        broken: d.broken,
        warm: d.warm_bits.iter().copied().map(f64::from_bits).collect(),
    };
    Ok(j)
}

pub(crate) fn joint_to_desc(j: &Joint) -> JointDesc {
    let kind = match &j.kind {
        JointKind::Distance { length } => JointKindDesc::Distance { length: *length },
        JointKind::Revolute {
            local_axis_a,
            local_axis_b,
            local_ref_a,
            local_ref_b,
            spring_hertz,
            spring_zeta,
            spring_angle,
        } => JointKindDesc::Revolute {
            local_axis_a: local_axis_a.to_array(),
            local_axis_b: local_axis_b.to_array(),
            local_ref_a: local_ref_a.to_array(),
            local_ref_b: local_ref_b.to_array(),
            spring_hertz: *spring_hertz,
            spring_zeta: *spring_zeta,
            spring_angle: *spring_angle,
        },
        JointKind::Prismatic {
            local_axis_a,
            local_axis_b,
            reference_translation,
        } => JointKindDesc::Prismatic {
            local_axis_a: local_axis_a.to_array(),
            local_axis_b: local_axis_b.to_array(),
            reference_translation: *reference_translation,
        },
        JointKind::Weld { q_rel } => JointKindDesc::Weld {
            q_rel: q_rel.to_array(),
        },
        JointKind::Wheel {
            local_axis_a,
            local_axis_b,
            spring_hertz,
            spring_zeta,
            rest_length,
        } => JointKindDesc::Wheel {
            local_axis_a: local_axis_a.to_array(),
            local_axis_b: local_axis_b.to_array(),
            spring_hertz: *spring_hertz,
            spring_zeta: *spring_zeta,
            rest_length: *rest_length,
        },
        JointKind::Mouse { target, max_force } => JointKindDesc::Mouse {
            target: target.to_array(),
            max_force: *max_force,
        },
    };
    JointDesc {
        body_a: body_id_u64(j.body_a),
        body_b: body_id_u64(j.body_b),
        local_anchor_a: j.local_anchor_a.to_array(),
        local_anchor_b: j.local_anchor_b.to_array(),
        kind,
        soft_hertz: j.soft.hertz,
        soft_zeta: j.soft.zeta,
        motor_enabled: j.motor.enabled,
        motor_target_speed: j.motor.target_speed,
        motor_max_force: j.motor.max_force,
        limit_enabled: j.limit.enabled,
        limit_lower: j.limit.lower,
        limit_upper: j.limit.upper,
        break_force: j.break_threshold.force,
        break_torque: j.break_threshold.torque,
        broken: j.broken,
        warm_bits: j.warm.iter().map(|x| x.to_bits()).collect(),
    }
}

pub(crate) fn apply_solver(settings: &SolverSettings, cw: &mut pe_collision::ContactWorld) {
    cw.solver_mode = SolverMode::parse(&settings.mode).unwrap_or(SolverMode::Pgs);
    cw.iterations = settings.iterations;
    cw.restitution_threshold = settings.restitution_threshold;
    cw.slop = settings.slop;
    cw.beta = settings.beta;
    cw.warm_start = settings.warm_start;
    cw.broad_phase = match settings.broad_phase.to_ascii_lowercase().as_str() {
        "brute" | "bruteforce" => BroadPhaseKind::Brute,
        "dynamic_tree" | "tree" | "aabb_tree" => BroadPhaseKind::DynamicTree,
        _ => BroadPhaseKind::Sap,
    };
    cw.soft_params = SoftStepParams {
        substeps: settings.soft_substeps,
        contact_hertz: settings.soft_contact_hertz,
        contact_zeta: settings.soft_contact_zeta,
        ..SoftStepParams::default()
    };
    cw.set_allow_sleep(settings.allow_sleep);
    cw.ccd = CcdParams {
        speculative: settings.ccd_speculative,
        speculative_distance: settings.ccd_speculative_distance,
        toi: settings.ccd_toi,
        toi_max_iters: settings.ccd_toi_max_iters,
        toi_tolerance: settings.ccd_toi_tolerance,
    };
}

pub(crate) fn solver_from_world(cw: &pe_collision::ContactWorld) -> SolverSettings {
    SolverSettings {
        mode: cw.solver_mode.as_str().into(),
        iterations: cw.iterations,
        restitution_threshold: cw.restitution_threshold,
        slop: cw.slop,
        beta: cw.beta,
        warm_start: cw.warm_start,
        broad_phase: match cw.broad_phase {
            BroadPhaseKind::Sap => "sap".into(),
            BroadPhaseKind::Brute => "brute".into(),
            BroadPhaseKind::DynamicTree => "dynamic_tree".into(),
        },
        soft_substeps: cw.soft_params.substeps,
        soft_contact_hertz: cw.soft_params.contact_hertz,
        soft_contact_zeta: cw.soft_params.contact_zeta,
        allow_sleep: cw.sleep.enabled,
        ccd_speculative: cw.ccd.speculative,
        ccd_speculative_distance: cw.ccd.speculative_distance,
        ccd_toi: cw.ccd.toi,
        ccd_toi_max_iters: cw.ccd.toi_max_iters,
        ccd_toi_tolerance: cw.ccd.toi_tolerance,
    }
}

pub(crate) fn bits3(a: [f64; 3]) -> [u64; 3] {
    [a[0].to_bits(), a[1].to_bits(), a[2].to_bits()]
}

pub(crate) fn from_bits3(a: [u64; 3]) -> [f64; 3] {
    [
        f64::from_bits(a[0]),
        f64::from_bits(a[1]),
        f64::from_bits(a[2]),
    ]
}

pub(crate) fn state_to_snapshot(st: &State, t: f64) -> StateSnapshot {
    let mut x_bits = Vec::with_capacity(st.n() * 3);
    let mut q_bits = Vec::with_capacity(st.n() * 4);
    let mut v_bits = Vec::with_capacity(st.n() * 3);
    let mut l_bits = Vec::with_capacity(st.n() * 3);
    for p in &st.x {
        for c in p.to_array() {
            x_bits.push(c.to_bits());
        }
    }
    for qq in &st.q {
        for c in qq.to_array() {
            q_bits.push(c.to_bits());
        }
    }
    for vv in &st.v {
        for c in vv.to_array() {
            v_bits.push(c.to_bits());
        }
    }
    for ll in &st.l {
        for c in ll.to_array() {
            l_bits.push(c.to_bits());
        }
    }
    StateSnapshot {
        t_bits: t.to_bits(),
        x_bits,
        q_bits,
        v_bits,
        l_bits,
    }
}

pub(crate) fn apply_state_snapshot(st: &mut State, snap: &StateSnapshot) -> Result<(), SceneError> {
    let n = st.n();
    if snap.x_bits.len() != n * 3
        || snap.q_bits.len() != n * 4
        || snap.v_bits.len() != n * 3
        || snap.l_bits.len() != n * 3
    {
        return Err(SceneError::Invalid {
            detail: format!(
                "state snapshot length mismatch for n={n}: x={} q={} v={} l={}",
                snap.x_bits.len(),
                snap.q_bits.len(),
                snap.v_bits.len(),
                snap.l_bits.len()
            ),
        });
    }
    for i in 0..n {
        st.x[i] = Vec3::new(
            f64::from_bits(snap.x_bits[3 * i]),
            f64::from_bits(snap.x_bits[3 * i + 1]),
            f64::from_bits(snap.x_bits[3 * i + 2]),
        );
        st.q[i] = Quat::new(
            f64::from_bits(snap.q_bits[4 * i]),
            f64::from_bits(snap.q_bits[4 * i + 1]),
            f64::from_bits(snap.q_bits[4 * i + 2]),
            f64::from_bits(snap.q_bits[4 * i + 3]),
        );
        st.v[i] = Vec3::new(
            f64::from_bits(snap.v_bits[3 * i]),
            f64::from_bits(snap.v_bits[3 * i + 1]),
            f64::from_bits(snap.v_bits[3 * i + 2]),
        );
        st.l[i] = Vec3::new(
            f64::from_bits(snap.l_bits[3 * i]),
            f64::from_bits(snap.l_bits[3 * i + 1]),
            f64::from_bits(snap.l_bits[3 * i + 2]),
        );
    }
    Ok(())
}
