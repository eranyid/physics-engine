//! Convert schema-v3 blast / fracture descriptors into `pe-blast` types.

use pe_blast::{
    BlastLoadParams, BlastSource, ExposedFace, OverpressureModel, ReflectionMix, GAMMA_AIR,
};
use pe_core::ExternalLoads;
use pe_core::World;
use pe_math::Vec3;

use crate::error::SceneError;
use crate::types::{BlastLoadParamsDesc, BlastLoadingDesc, BlastSourceDesc, ExposedFaceDesc};

fn v3(a: [f64; 3]) -> Vec3 {
    Vec3::from_slice(&a)
}

/// Convert a [`BlastSourceDesc`] to [`BlastSource`].
pub fn blast_source_from_desc(d: &BlastSourceDesc) -> BlastSource {
    BlastSource {
        origin: v3(d.origin),
        w_eq_kg: d.w_eq_kg,
        t0: d.t0,
        p0: d.p0,
        td_at_z1: d.td_at_z1,
        decay_b: d.decay_b,
    }
}

/// Convert an [`ExposedFaceDesc`].
pub fn exposed_face_from_desc(d: &ExposedFaceDesc) -> ExposedFace {
    ExposedFace {
        body: d.body,
        center: v3(d.center),
        normal: v3(d.normal),
        area: d.area,
    }
}

/// Parse load params (unknown strings → error).
pub fn blast_params_from_desc(d: &BlastLoadParamsDesc) -> Result<BlastLoadParams, SceneError> {
    let overpressure = match d.overpressure.as_str() {
        "kinney" => OverpressureModel::Kinney,
        "kingery_bulmash" => OverpressureModel::KingeryBulmash,
        other => {
            return Err(SceneError::Invalid {
                detail: format!("unknown blast overpressure model {other:?}"),
            })
        }
    };
    let reflection = match d.reflection.as_str() {
        "incident_only" => ReflectionMix::IncidentOnly,
        "incident_reflected" => ReflectionMix::IncidentReflected,
        other => {
            return Err(SceneError::Invalid {
                detail: format!("unknown blast reflection mix {other:?}"),
            })
        }
    };
    Ok(BlastLoadParams {
        overpressure,
        reflection,
        drag_cd: d.drag_cd,
        shadow_factor: d.shadow_factor,
        gamma: GAMMA_AIR,
    })
}

/// Clear `loads` and accumulate all faces from a blast loading block at time `t`.
pub fn apply_blast_loading_desc(
    loads: &ExternalLoads,
    world: &World,
    desc: &BlastLoadingDesc,
    t: f64,
) -> Result<(), SceneError> {
    let src = blast_source_from_desc(&desc.source);
    let params = blast_params_from_desc(&desc.params)?;
    let faces: Vec<ExposedFace> = desc.faces.iter().map(exposed_face_from_desc).collect();
    loads.clear();
    pe_blast::accumulate_blast_external_all(loads, world, &src, &faces, t, &params, &[]);
    Ok(())
}
