//! Stable C ABI for the physics engine (`pe_ffi`).
//!
//! # Thread safety
//!
//! A [`PeWorld`] must not be accessed concurrently from multiple threads.
//! Create / destroy / step / buffer I/O on a single owner thread. Distinct
//! worlds may be used from different threads. The library does not install a
//! global allocator lock beyond the Rust standard library.
//!
//! # Unwinding
//!
//! Every `extern "C"` entry point wraps its body in `catch_unwind`. Panics
//! become [`PE_ERR_PANIC`] and never unwind across the FFI boundary.

#![allow(clippy::missing_safety_doc)]
#![deny(missing_docs)]

mod error;
mod queries;
mod world;

pub use error::{
    PE_ERR_ARG, PE_ERR_BOUNDS, PE_ERR_NULL, PE_ERR_PANIC, PE_ERR_SCENE, PE_ERR_UNSUPPORTED, PE_OK,
};
pub use world::PeContactEvent;

use core::ffi::{c_char, c_int, c_void};
use core::ptr;
use core::slice;
use std::panic::{catch_unwind, AssertUnwindSafe};

use pe_collision::joints::{Joint, WORLD_BODY};
use pe_collision::shapes::Material;
use pe_collision::StaticCollider;
use pe_core::RigidBody;
use pe_math::{Quat, Vec3};
use pe_scene::SceneDocument;
use queries::{overlap_aabb, raycast};
use world::{parse_shape, PeWorld};

fn catch<F: FnOnce() -> c_int>(f: F) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(code) => code,
        Err(_) => PE_ERR_PANIC,
    }
}

fn catch_ptr<F: FnOnce() -> *mut PeWorld>(f: F) -> *mut PeWorld {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(p) => p,
        Err(_) => ptr::null_mut(),
    }
}

/// Create an empty world with Earth gravity. Returns null on panic.
#[no_mangle]
pub extern "C" fn pe_world_create() -> *mut PeWorld {
    catch_ptr(|| Box::into_raw(Box::new(PeWorld::new_empty())))
}

/// Destroy a world previously returned by [`pe_world_create`].
///
/// # Safety
/// `world` must be null or a valid pointer from [`pe_world_create`].
#[no_mangle]
pub unsafe extern "C" fn pe_world_destroy(world: *mut PeWorld) {
    let _ = catch(|| {
        if !world.is_null() {
            drop(Box::from_raw(world));
        }
        PE_OK
    });
}

/// Number of dynamic bodies.
///
/// # Safety
/// `world` must be valid or null (null → 0).
#[no_mangle]
pub unsafe extern "C" fn pe_world_body_count(world: *const PeWorld) -> c_int {
    if world.is_null() {
        return 0;
    }
    let w = &*world;
    w.cw.world.n as c_int
}

/// Add a dynamic body. Writes the new index to `out_index` when non-null.
///
/// Shape kinds: `0` sphere (`a`=radius), `1` box (`a,b,c`=half-extents),
/// `2` capsule (`a`=radius, `b`=half_height), `3` plane (statics only — rejected).
///
/// # Safety
/// Pointers must be valid or null as documented.
#[no_mangle]
pub unsafe extern "C" fn pe_world_add_body(
    world: *mut PeWorld,
    mass: f64,
    px: f64,
    py: f64,
    pz: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    qw: f64,
    qx: f64,
    qy: f64,
    qz: f64,
    wx: f64,
    wy: f64,
    wz: f64,
    ixx: f64,
    iyy: f64,
    izz: f64,
    shape_kind: c_int,
    sa: f64,
    sb: f64,
    sc: f64,
    restitution: f64,
    friction: f64,
    out_index: *mut u32,
) -> c_int {
    catch(|| {
        if world.is_null() {
            return PE_ERR_NULL;
        }
        if !(mass.is_finite() && mass > 0.0) {
            return PE_ERR_ARG;
        }
        if shape_kind == 3 {
            return PE_ERR_ARG;
        }
        let shape = match parse_shape(shape_kind, sa, sb, sc, 0.0) {
            Ok(s) => s,
            Err(e) => return e,
        };
        let body = RigidBody::new(mass)
            .with_position(Vec3::new(px, py, pz))
            .with_velocity(Vec3::new(vx, vy, vz))
            .with_orientation(Quat::new(qw, qx, qy, qz))
            .with_angular_velocity(Vec3::new(wx, wy, wz))
            .with_inertia_diag(ixx, iyy, izz);
        let mat = Material::new(restitution, friction);
        let w = &mut *world;
        let idx = w.cw.add_dynamic_body(body, shape, mat);
        if !out_index.is_null() {
            *out_index = idx as u32;
        }
        PE_OK
    })
}

/// Add a static collider (plane/box/sphere/capsule).
///
/// # Safety
/// `world` must be valid.
#[no_mangle]
pub unsafe extern "C" fn pe_world_add_static(
    world: *mut PeWorld,
    shape_kind: c_int,
    sa: f64,
    sb: f64,
    sc: f64,
    sd: f64,
    px: f64,
    py: f64,
    pz: f64,
    restitution: f64,
    friction: f64,
    out_index: *mut u32,
) -> c_int {
    catch(|| {
        if world.is_null() {
            return PE_ERR_NULL;
        }
        let shape = match parse_shape(shape_kind, sa, sb, sc, sd) {
            Ok(s) => s,
            Err(e) => return e,
        };
        let col = StaticCollider::new(shape)
            .with_position(Vec3::new(px, py, pz))
            .with_material(Material::new(restitution, friction));
        let w = &mut *world;
        let idx = w.cw.add_static(col);
        if !out_index.is_null() {
            *out_index = idx as u32;
        }
        PE_OK
    })
}

/// Add a distance joint between `body_a` and `body_b` (`u32::MAX` = world).
///
/// # Safety
/// `world` must be valid.
#[no_mangle]
pub unsafe extern "C" fn pe_world_add_distance_joint(
    world: *mut PeWorld,
    body_a: u32,
    body_b: u32,
    ax: f64,
    ay: f64,
    az: f64,
    bx: f64,
    by: f64,
    bz: f64,
    length: f64,
    out_index: *mut u32,
) -> c_int {
    catch(|| {
        if world.is_null() {
            return PE_ERR_NULL;
        }
        let ba = if body_a == u32::MAX {
            WORLD_BODY
        } else {
            body_a as usize
        };
        let bb = if body_b == u32::MAX {
            WORLD_BODY
        } else {
            body_b as usize
        };
        let j = Joint::distance(ba, bb, Vec3::new(ax, ay, az), Vec3::new(bx, by, bz), length);
        let w = &mut *world;
        let idx = w.cw.add_joint(j);
        if !out_index.is_null() {
            *out_index = idx as u32;
        }
        PE_OK
    })
}

/// Advance one contact step of size `dt` [s].
///
/// # Safety
/// `world` must be valid.
#[no_mangle]
pub unsafe extern "C" fn pe_world_step(world: *mut PeWorld, dt: f64) -> c_int {
    catch(|| {
        if world.is_null() {
            return PE_ERR_NULL;
        }
        if !(dt.is_finite() && dt > 0.0) {
            return PE_ERR_ARG;
        }
        let w = &mut *world;
        w.cw.step(dt);
        w.refresh_contacts();
        PE_OK
    })
}

/// Read transforms into caller SoA buffers (no per-body allocation).
///
/// Layout: `pos[3*i + {0,1,2}]`, `quat[4*i + {0..3}]` as `[w,x,y,z]`.
///
/// # Safety
/// Buffers must hold at least `count` bodies; `count` must be `≤ body_count`.
#[no_mangle]
pub unsafe extern "C" fn pe_world_get_transforms(
    world: *const PeWorld,
    pos: *mut f64,
    quat: *mut f64,
    count: usize,
) -> c_int {
    catch(|| {
        if world.is_null() || pos.is_null() || quat.is_null() {
            return PE_ERR_NULL;
        }
        let w = &*world;
        let n = w.cw.world.n;
        if count > n {
            return PE_ERR_BOUNDS;
        }
        let pos = slice::from_raw_parts_mut(pos, count * 3);
        let quat = slice::from_raw_parts_mut(quat, count * 4);
        for i in 0..count {
            let p = w.cw.world.state.x[i].to_array();
            let q = w.cw.world.state.q[i].to_array();
            pos[3 * i..3 * i + 3].copy_from_slice(&p);
            quat[4 * i..4 * i + 4].copy_from_slice(&q);
        }
        PE_OK
    })
}

/// Write transforms from caller SoA buffers (positions + quaternions).
///
/// # Safety
/// Buffers must hold at least `count` bodies.
#[no_mangle]
pub unsafe extern "C" fn pe_world_set_transforms(
    world: *mut PeWorld,
    pos: *const f64,
    quat: *const f64,
    count: usize,
) -> c_int {
    catch(|| {
        if world.is_null() || pos.is_null() || quat.is_null() {
            return PE_ERR_NULL;
        }
        let w = &mut *world;
        let n = w.cw.world.n;
        if count > n {
            return PE_ERR_BOUNDS;
        }
        let pos = slice::from_raw_parts(pos, count * 3);
        let quat = slice::from_raw_parts(quat, count * 4);
        for i in 0..count {
            w.cw.world.state.x[i] = Vec3::new(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
            w.cw.world.state.q[i] = Quat::new(
                quat[4 * i],
                quat[4 * i + 1],
                quat[4 * i + 2],
                quat[4 * i + 3],
            )
            .normalize();
            w.cw.wake_body(i);
        }
        PE_OK
    })
}

/// Raycast. Sets `*out_hit` to 1 on hit / 0 on miss when non-null; always `PE_OK` on success.
///
/// # Safety
/// Pointers must be valid when non-null.
#[no_mangle]
pub unsafe extern "C" fn pe_world_raycast(
    world: *const PeWorld,
    ox: f64,
    oy: f64,
    oz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    tmax: f64,
    out_hit: *mut c_int,
    out_t: *mut f64,
    out_point: *mut f64,
    out_normal: *mut f64,
    out_body: *mut u32,
) -> c_int {
    catch(|| {
        if world.is_null() {
            return PE_ERR_NULL;
        }
        let w = &*world;
        let hit = raycast(&w.cw, Vec3::new(ox, oy, oz), Vec3::new(dx, dy, dz), tmax);
        match hit {
            Some(h) => {
                if !out_hit.is_null() {
                    *out_hit = 1;
                }
                if !out_t.is_null() {
                    *out_t = h.t;
                }
                if !out_point.is_null() {
                    let p = slice::from_raw_parts_mut(out_point, 3);
                    p.copy_from_slice(&h.point.to_array());
                }
                if !out_normal.is_null() {
                    let n = slice::from_raw_parts_mut(out_normal, 3);
                    n.copy_from_slice(&h.normal.to_array());
                }
                if !out_body.is_null() {
                    *out_body = h.body as u32;
                }
                PE_OK
            }
            None => {
                if !out_hit.is_null() {
                    *out_hit = 0;
                }
                PE_OK
            }
        }
    })
}

/// Overlap AABB query. Writes up to `max_out` body indices into `out_bodies`.
///
/// # Safety
/// `out_bodies` must hold `max_out` u32s when non-null / max_out>0.
#[no_mangle]
pub unsafe extern "C" fn pe_world_overlap_aabb(
    world: *const PeWorld,
    lo_x: f64,
    lo_y: f64,
    lo_z: f64,
    hi_x: f64,
    hi_y: f64,
    hi_z: f64,
    out_bodies: *mut u32,
    max_out: usize,
    out_count: *mut usize,
) -> c_int {
    catch(|| {
        if world.is_null() {
            return PE_ERR_NULL;
        }
        let w = &*world;
        let aabb = pe_collision::shapes::Aabb {
            lo: Vec3::new(lo_x, lo_y, lo_z),
            hi: Vec3::new(hi_x, hi_y, hi_z),
        };
        let mut ids = Vec::new();
        overlap_aabb(&w.cw, aabb, &mut ids);
        let nwrite = ids.len().min(max_out);
        if nwrite > 0 {
            if out_bodies.is_null() {
                return PE_ERR_NULL;
            }
            let out = slice::from_raw_parts_mut(out_bodies, nwrite);
            for i in 0..nwrite {
                out[i] = ids[i] as u32;
            }
        }
        if !out_count.is_null() {
            *out_count = ids.len();
        }
        PE_OK
    })
}

/// Copy contact events from the last step into `out` (capacity `max_out`).
///
/// # Safety
/// `out` must hold `max_out` events when non-null.
#[no_mangle]
pub unsafe extern "C" fn pe_world_get_contacts(
    world: *const PeWorld,
    out: *mut PeContactEvent,
    max_out: usize,
    out_count: *mut usize,
) -> c_int {
    catch(|| {
        if world.is_null() {
            return PE_ERR_NULL;
        }
        let w = &*world;
        let n = w.last_contacts.len();
        let nwrite = n.min(max_out);
        if nwrite > 0 {
            if out.is_null() {
                return PE_ERR_NULL;
            }
            let dst = slice::from_raw_parts_mut(out, nwrite);
            dst.copy_from_slice(&w.last_contacts[..nwrite]);
        }
        if !out_count.is_null() {
            *out_count = n;
        }
        PE_OK
    })
}

/// Save world to a scene binary buffer. Writes required size to `out_len`.
///
/// # Safety
/// Pointers must be valid when non-null.
#[no_mangle]
pub unsafe extern "C" fn pe_world_save_scene(
    world: *const PeWorld,
    buf: *mut u8,
    buf_len: usize,
    out_len: *mut usize,
) -> c_int {
    catch(|| {
        if world.is_null() || out_len.is_null() {
            return PE_ERR_NULL;
        }
        let w = &*world;
        let doc = match SceneDocument::capture(&w.cw, w.forces.clone()) {
            Ok(d) => d,
            Err(_) => return PE_ERR_SCENE,
        };
        let bytes = match doc.to_binary() {
            Ok(b) => b,
            Err(_) => return PE_ERR_SCENE,
        };
        *out_len = bytes.len();
        if buf.is_null() || buf_len < bytes.len() {
            return PE_ERR_BOUNDS;
        }
        slice::from_raw_parts_mut(buf, bytes.len()).copy_from_slice(&bytes);
        PE_OK
    })
}

/// Load scene binary into an existing world handle (replaces contents).
///
/// # Safety
/// `buf` must hold `buf_len` bytes; `world` must be valid.
#[no_mangle]
pub unsafe extern "C" fn pe_world_load_scene(
    world: *mut PeWorld,
    buf: *const u8,
    buf_len: usize,
) -> c_int {
    catch(|| {
        if world.is_null() || buf.is_null() {
            return PE_ERR_NULL;
        }
        let bytes = slice::from_raw_parts(buf, buf_len);
        let doc = match SceneDocument::from_binary(bytes) {
            Ok(d) => d,
            Err(_) => return PE_ERR_SCENE,
        };
        let cw = match doc.instantiate() {
            Ok(w) => w,
            Err(_) => return PE_ERR_SCENE,
        };
        let w = &mut *world;
        w.forces = doc.forces;
        w.cw = cw;
        w.last_contacts.clear();
        PE_OK
    })
}

/// Library version string (static).
#[no_mangle]
pub extern "C" fn pe_version_string() -> *const c_char {
    c"0.1.0".as_ptr()
}

/// Opaque world pointer type alias for headers / docs.
pub type PeWorldHandle = *mut PeWorld;

// Reserved for future event userdata.
#[allow(dead_code)]
fn _reserved(_: *mut c_void) {}
