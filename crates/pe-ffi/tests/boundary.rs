//! FFI boundary validation: null pointers, bad args, NaN/Inf, oversized counts.
//!
//! Malformed input must return typed `PE_ERR_*` codes — never panic across the ABI.

use pe_ffi::{
    pe_version_string, pe_world_add_body, pe_world_add_static, pe_world_body_count,
    pe_world_create, pe_world_destroy, pe_world_get_transforms, pe_world_set_transforms,
    pe_world_step, PeWorld, PE_ERR_ARG, PE_ERR_BOUNDS, PE_ERR_NULL, PE_OK,
};
use std::ptr;

fn add_unit_sphere(world: *mut PeWorld) -> u32 {
    let mut idx = 999u32;
    let rc = unsafe {
        pe_world_add_body(
            world, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4,
            0.4, 0, 0.25, 0.0, 0.0, 0.5, 0.3, &mut idx,
        )
    };
    assert_eq!(rc, PE_OK);
    idx
}

#[test]
fn null_world_returns_err_null() {
    let mut idx = 0u32;
    let rc = unsafe {
        pe_world_add_body(
            ptr::null_mut(),
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            &mut idx,
        )
    };
    assert_eq!(rc, PE_ERR_NULL);
    assert_eq!(
        unsafe { pe_world_step(ptr::null_mut(), 1.0 / 60.0) },
        PE_ERR_NULL
    );
    assert_eq!(unsafe { pe_world_body_count(ptr::null()) }, 0);
    unsafe { pe_world_destroy(ptr::null_mut()) };
}

#[test]
fn nan_and_inf_inputs_are_arg_errors_not_panic() {
    let w = pe_world_create();
    assert!(!w.is_null());

    let mut idx = 0u32;
    assert_eq!(
        unsafe {
            pe_world_add_body(
                w,
                f64::NAN,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                &mut idx,
            )
        },
        PE_ERR_ARG
    );
    assert_eq!(
        unsafe {
            pe_world_add_body(
                w,
                1.0,
                f64::INFINITY,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                &mut idx,
            )
        },
        PE_ERR_ARG
    );
    // Non-PD inertia previously panicked → PE_ERR_PANIC; now typed ARG.
    assert_eq!(
        unsafe {
            pe_world_add_body(
                w, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0,
                1.0, 0, 0.1, 0.0, 0.0, 0.0, 0.0, &mut idx,
            )
        },
        PE_ERR_ARG
    );
    assert_eq!(
        unsafe {
            pe_world_add_body(
                w, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                1.0, 0, 0.1, 0.0, 0.0, 0.0, 0.0, &mut idx,
            )
        },
        PE_ERR_ARG
    );
    assert_eq!(
        unsafe {
            pe_world_add_body(
                w, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                1.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, &mut idx,
            )
        },
        PE_ERR_ARG
    );
    assert_eq!(
        unsafe {
            pe_world_add_static(
                w,
                3,
                0.0,
                0.0,
                1.0,
                0.0,
                f64::NAN,
                0.0,
                0.0,
                0.0,
                1.0,
                &mut idx,
            )
        },
        PE_ERR_ARG
    );
    assert_eq!(unsafe { pe_world_step(w, f64::NAN) }, PE_ERR_ARG);
    assert_eq!(unsafe { pe_world_step(w, 0.0) }, PE_ERR_ARG);
    assert_eq!(unsafe { pe_world_step(w, -0.1) }, PE_ERR_ARG);

    unsafe { pe_world_destroy(w) };
}

#[test]
fn oversized_transform_count_is_bounds() {
    let w = pe_world_create();
    add_unit_sphere(w);
    let mut pos = [0.0f64; 6];
    let mut quat = [0.0f64; 8];
    assert_eq!(
        unsafe { pe_world_get_transforms(w, pos.as_mut_ptr(), quat.as_mut_ptr(), 2) },
        PE_ERR_BOUNDS
    );
    assert_eq!(
        unsafe { pe_world_set_transforms(w, pos.as_ptr(), quat.as_ptr(), 2) },
        PE_ERR_BOUNDS
    );
    assert_eq!(
        unsafe { pe_world_get_transforms(w, pos.as_mut_ptr(), quat.as_mut_ptr(), 1) },
        PE_OK
    );
    unsafe { pe_world_destroy(w) };
}

#[test]
fn set_transforms_rejects_nan() {
    let w = pe_world_create();
    add_unit_sphere(w);
    let pos = [f64::NAN, 0.0, 0.0];
    let quat = [1.0, 0.0, 0.0, 0.0];
    assert_eq!(
        unsafe { pe_world_set_transforms(w, pos.as_ptr(), quat.as_ptr(), 1) },
        PE_ERR_ARG
    );
    let pos = [0.0, 0.0, 0.0];
    let quat = [0.0, 0.0, 0.0, 0.0];
    assert_eq!(
        unsafe { pe_world_set_transforms(w, pos.as_ptr(), quat.as_ptr(), 1) },
        PE_ERR_ARG
    );
    unsafe { pe_world_destroy(w) };
}

#[test]
fn null_transform_buffers_are_null_error() {
    let w = pe_world_create();
    add_unit_sphere(w);
    let mut pos = [0.0f64; 3];
    let mut quat = [0.0f64; 4];
    assert_eq!(
        unsafe { pe_world_get_transforms(w, ptr::null_mut(), quat.as_mut_ptr(), 1) },
        PE_ERR_NULL
    );
    assert_eq!(
        unsafe { pe_world_get_transforms(w, pos.as_mut_ptr(), ptr::null_mut(), 1) },
        PE_ERR_NULL
    );
    unsafe { pe_world_destroy(w) };
}

#[test]
fn version_string_is_non_null() {
    assert!(!pe_version_string().is_null());
}
