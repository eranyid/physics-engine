//! FFI error codes (C `int` / `PeError`).

/// Success.
pub const PE_OK: i32 = 0;
/// Null handle or pointer where non-null required.
pub const PE_ERR_NULL: i32 = 1;
/// Index / buffer length out of range.
pub const PE_ERR_BOUNDS: i32 = 2;
/// Invalid argument (NaN, non-positive mass, bad enum, …).
pub const PE_ERR_ARG: i32 = 3;
/// Internal panic caught at the FFI boundary (`catch_unwind`).
pub const PE_ERR_PANIC: i32 = 4;
/// Scene / serialization failure.
pub const PE_ERR_SCENE: i32 = 5;
/// Unsupported operation.
pub const PE_ERR_UNSUPPORTED: i32 = 6;
