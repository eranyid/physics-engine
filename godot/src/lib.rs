//! Godot 4 GDExtension for the physics engine (M25).
//!
//! Exposes a game-usable surface over `ContactWorld`: authored bodies/shapes,
//! joints, scene queries, contact signals, and fixed-step stepping with optional
//! visual interpolation. See `docs/GODOT.md`.

mod body;
mod convert;
mod joint;
mod shapes_util;
mod world;

use godot::prelude::*;

struct PeGodotExtension;

#[gdextension]
unsafe impl ExtensionLibrary for PeGodotExtension {}
