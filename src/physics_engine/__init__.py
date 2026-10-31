"""Rigorous rigid-body physics engine (SI units)."""
from .body import RigidBody, inertia_solid_sphere, inertia_solid_box, inertia_solid_cylinder
from .world import World
from .shapes import Sphere, Box, Capsule, Plane, Material
from .contact import ContactWorld, StaticCollider
from . import (
    forces, shapes, broadphase, narrowphase, contact, integrators, diagnostics,
    quaternion, units, atmosphere, aero, hydro,
)

__all__ = [
    "Sphere", "Box", "Capsule", "Plane", "Material", "ContactWorld", "StaticCollider",
    "shapes", "broadphase", "narrowphase", "contact",
    "RigidBody", "World", "forces", "integrators", "diagnostics", "quaternion", "units",
    "atmosphere", "aero", "hydro",
    "inertia_solid_sphere", "inertia_solid_box", "inertia_solid_cylinder",
]
