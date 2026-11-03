"""Type stubs for the ``physics_engine_rs`` PyO3 extension (abi3).

SI units throughout. See docs/DESIGN.md and the crate rustdoc for physics detail.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

__version__: str
G: float
STANDARD_GRAVITY: float

ArrayLike = Union[Sequence[float], NDArray[np.float64]]

class RigidBody:
    """Initial rigid-body description (SI)."""

    def __init__(
        self,
        mass: float,
        position: Optional[ArrayLike] = ...,
        velocity: Optional[ArrayLike] = ...,
        inertia_body: Optional[Any] = ...,
        orientation: Optional[ArrayLike] = ...,
        angular_velocity: Optional[ArrayLike] = ...,
        name: Optional[str] = ...,
    ) -> None: ...
    @property
    def mass(self) -> float: ...

class UniformGravity:
    """``F = m g`` with ``g`` [m/s²] (default Earth)."""

    def __init__(self, g: Optional[ArrayLike] = ...) -> None: ...

class NewtonianGravity:
    """Pairwise Newtonian gravity with optional softening."""

    def __init__(self, g_const: Optional[float] = ..., softening: float = ...) -> None: ...

class LinearDrag:
    """Linear drag ``F = -b (v - wind)``."""

    def __init__(self, b: float, wind: Optional[ArrayLike] = ...) -> None: ...

class QuadraticDrag:
    """Quadratic drag ``F = -k |u| u``."""

    def __init__(self, k: float, wind: Optional[ArrayLike] = ...) -> None: ...

class Spring:
    """Hookean spring from a body to a fixed world anchor."""

    def __init__(self, body: int, anchor: ArrayLike, k: float) -> None: ...

class ConstantTorque:
    """Constant world-frame torque on one body."""

    def __init__(self, body: int, tau: ArrayLike) -> None: ...

class World:
    """Stage-1 SoA rigid-body world."""

    def __init__(self, bodies: Sequence[RigidBody], forces: Sequence[Any]) -> None: ...
    def step(self, dt: float, integrator: str = ...) -> None:
        """Advance one step; releases the GIL."""
        ...
    def run(self, t_end: float, dt: float, integrator: str = ...) -> None:
        """Integrate to ``t_end``; releases the GIL."""
        ...
    @property
    def x(self) -> NDArray[np.float64]: ...
    @property
    def v(self) -> NDArray[np.float64]: ...
    @property
    def q(self) -> NDArray[np.float64]: ...
    def pack(self) -> NDArray[np.float64]: ...
    def state_hash(self) -> int: ...
    def kinetic_energy(self) -> float: ...
    def potential_energy(self) -> float: ...
    def total_energy(self) -> float: ...
    def linear_momentum(self) -> NDArray[np.float64]: ...

class Sphere:
    def __init__(self, radius: float) -> None: ...

class Box:
    def __init__(self, hx: float, hy: float, hz: float) -> None: ...

class Capsule:
    def __init__(self, radius: float, half_height: float) -> None: ...

class Plane:
    def __init__(self, normal: ArrayLike, offset: float = ...) -> None: ...

class Material:
    def __init__(self, restitution: float, friction: float) -> None: ...

class StaticCollider:
    def __init__(
        self,
        shape: Any,
        position: Optional[ArrayLike] = ...,
        material: Optional[Material] = ...,
    ) -> None: ...

class ContactWorld:
    """Stage-2 contact world wrapping a :class:`World`."""

    def __init__(
        self,
        world: World,
        shapes: Sequence[Any],
        statics: Sequence[StaticCollider],
        materials: Optional[Sequence[Material]] = ...,
    ) -> None: ...
    def step(self, dt: float) -> None:
        """One contact step; releases the GIL."""
        ...
    def run(self, t_end: float, dt: float) -> None: ...
    @property
    def x(self) -> NDArray[np.float64]: ...
    def state_hash(self) -> int: ...
    def kinetic_energy(self) -> float: ...

class WingPolar:
    def __init__(
        self,
        S: float,
        c: float,
        b: float,
        CD0: float = ...,
        e: float = ...,
        CL0: float = ...,
        CL_alpha: float = ...,
        **kwargs: Any,
    ) -> None: ...

def inertia_solid_sphere(mass: float, radius: float) -> list[list[float]]: ...
def inertia_solid_box(mass: float, a: float, b: float, c: float) -> list[list[float]]: ...
def inertia_solid_cylinder(mass: float, r: float, h: float) -> list[list[float]]: ...
def canonical_determinism_hash() -> int: ...
def canonical_collision_hash() -> int: ...
def canonical_soft_step_collision_hash() -> int: ...
def canonical_joints_hash() -> int: ...
def canonical_convex_hash() -> int: ...
def level_flight_trim(mass: float, polar: WingPolar, rho: float) -> dict[str, float]: ...

RHO0: float
