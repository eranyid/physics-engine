"""SI unit conventions. All engine quantities are SI; aliases document intent in signatures."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Annotated
import numpy as np
from scipy import constants as _c

Meters = Annotated[float, "m"]
Kilograms = Annotated[float, "kg"]
Seconds = Annotated[float, "s"]
Newtons = Annotated[float, "N"]
Joules = Annotated[float, "J"]
MetersPerSecond = Annotated[float, "m/s"]
Vec3 = Annotated[np.ndarray, "shape (3,)"]

G: float = _c.G                 # m^3 kg^-1 s^-2
STANDARD_GRAVITY: float = _c.g  # m/s^2


@dataclass(frozen=True)
class Dim:
    """Dimension exponents (mass, length, time) for dimensional-analysis checks."""
    M: float = 0
    L: float = 0
    T: float = 0

    def __mul__(self, o: "Dim") -> "Dim":
        return Dim(self.M + o.M, self.L + o.L, self.T + o.T)

    def __truediv__(self, o: "Dim") -> "Dim":
        return Dim(self.M - o.M, self.L - o.L, self.T - o.T)

    def __pow__(self, p: float) -> "Dim":
        return Dim(self.M * p, self.L * p, self.T * p)


DIMLESS = Dim()
MASS, LENGTH, TIME = Dim(M=1), Dim(L=1), Dim(T=1)
VELOCITY = LENGTH / TIME
ACCEL = VELOCITY / TIME
FORCE = MASS * ACCEL
ENERGY = FORCE * LENGTH
GRAV_CONST = LENGTH**3 / MASS / TIME**2
