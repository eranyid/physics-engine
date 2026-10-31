"""Aircraft aerodynamics: lift/drag polar with stall, thrust, body-frame forces/moments.

Body axes (right-handed, matching the engine's z-up world when level):
  +x nose forward, +y right wing, +z up through the cabin.
Angle of attack α = atan2(-v_z, v_x) from body velocity (positive nose-up).
Sideslip is ignored (planar longitudinal model).

Force generators follow the stage-1 protocol: apply(world, state, t, F, T).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import quaternion as quat
from . import atmosphere as atm
from .units import STANDARD_GRAVITY

RHO_SL = atm.RHO0


@dataclass
class WingPolar:
    """Lift curve with stall + parabolic drag polar C_D = C_D0 + C_L^2/(π e AR).

    S [m^2] reference area, c [m] mean aerodynamic chord, b [m] span (AR = b^2/S if
    aspect_ratio is None). Coefficients are dimensionless; angles in radians.
    """
    S: float
    c: float
    b: float
    CD0: float = 0.02
    e: float = 0.8                    # Oswald efficiency
    CL0: float = 0.0
    CL_alpha: float = 2.0 * np.pi     # per radian (thin-airfoil ≈ 2π)
    alpha_stall: float = np.deg2rad(15.0)
    CL_fall: float = 1.0              # post-stall dCL/dα magnitude (1/rad)
    aspect_ratio: float | None = None
    # Pitching-moment derivatives about the COM (body y): Cm0, Cm_α, Cm_q (per rad).
    Cm0: float = 0.0
    Cm_alpha: float = -0.5            # static pitch stiffness (stable if < 0 with CL_alpha > 0)
    Cm_q: float = -8.0                # pitch damping

    def __post_init__(self):
        if self.S <= 0 or self.c <= 0 or self.b <= 0:
            raise ValueError("S, c, b must be positive")
        if self.aspect_ratio is None:
            self.aspect_ratio = self.b * self.b / self.S
        if self.e <= 0 or self.aspect_ratio <= 0:
            raise ValueError("e and AR must be positive")

    def CL(self, alpha: float) -> float:
        """Lift coefficient vs angle of attack [rad], with a linear stall break."""
        a = float(alpha)
        a_s = self.alpha_stall
        cl_lin = self.CL0 + self.CL_alpha * a
        if abs(a) <= a_s:
            return cl_lin
        # Peak at stall, then fall toward a flat-plate-like slope.
        sign = 1.0 if a > 0 else -1.0
        cl_peak = self.CL0 + self.CL_alpha * sign * a_s
        return cl_peak - sign * self.CL_fall * (abs(a) - a_s)

    def CD(self, CL: float) -> float:
        """Parabolic drag polar including induced drag."""
        return self.CD0 + (CL * CL) / (np.pi * self.e * self.aspect_ratio)

    def Cm(self, alpha: float, q: float, V: float) -> float:
        """Pitching moment coefficient; q [rad/s] body pitch rate, V [m/s] airspeed."""
        V_safe = max(float(V), 1e-6)
        return self.Cm0 + self.Cm_alpha * alpha + self.Cm_q * (q * self.c / (2.0 * V_safe))

    def L_over_D(self, alpha: float) -> float:
        cl = self.CL(alpha)
        cd = self.CD(cl)
        return cl / cd if cd != 0 else np.inf


def dynamic_pressure(rho: float, V: float) -> float:
    """q_bar = 1/2 rho V^2 [Pa]."""
    return 0.5 * rho * V * V


class AircraftAero:
    """Body-frame aero forces/moments on one rigid body from a WingPolar + ISA density.

    Lift ⟂ freestream in the body x–z plane (positive "up" relative to velocity);
    drag ∥ −freestream; sideforce = 0. Moments: pitching M = q_bar S c Cm about +y.
    """

    def __init__(self, body: int, polar: WingPolar, atmosphere=None, wind=(0.0, 0.0, 0.0)):
        self.body = int(body)
        self.polar = polar
        self.atmosphere = atmosphere  # None => use atm.density(h)
        self.wind = np.asarray(wind, float)

    def _rho(self, h: float) -> float:
        if self.atmosphere is None:
            return atm.density(h)
        return float(self.atmosphere(h) if callable(self.atmosphere) else self.atmosphere)

    def air_data(self, world, s) -> dict:
        """Diagnostic: airspeed, alpha, CL, CD, Cm, lift, drag for the attached body."""
        i = self.body
        R = quat.to_matrix(s.q[i])
        v_rel = s.v[i] - self.wind
        v_b = R.T @ v_rel
        V = float(np.linalg.norm(v_b))
        if V < 1e-9:
            return {"V": 0.0, "alpha": 0.0, "CL": 0.0, "CD": self.polar.CD0, "Cm": self.polar.Cm0,
                    "lift": 0.0, "drag": 0.0, "rho": self._rho(float(s.x[i, 2]))}
        alpha = float(np.arctan2(-v_b[2], v_b[0]))
        omega_b = R.T @ world.omega(s)[i]
        q_rate = float(omega_b[1])
        rho = self._rho(float(s.x[i, 2]))
        CL = self.polar.CL(alpha)
        CD = self.polar.CD(CL)
        Cm = self.polar.Cm(alpha, q_rate, V)
        qbar = dynamic_pressure(rho, V)
        return {
            "V": V, "alpha": alpha, "CL": CL, "CD": CD, "Cm": Cm,
            "lift": qbar * self.polar.S * CL, "drag": qbar * self.polar.S * CD,
            "rho": rho, "q": q_rate,
        }

    def apply(self, world, s, t, F, T):
        i = self.body
        R = quat.to_matrix(s.q[i])          # body -> world
        v_rel = s.v[i] - self.wind
        v_b = R.T @ v_rel                   # body-frame velocity
        V = float(np.linalg.norm(v_b))
        if V < 1e-9:
            return
        # Unit freestream direction in body axes (aircraft velocity direction).
        e_v = v_b / V
        alpha = float(np.arctan2(-v_b[2], v_b[0]))
        omega_b = R.T @ world.omega(s)[i]
        q_rate = float(omega_b[1])
        rho = self._rho(float(s.x[i, 2]))
        polar = self.polar
        CL = polar.CL(alpha)
        CD = polar.CD(CL)
        Cm = polar.Cm(alpha, q_rate, V)
        qbar = dynamic_pressure(rho, V)
        # Freestream e_v = (cos α, 0, −sin α); lift "up" relative to V is (sin α, 0, cos α).
        e_lift = np.array([np.sin(alpha), 0.0, np.cos(alpha)])
        e_drag = -e_v
        f_b = qbar * polar.S * (CL * e_lift + CD * e_drag)
        m_b = np.array([0.0, qbar * polar.S * polar.c * Cm, 0.0])
        F[i] += R @ f_b
        T[i] += R @ m_b


class FixedThrust:
    """Constant thrust [N] along a body-fixed axis (default +x, nose)."""

    def __init__(self, body: int, thrust: float, axis_body=(1.0, 0.0, 0.0)):
        self.body = int(body)
        self.thrust = float(thrust)
        a = np.asarray(axis_body, float)
        self.axis = a / np.linalg.norm(a)

    def apply(self, world, s, t, F, T):
        i = self.body
        R = quat.to_matrix(s.q[i])
        F[i] += R @ (self.thrust * self.axis)


class PropellerThrust:
    """Simple propeller: T = min(T_static, η P / max(V, V_min)) · throttle · (ρ/ρ_sl).

    T_static [N], shaft power P [W], efficiency η (dimensionless), throttle in [0, 1].
    """

    def __init__(
        self,
        body: int,
        T_static: float,
        power: float,
        eta: float = 0.8,
        throttle: float = 1.0,
        V_min: float = 1.0,
        rho_sl: float = RHO_SL,
        atmosphere=None,
        axis_body=(1.0, 0.0, 0.0),
        wind=(0.0, 0.0, 0.0),
    ):
        self.body = int(body)
        self.T_static = float(T_static)
        self.power = float(power)
        self.eta = float(eta)
        self.throttle = float(throttle)
        self.V_min = float(V_min)
        self.rho_sl = float(rho_sl)
        self.atmosphere = atmosphere
        a = np.asarray(axis_body, float)
        self.axis = a / np.linalg.norm(a)
        self.wind = np.asarray(wind, float)

    def _rho(self, h: float) -> float:
        if self.atmosphere is None:
            return atm.density(h)
        return float(self.atmosphere(h) if callable(self.atmosphere) else self.atmosphere)

    def thrust_magnitude(self, V: float, rho: float) -> float:
        T_power = self.eta * self.power / max(V, self.V_min)
        T = min(self.T_static, T_power) * self.throttle * (rho / self.rho_sl)
        return max(0.0, T)

    def apply(self, world, s, t, F, T):
        i = self.body
        R = quat.to_matrix(s.q[i])
        V = float(np.linalg.norm(s.v[i] - self.wind))
        rho = self._rho(float(s.x[i, 2]))
        F[i] += R @ (self.thrust_magnitude(V, rho) * self.axis)


class JetThrust:
    """Simple jet: T = T_sl · throttle · (ρ/ρ_sl)^n along a body axis."""

    def __init__(
        self,
        body: int,
        T_sl: float,
        throttle: float = 1.0,
        n: float = 0.7,
        rho_sl: float = RHO_SL,
        atmosphere=None,
        axis_body=(1.0, 0.0, 0.0),
    ):
        self.body = int(body)
        self.T_sl = float(T_sl)
        self.throttle = float(throttle)
        self.n = float(n)
        self.rho_sl = float(rho_sl)
        self.atmosphere = atmosphere
        a = np.asarray(axis_body, float)
        self.axis = a / np.linalg.norm(a)

    def _rho(self, h: float) -> float:
        if self.atmosphere is None:
            return atm.density(h)
        return float(self.atmosphere(h) if callable(self.atmosphere) else self.atmosphere)

    def thrust_magnitude(self, rho: float) -> float:
        return max(0.0, self.T_sl * self.throttle * (rho / self.rho_sl) ** self.n)

    def apply(self, world, s, t, F, T):
        i = self.body
        R = quat.to_matrix(s.q[i])
        rho = self._rho(float(s.x[i, 2]))
        F[i] += R @ (self.thrust_magnitude(rho) * self.axis)


def level_flight_trim(mass: float, polar: WingPolar, rho: float, g: float = STANDARD_GRAVITY):
    """Solve steady level flight: L = W, T = D. Returns V, alpha, thrust, CL, CD, L/D.

    Assumes small α so lift ≈ vertical; solves CL = W / (q S) with CL in the linear range,
    then α = (CL − CL0)/CL_α and T = D.
    """
    W = mass * g
    # Need q such that required CL is achievable: CL = W/(q S) ∈ (−CL_stall, CL_stall).
    CL_stall = polar.CL(polar.alpha_stall)
    q_min = W / (polar.S * CL_stall) if CL_stall > 0 else np.inf
    # Use the α that maximises L/D in the linear regime as a convenient trim point,
    # or solve for a prescribed V — here we pick the CL that maximises L/D analytically
    # for a parabolic polar: CL_opt = sqrt(π e AR CD0), then V from L=W.
    CL = np.sqrt(np.pi * polar.e * polar.aspect_ratio * polar.CD0)
    CL = min(CL, 0.95 * CL_stall)
    q = W / (polar.S * CL)
    V = float(np.sqrt(2.0 * q / rho))
    alpha = (CL - polar.CL0) / polar.CL_alpha
    CD = polar.CD(CL)
    D = q * polar.S * CD
    return {
        "V": V,
        "alpha": float(alpha),
        "thrust": float(D),
        "CL": float(CL),
        "CD": float(CD),
        "L_over_D": float(CL / CD),
        "q_bar": float(q),
        "q_min_stall": float(q_min),
    }


def terminal_velocity_vertical(mass: float, polar: WingPolar, rho: float, alpha: float = -np.pi / 2,
                               g: float = STANDARD_GRAVITY) -> float:
    """Speed where |Drag(+Lift projected)| balances weight for a given α (e.g. flat spin/dive).

    For a pure vertical dive with α = 0 (nose down, body x along −z_world is a separate
    orientation problem). Here we balance weight against aero force magnitude along −velocity
    for a prescribed α with freestream along body x (α measured as usual): net opposing
    coefficient is CD at that α (lift is perpendicular and does no work along V).
    So mg = 1/2 rho V^2 S CD(α)  =>  V = sqrt(2 mg / (rho S CD)).
    """
    CD = polar.CD(polar.CL(alpha))
    return float(np.sqrt(2.0 * mass * g / (rho * polar.S * CD)))
