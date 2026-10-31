"""Observed convergence order on the harmonic oscillator x'' = -(k/m) x."""
import numpy as np
import pytest
from physics_engine import RigidBody, World, forces, integrators, diagnostics

M, K = 2.0, 8.0          # kg, N/m -> omega = 2 rad/s
OMEGA = np.sqrt(K / M)
T_END = 2.0              # s


def final_error(integ, dt):
    w = World([RigidBody(mass=M, position=[1.0, 0, 0])], [forces.Spring(0, [0, 0, 0], K)])
    w.run(T_END, dt, integ)
    x_ex = np.cos(OMEGA * T_END)
    v_ex = -OMEGA * np.sin(OMEGA * T_END)
    return np.hypot(w.state.x[0, 0] - x_ex, (w.state.v[0, 0] - v_ex) / OMEGA)


def observed_order(integ, dts):
    errs = np.array([final_error(integ, dt) for dt in dts])
    return np.polyfit(np.log(dts), np.log(errs), 1)[0]


@pytest.mark.parametrize("name,expected,dts", [
    ("explicit_euler", 1, [1e-3, 5e-4, 2.5e-4, 1.25e-4]),
    ("symplectic_euler", 1, [1e-3, 5e-4, 2.5e-4, 1.25e-4]),
    ("velocity_verlet", 2, [1e-2, 5e-3, 2.5e-3, 1.25e-3]),
    ("rk4", 4, [8e-2, 4e-2, 2e-2, 1e-2]),
])
def test_convergence_order(name, expected, dts):
    p = observed_order(integrators.ALL[name], dts)
    print(f"{name}: observed order {p:.3f}")
    assert p == pytest.approx(expected, abs=0.1)


def test_oscillator_energy_behaviour():
    """Euler gains energy; symplectic schemes keep it bounded over 100 periods."""
    def run(integ, dt=0.01, periods=100):
        w = World([RigidBody(mass=M, position=[1.0, 0, 0])], [forces.Spring(0, [0, 0, 0], K)])
        E0 = diagnostics.total_energy(w)
        Es = w.run(periods * 2 * np.pi / OMEGA, dt, integ, record=diagnostics.total_energy)
        return np.max(np.abs(np.array(Es) - E0)) / E0, Es[-1] / E0 - 1
    _, euler_drift = run(integrators.explicit_euler)
    assert euler_drift > 1.0  # (1 + (omega dt)^2)^(N/2) - 1, grows exponentially
    for integ, bound in ((integrators.symplectic_euler, 0.011), (integrators.velocity_verlet, 1.1e-4)):
        max_err, _ = run(integ)
        assert max_err < bound  # symplectic: O(omega dt) and O((omega dt)^2) bounded oscillation
