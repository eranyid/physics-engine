"""Two-body Kepler problem in SI (Sun-Earth-like masses)."""
import numpy as np
import pytest
from physics_engine import RigidBody, World, forces, integrators, diagnostics
from physics_engine.units import G

M_SUN, M_EARTH, AU = 1.98847e30, 5.9722e24, 1.495978707e11  # kg, kg, m


def two_body(a, e):
    """Bodies starting at periapsis, COM at rest at origin. Returns world and period [s]."""
    M = M_SUN + M_EARTH
    rp = a * (1 - e)
    vp = np.sqrt(G * M * (1 + e) / rp)  # vis-viva at periapsis, relative speed
    f1, f2 = M_EARTH / M, M_SUN / M
    bodies = [RigidBody(mass=M_SUN, position=[-f1 * rp, 0, 0], velocity=[0, -f1 * vp, 0]),
              RigidBody(mass=M_EARTH, position=[f2 * rp, 0, 0], velocity=[0, f2 * vp, 0])]
    return World(bodies, [forces.NewtonianGravity()]), 2 * np.pi * np.sqrt(a**3 / (G * M))


def analytic_energy(a):
    return -G * M_SUN * M_EARTH / (2 * a)


@pytest.mark.parametrize("e", [0.0, 0.5])
def test_orbit_period_energy_angular_momentum(e):
    w, P = two_body(AU, e)
    assert P / 86400 == pytest.approx(365.25, rel=2e-3)  # one year
    E0 = diagnostics.total_energy(w)
    assert E0 == pytest.approx(analytic_energy(AU), rel=1e-12)
    L0 = diagnostics.angular_momentum(w)
    x0 = w.state.x[1] - w.state.x[0]
    w.run(P, P / 20000, integrators.rk4)
    rel = w.state.x[1] - w.state.x[0]
    assert np.linalg.norm(rel - x0) / AU < 1e-8         # returns to periapsis after exactly P
    assert diagnostics.total_energy(w) == pytest.approx(E0, rel=1e-10)
    np.testing.assert_allclose(diagnostics.angular_momentum(w), L0, rtol=1e-10, atol=1e-10 * np.linalg.norm(L0))
    if e > 0:  # apoapsis distance a(1+e) half a period in
        w2, _ = two_body(AU, e)
        w2.run(P / 2, P / 20000, integrators.rk4)
        assert np.linalg.norm(w2.state.x[1] - w2.state.x[0]) == pytest.approx(AU * (1 + e), rel=1e-9)


@pytest.mark.slow
def test_symplectic_vs_rk4_long_run_energy():
    """Coarse dt, 200 orbits, e=0.5: Verlet energy error stays bounded and does not grow;
    RK4 drifts secularly (monotonic loss)."""
    e, n_orbits, steps_per_orbit = 0.5, 200, 200
    out = {}
    for name in ("velocity_verlet", "rk4"):
        w, P = two_body(AU, e)
        E0 = diagnostics.total_energy(w)
        Es = np.array(w.run(n_orbits * P, P / steps_per_orbit, integrators.ALL[name],
                            record=diagnostics.total_energy))
        err = (Es - E0) / abs(E0)
        k = len(err) // 10
        out[name] = (np.max(np.abs(err[:k])), np.max(np.abs(err[-k:])), err[-1])
        print(name, "first-10%% max |dE/E|=%.3e last-10%% max=%.3e final=%.3e" % out[name])
    v_first, v_last, _ = out["velocity_verlet"]
    r_first, r_last, r_final = out["rk4"]
    assert v_last < 1.5 * v_first           # bounded, no secular growth
    assert r_last > 5 * r_first             # secular drift
    assert r_final < 0 and abs(r_final) > 0.9 * r_last  # RK4 energy loss accumulates (dissipative)
