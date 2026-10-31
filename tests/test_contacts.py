"""Contact dynamics vs closed-form results."""
import numpy as np
import pytest
from physics_engine import (RigidBody, World, ContactWorld, StaticCollider, Sphere, Box, Plane,
                            Material, quaternion as quat, diagnostics as dg, units as u)
from physics_engine.forces import UniformGravity

g = u.STANDARD_GRAVITY
GROUND = StaticCollider(Plane((0, 0, 1), 0.0), material=Material(0.0, 1.0))


def ball(m, r, x, v=(0, 0, 0), w=(0, 0, 0)):
    return RigidBody(m, Sphere(r).inertia(m), position=x, velocity=v, angular_velocity=w)


def make(bodies, shapes, statics=(), mats=None, gravity=True, **kw):
    w = World(bodies, [UniformGravity()] if gravity else [])
    return ContactWorld(w, shapes, statics, mats, **kw)


# ---------------------------------------------------------------- head-on collisions
@pytest.mark.parametrize("e", [1.0, 0.5, 0.0])
def test_head_on_spheres_momentum_energy_restitution(e):
    m1, m2, v1, v2 = 2.0, 3.0, 4.0, -1.0
    cw = make([ball(m1, 0.5, [-1, 0, 0], [v1, 0, 0]), ball(m2, 0.5, [1, 0, 0], [v2, 0, 0])],
              [Sphere(0.5)] * 2, mats=[Material(e, 0.0)] * 2, gravity=False, restitution_threshold=0.0)
    p0, T0 = dg.linear_momentum(cw.world), dg.kinetic_energy(cw.world)
    cw.run(1.0, 1e-3)
    s = cw.world.state
    p1, T1 = dg.linear_momentum(cw.world), dg.kinetic_energy(cw.world)
    # closed form 1-D collision
    u1 = (m1 * v1 + m2 * v2 + m2 * e * (v2 - v1)) / (m1 + m2)
    u2 = (m1 * v1 + m2 * v2 + m1 * e * (v1 - v2)) / (m1 + m2)
    np.testing.assert_allclose(s.v[:, 0], [u1, u2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(p1, p0, atol=1e-12)
    assert (s.v[1, 0] - s.v[0, 0]) / (v1 - v2) == pytest.approx(e, abs=1e-12)
    dT = 0.5 * m1 * m2 / (m1 + m2) * (1 - e * e) * (v1 - v2) ** 2   # energy lost
    assert T0 - T1 == pytest.approx(dT, abs=1e-10)
    assert np.allclose(s.v[:, 1:], 0) and np.allclose(s.L, 0)


def test_head_on_off_axis_hit_conserves_angular_momentum_about_contact_point():
    """Sphere hits a free box off-centre with friction: L about the (fixed) impact point and
    P are conserved; the box picks up spin."""
    mb, ms = 4.0, 1.0
    box = RigidBody(mb, Box((0.5, 0.5, 0.5)).inertia(mb), position=[0, 0, 0])
    sph = ball(ms, 0.2, [-2, 0.3, 0.1], [5, 0, 0], [0, 0, 20])
    cw = make([box, sph], [Box((0.5, 0.5, 0.5)), Sphere(0.2)],
              mats=[Material(0.6, 0.4)] * 2, gravity=False, beta=0.0)
    o = np.array([-0.5, 0.3, 0.1])
    H0, P0, T0 = dg.angular_momentum(cw.world, origin=o), dg.linear_momentum(cw.world), dg.kinetic_energy(cw.world)
    cw.run(0.6, 1e-3)
    H1, P1 = dg.angular_momentum(cw.world, origin=o), dg.linear_momentum(cw.world)
    np.testing.assert_allclose(P1, P0, atol=1e-12)
    np.testing.assert_allclose(H1, H0, atol=1e-12)
    assert abs(dg.body_angular_velocity(cw.world)[0, 2]) > 0.1   # box spins
    assert dg.kinetic_energy(cw.world) < T0


def test_position_correction_keeps_momenta():
    """Overlapping spheres at rest are pushed apart without gaining kinetic energy."""
    cw = make([ball(1.0, 0.5, [0, 0, 0]), ball(3.0, 0.5, [0.8, 0.1, 0])], [Sphere(0.5)] * 2, gravity=False)
    cw.run(2.0, 1e-2)
    s = cw.world.state
    assert np.linalg.norm(s.x[1] - s.x[0]) > 1.0 - 2e-4
    assert dg.kinetic_energy(cw.world) == 0.0
    np.testing.assert_allclose(1.0 * s.x[0] + 3.0 * s.x[1], [2.4, 0.3, 0], atol=1e-12)  # COM fixed


# ---------------------------------------------------------------- bouncing ball
def test_ball_bounce_heights_and_times():
    e, r, h, dt = 0.8, 0.1, 1.0, 1e-4
    cw = make([ball(1.0, r, [0, 0, r + h])], [Sphere(r)],
              [StaticCollider(Plane((0, 0, 1)), material=Material(e, 0.0))], [Material(e, 0.0)])
    zs, vs, ts = [], [], []
    cw.run(2.5, dt, lambda c: (zs.append(c.world.state.x[0, 2]), vs.append(c.world.state.v[0, 2]), ts.append(c.t)))
    zs, vs, ts = map(np.array, (zs, vs, ts))
    bounce = np.where((vs[1:] > 0) & (vs[:-1] <= 0))[0] + 1
    apex = np.where((vs[1:] <= 0) & (vs[:-1] > 0))[0] + 1
    t1, v1 = np.sqrt(2 * h / g), np.sqrt(2 * g * h)
    k = np.arange(len(bounce))
    t_exp = t1 + 2 * v1 / g * np.array([sum(e ** j for j in range(1, i + 1)) for i in k])
    h_exp = h * e ** (2 * (np.arange(len(apex)) + 1))
    assert len(bounce) >= 4 and len(apex) >= 4
    np.testing.assert_allclose(ts[bounce[:4]], t_exp[:4], rtol=1e-3)
    np.testing.assert_allclose(zs[apex[:4]] - r, h_exp[:4], rtol=5e-3)
    ratios = (zs[apex[1:4]] - r) / (zs[apex[0:3]] - r)
    np.testing.assert_allclose(ratios, e * e, rtol=3e-3)


def test_inelastic_ball_comes_to_rest_without_drift():
    cw = make([ball(1.0, 0.1, [0, 0, 0.5])], [Sphere(0.1)], [GROUND])
    cw.run(3.0, 1 / 240)
    s = cw.world.state
    assert s.x[0, 2] == pytest.approx(0.1, abs=2 * cw.slop)
    assert abs(s.v[0, 2]) < g / 240 + 1e-12


# ---------------------------------------------------------------- friction on an incline
def incline(theta, mu, T=1.0, dt=1 / 480):
    n = np.array([np.sin(theta), 0, np.cos(theta)])
    tdown = np.array([np.cos(theta), 0, -np.sin(theta)])
    hb = np.array([0.2, 0.2, 0.05])   # low box: no tipping
    q = quat.from_axis_angle([0, 1, 0], theta)
    box = RigidBody(1.0, Box(tuple(hb)).inertia(1.0), position=n * (hb[2] - 1e-6), orientation=q)
    cw = make([box], [Box(tuple(hb))], [StaticCollider(Plane(tuple(n)), material=Material(0, mu))],
              [Material(0, mu)])
    cw.run(T, dt)
    return cw.world.state, tdown


@pytest.mark.parametrize("theta,mu", [(np.radians(30), 0.3), (np.radians(40), 0.5), (np.radians(20), 0.1)])
def test_block_slides_with_coulomb_acceleration(theta, mu):
    T = 1.0
    s, tdown = incline(theta, mu, T)
    a = g * (np.sin(theta) - mu * np.cos(theta))
    assert s.v[0] @ tdown == pytest.approx(a * T, rel=2e-3)
    assert s.x[0] @ tdown == pytest.approx(0.5 * a * T * T, rel=1e-2)


@pytest.mark.parametrize("theta,mu", [(np.radians(30), 0.6), (np.radians(10), 0.2)])
def test_block_holds_when_tan_below_mu(theta, mu):
    assert np.tan(theta) < mu
    s1, tdown = incline(theta, mu, T=0.5, dt=1 / 240)
    s2, _ = incline(theta, mu, T=2.0, dt=1 / 240)
    assert abs(s1.x[0] @ tdown) < 1e-4            # settling transient only
    assert abs((s2.x[0] - s1.x[0]) @ tdown) < 1e-9  # no creep afterwards
    assert np.linalg.norm(s2.v[0]) < 1e-6


# ---------------------------------------------------------------- rolling
def test_sliding_sphere_transitions_to_rolling():
    """Sphere launched without spin: v(t) = v0 - mu g t until v = 5/7 v0 at t* = 2 v0 / (7 mu g)."""
    v0, mu, r, dt = 3.0, 0.3, 0.1, 1e-3
    cw = make([ball(1.0, r, [0, 0, r - 1e-6], [v0, 0, 0])], [Sphere(r)],
              [StaticCollider(Plane((0, 0, 1)), material=Material(0, mu))], [Material(0, mu)])
    tstar = 2 * v0 / (7 * mu * g)
    rec = cw.run(1.0, dt, lambda c: (c.t, c.world.state.v[0, 0], dg.body_angular_velocity(c.world)[0, 1]))
    t, v, w = map(np.array, zip(*rec))
    sl = t < tstar - 2 * dt
    np.testing.assert_allclose(v[sl], v0 - mu * g * t[sl], atol=1e-9)
    ro = t > tstar + 2 * dt
    np.testing.assert_allclose(v[ro], 5 / 7 * v0, rtol=1e-5)
    np.testing.assert_allclose(w[ro] * r, v[ro], rtol=1e-5)   # rolling (lever arm r - depth/2)


def test_sphere_rolls_down_incline_at_five_sevenths_g_sin():
    theta, r, T = np.radians(20), 0.1, 1.0
    n = np.array([np.sin(theta), 0, np.cos(theta)])
    tdown = np.array([np.cos(theta), 0, -np.sin(theta)])
    cw = make([ball(1.0, r, n * r)], [Sphere(r)], [StaticCollider(Plane(tuple(n)), material=Material(0, 1.0))],
              [Material(0, 1.0)])
    cw.run(T, 1e-3)
    assert cw.world.state.v[0] @ tdown == pytest.approx(5 / 7 * g * np.sin(theta) * T, rel=1e-3)


# ---------------------------------------------------------------- resting / stacking
def test_box_resting_on_plane_does_not_drift():
    hb = (0.5, 0.3, 0.2)
    cw = make([RigidBody(10.0, Box(hb).inertia(10.0), position=[0.0, 0.0, 0.2])], [Box(hb)], [GROUND])
    cw.run(5.0, 1 / 240)
    s = cw.world.state
    assert np.linalg.norm(s.x[0, :2]) < 1e-9
    assert s.x[0, 2] == pytest.approx(0.2, abs=2 * cw.slop)
    assert np.linalg.norm(quat.to_matrix(s.q[0]) - np.eye(3)) < 1e-7
    assert np.linalg.norm(s.v[0]) < g / 240 + 1e-9


@pytest.mark.slow
def test_three_box_stack_is_stable():
    hb = (0.5, 0.5, 0.5)
    bodies = [RigidBody(1.0, Box(hb).inertia(1.0), position=[0.0, 0.0, 0.5 + i]) for i in range(3)]
    cw = make(bodies, [Box(hb)] * 3, [GROUND], [Material(0, 0.6)] * 3, iterations=30)
    cw.run(5.0, 1 / 240)
    s = cw.world.state
    assert np.max(np.abs(s.x[:, :2])) < 1e-4
    np.testing.assert_allclose(s.x[:, 2], [0.5, 1.5, 2.5], atol=5e-3)


@pytest.mark.slow
def test_box_dropped_rotated_settles_flat():
    hb = (0.3, 0.3, 0.3)
    q = quat.from_axis_angle([1, 1, 0], 0.4)
    cw = make([RigidBody(1.0, Box(hb).inertia(1.0), position=[0, 0, 1.0], orientation=q)], [Box(hb)], [GROUND])
    cw.run(4.0, 1 / 240)
    s = cw.world.state
    assert s.x[0, 2] == pytest.approx(0.3, abs=2e-3)
    assert np.linalg.norm(s.v[0]) < 0.05 and dg.kinetic_energy(cw.world) < 1e-3


# ---------------------------------------------------------------- determinism and units
def test_contact_simulation_is_bit_deterministic():
    def run():
        rng = np.random.default_rng(1)
        bodies, shapes = [], []
        for i in range(6):
            bodies.append(RigidBody(1.0, Box((0.2, 0.2, 0.2)).inertia(1.0), position=rng.uniform([-1, -1, 0.5], [1, 1, 2]),
                                    orientation=quat.normalize(rng.normal(size=4))))
            shapes.append(Box((0.2, 0.2, 0.2)))
        cw = make(bodies, shapes, [GROUND], [Material(0.3, 0.5)] * 6)
        cw.run(0.5, 1 / 240)
        return cw.world.state.pack()
    a, b = run(), run()
    assert np.array_equal(a, b)


def test_contact_quantities_have_correct_dimensions():
    IMPULSE = u.MASS * u.VELOCITY
    assert IMPULSE == u.FORCE * u.TIME
    # angular impulse r x P changes angular momentum (kg m^2/s)
    assert u.LENGTH * IMPULSE == u.MASS * u.LENGTH**2 / u.TIME
    # effective mass 1/(1/m + n.(I^-1 (r x n)) x r): both terms are 1/mass
    assert (u.LENGTH**2 / (u.MASS * u.LENGTH**2)) == u.MASS ** -1
    # sliding acceleration g (sin - mu cos), bounce height e^2 h, bounce interval 2 e v/g
    assert u.ACCEL * u.DIMLESS == u.ACCEL and u.VELOCITY / u.ACCEL == u.TIME
