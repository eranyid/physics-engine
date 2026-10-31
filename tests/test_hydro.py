"""Ship hydrostatics vs Archimedes / metacentric / period closed forms (SI)."""
import numpy as np
import pytest
from physics_engine import RigidBody, World, forces, integrators, hydro, quaternion as quat
from physics_engine.units import STANDARD_GRAVITY as g0
from physics_engine.body import inertia_solid_box


def test_box_floating_draft_archimedes():
    hull = hydro.BoxHull(length=20.0, beam=5.0, depth=3.0)
    rho, mass = 1000.0, 50_000.0  # kg → draft = m/(ρ L B) = 0.5 m
    T = hull.upright_draft_for_mass(mass, rho)
    assert T == pytest.approx(0.5, rel=1e-15)
    # Geometric centre so keel is at z = -T when water at 0: z_c = -T + D/2
    z_c = -T + 0.5 * hull.depth
    sub = hull.submerged(np.array([0.0, 0.0, z_c]), quat.identity(), 0.0)
    assert sub.volume == pytest.approx(mass / rho, rel=1e-12)
    assert sub.A_wp == pytest.approx(hull.length * hull.beam)
    assert sub.cob_world[2] == pytest.approx(-T + 0.5 * T, rel=1e-12)  # KB = T/2 from keel


def test_cylinder_floating_draft_archimedes():
    hull = hydro.CylinderHull(radius=2.0, height=4.0)
    rho, mass = 1000.0, 8000.0
    T = hull.upright_draft_for_mass(mass, rho)
    assert T == pytest.approx(mass / (rho * np.pi * 4.0), rel=1e-15)
    z_c = -T + 0.5 * hull.height
    sub = hull.submerged(np.array([0.0, 0.0, z_c]), quat.identity(), 0.0)
    assert sub.volume == pytest.approx(mass / rho, rel=1e-12)
    assert sub.A_wp == pytest.approx(np.pi * hull.radius ** 2)


def test_buoyancy_force_equals_weight_at_equilibrium_draft():
    hull = hydro.BoxHull(10.0, 4.0, 2.0)
    rho, mass = hydro.RHO_FRESHWATER, 20_000.0
    T = hull.upright_draft_for_mass(mass, rho)
    z_c = -T + 0.5 * hull.depth
    body = RigidBody(mass=mass, inertia_body=inertia_solid_box(mass, 10, 4, 2),
                     position=[0, 0, z_c])
    w = World([body], [
        forces.UniformGravity(g=(0, 0, -g0)),
        hydro.Buoyancy(0, hull, rho=rho, water_level=0.0),
    ])
    F, _ = w.forces(w.state, 0.0)
    np.testing.assert_allclose(F[0], [0, 0, 0], atol=1e-8)


def test_rectangular_barge_metacentric_height():
    """GM = KB + BM − KG with BM = I/V = B²/(12 T)."""
    hull = hydro.BoxHull(length=30.0, beam=8.0, depth=4.0)
    draft, KG = 2.0, 1.5
    KB = hull.upright_KB(draft)
    BM = hull.upright_BM_roll(draft)
    GM = hull.upright_GM(draft, KG)
    V = hull.length * hull.beam * draft
    I = hull.length * hull.beam ** 3 / 12.0
    assert KB == pytest.approx(1.0)
    assert BM == pytest.approx(I / V, rel=1e-15)
    assert BM == pytest.approx(hull.beam ** 2 / (12.0 * draft), rel=1e-15)
    assert GM == pytest.approx(hydro.metacentric_height(KB, BM, KG), rel=1e-15)
    assert GM == pytest.approx(KB + BM - KG, rel=1e-15)
    assert GM > 0


def test_righting_arm_small_angle_GZ_equals_GM_sin_phi():
    hull = hydro.BoxHull(30.0, 8.0, 4.0)
    draft, KG = 2.0, 1.5
    GM = hull.upright_GM(draft, KG)
    for phi in (0.01, 0.05, 0.1, np.deg2rad(5.0)):
        gz_formula = hydro.righting_arm(GM, phi)
        assert hull.righting_arm_small(GM, phi) == pytest.approx(GM * np.sin(phi))
        gz_num = hydro.box_righting_arm_curve(hull, draft, KG, [phi])[0]
        assert gz_num == pytest.approx(gz_formula, rel=2e-2, abs=1e-3)


def test_righting_moment_from_GZ():
    rho, V, GZ = 1025.0, 100.0, 0.2
    M = hydro.righting_moment(rho, V, GZ)
    assert M == pytest.approx(rho * g0 * V * GZ)


def test_capsize_beyond_angle_of_vanishing_stability():
    """Low-freeboard barge: GZ > 0 below the AVS and GZ < 0 past it (capsize)."""
    hull = hydro.BoxHull(length=20.0, beam=8.0, depth=2.5)
    draft, KG = 2.0, 1.8
    GM = hull.upright_GM(draft, KG)
    assert GM > 0
    avs = hydro.angle_of_vanishing_stability(hull, draft, KG, phi_max=np.deg2rad(89.0), n=180)
    assert np.deg2rad(50.0) < avs < np.deg2rad(85.0)
    gz_before = hydro.box_righting_arm_curve(hull, draft, KG, [0.5 * avs])[0]
    gz_after = hydro.box_righting_arm_curve(hull, draft, KG, [avs + np.deg2rad(5.0)])[0]
    assert gz_before > 0
    assert gz_after < 0


def test_heave_period_formula_and_simulation():
    hull = hydro.BoxHull(12.0, 4.0, 2.0)
    rho, mass = 1000.0, 24_000.0
    T_draft = hull.upright_draft_for_mass(mass, rho)
    A_wp = hull.length * hull.beam
    T_heave = hydro.heave_period(mass, rho, A_wp)
    assert T_heave == pytest.approx(2 * np.pi * np.sqrt(mass / (rho * g0 * A_wp)), rel=1e-15)

    z_eq = -T_draft + 0.5 * hull.depth
    body = RigidBody(mass=mass, inertia_body=inertia_solid_box(mass, 12, 4, 2),
                     position=[0, 0, z_eq + 0.05])  # 5 cm heave offset
    w = World([body], [
        forces.UniformGravity(),
        hydro.Buoyancy(0, hull, rho=rho),
    ])
    # Sample z(t); estimate period from successive upward zero-crossing of (z - z_eq).
    dt, t_end = 1e-3, T_heave * 3.5
    zs, ts = [], []
    while w.t < t_end:
        w.step(dt, integrators.rk4)
        zs.append(w.state.x[0, 2] - z_eq)
        ts.append(w.t)
    zs, ts = np.asarray(zs), np.asarray(ts)
    # Find zero crossings with positive slope.
    crosses = []
    for i in range(1, len(zs)):
        if zs[i - 1] <= 0.0 < zs[i]:
            crosses.append(ts[i])
    assert len(crosses) >= 2
    T_meas = crosses[1] - crosses[0]
    assert T_meas == pytest.approx(T_heave, rel=2e-2)


def test_roll_period_formula():
    hull = hydro.BoxHull(30.0, 8.0, 4.0)
    draft, KG = 2.0, 1.5
    GM = hull.upright_GM(draft, KG)
    # Radius of gyration about x for a solid box: k^2 = (B^2 + D^2)/12
    k = np.sqrt((hull.beam ** 2 + hull.depth ** 2) / 12.0)
    T = hydro.roll_period(k, GM)
    assert T == pytest.approx(2 * np.pi * k / np.sqrt(g0 * GM), rel=1e-15)
    assert T > 0


@pytest.mark.slow
def test_roll_oscillation_period_simulation():
    hull = hydro.BoxHull(20.0, 6.0, 3.0)
    draft, KG = 1.2, 1.0
    GM = hull.upright_GM(draft, KG)
    assert GM > 0
    mass = hydro.RHO_FRESHWATER * hull.length * hull.beam * draft
    # Inertia about COM; place COM at KG from keel ⇒ geometric offset.
    com_body = np.array([0.0, 0.0, -0.5 * hull.depth + KG])
    I_xx = mass * ((hull.beam ** 2 + hull.depth ** 2) / 12.0)
    I = np.diag([I_xx, I_xx * 1.5, I_xx * 1.2])
    k = np.sqrt(I_xx / mass)
    T_roll = hydro.roll_period(k, GM)
    # Small heel initial condition.
    phi0 = np.deg2rad(3.0)
    q0 = quat.from_axis_angle([1.0, 0.0, 0.0], phi0)
    z_com = -draft + KG
    body = RigidBody(mass=mass, inertia_body=I, position=[0, 0, z_com], orientation=q0)
    w = World([body], [
        forces.UniformGravity(),
        hydro.Buoyancy(0, hull, rho=hydro.RHO_FRESHWATER, com_body=com_body),
    ])
    dt, t_end = 5e-3, T_roll * 3.5
    phis, ts = [], []
    while w.t < t_end:
        w.step(dt, integrators.rk4)
        # Extract roll from quaternion (rotation about x).
        R = quat.to_matrix(w.state.q[0])
        phi = float(np.arctan2(R[2, 1], R[2, 2]))  # heel of body z in yz
        # Better: body y's world z component.
        phi = float(np.arcsin(np.clip(R[2, 1], -1, 1)))
        phis.append(phi)
        ts.append(w.t)
    phis, ts = np.asarray(phis), np.asarray(ts)
    crosses = [ts[i] for i in range(1, len(phis)) if phis[i - 1] <= 0.0 < phis[i]]
    assert len(crosses) >= 2
    T_meas = crosses[1] - crosses[0]
    # Soft restoring (large-angle / damping-free) — allow ~8 % for discrete hydrostatics.
    assert T_meas == pytest.approx(T_roll, rel=0.08)


def test_convex_mesh_box_matches_analytic_upright():
    hull = hydro.BoxHull(4.0, 2.0, 2.0)
    # Build equivalent mesh and compare submerged volume at a known pose.
    hx, hy, hz = 2.0, 1.0, 1.0
    verts = np.array([
        [-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, -hz], [-hx, hy, hz],
        [hx, -hy, -hz], [hx, -hy, hz], [hx, hy, -hz], [hx, hy, hz],
    ])
    faces = [
        [0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5],
    ]
    mesh = hydro.ConvexMeshHull(verts, faces)
    x = np.array([0.0, 0.0, 0.0])  # half submerged
    # Force mesh path by using a tiny heel so BoxHull analytic path isn't required for mesh.
    q = quat.from_axis_angle([1, 0, 0], 1e-4)
    sub_m = mesh.submerged(x, q, 0.0)
    sub_b = hull.submerged(x, quat.identity(), 0.0)
    assert sub_m.volume == pytest.approx(sub_b.volume, rel=1e-3)
    assert sub_m.cob_world[2] == pytest.approx(sub_b.cob_world[2], abs=2e-3)


def test_hull_drag_opposes_surge():
    body = RigidBody(mass=1000.0, inertia_body=np.eye(3) * 100.0, velocity=[5.0, 0, 0])
    drag = hydro.HullDrag(0, S=20.0, L=10.0, C_f=0.004, C_w=0.002, rho=1000.0)
    w = World([body], [drag])
    F, _ = w.forces(w.state, 0.0)
    V = 5.0
    R = 0.5 * 1000.0 * (0.004 + 0.002) * 20.0 * V * V
    np.testing.assert_allclose(F[0], [-R, 0, 0], rtol=1e-12)


def test_added_mass_helpers_positive():
    assert hydro.box_added_mass_heave(1000.0, 40.0, 1.0) > 0
    assert hydro.box_added_mass_surge(1000.0, 40.0) == pytest.approx(0.5 * 1000.0 * 40.0)


def test_dimension_checks_hydro_aero():
    from physics_engine import units as u
    # Heave period ~ time
    assert (u.MASS / (u.MASS / u.LENGTH**3 * u.ACCEL * u.LENGTH**2)) ** 0.5 == u.TIME
    # Dynamic pressure * area = force
    assert (u.MASS / u.LENGTH**3) * (u.VELOCITY ** 2) * u.LENGTH**2 == u.FORCE
    # BM = I/V ~ length
    assert (u.LENGTH**4) / (u.LENGTH**3) == u.LENGTH
