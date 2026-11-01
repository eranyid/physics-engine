"""Python tests for Cummins retardation memory (physics_engine_rs)."""

import math

import numpy as np
import pytest

pe = pytest.importorskip("physics_engine_rs")


def test_cummins_hash_locked():
    a = pe.canonical_cummins_hash()
    b = pe.canonical_cummins_hash()
    assert a == b
    assert a == 0x1EE2F57B9C89C3AE


def test_cummins_k_b_consistency_bound():
    rel = pe.cummins_k_b_consistency(40.0, 8.0, 2.0)
    # Measured mid-band relative L2 ≈ 0.0095; tol 8% locked with Rust oracle.
    assert rel < 0.08


def test_cummins_free_decay_dissipates():
    length, beam, draft = 30.0, 6.0, 1.5
    rho = pe.RHO_FRESHWATER
    mass = rho * length * beam * draft
    z = pe.cummins_free_decay(length, beam, draft, mass, z0=0.2, dt=0.05, n_steps=400)
    z = np.asarray(z)
    assert z[0] == pytest.approx(0.2)
    assert abs(z[-1]) < abs(z[0])


def test_cummins_kernel_positive_at_zero():
    dt, k, a_inf, c = pe.cummins_box_kernel(40.0, 8.0, 2.0, dt=0.05, n_time=100)
    assert dt == pytest.approx(0.05)
    assert k[0] > 0.0
    assert a_inf > 0.0
    assert c > 0.0
