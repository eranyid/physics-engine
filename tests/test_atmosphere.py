"""ISA atmosphere vs published ICAO / ISO table values (SI)."""
import pytest
from physics_engine import atmosphere as atm


# Published ISA table (Torenbeek / ICAO Doc 7488), geopotential altitude.
# Columns: h [m], T [K], p [Pa], rho [kg/m^3], a [m/s]
ISA_TABLE = [
    (0,     288.15, 101_325, 1.2250, 340.29),
    (1000,  281.65,  89_874, 1.1117, 336.43),
    (5000,  255.65,  54_020, 0.7361, 320.53),
    (11_000, 216.65, 22_632, 0.3639, 295.07),
    (20_000, 216.65,   5475, 0.0880, 295.07),
]


@pytest.mark.parametrize("h,T,p,rho,a", ISA_TABLE)
def test_isa_table_values(h, T, p, rho, a):
    assert atm.temperature(h) == pytest.approx(T, abs=0.01)
    assert atm.pressure(h) == pytest.approx(p, rel=2e-4, abs=2.0)
    assert atm.density(h) == pytest.approx(rho, rel=2e-4, abs=2e-4)
    assert atm.speed_of_sound(h) == pytest.approx(a, rel=2e-4, abs=0.05)


def test_isa_sea_level_and_ideal_gas():
    assert atm.T0 == 288.15
    assert atm.P0 == 101_325.0
    assert atm.RHO0 == 1.225
    assert atm.density(0.0) == pytest.approx(atm.P0 / (atm.R_AIR * atm.T0), rel=1e-15)
    assert atm.R_AIR == pytest.approx(287.05287, rel=1e-6)


def test_isa_tropopause_continuity():
    h = atm.H_TROPOPAUSE
    # Approach from below and evaluate at boundary — single-valued.
    assert atm.temperature(h - 1e-6) == pytest.approx(atm.temperature(h), abs=1e-6)
    assert atm.pressure(h - 1e-3) == pytest.approx(atm.pressure(h), rel=1e-6)


def test_isa_rejects_above_model():
    with pytest.raises(ValueError):
        atm.temperature(20_001.0)


def test_state_dict_keys():
    s = atm.state(0.0)
    assert set(s) == {"temperature", "pressure", "density", "speed_of_sound"}
