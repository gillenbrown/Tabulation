import pytest
from pytest import approx
from tabulation import SSPYields


@pytest.fixture
def ssp():
    return SSPYields("Kroupa", 1, 50, 1,
                     "Raiteri_96",
                     "Kobayashi_06", 0.5, 8, 50,
                     "ART", "Nomoto_18",
                     "NuGrid", 1, 8)


# ----------------------------------------------------------

# winds from SSP

# ----------------------------------------------------------
def test_wind_constant_early_times(ssp):
    # at early times when no stars have died yet, the mass loss rate should
    # be constant
    metallicity = 0.004
    time_early = 0
    time_late = ssp.lifetimes.lifetime(55, metallicity)

    wind_early = ssp.mass_loss_rate_winds(time_early, metallicity)
    wind_late = ssp.mass_loss_rate_winds(time_late, metallicity)

    assert approx(wind_early) == wind_late


def test_wind_decay(ssp):
    # Once stars start dying winds should go down
    metallicity = 0.004
    time_early = ssp.lifetimes.lifetime(55, metallicity)
    time_late = ssp.lifetimes.lifetime(20, metallicity)

    wind_early = ssp.mass_loss_rate_winds(time_early, metallicity)
    wind_late = ssp.mass_loss_rate_winds(time_late, metallicity)

    assert wind_early > wind_late


def test_wind_all_done(ssp):
    # Make sure winds are over once the last star explodes as SN
    metallicity = 0.004
    time = ssp.lifetimes.lifetime(7.99, metallicity)

    assert 0 == ssp.mass_loss_rate_winds(time, metallicity)
