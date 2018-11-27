import pytest
from pytest import approx
from scipy import integrate

from tabulation import SSPYields


@pytest.fixture
def ssp():
    return SSPYields("Kroupa", 1, 50, 1,
                     "Raiteri_96",
                     "Kobayashi_06", 0.5, 8, 50,
                     "ART", "Nomoto_18",
                     "NuGrid", 1, 8)


ssp_obj = SSPYields("Kroupa", 1, 50, 1,
                    "Raiteri_96",
                    "Kobayashi_06", 0.5, 8, 50,
                    "ART", "Nomoto_18",
                    "NuGrid", 1, 8)


@pytest.mark.parametrize("m_lo,m_hi", [[10, 45], [23, 45], [10, 27], [17, 32]])
def test_integral_easy(m_lo, m_hi):
    # see if the integral matches the full integral for an easy example
    def linear(x):
        return x

    real_integral = 0.5 * (m_hi**2 - m_lo**2)  # analytic solution
    test_integral = ssp_obj._integrate_mass_smart(linear, m_lo, m_hi, "massive")

    assert approx(real_integral) == test_integral

@pytest.mark.parametrize("m_lo,m_hi", [[10, 45], [23, 45], [10, 27], [17, 32]])
def test_integral_actual(m_lo, m_hi):
    def ejecta(m):
        return ssp_obj.sn_ii_model.elemental_ejecta_mass(m, 0.02, "total")

    real_integral = integrate.quad(ejecta, m_lo, m_hi)[0]
    test_integral = ssp_obj._integrate_mass_smart(ejecta, m_lo, m_hi, "massive")

    assert approx(real_integral) == test_integral


@pytest.mark.parametrize("m_lo,m_hi", [[10, 45], [23, 45], [10, 27], [17, 32]])
def test_integral_tricky(m_lo, m_hi):
    def ejecta(m):
        return ssp_obj.sn_ii_model.elemental_ejecta_mass(m, 0.02, "C")

    real_integral_result = integrate.quad(ejecta, m_lo, m_hi)
    real_integral = real_integral_result[0]
    error = real_integral_result[1]
    test_integral = ssp_obj._integrate_mass_smart(ejecta, m_lo, m_hi, "massive")

    # don't be too stringent, since the real integrand might actually be
    # less accurate
    assert approx(real_integral, abs=2 * error) == test_integral


# ----------------------------------------------------------

# winds from SSP

# ----------------------------------------------------------
def test_wind_constant_early_times():
    # at early times when no stars have died yet, the mass loss rate should
    # be constant
    metallicity = 0.004
    time_early = 0
    time_late = ssp_obj.lifetimes.lifetime(55, metallicity)

    wind_early = ssp_obj.mass_loss_rate_winds(time_early, metallicity)
    wind_late = ssp_obj.mass_loss_rate_winds(time_late, metallicity)

    assert approx(wind_early) == wind_late


def test_wind_decay():
    # Once stars start dying winds should go down
    metallicity = 0.004
    time_early = ssp_obj.lifetimes.lifetime(55, metallicity)
    time_late = ssp_obj.lifetimes.lifetime(20, metallicity)

    wind_early = ssp_obj.mass_loss_rate_winds(time_early, metallicity)
    wind_late = ssp_obj.mass_loss_rate_winds(time_late, metallicity)

    assert wind_early > wind_late


def test_wind_all_done():
    # Make sure winds are over once the last star explodes as SN
    metallicity = 0.004
    time = ssp_obj.lifetimes.lifetime(7.99, metallicity)

    assert 0 == ssp_obj.mass_loss_rate_winds(time, metallicity)
