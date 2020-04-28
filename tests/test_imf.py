import pytest
from tabulation import IMF
import numpy as np
from scipy import integrate

n_random = 10

@pytest.mark.parametrize("total_mass",10**np.random.uniform(3, 6, n_random))
def test_normalization(total_mass):
    imf = IMF("Kroupa", 0.08, 50, total_mass)

    total_mass_imf = integrate.quad(imf.normalized_m_dn_dm, 0.08, 50)[0]
    assert pytest.approx(total_mass) == total_mass_imf


@pytest.mark.parametrize("total_mass",10**np.random.uniform(3, 6, n_random))
def test_normalization_changes(total_mass):
    imf = IMF("Kroupa", 0.08, 50, total_mass)
    total_mass_imf = integrate.quad(imf.normalized_m_dn_dm, 0.08, 50)[0]
    assert pytest.approx(total_mass) == total_mass_imf


@pytest.mark.parametrize("total_mass",10**np.random.uniform(3, 6, n_random))
def test_normalization_adjustment(total_mass):
    imf = IMF("Kroupa", 0.08, 50, 1)

    imf.normalize(total_mass)

    total_mass_imf = integrate.quad(imf.normalized_m_dn_dm, 0.08, 50)[0]
    assert pytest.approx(total_mass) == total_mass_imf


@pytest.mark.parametrize("total_mass",10**np.random.uniform(3, 6, n_random))
def test_normalization_scaling(total_mass):
    imf = IMF("Kroupa", 0.08, 50, 1)

    total_mass_1 = total_mass * integrate.quad(imf.normalized_m_dn_dm, 0.08, 50)[0]

    imf.normalize(total_mass)

    total_mass_2 = integrate.quad(imf.normalized_m_dn_dm, 0.08, 50)[0]
    assert pytest.approx(total_mass_1) == total_mass_2
    assert pytest.approx(total_mass) == total_mass_1


@pytest.mark.parametrize("total_mass", 10 ** np.random.uniform(3, 6, n_random))
def test_normalization_scaling(total_mass):
    imf = IMF("Kroupa", 0.08, 50, 1)

    total_mass_1 = total_mass * \
                   integrate.quad(imf.normalized_m_dn_dm, 0.08, 50)[0]

    imf.normalize(total_mass)

    total_mass_2 = integrate.quad(imf.normalized_m_dn_dm, 0.08, 50)[0]
    assert pytest.approx(total_mass_1) == total_mass_2
    assert pytest.approx(total_mass) == total_mass_1


@pytest.mark.parametrize("total_mass", 10 ** np.random.uniform(3, 6, n_random))
@pytest.mark.parametrize("m1", np.random.uniform(0.08, 50, n_random))
def test_normalization_number_stars(total_mass, m1):
    imf = IMF("Kroupa", 0.08, 50, 1)

    m2 = np.random.uniform(m1, 50)

    n1 = total_mass * integrate.quad(imf.normalized_dn_dm, m1, m2)[0]

    imf.normalize(total_mass)

    n2 = integrate.quad(imf.normalized_dn_dm, m1, m2)[0]

    assert pytest.approx(n1) == n2


def test_power_law_slope_high_end():
    imf = IMF("Kroupa", 0.08, 50, 1)
    phi_10 = imf.normalized_dn_dm(10)
    phi_20 = imf.normalized_dn_dm(20)

    assert phi_10 / phi_20 == pytest.approx((10 / 20)**-2.3)


def test_power_law_slope_low_end():
    imf = IMF("Kroupa", 0.08, 50, 1)
    phi_01 = imf.normalized_dn_dm(0.1)
    phi_02 = imf.normalized_dn_dm(0.2)

    assert phi_01 / phi_02 == pytest.approx((0.1 / 0.2)**-1.3)


def test_imf_bounds_equal():
    with pytest.raises(ValueError):
        IMF("Kroupa", 1, 1)


def test_imf_bounds_reversed():
    with pytest.raises(ValueError):
        IMF("Kroupa", 2, 1)