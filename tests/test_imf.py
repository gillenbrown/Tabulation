import pytest
from tabulation import IMF
import numpy as np
from scipy import integrate


def test_normalization():
    total_mass = np.random.uniform(10, 50)
    imf = IMF("Kroupa", 0.08, 50, total_mass)

    def integrand(m):
        return m * imf.normalized_dn_dm(m)

    total_mass_imf = integrate.quad(integrand, 0.08, 50)[0]
    assert pytest.approx(total_mass) == total_mass_imf


def test_imf_bounds_equal():
    with pytest.raises(ValueError):
        IMF("Kroupa", 1, 1)


def test_imf_bounds_reversed():
    with pytest.raises(ValueError):
        IMF("Kroupa", 2, 1)