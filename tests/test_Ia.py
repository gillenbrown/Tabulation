import pytest
from tabulation import SNIa
from scipy import integrate


def test_normalization():
    sn_rate = SNIa("ART", "Nomoto_18")
    # Want to integrate to infinity, but scipy runs into trouble doing that,
    # so I'll just do a very long time with larger tolerance.
    integral = integrate.quad(sn_rate.sn_rate, 0, 10**13)[0]
    assert pytest.approx(1, rel=0.05) == integral
