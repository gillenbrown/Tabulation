import pytest
from tabulation import SNIa
from scipy import integrate


@pytest.fixture
def sn_ia():
    return SNIa("ART", "Nomoto_18", 0.015, 3, 8)


def test_normalization(sn_ia):
    # Want to integrate to infinity, but scipy runs into trouble doing that,
    # so I'll just do a very long time with larger tolerance.
    integral = integrate.quad(sn_ia.sn_dtd, 0, 10 ** 13)[0]
    assert pytest.approx(1, rel=0.05) == integral


def test_ejected_mass_error_checking(sn_ia):
    for z in [0.0001, 0.001, 0.01, 0.03]:
        with pytest.raises(ValueError):
            sn_ia.ejected_mass("C", z)


def test_ejected_mass_correct(sn_ia):
    assert sn_ia.ejected_mass("C", 0.02) == 4.75E-2 + 5.17E-8
    assert sn_ia.ejected_mass("N", 0.02) == 1.1E-5 + 5.46E-8
    assert sn_ia.ejected_mass("O", 0.02) == 5.0E-2 + 4.6E-6 + 1.43E-7
    assert sn_ia.ejected_mass("Fe", 0.02) == sum([0.131, 0.741, 2.7E-2,
                                                  6.24E-4, 1.21E-8])

    assert sn_ia.ejected_mass("C", 0.002) == 6.67E-2 + 1.28E-12
    assert sn_ia.ejected_mass("N", 0.002) == 7.83E-10 + 1.32E-8
    assert sn_ia.ejected_mass("O", 0.002) == 9.95E-2 + 1.32E-11 + 7.60E-13
    assert sn_ia.ejected_mass("Fe", 0.002) == sum([0.18, 0.683, 1.85E-2,
                                                   5.64E-4, 1.10E-8])


def test_ejected_mass_elts_not_present(sn_ia):
    assert sn_ia.ejected_mass("H", 0.02) == 0
    assert sn_ia.ejected_mass("He", 0.02) == 0

    assert sn_ia.ejected_mass("H", 0.002) == 0
    assert sn_ia.ejected_mass("He", 0.002) == 0


def test_ejected_mass_total_metals(sn_ia):
    assert 1.2 < sn_ia.ejected_mass("total_metals", 0.02) < 1.4
    assert 1.2 < sn_ia.ejected_mass("total_metals", 0.002) < 1.4
