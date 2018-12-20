import pytest
from tabulation import SNIa, Lifetimes, IMF
from scipy import integrate


imf = IMF("Kroupa", 0.1, 50)
lifetimes = Lifetimes("Raiteri_96")
number_sn_ia = 1.6E-3


@pytest.fixture
def sn_ia_old():
    return SNIa("old ART", "Nomoto_18", lifetimes, imf,
                exploding_fraction=0.015, min_mass=3, max_mass=8)


@pytest.fixture
def sn_ia_new():
    return SNIa("ART power law", "Nomoto_18", lifetimes, imf,
                number_sn_ia=number_sn_ia)


# hack stolen from github to use parametrize on fixtures
# https://github.com/pytest-dev/pytest/issues/349#issuecomment-189370273
@pytest.fixture(params=['sn_ia_old', 'sn_ia_new'])
def both_sn(request):
    return request.getfuncargvalue(request.param)


def test_number_sn_old(sn_ia_old):
    assert number_sn_ia / 10 < sn_ia_old.number_sn_Ia < number_sn_ia


def test_number_sn_new(sn_ia_new):
    assert sn_ia_new.number_sn_Ia == number_sn_ia


def test_normalization_old(sn_ia_old):
    # Want to integrate to infinity, but scipy runs into trouble doing that,
    # so I'll just do a very long time with larger tolerance.
    integral = integrate.quad(sn_ia_old.old_art_phi_per_dt, 0, 10 ** 13)[0]
    assert pytest.approx(1, rel=0.05) == integral


def test_normalization_new(sn_ia_new):
    integral = integrate.quad(sn_ia_new.sn_dtd, 0, 13.79E9, args=(0.02))[0]
    assert pytest.approx(number_sn_ia, rel=0.05) == integral


def test_sn_ia_rate_turn_on(sn_ia_new):
    z = 0.01
    age_8 = lifetimes.lifetime(8.0, z)
    assert sn_ia_new.sn_dtd(0, z) == 0
    assert sn_ia_new.sn_dtd(age_8 / 2, z) == 0
    assert sn_ia_new.sn_dtd(age_8 - 1, z) == 0
    assert sn_ia_new.sn_dtd(age_8 + 1, z) != 0


def test_sn_ia_rate_new_late_times_no_z_change(sn_ia_new):
    age = 1E9
    rate_1 = sn_ia_new.sn_dtd(age, 0)
    rate_2 = sn_ia_new.sn_dtd(age, 0.0001)
    rate_3 = sn_ia_new.sn_dtd(age, 0.001)
    rate_4 = sn_ia_new.sn_dtd(age, 0.02)
    assert rate_1 == rate_2 == rate_3 == rate_4


def test_sn_ia_rate_new_late_times(sn_ia_new):
    # plotted this and guessed at values
    z = 0.02
    assert 1E-12 < sn_ia_new.sn_dtd(1E8, z) < 1E11
    assert 1E-13 < sn_ia_new.sn_dtd(1E9, z) < 1E12
    assert 1E-14 < sn_ia_new.sn_dtd(1E9, z) < 1E13


def test_ejected_mass_error_checking(both_sn):
    for z in [0.0001, 0.001, 0.01, 0.03]:
        with pytest.raises(ValueError):
            both_sn.ejected_mass("C", z)


def test_ejected_mass_correct(both_sn):
    assert both_sn.ejected_mass("C", 0.02) == 4.75E-2 + 5.17E-8
    assert both_sn.ejected_mass("N", 0.02) == 1.1E-5 + 5.46E-8
    assert both_sn.ejected_mass("O", 0.02) == 5.0E-2 + 4.6E-6 + 1.43E-7
    assert both_sn.ejected_mass("Fe", 0.02) == sum([0.131, 0.741, 2.7E-2,
                                                    6.24E-4, 1.21E-8])

    assert both_sn.ejected_mass("C", 0.002) == 6.67E-2 + 1.28E-12
    assert both_sn.ejected_mass("N", 0.002) == 7.83E-10 + 1.32E-8
    assert both_sn.ejected_mass("O", 0.002) == 9.95E-2 + 1.32E-11 + 7.60E-13
    assert both_sn.ejected_mass("Fe", 0.002) == sum([0.18, 0.683, 1.85E-2,
                                                     5.64E-4, 1.10E-8])


def test_ejected_mass_elts_not_present(both_sn):
    assert both_sn.ejected_mass("H", 0.02) == 0
    assert both_sn.ejected_mass("He", 0.02) == 0

    assert both_sn.ejected_mass("H", 0.002) == 0
    assert both_sn.ejected_mass("He", 0.002) == 0


def test_ejected_mass_total_metals(both_sn):
    assert 1.2 < both_sn.ejected_mass("total_metals", 0.02) < 1.4
    assert 1.2 < both_sn.ejected_mass("total_metals", 0.002) < 1.4
