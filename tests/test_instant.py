import pytest
from pytest import approx
from tabulation import SNII, SNOverall, AGB
import numpy as np
import yields

sn_ii_kobayashi_obj = SNII("kobayashi_06_sn", 8, 50)
hn_ii_kobayashi_obj = SNII("kobayashi_06_hn", 8, 50)
agb_nugrid_obj = AGB("nugrid", 0, 8)


@pytest.fixture
def sn_ii_kobayashi():
    return SNII("kobayashi_06_sn", 8, 50)


@pytest.fixture
def hn_ii_kobayashi():
    return SNII("kobayashi_06_hn", 8, 50)


@pytest.fixture
def agb_nugrid():
    return AGB("nugrid", 0, 8)


@pytest.fixture
def sn_overall():
    return SNOverall("kobayashi_06", 0.45, 8, 50)


yields_models = {"sn": dict(), "hn": dict(), "agb": dict()}
yields_models["sn"][13] = yields.Yields("kobayashi_06_II_13")
yields_models["sn"][18] = yields.Yields("kobayashi_06_II_18")
yields_models["sn"][20] = yields.Yields("kobayashi_06_II_20")
yields_models["sn"][25] = yields.Yields("kobayashi_06_II_25")
yields_models["sn"][30] = yields.Yields("kobayashi_06_II_30")
yields_models["sn"][40] = yields.Yields("kobayashi_06_II_40")
yields_models["hn"][20] = yields.Yields("kobayashi_06_II_20_hn")
yields_models["hn"][25] = yields.Yields("kobayashi_06_II_25_hn")
yields_models["hn"][30] = yields.Yields("kobayashi_06_II_30_hn")
yields_models["hn"][40] = yields.Yields("kobayashi_06_II_40_hn")
yields_models["agb"][1] = yields.Yields("nugrid_agb_1")
yields_models["agb"][5] = yields.Yields("nugrid_agb_5")
yields_models["agb"][6] = yields.Yields("nugrid_agb_6")
yields_models["agb"][7] = yields.Yields("nugrid_agb_7")


def get_if_between(a, b, test_val):
    """Returns True is test_val is between a and b"""
    if a < b:
        return a <= test_val <= b
    else:
        return b <= test_val <= a


def get_closer(a, b, test_val):
    """Returns which ever of a or b is closer to test_val"""
    diff_a = abs(a - test_val)
    diff_b = abs(b - test_val)
    if diff_a < diff_b:
        return a
    else:
        return b


# ----------------------------------------------------------

# error checking

# ----------------------------------------------------------
def test_min_max_limits_equal_sn():
    with pytest.raises(ValueError):
        SNII("Kobayashi_06_sn", 15, 15)


def test_min_max_limits_reversed_sn():
    with pytest.raises(ValueError):
        SNII("Kobayashi_06_sn", 20, 10)


def test_min_max_limits_equal_hn():
    with pytest.raises(ValueError):
        SNII("Kobayashi_06_hn", 15, 15)


def test_min_max_limits_reversed_hn():
    with pytest.raises(ValueError):
        SNII("Kobayashi_06_hn", 20, 10)


def test_min_max_limits_equal_agb():
    with pytest.raises(ValueError):
        AGB("NuGrid", 15, 15)


def test_min_max_limits_reversed_agb():
    with pytest.raises(ValueError):
        AGB("NuGrid", 20, 10)


# ----------------------------------------------------------

# Testing individual models

# ----------------------------------------------------------
def test_init_sn_model(sn_ii_kobayashi):
    assert sn_ii_kobayashi.masses == [13, 15, 18, 20, 25, 30, 40]
    assert sn_ii_kobayashi.metallicities == [0, 0.001, 0.004, 0.02]


def test_init_hn_model(hn_ii_kobayashi):
    assert hn_ii_kobayashi.masses == [20, 25, 30, 40]
    assert hn_ii_kobayashi.metallicities == [0, 0.001, 0.004, 0.02]


def test_init_agb_model(agb_nugrid):
    assert agb_nugrid.masses == [1, 1.65, 2, 3, 4, 5, 6, 7]
    assert agb_nugrid.metallicities == [0.0001, 0.001, 0.006, 0.01, 0.02]


@pytest.mark.parametrize("my_model,yields_models,mass",
                         [[sn_ii_kobayashi_obj, yields_models["sn"], 25],
                          [sn_ii_kobayashi_obj, yields_models["sn"], 30],
                          [hn_ii_kobayashi_obj, yields_models["hn"], 25],
                          [hn_ii_kobayashi_obj, yields_models["hn"], 30],
                          [agb_nugrid_obj, yields_models["agb"], 5],
                          [agb_nugrid_obj, yields_models["agb"], 6]])
def test_ejected_masses_exact(my_model, yields_models, mass):
    """Test the ejected mass when they align with real SN models"""
    z = 0.001
    real_ejecta = yields_models[mass].total_end_ejecta[z]
    my_ejecta = my_model.ejecta(mass, z)

    assert approx(real_ejecta) == my_ejecta


def test_ejected_masses_sn_below_8(sn_ii_kobayashi):
    for z in sn_ii_kobayashi.metallicities:
        assert 0 == sn_ii_kobayashi.ejecta(7.99, z)


def test_ejected_masses_hn_below_20(hn_ii_kobayashi):
    for z in hn_ii_kobayashi.metallicities:
        assert 0 == hn_ii_kobayashi.ejecta(19.99, z)


def test_ejected_masses_sn_between_8_13(sn_ii_kobayashi):
    rand_mass = np.random.uniform(8, 13, 1)

    # the ejecta masses should be the fraction of ejected mass at the 8 solar
    # mass model times the mass
    y_m = yields_models["sn"][13]
    for z in sn_ii_kobayashi.metallicities:
        frac_ejected = y_m.total_end_ejecta[z] / y_m.mass
        true_ejected = frac_ejected * rand_mass
        my_ejected = sn_ii_kobayashi.ejecta(rand_mass, z)

        assert approx(true_ejected) == my_ejected


@pytest.mark.parametrize("my_model,yields_models",
                         [[sn_ii_kobayashi_obj, yields_models["sn"]],
                          [hn_ii_kobayashi_obj, yields_models["hn"]]])
def test_ejected_masses_sn_above_40(my_model, yields_models):
    rand_mass = np.random.uniform(40, 50, 1)

    # the ejecta masses should be the fraction of ejected mass at the 8 solar
    # mass model times the mass
    y_m = yields_models[40]
    for z in my_model.metallicities:
        frac_ejected = y_m.total_end_ejecta[z] / y_m.mass
        true_ejected = frac_ejected * rand_mass
        my_ejected = my_model.ejecta(rand_mass, z)

        assert approx(true_ejected) == my_ejected


def test_ejected_masses_agb_below_1(agb_nugrid):
    rand_mass = np.random.uniform(0, 1, 1)

    # the ejecta masses should be the fraction of ejected mass at the 8 solar
    # mass model times the mass
    y_m = yields_models["agb"][1]
    for z in agb_nugrid.metallicities:
        frac_ejected = y_m.total_end_ejecta[z] / y_m.mass
        true_ejected = frac_ejected * rand_mass
        my_ejected = agb_nugrid.ejecta(rand_mass, z)

        assert approx(true_ejected) == my_ejected


def test_ejected_masses_agb_above_8(agb_nugrid):
    for z in agb_nugrid.metallicities:
        assert 0 == agb_nugrid.ejecta(8.01, z)


@pytest.mark.parametrize("my_model,yields_models,m_up,m_low",
                         [[sn_ii_kobayashi_obj, yields_models["sn"], 25, 30],
                          [hn_ii_kobayashi_obj, yields_models["hn"], 25, 30],
                          [agb_nugrid_obj, yields_models["agb"], 5, 6]])
def test_ejected_masses_interp(my_model, yields_models, m_up, m_low):
    """Test the ejected mass when they dont align with real SN models
    We test this by just checking it's in between the two edges."""
    z = 0.001
    ejecta_a = yields_models[m_low].total_end_ejecta[z]
    ejecta_b = yields_models[m_up].total_end_ejecta[z]

    my_ejecta = my_model.ejecta(np.random.uniform(m_low, m_up, 1), z)

    assert get_if_between(ejecta_a, ejecta_b, my_ejecta)


@pytest.mark.parametrize("my_model,yields_models,m_low,m_high",
                         [[sn_ii_kobayashi_obj, yields_models["sn"], 25, 30],
                          [hn_ii_kobayashi_obj, yields_models["hn"], 25, 30],
                          [agb_nugrid_obj, yields_models["agb"], 5, 6]])
def test_mass_fractions(my_model, yields_models, m_low, m_high):
    """Test the mass fractions when they dont align with real SN models
        It should be at the closest real model."""
    z = 0.001
    for mass in np.arange(m_low-0.4, m_high+0.4, 0.2):
        # get my values
        my_m_frac = my_model.mass_fractions("Fe", z, mass)

        # get the mass of the closest model
        m_b = get_closer(m_low, m_high, mass)
        # then the mass fractions of that mass
        real_m_frac = yields_models[m_b].mass_fraction("Fe", z,
                                                       metal_only=False)

        assert my_m_frac == approx(real_m_frac[0], rel=1E-3)


@pytest.mark.parametrize("my_model,yields_models,m_low",
                         [[sn_ii_kobayashi_obj, yields_models["sn"], 13],
                          [hn_ii_kobayashi_obj, yields_models["hn"], 20],
                          [agb_nugrid_obj, yields_models["agb"], 1]])
def test_mass_fractions_below_range(my_model, yields_models, m_low):
    z = 0.001
    m_rand = np.random.uniform(0, m_low, 1)
    # get my values
    my_m_frac = my_model.mass_fractions("Fe", z, m_rand)

    # then the mass fractions of that mass
    real_m_frac = yields_models[m_low].mass_fraction("Fe", z, metal_only=False)

    assert my_m_frac == approx(real_m_frac[0], rel=1E-3)


@pytest.mark.parametrize("my_model,yields_models,m_high",
                         [[sn_ii_kobayashi_obj, yields_models["sn"], 40],
                          [hn_ii_kobayashi_obj, yields_models["hn"], 40],
                          [agb_nugrid_obj, yields_models["agb"], 7]])
def test_mass_fractions_above_range(my_model, yields_models, m_high):
    z = 0.001
    m_rand = np.random.uniform(m_high, 120, 1)
    # get my values
    my_m_frac = my_model.mass_fractions("Fe", z, m_rand)

    # then the mass fractions of that mass
    real_m_frac = yields_models[m_high].mass_fraction("Fe", z, metal_only=False)

    assert my_m_frac == approx(real_m_frac[0], rel=1E-3)


@pytest.mark.parametrize("my_model,yields_models,mass",
                         [[sn_ii_kobayashi_obj, yields_models["sn"], 25],
                          [sn_ii_kobayashi_obj, yields_models["sn"], 30],
                          [hn_ii_kobayashi_obj, yields_models["hn"], 25],
                          [hn_ii_kobayashi_obj, yields_models["hn"], 30]])
def test_wind_masses_exact(my_model, yields_models, mass):
    """Test the ejected mass when they align with real SN models"""
    z = 0.001
    real_ejecta = yields_models[mass].wind_ejecta[z]
    my_ejecta = my_model.winds(mass, z)

    assert approx(real_ejecta) == my_ejecta


@pytest.mark.parametrize("my_model", [sn_ii_kobayashi_obj, hn_ii_kobayashi_obj])
def test_wind_masses_sn_below_8(my_model):
    for z in my_model.metallicities:
        assert 0 == my_model.winds(7.99, z)


def test_wind_masses_sn_between_8_13(sn_ii_kobayashi):
    rand_mass = np.random.uniform(8, 13, 1)

    # the ejecta masses should be the fraction of ejected mass at the 8 solar
    # mass model times the mass
    y_m = yields_models["sn"][13]
    for z in sn_ii_kobayashi.metallicities:
        frac_ejected = y_m.wind_ejecta[z] / y_m.mass
        true_ejected = frac_ejected * rand_mass
        my_ejected = sn_ii_kobayashi.winds(rand_mass, z)

        assert approx(true_ejected) == my_ejected


@pytest.mark.parametrize("my_model,yields_models",
                         [[sn_ii_kobayashi_obj, yields_models["sn"]],
                          [hn_ii_kobayashi_obj, yields_models["hn"]]])
def test_wind_masses_sn_above_40(my_model, yields_models):
    rand_mass = np.random.uniform(40, 50, 1)

    # the ejecta masses should be the fraction of ejected mass at the 8 solar
    # mass model times the mass
    y_m = yields_models[40]
    for z in my_model.metallicities:
        frac_ejected = y_m.wind_ejecta[z] / y_m.mass
        true_ejected = frac_ejected * rand_mass
        my_ejected = my_model.winds(rand_mass, z)

        assert approx(true_ejected) == my_ejected


@pytest.mark.parametrize("my_model,yields_models,m_up,m_low",
                         [[sn_ii_kobayashi_obj, yields_models["sn"], 25, 30],
                          [hn_ii_kobayashi_obj, yields_models["hn"], 25, 30]])
def test_wind_masses_interp(my_model, yields_models, m_up, m_low):
    """Test the ejected mass when they dont align with real SN models
    We test this by just checking it's in between the two edges."""
    z = 0.001
    ejecta_a = yields_models[m_low].wind_ejecta[z]
    ejecta_b = yields_models[m_up].wind_ejecta[z]

    my_ejecta = my_model.winds(np.random.uniform(m_low, m_up, 1), z)

    assert get_if_between(ejecta_a, ejecta_b, my_ejecta)


ejecta_check_sn = [[13, 0,     "C",  7.41E-2 + 8.38E-8],
                   [15, 0,     "N",  1.86E-3 + 6.86E-8],
                   [18, 0.001, "O",  4.22E-1 + 2.42E-5 + 3.06E-4],
                   [20, 0.001, "Fe", 2.37E-3 + 7.09E-2 + 7.21E-4 + 4.99E-5],
                   [25, 0.004, "C",  1.32E-1 + 3.83E-4],
                   [30, 0.004, "N",  2.01E-2 + 4.98E-6],
                   [40, 0.02,  "O",  7.33E-0 + 9.72E-4 + 1.23E-2],
                   [13, 0.02,  "Fe", 1.98E-3 + 8.32E-2 + 2.22E-3 + 1.21E-4]]

@pytest.mark.parametrize("m,z,elt,answer", ejecta_check_sn)
def test_sn_ejected_masses(sn_ii_kobayashi, m, z, elt, answer):
    """Get items right from the table."""
    my_answer = sn_ii_kobayashi.elemental_ejecta_mass(m, z, elt)
    assert my_answer == approx(answer, rel=2E-3)


ejecta_check_hn = [[20, 0,     "C",  1.90E-1 + 1.18E-8],
                   [25, 0.001, "N",  9.20E-3 + 7.24E-6],
                   [30, 0.004, "O",  3.82E-0 + 1.20E-4 + 4.38E-5],
                   [40, 0.02,  "Fe", 5.89E-3 + 2.77E-1 + 8.90E-3 + 1.36E-3]]

@pytest.mark.parametrize("m,z,elt,answer", ejecta_check_hn)
def test_hn_ejected_masses(hn_ii_kobayashi, m, z, elt, answer):
    """Get items right from the table."""
    my_answer = hn_ii_kobayashi.elemental_ejecta_mass(m, z, elt)
    assert my_answer == approx(answer, rel=2E-3)


# These are only rough estimates, since the values given in Kobayashi are
# only to 3 sig figs. This is calcuted by M_final - M_p - M_4He. Deuterium and
# 3He are insignificant
metals_check_sn = [[13, 0,     13.00, 1.57, 6.60, 4.01],
                   [15, 0,     15.00, 1.48, 7.58, 4.40],
                   [18, 0.001, 17.84, 1.70, 8.46, 6.54],
                   [20, 0.001, 19.72, 1.85, 8.43, 5.94],
                   [25, 0.004, 24.03, 1.68, 10.2, 8.48],
                   [30, 0.004, 27.59, 2.56, 10.1, 7.92],
                   [40, 0.02,  21.83, 2.21, 3.55, 4.71],
                   [13, 0.02,  12.73, 1.60, 6.16, 4.30]]


@pytest.mark.parametrize("m,z,m_final,m_cut,m_h,m_he", metals_check_sn)
def test_sn_ejected_metals(sn_ii_kobayashi, m, z, m_final, m_cut, m_h, m_he):
    """Get items right from the table."""
    metal_ejecta = m_final - (m_cut + m_h + m_he)
    my_answer = sn_ii_kobayashi.elemental_ejecta_mass(m, z, "total_metals")
    assert my_answer == approx(metal_ejecta, rel=1E-2)


# These are only rough estimates, since the values given in Kobayashi are
# only to 3 sig figs. This is calcuted by M_final - M_p - M_4He. Deuterium and
# 3He are insignificant
metals_check_hn = [[20, 0    , 20.00, 1.88, 8.77, 5.96],
                   [25, 0.001, 24.45, 2.15, 9.80, 7.00],
                   [30, 0.004, 27.55, 4.05, 10.1, 7.93],
                   [40, 0.02,  21.84, 2.67, 3.55, 4.78]]


@pytest.mark.parametrize("m,z,m_final,m_cut,m_h,m_he", metals_check_hn)
def test_hn_ejected_metals(hn_ii_kobayashi, m, z, m_final, m_cut, m_h, m_he):
    """Get items right from the table."""
    metal_ejecta = m_final - (m_cut + m_h + m_he)
    my_answer = hn_ii_kobayashi.elemental_ejecta_mass(m, z, "total_metals")
    assert my_answer == approx(metal_ejecta, rel=1E-2)


# ----------------------------------------------------------

# Testing combined SN model set

# ----------------------------------------------------------
def test_min_max_limits_equal_overall():
    with pytest.raises(ValueError):
        SNOverall("Kobayashi_06", 0.5, 15, 15)


def test_min_max_limits_reversed_overall():
    with pytest.raises(ValueError):
        SNOverall("Kobayashi_06", 0.5, 20, 10)


def test_init_sn_overall(sn_overall):
    """ Test initialization """
    assert sn_overall._hn_fraction == 0.45
    assert sn_overall._sn_fraction == 0.55


def test_hn_fraction(sn_overall):
    assert sn_overall.hn_fraction(9.9) == 0
    assert sn_overall.sn_fraction(9.9) == 1

    assert sn_overall.hn_fraction(19.9) == 0
    assert sn_overall.sn_fraction(19.9) == 1

    assert sn_overall.hn_fraction(20) == 0.45
    assert sn_overall.sn_fraction(20) == 0.55

    assert sn_overall.hn_fraction(20.1) == 0.45
    assert sn_overall.sn_fraction(20.1) == 0.55

    assert sn_overall.hn_fraction(120) == 0.45
    assert sn_overall.sn_fraction(120) == 0.55


def test_integrand_example_with_hn(sn_overall):
    """See if the integrand calculation is working at a given mass"""
    z = 0.004
    m = 20

    hn_frac = sn_overall._hn_fraction
    sn_frac = 1.0 - hn_frac

    real_sn_mass_frac = yields_models["sn"][m].mass_fraction("C", z,
                                                             metal_only=False)
    real_hn_mass_frac = yields_models["hn"][m].mass_fraction("C", z,
                                                             metal_only=False)

    real_sn_ejecta = yields_models["sn"][m].total_end_ejecta[z]
    real_hn_ejecta = yields_models["hn"][m].total_end_ejecta[z]

    hn_term = hn_frac * real_hn_mass_frac * real_hn_ejecta
    sn_term = sn_frac * real_sn_mass_frac * real_sn_ejecta
    total_integrand = hn_term + sn_term

    my_integrand = sn_overall.elemental_ejecta_mass(m, z, "C")

    assert my_integrand == approx(total_integrand)


def test_integrand_example_with_hn_total(sn_overall):
    """See if the integrand calculation is working at a given mass"""
    z = 0.004
    m = 20

    hn_frac = sn_overall._hn_fraction
    sn_frac = 1.0 - hn_frac

    real_sn_ejecta = yields_models["sn"][m].total_end_ejecta[z]
    real_hn_ejecta = yields_models["hn"][m].total_end_ejecta[z]

    hn_term = hn_frac * real_hn_ejecta
    sn_term = sn_frac * real_sn_ejecta
    total_integrand = hn_term + sn_term

    my_integrand = sn_overall.elemental_ejecta_mass(m, z, "total")

    assert my_integrand == approx(total_integrand)


def test_integrand_example_without_hn(sn_overall):
    """See if the integrand calculation is working at a given mass"""
    z = 0.004
    m = 18

    real_sn_mass_frac = yields_models["sn"][m].mass_fraction("N", z,
                                                             metal_only=False)
    real_sn_ejecta = yields_models["sn"][m].total_end_ejecta[z]
    sn_term = real_sn_mass_frac * real_sn_ejecta

    my_integrand = sn_overall.elemental_ejecta_mass(m, z, "N")

    assert my_integrand == approx(sn_term)


def test_integrand_example_without_hn_total(sn_overall):
    """See if the integrand calculation is working at a given mass"""
    z = 0.004
    m = 18

    real_sn_ejecta = yields_models["sn"][m].total_end_ejecta[z]

    my_integrand = sn_overall.elemental_ejecta_mass(m, z, "total")

    assert my_integrand == approx(real_sn_ejecta)


def test_sn_hn_fractions_sum(sn_overall):
    for m in np.arange(0, 100, 1):
        assert 1.0 == sn_overall.sn_fraction(m) + sn_overall.hn_fraction(m)
