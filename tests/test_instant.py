import pytest
from pytest import approx
from tabulation import SNII, MassiveOverall, AGB
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
def massive_overall():
    return MassiveOverall("kobayashi_06", 0.45, 8, 50)


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


def test_ejected_masses_sn_above_50(sn_ii_kobayashi):
    for z in sn_ii_kobayashi.metallicities:
        assert 0 == sn_ii_kobayashi.ejecta(50.01, z)


def test_ejected_masses_hn_below_20(hn_ii_kobayashi):
    for z in hn_ii_kobayashi.metallicities:
        assert 0 == hn_ii_kobayashi.ejecta(19.99, z)


def test_ejected_masses_hn_above_50(hn_ii_kobayashi):
    for z in hn_ii_kobayashi.metallicities:
        assert 0 == hn_ii_kobayashi.ejecta(50.01, z)


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


def test_wind_masses_sn_below_8(sn_ii_kobayashi):
    for z in sn_ii_kobayashi.metallicities:
        assert 0 == sn_ii_kobayashi.winds(7.99, z)


def test_wind_masses_hn_below_20(hn_ii_kobayashi):
    for z in hn_ii_kobayashi.metallicities:
        assert 0 == hn_ii_kobayashi.winds(19.99, z)


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
                   [13, 0.02,  "Fe", 1.98E-3 + 8.32E-2 + 2.22E-3 + 1.21E-4],
                   [15, 0,     "Mg", 6.82E-2 + 2.98E-4 + 3.99E-4],
                   [18, 0.001, "Mg", 5.93E-2 + 9.46E-4 + 9.27E-4],
                   [20, 0.004, "S",  5.15E-2 + 1.89E-4 + 9.12E-4 + 1.36E-6],
                   [25, 0.02,  "S",  4.99E-2 + 3.25E-4 + 2.26E-3 + 2.42E-5],
                   [30, 0,     "Ca", 1.74E-2 + 8.62E-7 + 1.93E-9 + 5.44E-6],
                   [40, 0.001, "Ca", 3.66E-2 + 2.29E-5 + 2.81E-7 + 1.10E-5]]


@pytest.mark.parametrize("m,z,elt,answer", ejecta_check_sn)
def test_sn_ejected_masses(sn_ii_kobayashi, m, z, elt, answer):
    """Get items right from the table."""
    my_answer = sn_ii_kobayashi.elemental_ejecta_mass(m, z, elt)
    assert my_answer == approx(answer, rel=2E-3)


ejecta_check_hn = [[20, 0,     "C",  1.90E-1 + 1.18E-8],
                   [25, 0.001, "N",  9.20E-3 + 7.24E-6],
                   [30, 0.004, "O",  3.82E-0 + 1.20E-4 + 4.38E-5],
                   [40, 0.02,  "Fe", 5.89E-3 + 2.77E-1 + 8.90E-3 + 1.36E-3],
                   [20, 0.02,  "Mg", 6.88E-2 + 1.12E-2 + 7.52E-3],
                   [25, 0.004, "S",  4.02E-2 + 3.05E-4 + 1.25E-3],
                   [30, 0.001, "Ca", 1.10E-2 + 8.11E-6 + 6.45E-7 + 2.44E-4]]


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
metals_check_hn = [[20, 0,     20.00, 1.88, 8.77, 5.96],
                   [25, 0.001, 24.45, 2.15, 9.80, 7.00],
                   [30, 0.004, 27.55, 4.05, 10.1, 7.93],
                   [40, 0.02,  21.84, 2.67, 3.55, 4.78]]


@pytest.mark.parametrize("m,z,m_final,m_cut,m_h,m_he", metals_check_hn)
def test_hn_ejected_metals(hn_ii_kobayashi, m, z, m_final, m_cut, m_h, m_he):
    """Get items right from the table."""
    metal_ejecta = m_final - (m_cut + m_h + m_he)
    my_answer = hn_ii_kobayashi.elemental_ejecta_mass(m, z, "total_metals")
    assert my_answer == approx(metal_ejecta, rel=1E-2)


ejecta_check_agb = [[1,    0.0001, "C",  8.708E-03],
                    [1.65, 0.0001, "N",  5.710E-05],
                    [2,    0.001,  "O",  1.184E-02],
                    [3,    0.001,  "Fe", 3.116E-05],
                    [4,    0.006,  "C",  9.100E-03],
                    [5,    0.006,  "N",  7.439E-03],
                    [6,    0.01,   "O",  1.943E-02],
                    [7,    0.01,   "Fe", 4.227E-03],
                    [1,    0.02,   "C",  1.353E-03],
                    [7,    0.02,   "N",  4.746E-02]]


@pytest.mark.parametrize("m,z,elt,answer", ejecta_check_agb)
def test_agb_ejected_masses(agb_nugrid, m, z, elt, answer):
    """Get items right from the table."""
    my_answer = agb_nugrid.elemental_ejecta_mass(m, z, elt)
    assert my_answer == approx(answer, rel=1E-3)


#
# Get the total metals lost. This is calculated by M - M_remnant - M_H - M_He.
metals_check_agb = [[1, 0.0001, 5.296E-01, 3.289E-01, 1.308E-01],
                    [3, 0.001,  8.242E-01, 1.570E+00, 5.920E-01],
                    [5, 0.006,  9.533E-01, 2.650E+00, 1.371E+00],
                    [6, 0.01,   9.938E-01, 3.133E+00, 1.805E+00],
                    [7, 0.02,   1.066E+00, 3.568E+00, 2.239E+00]]


@pytest.mark.parametrize("m,z,m_remnant,m_h,m_he", metals_check_agb)
def test_agb_ejected_metals(agb_nugrid, m, z, m_remnant, m_h, m_he):
    """Get items right from the table."""
    metal_ejecta = m - (m_remnant + m_h + m_he)
    my_answer = agb_nugrid.elemental_ejecta_mass(m, z, "total_metals")
    assert my_answer == approx(metal_ejecta, rel=5E-2)


# ----------------------------------------------------------

# SN energies

# ----------------------------------------------------------
@pytest.mark.parametrize("z", [0, 0.001, 0.004, 0.02])
def test_sn_energies_boundaries(sn_ii_kobayashi, z):
    # Outside the boundary there is no SN and no energy
    assert sn_ii_kobayashi.energy_released_erg(7.99, z) == 0
    assert sn_ii_kobayashi.energy_released_erg(50.01, z) == 0


@pytest.mark.parametrize("z", [0, 0.001, 0.004, 0.02])
def test_sn_energies(sn_ii_kobayashi, z):
    # These should all have the same energies, at all metallicities and masses
    for m in np.random.uniform(sn_ii_kobayashi.mass_boundary_low,
                               sn_ii_kobayashi.mass_boundary_high, 100):
        assert sn_ii_kobayashi.energy_released_erg(m, z) == approx(1E51)


@pytest.mark.parametrize("z", [0, 0.001, 0.004, 0.02])
def test_hn_energies_boundaries(hn_ii_kobayashi, z):
    # Outside the boundary there is no SN and no energy
    assert hn_ii_kobayashi.energy_released_erg(7.99, z) == 0
    assert hn_ii_kobayashi.energy_released_erg(50.01, z) == 0


@pytest.mark.parametrize("z", [0, 0.001, 0.004, 0.02])
@pytest.mark.parametrize("mass,energy", [[20, 10E51], [25, 10E51],
                                         [30, 20E51], [40, 30E51]])
def test_hn_energies_exact(hn_ii_kobayashi, z, mass, energy):
    assert hn_ii_kobayashi.energy_released_erg(mass, z) == approx(energy)


@pytest.mark.parametrize("z", [0, 0.001, 0.004, 0.02])
def test_hn_energies_low_end(hn_ii_kobayashi, z):
    # Everything between 20 and 25 has the same energy
    for mass in np.arange(20, 25, 0.1):
        assert hn_ii_kobayashi.energy_released_erg(mass, z) == approx(10E51)


@pytest.mark.parametrize("z", [0, 0.001, 0.004, 0.02])
def test_hn_energies_high_end(hn_ii_kobayashi, z):
    # Everything above 40 should have the same energy
    for mass in np.arange(40, 50, 0.1):
        assert hn_ii_kobayashi.energy_released_erg(mass, z) == approx(30E51)


energy_checks = [[25, 30, 1E52, 2E52],
                 [30, 40, 2E52, 3E52]]


@pytest.mark.parametrize("z", [0, 0.001, 0.004, 0.02])
@pytest.mark.parametrize("m_low,m_high,e_low,e_high", energy_checks)
def test_hn_energies_between_general(hn_ii_kobayashi, m_low, m_high,
                                     e_low, e_high, z):
    """Test the energies when they dont align with real SN models
       We interpolate the energies, so the results should be between the
       two models given."""
    for mass in np.arange(m_low+0.1, m_high, 0.1):
        energy = hn_ii_kobayashi.energy_released_erg(mass, z)
        assert e_low < energy < e_high


# ----------------------------------------------------------

# Testing combined SN model set

# ----------------------------------------------------------
def test_min_max_limits_equal_overall():
    with pytest.raises(ValueError):
        MassiveOverall("Kobayashi_06", 0.5, 15, 15)


def test_min_max_limits_reversed_overall():
    with pytest.raises(ValueError):
        MassiveOverall("Kobayashi_06", 0.5, 20, 10)


def test_init_massive_overall(massive_overall):
    """ Test initialization """
    assert massive_overall._hn_fraction == 0.45
    assert massive_overall._sn_fraction == 0.55


def test_hn_fraction(massive_overall):
    assert massive_overall.hn_fraction(9.9) == 0
    assert massive_overall.sn_fraction(9.9) == 1

    assert massive_overall.hn_fraction(19.9) == 0
    assert massive_overall.sn_fraction(19.9) == 1

    assert massive_overall.hn_fraction(20) == 0.45
    assert massive_overall.sn_fraction(20) == 0.55

    assert massive_overall.hn_fraction(20.1) == 0.45
    assert massive_overall.sn_fraction(20.1) == 0.55

    assert massive_overall.hn_fraction(120) == 0.45
    assert massive_overall.sn_fraction(120) == 0.55


def test_integrand_example_with_hn(massive_overall):
    """See if the integrand calculation is working at a given mass"""
    z = 0.004
    m = 20

    hn_frac = massive_overall._hn_fraction
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

    my_integrand = massive_overall.elemental_ejecta_mass(m, z, "C")

    assert my_integrand == approx(total_integrand)


def test_integrand_example_with_hn_total(massive_overall):
    """See if the integrand calculation is working at a given mass"""
    z = 0.004
    m = 20

    hn_frac = massive_overall._hn_fraction
    sn_frac = 1.0 - hn_frac

    real_sn_ejecta = yields_models["sn"][m].total_end_ejecta[z]
    real_hn_ejecta = yields_models["hn"][m].total_end_ejecta[z]

    hn_term = hn_frac * real_hn_ejecta
    sn_term = sn_frac * real_sn_ejecta
    total_integrand = hn_term + sn_term

    my_integrand = massive_overall.elemental_ejecta_mass(m, z, "total")

    assert my_integrand == approx(total_integrand)


def test_integrand_example_with_hn_winds(massive_overall):
    """See if the integrand calculation is working at a given mass"""
    z = 0.004
    m = 20

    hn_frac = massive_overall._hn_fraction
    sn_frac = 1.0 - hn_frac

    real_wind_ejecta_sn = yields_models["sn"][m].wind_ejecta[z]
    real_wind_ejecta_hn = yields_models["hn"][m].wind_ejecta[z]

    hn_term = hn_frac * real_wind_ejecta_hn
    sn_term = sn_frac * real_wind_ejecta_sn
    total_integrand = hn_term + sn_term

    my_integrand = massive_overall.wind_mass(m, z)

    assert my_integrand == approx(total_integrand)


def test_integrand_example_without_hn(massive_overall):
    """See if the integrand calculation is working at a given mass"""
    z = 0.004
    m = 18

    real_sn_mass_frac = yields_models["sn"][m].mass_fraction("N", z,
                                                             metal_only=False)
    real_sn_ejecta = yields_models["sn"][m].total_end_ejecta[z]
    sn_term = real_sn_mass_frac * real_sn_ejecta

    my_integrand = massive_overall.elemental_ejecta_mass(m, z, "N")

    assert my_integrand == approx(sn_term)


def test_integrand_example_without_hn_total(massive_overall):
    """See if the integrand calculation is working at a given mass"""
    z = 0.004
    m = 18

    real_sn_ejecta = yields_models["sn"][m].total_end_ejecta[z]

    my_integrand = massive_overall.elemental_ejecta_mass(m, z, "total")

    assert my_integrand == approx(real_sn_ejecta)


def test_integrand_example_without_hn_winds(massive_overall):
    """See if the integrand calculation is working at a given mass"""
    z = 0.004
    m = 18

    real_wind_ejecta = yields_models["sn"][m].wind_ejecta[z]

    my_integrand = massive_overall.wind_mass(m, z)

    assert my_integrand == approx(real_wind_ejecta)


def test_sn_hn_fractions_sum(massive_overall):
    for m in np.arange(0, 100, 1):
        assert 1.0 == massive_overall.sn_fraction(m) + \
                      massive_overall.hn_fraction(m)


energy_checks_overall = [[8, 1E51], [13, 1E51], [19.99, 1E51], [20, 5.05E51],
                         [25, 5.05E51], [30, 9.55E51], [40, 14.05E51],
                         [49.9, 14.05E51]]


@pytest.mark.parametrize("z", [0, 0.001, 0.004, 0.02])
@pytest.mark.parametrize("m,e_true", energy_checks_overall)
def test_integrand_energies(massive_overall, z, m, e_true):
    assert massive_overall.energy_released_erg(m, z) == approx(e_true)

# TODO: think of splitting up integrals
