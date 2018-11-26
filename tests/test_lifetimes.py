import pytest
from tabulation import Lifetimes
import numpy as np


def test_inversion_to_mass():
    lifetimes = Lifetimes("raiteri_96")
    rand_mass = np.random.uniform(1, 50)
    rand_z = np.random.uniform(1E-4, 0.02)

    life = lifetimes.lifetime(rand_mass, rand_z)
    inverted_mass = lifetimes.turnoff_mass(life, rand_z)

    assert pytest.approx(rand_mass) == inverted_mass


def test_inversion_to_time():
    lifetimes = Lifetimes("raiteri_96")
    rand_time = 10**np.random.uniform(7, 10)
    rand_z = np.random.uniform(1E-4, 0.02)

    mass = lifetimes.turnoff_mass(rand_time, rand_z)
    inverted_life = lifetimes.lifetime(mass, rand_z)

    assert pytest.approx(rand_time) == inverted_life
