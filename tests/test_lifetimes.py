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


def test_zero_time():
    lifetimes = Lifetimes("raiteri_96")
    rand_z = np.random.uniform(1E-4, 0.02)
    mass = lifetimes.turnoff_mass(0, rand_z)

    assert mass == 120


def test_two_mass_inversions():
    lifetimes = Lifetimes("raiteri_96")
    rand_z = np.random.uniform(1E-4, 0.02)
    mass_0 = lifetimes.turnoff_mass(4E6, rand_z)
    mass_1 = lifetimes.turnoff_mass(5E6, rand_z)

    assert mass_1 < mass_0

def test_example_10():
    lifetimes = Lifetimes("raiteri_96")
    test_answer = lifetimes.lifetime(10, 0.02)
    true_answer = 10**7.40388943
    assert test_answer == pytest.approx(true_answer, rel=1E-5)

def test_example_1():
    lifetimes = Lifetimes("raiteri_96")
    test_answer = lifetimes.lifetime(1, 0.02)
    true_answer = 10**9.978444275
    assert test_answer == pytest.approx(true_answer, rel=1E-5)

def test_example_0_8():
    lifetimes = Lifetimes("raiteri_96")
    test_answer = lifetimes.lifetime(0.8, 0.02)
    true_answer = 10**10.31758431
    assert test_answer == pytest.approx(true_answer, rel=1E-5)