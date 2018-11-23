import numpy as np
from scipy import optimize


class Lifetimes(object):
    """
    Class holding the functions to get the lifetimes of stars.
    """
    def __init__(self, name):
        """
        Initialize a certain lifetime set.

        :param name: Name of the lifetime set being used.
        """
        # parse the name
        if name.lower() == "raiteri_96":
            self.lifetime = self.raiteri_96
        else:
            raise ValueError("Lifetime not supported")

    def turnoff_mass(self, time, z):
        """
        Invert the lifetimes to get the mass of the star leaving the main
        sequence at a given time.

        :param time: Time in years.
        :param z: Metallicity at which to evaluate the lifetimes.
        :return: Mass in solar masses of the star leaving the main sequence.
        """
        # We'll use scipy's minimize, which will be used to find the stellar
        # mass that best returns the correct lifetimes.
        # make a function that we can use to minimize.
        def age_error(m):
            lifetime = self.lifetime(m, z)
            # calculate the absolute error in log lifetime.
            return np.abs(np.log10(lifetime) - np.log10(time))

        turnoff_result = optimize.minimize(age_error, x0=np.array([2]),
                                           bounds=[[0.61, 119.9]])
        # turnoff_result is a structure that scipy uses. We want the answer
        # it contains.
        return turnoff_result.x

    @staticmethod
    def raiteri_96(m, z):
        """
        Metallicity dependent lifetimes from Raiteri+ 1996 and currently
        implemented in ART.

        :param m: Mass of the star in solar masses
        :param z: total metallicity of the star
        :return: Lifetime of the star, in years.
        """
        log_z = np.log10(z)
        log_m = np.log10(m)

        # set boundary conditions
        if log_z < -4.155:
            log_z = -4.155
        elif log_z > -1.523:
            log_z = -1.523

        if m < 0.6:
            raise ValueError("Mass too low")
        elif m > 120:
            raise ValueError("Mass too big")

        # specific formulae used in the paper.
        a0 = 10.13 + 0.07547 * log_z - 0.008084 * log_z * log_z
        a1 = -4.424 - 0.7939 * log_z - 0.1187 * log_z * log_z
        a2 = 1.262 + 0.3385 * log_z + 0.05417 * log_z * log_z
        log_lifetime = a0 + a1 * log_m + a2 * log_m * log_m
        return 10**log_lifetime
