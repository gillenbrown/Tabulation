from scipy import integrate
import numpy as np

class IMF(object):
    def __init__(self, name, m_low, m_high, total_mass=1):
        if name.lower() == "kroupa":
            self.dn_dm = np.vectorize(self.kroupa_dn_dm)
        else:
            raise ValueError("IMF not supported.")
        self.normalize(m_low, m_high, total_mass)

    def normalized_dn_dm(self, mass):
        """
        Return the normalized IMF (dn/dm) at a given mass.

        :param mass: Stellar mass, in units of solar masses
        :return: IMF evaluated at `mass`
        """
        return self.normalization * self.dn_dm(mass)

    def normalize(self, m_low, m_high, total_mass=1):
        """
        Find the normalization for the IMF so it integrates to a given total.

        :param m_low: Lower IMF mass cutoff.
        :param m_high: Upper IMF mass cutoff.
        :param total_mass: Total mass the IMF should integrate to. Defaults to 1
        :return: None, but sets the self.normalization parameter
        """
        # We want it such that:
        # total_mass = N * integral of IMF from low to high
        # so
        # N = total_mass / integral
        def mass_integrand(m):
            return m * self.dn_dm(m)

        integral = integrate.quad(mass_integrand, m_low, m_high)[0]
        self.normalization = total_mass / integral

    @staticmethod
    def kroupa_dn_dm(mass):
        """
        Kroupa IMF function (dn_dm) that is not normalized.

        :param mass: Stellar mass, in solar masses.
        :return: IMF (not normalized)
        """
        if (mass < 0.5):
            return (mass / 0.08) ** -1.3
        else:
            return (0.5 / 0.08) ** -1.3 * (mass / 0.5) ** -2.3