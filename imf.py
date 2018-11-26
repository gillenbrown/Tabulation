from scipy import integrate
import numpy as np


class IMF(object):
    """
    Class representing the initial mass function.
    """
    def __init__(self, name, m_low, m_high, total_mass=1):
        """
        Initialize the IMF with a given form, mass range, and normalization.
        The normalization takes the form of the total mass present in an SSP
        with this IMF. This defaults to 1 if not set.

        :param name: Name of the IMF model to be used.
        :param m_low: Minumum stellar mass in the IMF.
        :param m_high: Maximum stellar mass in the IMF.
        :param total_mass: Total mass of the SSP. Used to normalize quantities.
                           Defaults to 1 if not set.
        """
        # parse the model name.
        if name.lower() == "kroupa":
            self.dn_dm = np.vectorize(self.kroupa_dn_dm)
        else:
            raise ValueError("IMF not supported.")

        # check the limits
        if m_low >= m_high:
            raise ValueError("IMF minumum mass must be lower than maximum mass")

        # use the total mass to get the normalization.
        self.normalization = None
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
            """
            Integrand to calculate the total mass of the IMF.
            if m * phi(m) at all masses.

            :param m: Stellar mass.
            :return: m * phi(m) at all masses.
            """
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
        if mass < 0.5:
            return (mass / 0.08) ** -1.3
        else:
            # first term needed to match the value of the other half
            # of the function.
            return (0.5 / 0.08) ** -1.3 * (mass / 0.5) ** -2.3
