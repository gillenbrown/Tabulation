import numpy as np
import yields


class SNIa(object):
    """
    Class holding the SNIa delay time distribution and yields.
    """
    def __init__(self, dtd_name, yield_name, exploding_fraction,
                 min_mass, max_mass):
        """
        Initialize the SN Ia model.

        :param dtd_name: Name of the delay time distribution. Currently only the
                         one from ART is supported.
        :type dtd_name: str
        :param yield_name: Name of the yield set being used.
        :type yield_name: str
        """
        # parse the DTD
        if dtd_name.lower() == "art":
            self.sn_dtd = np.vectorize(self.art_phi_per_dt, otypes=[float])
        else:
            raise ValueError("SN Ia DTD not supported.")

        # parse the yields
        if yield_name.lower() == "nomoto_18":
            self.model = yields.Yields("nomoto_18_Ia_W7")
            self.metallicities = [0.002, 0.02]
        else:
            raise ValueError("SN Ia model set not recognized")

        self.exploding_fraction = exploding_fraction
        self.min_mass = min_mass
        self.max_mass = max_mass

    @staticmethod
    def art_phi_per_dt(age):
        """
        SN rate as coded in ART. Is normalized to 1 at infinity.

        This function returns phi / dt.

        :param age: Age (in years) at which to get the SN Ia rate.
        :return: rate, normalized to 1.
        """
        t_eject = 2E8  # years
        # no SN Ia early on
        if age < 0.1 * t_eject:
            return 0

        # make the heart of the DTD. The last term is to make it normalized.
        def sub_func(x):
            return np.exp(-x ** 2) * np.sqrt(x ** 3) / 1.812804954

        normalized_age = t_eject / age
        return sub_func(normalized_age) / t_eject

    def ejected_mass(self, element, metallicity):
        """
        Get the ejected mass of a given element in a SN Ia explosion.

        :param element: element to get the ejected mass of
        :param metallicity: Metallicity of the progenitor
        :return: Ejected mass of that element in solar masses
        """
        if metallicity not in self.metallicities:
            raise ValueError("Can only get values at the exact metallicities.")

        # this is basically a convenience function to just get the model.
        self.model.set_metallicity(metallicity)
        if element == "total_metals":
            return self.model.ejecta_sum(metal_only=True)
        else:
            try:
                return self.model.abundances[element]
            except KeyError:  # element not present
                return 0
