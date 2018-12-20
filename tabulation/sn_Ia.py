import numpy as np
import yields
from scipy import integrate


class SNIa(object):
    """
    Class holding the SNIa delay time distribution and yields.
    """
    def __init__(self, dtd_name, yield_name, lifetimes_obj, imf_obj,
                 **kwargs):
        """
        Initialize the SN Ia model.

        :param dtd_name: Name of the delay time distribution. Currently can be
                         either "old art" for the old prescription or
                         "new art power law" for the new one.
        :type dtd_name: str
        :param yield_name: Name of the yield set being used.
        :type yield_name: str
        :param lifetimes_obj: Already created Lifetimes object to be used by
                              the delay time distribution
        :type lifetimes_obj: tabulation.Lifetime
        :param imf_obj: Already created IMF object to be used to calibrate the
                        number of SN (for the old ART prescription)
        :type imf_obj: tabulation.IMF
        :param kwargs: Additional parameters to be passed to the SN Ia DTD
                       function. For "old art" we need "min_mass", "max_mass",
                       and "exploding_fraction" (the fraction of stars between
                       min_mass and max_mass that explode as SN Ia. For
                       "art power law" we only need the overall number of SNIa
                       that explode in a unit mass SSP: "number_sn_ia"
        """
        # parse the DTD
        if dtd_name.lower() == "old art":
            self.sn_dtd = np.vectorize(self.old_art_dtd, otypes=[float])
        elif dtd_name.lower() == "art power law":
            self.sn_dtd = np.vectorize(self.art_power_law_dtd, otypes=[float])
        else:
            raise ValueError("SN Ia DTD not supported.")

        # store the lifetime objects
        self.lifetimes = lifetimes_obj
        self.imf = imf_obj

        # parse the yields
        if yield_name.lower() == "nomoto_18":
            self.model = yields.Yields("nomoto_18_Ia_W7")
            self.metallicities = [0.002, 0.02]
        else:
            raise ValueError("SN Ia model set not recognized")

        # set the number of SN Ia per unit mass SSP. For old art we need to
        # calculate it.
        if dtd_name.lower() == "old art":
            num_in_range = integrate.quad(self.imf.normalized_dn_dm,
                                          kwargs["min_mass"],
                                          kwargs["max_mass"])[0]
            self.number_sn_Ia = num_in_range * kwargs["exploding_fraction"]
        elif dtd_name == "ART power law":
            self.number_sn_Ia = kwargs["number_sn_ia"]

    def old_art_dtd(self, age):
        """
        SN Ia rate previously coded in ART. This is normalized to produce
        the desired number of SN Ia (as specified by min_mass, max_mass, and
        exploding_fraction) at infinity.

        :param age: Age (in years) at which to get the SN rate.
        :return: SN Ia rate, in units of SN per year per unit stellar mass.
        """
        return self.number_sn_Ia * self.old_art_phi_per_dt(age)

    @staticmethod
    def old_art_phi_per_dt(age):
        """
        SN rate as previously coded in ART. Is normalized to 1 at infinity.

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

    def art_power_law_dtd(self, age, z):
        """
        SN rate going in the new ART prescription.

        This is a power law with index -1.13, normalized to produce the desired
        number of supernovae after one Hubble time.

        See http://nbviewer.jupyter.org/github/gillenbrown/Tabulation/blob/master/notebooks/sn_Ia.ipynb
        for more info.

        :param age: Age (in years) at which to get the SN Ia rates.
        :param z: Metallicity of the progenitor. This is used to determine the
                  lifetime at which SN Ia start.
        :return: SN Ia rate in supernovae per year per unit stellar mass.
        """
        new_art_s = 1.13
        min_age = self.lifetimes.lifetime(8.0, z)
        if age < min_age:
            return 0
        else:
            return self.number_sn_Ia * 2.3480851917 * age ** (-new_art_s)

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
