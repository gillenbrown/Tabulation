import numpy as np
import yields


class SNIa(object):
    """
    Class holding the SNIa delay time distribution and yields.
    """
    def __init__(self, dtd_name, yield_name):
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
            self.sn_rate = np.vectorize(self.art_rate, otypes=[float])
        else:
            raise ValueError("SN Ia DTD not supported.")

        # parse the yields
        if yield_name.lower() == "nomoto_18":
            self.model = yields.Yields("nomoto_18_Ia_W7")
        else:
            raise ValueError("SN Ia model set not recognized")

    @staticmethod
    def art_rate(age):
        """
        SN rate as coded in ART. Is normalized to 1 at infinity.

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
