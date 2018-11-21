import numpy as np

class SN_Ia(object):
    def __init__(self, name):
        if name.lower() == "art":
            self.sn_rate = np.vectorize(self.art_rate, otypes=[float])
        else:
            raise ValueError("SN Ia DTD not supported.")

    def _testing_sn_rate_age_log(self, log_age):
        return self.sn_rate(10**log_age)

    @staticmethod
    def art_rate(age):
        t_eject = 2E8  # years
        if age < 0.1 * t_eject:
            return 0

        def sub_func(x):
            """
            This function is normalized to integrate to 1 at infinity.
            """
            return np.exp(-x ** 2) * np.sqrt(x ** 3) / 1.812804954

        normalized_age = t_eject / age
        return sub_func(normalized_age) / t_eject


