from scipy import integrate

from .sn_Ia import SNIa
from .instant import SNOverall, AGB
from .lifetimes import Lifetimes
from .imf import IMF


class SSPYields(object):
    """
    Class representing the stellar yields from an SSP
    """
    def __init__(self, imf_name, imf_min_mass, imf_max_mass, total_mass,
                 lifetimes_name,
                 sn_ii_yields, sn_ii_hn_fraction,
                 sn_ii_min_mass, sn_ii_max_mass,
                 sn_ia_dtd, sn_ia_yields,
                 agb_yields, agb_min_mass, agb_max_mass):
        """

        :param imf_name: Name of the IMF model to choose. Available choices are:
                         "Kroupa".
        :type imf_name: str
        :param imf_min_mass: Lower mass limit of the IMF.
        :type imf_min_mass: float
        :param imf_max_mass: Upper mass limit of the IMF.
        :type imf_max_mass: float
        :param total_mass: Total stellar mass of the SSP.
        :type total_mass: float
        :param lifetimes_name: Name of the models used to calculate the
                               stellar lifetimes. Available choices are:
                               "Raiteri_96".
        :type lifetimes_name: str
        :param sn_ii_yields: Name of the yield set used for SN II. Available
                             choices are: "Kobayashi_06".
        :type sn_ii_yields: str
        :param sn_ii_hn_fraction: Fraction of massive stars above the minimum
                                  mass for HN (which is set by the yield set)
                                  that explode as HN. The rest explode as SN.
        :type sn_ii_hn_fraction: float
        :param sn_ii_min_mass: Minimum stellar mass that explodes as SN.
        :type sn_ii_min_mass: float
        :param sn_ii_max_mass: Maximum stellar mass that explodes as SN.
        :type sn_ii_max_mass: float
        :param sn_ia_dtd: Name of the delay time distribution for SN IA.
                          Available choices are: "ART"
        :type sn_ia_dtd: str
        :param sn_ia_yields: Name of the yield set used for SN Ia. Available
                             choices are: "Nomoto_18".
        :type sn_ia_yields: str
        :param agb_yields: Name of the yield set used for SN II. Available
                           choices are: "NuGrid".
        :type agb_yields: str
        :param agb_min_mass: Minimum stellar mass that ejects mass in AGB phase.
        :type agb_min_mass: float
        :param agb_min_mass: Maximum stellar mass that ejects mass in AGB phase.
        :type agb_min_mass: float
        """
        self.imf = IMF(imf_name, imf_min_mass, imf_max_mass, total_mass)
        self.lifetimes = Lifetimes(lifetimes_name)
        self.sn_ii_model = SNOverall(sn_ii_yields, sn_ii_hn_fraction,
                                     sn_ii_min_mass, sn_ii_max_mass)
        self.sn_ia_model = SNIa(sn_ia_dtd, sn_ia_yields)
        self.agb_model = AGB(agb_yields, agb_min_mass, agb_max_mass)

    def sn_mass_lost(self, element, time_1, time_2, metallicity):
        """
        Calculate the mass loss in SN for a given element.

        This will have units of stellar masses.

        This calculation calculates the mass lost in an element in the given
        timestep by integrating over the IMF-weighted mass lost by stars dying
        in the the time interval (time_2, time_2) to get the total mass loss.

        :param element: Element to get the ejecta for. Can pass "total" to get
                        the total ejecta.
        :param time_1: Starting time boundary for mass loss
        :param time_2: Ending time boundary.
        :param metallicity: Metallicity of the progenitors.
        :return:
        """
        # get the mass limits of the timesteps the user passed in. The
        # lower time corresponds to the higher stellar mass
        m_low = self.lifetimes.turnoff_mass(time_2, metallicity)
        m_high = self.lifetimes.turnoff_mass(time_1, metallicity)

        # we want to integrate the instantaneous mass loss to get the
        # total mass loss, so we define the IMF-weighted mass loss
        def instantaneous_mass_loss(mass):
            m_ej_per_sn = self.sn_ii_model.elemental_ejecta_mass(mass,
                                                                 metallicity,
                                                                 element)
            imf_weight = self.imf.normalized_dn_dm(mass)

            return m_ej_per_sn * imf_weight

        # integrate this between our mass limits
        total_mass_loss = integrate.quad(instantaneous_mass_loss, m_low, m_high)
        return total_mass_loss[0]

    def sn_ejecta_rate(self, element, time, timestep, metallicity):
        """
        Calculate the mass loss rate in SN for a given element.

        This will have units of stellar masses per year.

        This calculation calculates the mass lost in an element in the given
        timestep by integrating over the mass lost by stars dying in the the
        time interval (time, time + timestep) to get the total mass loss, then
        dividing by the time to get the average mass loss rate in this timestep.

        :param element: Element to get the ejecta for. Can pass "total" to get
                        the total ejecta.
        :param time: Time to evaluate the mass loss rate at (in years)
        :param timestep: Timestep used to calculate the rate, as described
                         above, in years.
        :param metallicity: Metallicity of the progenitors.
        :return: Mass loss rate in stellar masses per year.
        """
        # the mss loss rate is the mass lost divided by the timestep. We first
        # get the total mass lost
        mass_lost = self.sn_mass_lost(element, time, time+timestep, metallicity)
        return mass_lost / timestep
