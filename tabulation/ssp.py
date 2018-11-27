from scipy import integrate

from .sn_Ia import SNIa
from .instant import MassiveOverall, AGB
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
        self.sn_ii_model = MassiveOverall(sn_ii_yields, sn_ii_hn_fraction,
                                          sn_ii_min_mass, sn_ii_max_mass)
        self.sn_ia_model = SNIa(sn_ia_dtd, sn_ia_yields)
        self.agb_model = AGB(agb_yields, agb_min_mass, agb_max_mass)
        
    def mass_lost_end_ms(self, element, time_1, time_2, metallicity,
                         source):
        """
        Calculate the mass loss for a given element over a given time from
        ejecta sources that happen at the end of the main sequence.

        This will have units of stellar masses.

        This calculation calculates the mass lost in an element in the given
        timestep by integrating over the IMF-weighted mass lost by stars dying
        in the the time interval (time_2, time_2) to get the total mass loss.

        :param element: Element to get the ejecta for. Can pass "total" to get
                        the total ejecta.
        :param time_1: Starting time boundary for mass loss
        :param time_2: Ending time boundary.
        :param metallicity: Metallicity of the progenitors.
        :param source: What source the elements come from. Can be "AGB" or "SN".
        :return: Mass lost at the end of the main sequence, in solar masses.
        """
        # get the mass limits of the timesteps the user passed in. The
        # lower time corresponds to the higher stellar mass
        m_low = self.lifetimes.turnoff_mass(time_2, metallicity)
        m_high = self.lifetimes.turnoff_mass(time_1, metallicity)

        if source == "AGB":
            model = self.agb_model
        elif source == "SNII":
            model = self.sn_ii_model
        else:
            raise ValueError("This source not supported.")

        # we want to integrate the instantaneous mass loss to get the
        # total mass loss, so we define the IMF-weighted mass loss
        def instantaneous_mass_loss(mass):
            m_ej_per_star = model.elemental_ejecta_mass(mass, metallicity,
                                                        element)
            imf_weight = self.imf.normalized_dn_dm(mass)

            return m_ej_per_star * imf_weight

        # integrate this between our mass limits
        total_mass_loss = integrate.quad(instantaneous_mass_loss, m_low, m_high)
        return total_mass_loss[0]

    def mass_loss_rate_end_ms(self, element, time, timestep, metallicity,
                              source):
        """
        Calculate the mass loss rate for a given element from a given source.

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
        :param source: What source the elements come from. Can be "AGB" or "SN".
        :return: Mass loss rate in stellar masses per year.
        """
        # the mss loss rate is the mass lost divided by the timestep. We first
        # get the total mass lost
        mass_lost = self.mass_lost_end_ms(element, time, time + timestep,
                                          metallicity, source)
        return mass_lost / timestep

    def mass_loss_rate_winds(self, time, metallicity):
        """
        Get the mass loss rate in winds at a given time.

        This will have units of solar masses.

        This calculation assumes that massive stars lose mass at a constant
        rate throughout their main sequence lifetime. The mass loss rate for
        a given stellar mass is therefore just the total mass lost in winds
        (as provided in yield tables) divided by its lifetime. To get the mass
        lost in all stars at a given time, we integrate over all stars active
        at this time, weighted by the IMF.

        :param time: Time at which to get the mass loss rate.
        :param metallicity: Metallicity of the progenitors.
        :return: Mass lost in winds during this time, in solar masses.
        """
        # Define the mass limits of the integral. The upper limit is the star
        # evolving off the main sequence at this time. At early times this is
        # just the maximum mass of the IMF. The lower limit is just the m
        # boundary for SN.
        if time < self.lifetimes.min_time(metallicity):  # check for too early
            m_upper_limit = self.sn_ii_model.sn.mass_boundary_high
        else:  # do the regular comparison
            m_upper_limit = min(self.lifetimes.turnoff_mass(time, metallicity),
                                self.sn_ii_model.sn.mass_boundary_high)
        m_lower_limit = self.sn_ii_model.sn.mass_boundary_low
        if m_upper_limit < m_lower_limit:
            return 0

        # we want to integrate over mass to get the total wind mass rate
        def wind_mass_loss_at_a_mass(mass):
            m_wind = self.sn_ii_model.wind_mass(mass, metallicity)
            wind_time = self.lifetimes.lifetime(mass, metallicity)
            imf_weight = self.imf.normalized_dn_dm(mass)

            return m_wind * imf_weight / wind_time

        # integrate this between our mass limits
        total_mass_loss = integrate.quad(wind_mass_loss_at_a_mass,
                                         m_lower_limit, m_upper_limit)
        return total_mass_loss[0]
