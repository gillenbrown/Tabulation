from scipy import integrate
import numpy as np

from .sn_Ia import SNIa
from .instant import MassiveOverall, AGB
from .lifetimes import Lifetimes
from .imf import IMF


def test_if_between(a, b, test_val):
    """Returns True is test_val is between a and b"""
    if a < b:
        return a <= test_val <= b
    else:
        return b <= test_val <= a


class SSPYields(object):
    """
    Class representing the stellar yields from an SSP
    """
    def __init__(self, imf_name, imf_min_mass, imf_max_mass, total_mass,
                 lifetimes_name,
                 sn_ii_yields, sn_ii_hn_fraction,
                 sn_ii_min_mass, sn_ii_max_mass,
                 sn_ia_dtd, sn_ia_yields, sn_ia_kwargs,
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
        :param sn_ia_kwargs: Dictionary of additional parameters to be passed
                             to the SN Ia DTD function. For the "old art"
                             DTD we need "min_mass", "max_mass", and
                             "exploding_fraction" (the fraction of stars between
                             min_mass and max_mass that explode as SN Ia. For
                             "art power law" we only need the overall number of
                             SNIa that explode in a unit mass SSP:
                             "number_sn_ia"
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
        self.sn_ia_model = SNIa(sn_ia_dtd, sn_ia_yields, self.lifetimes,
                                self.imf, **sn_ia_kwargs)
        self.agb_model = AGB(agb_yields, agb_min_mass, agb_max_mass)

    def _integrate_mass_smart(self, func, lower_mass_limit, upper_mass_limit,
                              source):
        """
        Integrates a function with respect to mass in a smart way that avoids
        discontinuities.

        It does this simply by splitting the integral at locations where there
        are discontinuities in the mass fractions. This happens because we use
        nearest neighbor interpolation.

        :param func: function to integrate
        :param lower_mass_limit: Lower mass limit of the integration
        :param upper_mass_limit: Upper mass limit of the integration
        :param source: What kind of stellar models this uses. Can be either
                       "massive" for the massive star set or "low_mass" for the
                       AGB model set.
        :return:
        """
        if source == "massive":
            masses = self.sn_ii_model.sn.masses
        elif source == "low_mass":
            masses = self.agb_model.masses
        else:
            raise ValueError("source not recognized.")

        # make the boundaries where the mass fractions shift
        boundaries = [np.mean([masses[idx], masses[idx+1]])
                      for idx in range(len(masses) - 1)]

        # we want to split the integral at the discontinuities. Find all the
        # limits we want to integrate between. The edges the user specified
        # are automatically included
        limits = [lower_mass_limit, upper_mass_limit]
        for b in boundaries:
            if test_if_between(lower_mass_limit, upper_mass_limit, b):
                limits.append(b)

        # sort the limits to get them in order for easier use
        limits.sort()

        # then go through 2 at a time and do the integration
        total_integral = 0
        for idx in range(len(limits) - 1):
            m_low = limits[idx]
            m_high = limits[idx+1]

            total_integral += integrate.quad(func, m_low, m_high)[0]

        return total_integral

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
            integrate_source = "low_mass"
        elif source == "SNII":
            model = self.sn_ii_model
            integrate_source = "massive"
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
        return self._integrate_mass_smart(instantaneous_mass_loss,
                                          m_low, m_high,
                                          source=integrate_source)

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
        return self._integrate_mass_smart(wind_mass_loss_at_a_mass,
                                          m_lower_limit, m_upper_limit,
                                          source="massive")

    def mass_loss_rate_snia(self, element, time, metallicity):
        """
        Get the mass loss rate of a given element by SNIa.

        This follows the prescription in ART. There the calculation of the
        amount of metals to add to a cell is
        phi * snIa.metals * stellar_mass_of_particle

        The calculation of snIa.metals means this works out to
        phi * metal_mass_per_sn * number_SNIa

        To turn this into a rate we can use the phi/dt, which is what I have
        coded into my SN Ia class. So this calculation turns out to be simple.
        rate = phi/dt * mass_in_elt_per_sn * number_SNIa

        :param element: Element to get the ejecta for
        :param time: Time at which to get the ejecta
        :param metallicity: Metallicity at which to get the ejecta
        :return: Mass loss rate for the given element by SNIa.
        """
        sn_rate = self.sn_ia_model.sn_dtd(time)
        mass_per_sn = self.sn_ia_model.ejected_mass(element, metallicity)
        return sn_rate * mass_per_sn
