import numpy as np
import yields
from scipy import interpolate

elts = ["He", "C", "N", "O", "Fe"]

class Instantaneous_Ejecta(object):
    """
    Parent class for objects that eject everything the instant they leave the
    main sequence.
    """


    def __init__(self):
        self.masses = []
        self.metallicities = []

        self.models = dict()  # keys of mass, values of Yield objects
        # we then have a bunch of information about the yields. For my
        # formalism, the keys here are the metallicities, and the values
        # will be functions of mass
        # First is mass fractions, which holds the fraction of the total ejected
        # mass in a given element.
        self.mass_fracs = {elt: dict() for elt in elts}
        # then the SN ejecta, which is just the ejected mass
        self.ejecta = lambda m, z: 0

    def _make_ejecta_mass_interps(self, mass_boundary_low, mass_boundary_high):
        # requires models to already exist
        interps = dict()
        for z in self.metallicities:
            # Make the interpolation object for the place where we have models
            ejecta_masses = [self.models[m].total_end_ejecta[z]
                             for m in self.masses]
            interp = interpolate.interp1d(x=self.masses, y=ejecta_masses,
                                          kind="linear")
            interps[z] = interp

        def fill_func_low(mass, z):
            low_model = self.models[min(self.masses)]
            frac_lost_low = low_model.total_end_ejecta[z] / low_model.mass
            return mass * frac_lost_low

        def fill_func_high(mass, z):
            high_model = self.models[max(self.masses)]
            frac_lost_high = high_model.total_end_ejecta[
                                 z] / high_model.mass
            return mass * frac_lost_high

        def ejecta_mass_temp(mass, z):
            if mass < mass_boundary_low:
                return 0
            elif mass > mass_boundary_high:
                return 0
            elif mass < min(self.masses):
                return fill_func_low(mass, z)
            elif mass > max(self.masses):
                return fill_func_high(mass, z)
            else:  # this is where we have good results, so use them.
                return interps[z](mass)


        self.ejecta = ejecta_mass_temp

    def _make_mass_fractions(self):
        for elt in elts:
            for z in self.metallicities:
                # Make the interpolation object for where we have models
                mass_fracs = [self.models[m].mass_fraction(elt, z,
                                                           metal_only=False)[0]
                              for m in sorted(self.masses)]

                fill_values = (mass_fracs[0], mass_fracs[-1])
                interp = interpolate.interp1d(x=self.masses, y=mass_fracs,
                                              kind="nearest",
                                              fill_value=fill_values,
                                              bounds_error=False)
                self.mass_fracs[elt][z] = interp

    def get_mass_fractions(self, element, metallicity, mass):
        return self.mass_fracs[element][metallicity](mass)



class SN_II(Instantaneous_Ejecta):
    def __init__(self, name):
        self.winds = lambda m, z: 0

        super().__init__()
        if name.lower() == "kobayashi_06_sn":
            self.masses = [13, 15, 18, 20, 25, 30, 40]
            self.metallicities = [0, 0.001, 0.004, 0.02]
            label_fmt = "kobayashi_06_II_{}"
        elif name.lower() == "kobayashi_06_hn":
            self.masses = [20, 25, 30, 40]
            self.metallicities = [0, 0.001, 0.004, 0.02]
            label_fmt = "kobayashi_06_II_{}_hn"
        else:
            raise ValueError("SN not recognized.")

        for m in self.masses:
            self.models[m] = yields.Yields(label_fmt.format(m))

        self._make_ejecta_mass_interps(8, 1000)
        self._make_wind_masses(8, 1000)
        self._make_mass_fractions()


    def _make_wind_masses(self, mass_boundary_low, mass_boundary_high):
        # requires models to already exist
        interps = dict()
        for z in self.metallicities:
            # Make the interpolation object for the place where we have models
            ejecta_masses = [self.models[m].wind_ejecta[z]
                             for m in self.masses]
            interp = interpolate.interp1d(x=self.masses, y=ejecta_masses,
                                          kind="linear")
            interps[z] = interp

        def fill_func_low(mass, z):
            low_model = self.models[min(self.masses)]
            frac_lost_low = low_model.wind_ejecta[z] / low_model.mass
            return mass * frac_lost_low

        def fill_func_high(mass, z):
            high_model = self.models[max(self.masses)]
            frac_lost_high = high_model.wind_ejecta[z] / high_model.mass
            return mass * frac_lost_high

        def ejecta_mass_temp(mass, z):
            if mass < mass_boundary_low:
                return 0
            elif mass > mass_boundary_high:
                return 0
            elif mass < min(self.masses):
                return fill_func_low(mass, z)
            elif mass > max(self.masses):
                return fill_func_high(mass, z)
            else:  # this is where we have good results, so use them.
                return interps[z](mass)

        self.winds = ejecta_mass_temp

class AGB(Instantaneous_Ejecta):
    def __init__(self, name):
        super().__init__()
        if name.lower() == "nugrid":
            self.masses = [1, 1.65, 2, 3, 4, 5, 6, 7]
            self.metallicities = [0.0001, 0.001, 0.006, 0.01, 0.02]
            label_fmt = "nugrid_agb_{}"
        else:
            raise ValueError("AGB not recognized.")

        for m in self.masses:
            self.models[m] = yields.Yields(label_fmt.format(m))

        self._make_ejecta_mass_interps(0, 8)
        self._make_mass_fractions()




class SN_Overall(object):
    def __init__(self, name, hn_fraction):
        self.name = name
        if name.lower() == "kobayashi_06":
            self.sn = SN_II("kobayashi_06_sn")
            self.hn = SN_II("kobayashi_06_hn")
        else:
            raise ValueError("SN II not supported")

        if hn_fraction < 0 or hn_fraction > 1:
            raise ValueError("Hypernova fraction must be between 0 and 1.")


        self._hn_fraction = hn_fraction
        self._sn_fraction = 1.0 - hn_fraction

    def hn_fraction(self, mass):
        if mass >= min(self.hn.masses):
            return self._hn_fraction
        else:
            return 0

    def sn_fraction(self, mass):
        if mass >= min(self.hn.masses):
            return self._sn_fraction
        else:
            return 1

    def get_elemental_ejecta_mass(self, mass, metallicity, element):
        m_ej_hn = self.hn.ejecta(mass, metallicity)
        m_ej_sn = self.sn.ejecta(mass, metallicity)

        f_elt_hn = self.hn.get_mass_fractions(element, metallicity, mass)
        f_elt_sn = self.sn.get_mass_fractions(element, metallicity, mass)

        hn_term = self.hn_fraction(mass) * m_ej_hn * f_elt_hn
        sn_term = self.sn_fraction(mass) * m_ej_sn * f_elt_sn

        return hn_term + sn_term
