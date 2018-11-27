import yields
from scipy import interpolate

elts = ["He", "C", "N", "O", "Fe", "total_metals"]


class InstantaneousEjecta(object):
    """
    Parent class for objects that eject everything the instant they leave the
    main sequence.
    """
    def __init__(self, mass_boundary_low, mass_boundary_high):
        """
        Initialize the basic attributes of the class.

        :param mass_boundary_low: Mass below which there are zero ejecta. This
                                  is best throught of as a separation between
                                  SN and AGB.
        :param mass_boundary_high: Mass above which there are zero ejecta. This
                                   is best throught of as a separation between
                                   SN and AGB, or the maximum SN mass.
        """

        self.masses = []  # masses of the stellar models
        self.metallicities = []  # metallicities at which we have models.

        self.mass_boundary_low = mass_boundary_low
        self.mass_boundary_high = mass_boundary_high

        # check that mass boundaries aren't inconsistent
        if self.mass_boundary_low >= self.mass_boundary_high:
            raise ValueError("The minimum mass needs to be smaller than the"
                             "maximum mass for a yield set.")

        self.models = dict()  # keys of mass, values of Yield objects
        # we then have a bunch of information about the yields.
        # First is mass fractions, which holds the fraction of the total ejected
        # mass in a given element. The keys here are the elements, and the
        # values are dictionaries, which themselves will have keys of
        # metallicity and values that are interpolation objects
        self._mass_fracs = {elt: dict() for elt in elts}
        # then is the interpolation object to get the total ejecta. This has
        # keys of metallicities, and values that are interpolation objects
        self._ejecta_interp = dict()

    def _make_ejecta_mass_interps(self):
        """
        Make the interpolation objects needed for the calculation of ejecta mass

        This requires that models already exist, so it can't be done in the
        parent class constructor.

        This is done only in the range where stellar models exist. The rest of
        the handling is taken care of in the `ejecta` function.

        :return: None
        """
        for z in self.metallicities:
            # get the ejected mass for all models
            ejecta_masses = [self.models[m].total_end_ejecta[z]
                             for m in self.masses]
            # then interpolate between them. We do not want to allow
            # extrapolation, although this should only be a check since we
            # should do that if statement elsewhere.
            interp = interpolate.interp1d(x=self.masses, y=ejecta_masses,
                                          kind="linear", bounds_error=True)
            self._ejecta_interp[z] = interp

    def ejecta(self, mass, z):
        """
        Calculate the ejected mass for a given stellar mass and metallicity.

        This represents the mass lost at the instant the star leaves the main
        sequence, so this would be the SN explosion for massive stars and
        AGB winds for lower mass stars. Massive star winds are not included
        in this.

        If the mass requested is outside the allowed mass range for this type
        of object, the ejecta will be zero. If it is within the stellar mass
        range specified by the models, the total ejected mass will be
        interpolated between the stellar models. In the range outside the range
        of the models but inside the allowed mass range, we do a slightly more
        complicated thing. We calculate the fraction of the total stellar mass
        ejected for the extremal (largest or smallest) stellar model. This is
        then multiplied by the mass requested. This is a way of extrapolating
        that makes the least assumptions. It extends the model without actually
        extrapolating any quantity from the models.

        :param mass: Mass of the star.
        :param z: Metallicity of the star.
        :return: Total mass ejected at the end of the main sequence.
        """
        # if outside the allowed range, return zero.
        if mass < self.mass_boundary_low:
            return 0
        elif mass > self.mass_boundary_high:
            return 0
        # if in the region inside the allowed range but outside the range
        # where models exist, we do the thing described in the docstring.
        elif mass < min(self.masses):
            # get the smallest mass model.
            low_model = self.models[min(self.masses)]
            # get the fraction of the stellar mass that was ejected.
            frac_low = low_model.total_end_ejecta[z] / low_model.mass
            # then say the ejected mass is this fraction times the mass of the
            # new star desired by the user.
            return mass * frac_low
        elif mass > max(self.masses):  # similar to last elif
            high_model = self.models[max(self.masses)]
            frac_high = high_model.total_end_ejecta[z] / high_model.mass
            return mass * frac_high
        else:  # this is where we have good results, so use them.
            return self._ejecta_interp[z](mass)

    def _make_mass_fractions(self):
        """
        Make the mass fraction interpolation objects.

        Here mass fraction is the fraction of the ejected mass that is in
        a given element: M_{element,ejected} / M_{tot,ejected}

        This requires that the models already exist, so it can't be done in the
        parent class constructor.

        We do simple nearest neighbor interpolation. In the range outside the
        range of the stellar models, we just use the mass fraction of the
        extremal (biggest or smallest) model. This is fine since we will have
        the total ejecta be larger or smaller. This method extends the model
        without extrapolating it.

        :return: None
        """
        for elt in elts:
            for z in self.metallicities:
                # get the mass fraction for all models.
                if elt == "total_metals":
                    mass_fracs = []
                    for m in sorted(self.masses):
                        self.models[m].set_metallicity(z)
                        met_ejecta = self.models[m].ejecta_sum(metal_only=True)
                        tot_ejecta = self.models[m].ejecta_sum(metal_only=False)
                        mass_fracs.append(met_ejecta / tot_ejecta)
                else:
                    # The mass fraction function returns an array. The [0]
                    # index gets rid of that.
                    mass_fracs = [self.models[m].mass_fraction(elt, z, False)[0]
                                  for m in sorted(self.masses)]

                # outside the range, just use extremal values.
                fill_values = (mass_fracs[0], mass_fracs[-1])
                # make the interpolation object, as described above. We let it
                # fill values outside the range using extremal values.
                interp = interpolate.interp1d(x=self.masses, y=mass_fracs,
                                              kind="nearest",
                                              fill_value=fill_values,
                                              bounds_error=False)
                self._mass_fracs[elt][z] = interp

    def mass_fractions(self, element, metallicity, mass):
        """
        Get the ejected mass fraction for an element for a stellar model at a
        given stellar mass and metallicity.

        Here mass fraction is the fraction of the ejected mass that is in
        a given element: M_{element,ejected} / M_{tot,ejected}

        We do simple nearest neighbor interpolation. In the range outside the
        range of the stellar models, we just use the mass fraction of the
        extremal (biggest or smallest) model. This is fine since we will have
        the total ejecta be larger or smaller. This method extends the model
        without extrapolating it.

        :param element: Element desired.
        :param metallicity: Metalliicty of stellar model
        :param mass: Mass of stellar model
        :return: Mass fraction of the element in the given stellar model.
        :rtype: float
        """
        # simply call the interpolation objects we made earlier.
        return self._mass_fracs[element][metallicity](mass)

    def elemental_ejecta_mass(self, mass, metallicity, element):
        """
        Get the mass of a given element ejected at given stellar mass and
        metallicity.

        The mass of an ejected element is the total ejected mass times the
        mass fraction of that element.

        :param mass: Stellar mass of the supernova progenitor.
        :param metallicity: Metallicity of the supernova progenitor.
        :param element: Element to get the ejected mass of. To get the total
                        mass ejected, pass "total", which will use 1 for the
                        mass fractions above.
        :return: Ejected mass of that element.
        """
        # get the mass fractions in a given element
        if element == "total":
            frac = 1
        else:
            frac = self.mass_fractions(element, metallicity, mass)
        eject = self.ejecta(mass, metallicity)
        return frac * eject


class SNII(InstantaneousEjecta):
    """
    Subclass specific to supernovae, which also have stellar winds pre-explosion
    """
    def __init__(self, name, min_mass, max_mass):
        """
        Initialize the SNII object.

        :param name: Name of the SN yield set
        :param min_mass: Minimum mass that explodes as SN
        :param max_mass: Maximum mass that explodes as SN
        """
        # initialize the wind model
        self._wind_interps = dict()
        # then call the parent class constructor
        super().__init__(min_mass, max_mass)

        # parse the model set
        if name.lower() == "kobayashi_06_sn":
            self.masses = [13, 15, 18, 20, 25, 30, 40]  # stellar masses
            self.metallicities = [0, 0.001, 0.004, 0.02]
            label_fmt = "kobayashi_06_II_{}"  # format will be filled with mass
        elif name.lower() == "kobayashi_06_hn":
            self.masses = [20, 25, 30, 40]  # stellar masses
            self.metallicities = [0, 0.001, 0.004, 0.02]
            self.mass_boundary_low = min(self.masses)
            label_fmt = "kobayashi_06_II_{}_hn"
        else:
            raise ValueError("SN not recognized.")

        # make the SN models
        for m in self.masses:
            self.models[m] = yields.Yields(label_fmt.format(m))

        # then fill the interpolation objects
        self._make_ejecta_mass_interps()
        self._make_mass_fractions()
        self._make_wind_masses()

    def winds(self, mass, z):
        """
        Calculate the ejected mass in winds for a given stellar mass and
        metallicity.

        This represents the total mass of winds that were ejected throughout
        the star's lifetime. No time resolution is considered.

        If the mass requested is outside the allowed mass range for this type
        of object, the ejecta will be zero. If it is within the stellar mass
        range specified by the models, the total wind ejected mass will be
        interpolated between the stellar models. In the range outside the range
        of the models but inside the allowed mass range, we do a slightly more
        complicated thing. We calculate the fraction of the total stellar mass
        ejected as windsfor the extremal (largest or smallest) stellar model.
        This is then multiplied by the mass requested. This is a way of
        extending that makes the least assumptions. It extends the model
        without actually extrapolating any quantity from the models.

        :param mass: Mass of the star.
        :param z: Metallicity of the star.
        :return: Total mass ejected in the winds throughout the star's lifetime
        """
        # if outside the allowed range, return zero.
        if mass < self.mass_boundary_low:
            return 0
        elif mass > self.mass_boundary_high:
            return 0
        # if in the region inside the allowed range but outside the range
        # where models exist, we do the thing described in the docstring.
        elif mass < min(self.masses):
            # get the lowest mass model
            low_model = self.models[min(self.masses)]
            # get the fraction of mass lost as winds
            frac_lost_low = low_model.wind_ejecta[z] / low_model.mass
            # then say the ejected mass is this fraction times the mass of the
            # new star desired by the user.
            return mass * frac_lost_low
        elif mass > max(self.masses):   # similar to last elif
            high_model = self.models[max(self.masses)]
            frac_lost_high = high_model.wind_ejecta[z] / high_model.mass
            return mass * frac_lost_high
        else:  # this is where we have good results, so use them.
            return self._wind_interps[z](mass)

    def _make_wind_masses(self):
        """
        Make the interpolation objects needed for the calculation of wind mass

        This requires that models already exist, so it can't be done in the
        parent class constructor.

        This is done only in the range where stellar models exist. The rest of
        the handling is taken care of in the `winds` function.

        :return: None
        """
        for z in self.metallicities:
            # Make the interpolation object for the place where we have models
            ejecta_masses = [self.models[m].wind_ejecta[z]
                             for m in self.masses]
            # then interpolate between them. We do not want to allow
            # extrapolation, although this should only be a check since we
            # should do that if statement elsewhere.
            interp = interpolate.interp1d(x=self.masses, y=ejecta_masses,
                                          kind="linear", bounds_error=True)
            self._wind_interps[z] = interp


class AGB(InstantaneousEjecta):
    """
    Class for AGB stars. Only handles the parsing of models.
    No special methods are added.
    """
    def __init__(self, name, min_mass, max_mass):
        """
        Initialize the AGB object.

        :param name: Name of the AGB yield set
        :param min_mass: Minimum mass that goes as AGB.
        :param max_mass: Maximum mass that is an AGB.
        """
        super().__init__(min_mass, max_mass)
        # handle the NuGrid model set
        if name.lower() == "nugrid":
            self.masses = [1, 1.65, 2, 3, 4, 5, 6, 7]
            self.metallicities = [0.0001, 0.001, 0.006, 0.01, 0.02]
            label_fmt = "nugrid_agb_{}"
        else:
            raise ValueError("AGB not recognized.")

        # make the models
        for m in self.masses:
            self.models[m] = yields.Yields(label_fmt.format(m))

        # then fill the interpolation object
        self._make_ejecta_mass_interps()
        self._make_mass_fractions()


class MassiveOverall(object):
    """
    Class handling the total SN ejecta, which can consist of both supernovae
    and hypernovae.
    """
    def __init__(self, name, hn_fraction, min_mass, max_mass):
        """
        Initialize the SN combined set.

        :param name: Name of the SN model set. One set will be used for both
                     SN and HN.
        :param hn_fraction: Fraction of massive stars above the minimum mass
                            for HN (which is set by the yield set) that explode
                            as HN. The rest explode as SN.
        :param min_mass: Minimum stellar mass that explodes as SN.
        :param max_mass: Maximum stellar mass that explodes as SN.
        """
        self.name = name
        # Parse the yield set
        if name.lower() == "kobayashi_06":
            # make separate objects for both SN and HN
            self.sn = SNII("kobayashi_06_sn", min_mass, max_mass)
            self.hn = SNII("kobayashi_06_hn", min_mass, max_mass)
        else:
            raise ValueError("SN II not supported")

        # check that the hypernovae fraction is appropriate
        if hn_fraction <= 0 or hn_fraction >= 1:
            raise ValueError("Hypernova fraction must be between 0 and 1.")

        # set the variable for the hypernova fraction. This shouldn't be used
        # by the user, the function that makes the mass dependence is what
        # should be used.
        self._hn_fraction = hn_fraction
        self._sn_fraction = 1.0 - hn_fraction

    def hn_fraction(self, mass):
        """
        Handles the mass dependent hypernova fraction.

        This is a piecewise function. The split happens at the mass of the
        lowest mass hypernova model. Below this mass, the hypernova fraction
        is zero. At and above this mass, the hypernova fraction will be the
        single value set by the user.

        :param mass: Mass at which to get the hypernova fraction.
        :return: Value of the hypernova fraction at a given mass.
        """
        # above the minimum mass HN model, return the value the user set
        if mass >= min(self.hn.masses):
            return self._hn_fraction
        else:  # below this, return zero.
            return 0

    def sn_fraction(self, mass):
        """
        Analagous to the `hn_fraction` function, just for SN. This will be
        `1 - hn_fraction(mass)` at all masses.

        :param mass: Mass at which to get the SN fraction.
        :return: Value of the supernova fraction at a given mass.
        """
        return 1 - self.hn_fraction(mass)

    def elemental_ejecta_mass(self, mass, metallicity, element):
        """
        Get the mass of a given element ejected by supernova and hypernova
        of a given stellar mass and metallicity.

        This is calculated in the following way:
        The mass of an ejected element in a given supernova is the total
        ejected mass times the mass fraction of that element. This is calculated
        separately for both SN and HN. To combine these, we multiply the SN
        fraction times the SN ejected elemental mass, and similarly for the HN
        fraction. Then these are added together. Mathematically, this is:
        f_{SN} * M_{ej,total,SN} * f_{elt,SN} +
        f_{HN} * M_{ej,total,HN} * f_{elt,HN}

        This goes directly into the calculation of the time-resolved yields that
        are tabulated.

        :param mass: Stellar mass of the supernova progenitor.
        :param metallicity: Metallicity of the supernova progenitor.
        :param element: Element to get the ejected mass of. To get the total
                        mass ejected, pass "total", which will use 1 for the
                        mass fractions above.
        :return: Ejected mass of that element.
        """
        # get the SN and HN terms
        sn_ejected_mass = self.sn.elemental_ejecta_mass(mass, metallicity,
                                                        element)
        hn_ejected_mass = self.hn.elemental_ejecta_mass(mass, metallicity,
                                                        element)

        # then construct the total for each type of supernova
        hn_term = self.hn_fraction(mass) * hn_ejected_mass
        sn_term = self.sn_fraction(mass) * sn_ejected_mass

        return hn_term + sn_term

    def wind_mass(self, mass, metallicity):
        """
        Get the total mass ejected as winds by massive stars of a given stellar
        mass and metallicity.

        This is calculated in the following way:
        The mass of an ejected element in a given supernova is the total
        ejected mass times the mass fraction of that element. This is calculated
        separately for both SN and HN. To combine these, we multiply the SN
        fraction times the SN ejected elemental mass, and similarly for the HN
        fraction. Then these are added together. Mathematically, this is:
        f_{SN} * M_{ej,total,SN} * f_{elt,SN} +
        f_{HN} * M_{ej,total,HN} * f_{elt,HN}

        This goes directly into the calculation of the time-resolved yields that
        are tabulated.

        :param mass: Stellar mass of the supernova progenitor.
        :param metallicity: Metallicity of the supernova progenitor.
        :return: Ejected mass of that element.
        """
        # get the SN and HN terms
        sn_ejected_mass = self.sn.winds(mass, metallicity)
        hn_ejected_mass = self.hn.winds(mass, metallicity)

        # then construct the total for each type of supernova
        hn_term = self.hn_fraction(mass) * hn_ejected_mass
        sn_term = self.sn_fraction(mass) * sn_ejected_mass

        return hn_term + sn_term
