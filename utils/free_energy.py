"""Module that contains classes describing different free energies.
"""

import numpy as np
import fipy as fp


class DoubleWellExpansion(object):
    """Free energy of three component system with a double well expansion about critical point.

    This class describes the free energy density of a two component system given by the below expression:

    .. math::

       f[c_1, c_2] =  \\frac{1}{6 \\bar{c}_1^3} (c_1-\\bar{c}_1)^4 - \\beta(c_1-\\bar{c}_1)^2 + \\beta w^2 |\\nabla c_1|^2 
                    + c_2 \log c_2 + \\chi c_1 c_2 

    Interactions between molecules of species 1 are described by a quartic-well potential. If :math:`\\beta > 2.0`, then
    we get a double-well and species 1 can phase separate by itself.

    The cross interactions between the species are described by a mean-field product of concentrations with the
    interaction strength captured by a Flory parameter :math:`\\chi`
    
    The self interactions between molecules of species two are repulsive, drive by the entropy term with a degree of 
    polymerization that is :math:`N_2`
    
    """

    def __init__(self, beta, chi, N2, w2, c_bar_1):
        """Initialize an object of :class:`TwoCompDoubleWellFHCrossQuadratic`.

        Args:
            beta (float): Parameter associated with the quadratic term :math:`- \\beta (c_1-\\bar{c}_1)^2` of species 1

            chi (float): Parameter that describes the cross-interactions between the species :math:`\\chi c_1 c_2`

            N2 (float): Parameter that describes the degree of polymerization of species 2

            w2 (float): Parameter that describes the interface width in the surface tension associated with species 1 
            :math:`\\beta w^2 |\\nabla c_1|^2`

            c_bar_1 (float): Critical concentration of species 1 at the onset of phase separation
        """

        # Ensure that the parameter N_2 and w2 are always positive
        # Otherwise, we will get nonsense results in the simulations
        assert N2 > 0, "The parameter N_2 is negative. Please supply a positive value"
        assert w2 > 0, "The parameter w2 is negative. Please supply a positive value"

        # Assign all free energy parameters to private variables
        self._beta = beta
        self._chi = chi
        # self._N2 = N2
        self._w2 = w2
        self._c_bar_1 = c_bar_1
        # Define a surface energy parameter that is just the product of beta and w2
        self._kappa = 2.0*self._beta*self._w2

    @property
    def kappa(self):
        """Getter for the private variable self._kappa.
        This is used to set up the surface tension term in the dynamical equations"""
        return self._kappa

    def calculate_fe(self, c_vector):
        """Calculate free energy according to the expression in class description.

        Args:
            c_vector (numpy.ndarray): A 2x1 vector of species concentrations that looks like :math:`[c_1, c_2]`.
            The concentration variables :math:`c_1` and :math:`c_2` must be instances of the class
            :class:`fipy.CellVariable` or equivalent. These instances should have an attribute called :attr:`.grad.mag`
            that returns the magnitude of gradient of the concentration field for every position in the mesh to compute
            the surface tension contribution of the free energy

        Returns:
            free_energy (fipy.cellVariable): Free energy value
        """

        # Check that c_vector satisfies the necessary conditions
        assert len(c_vector) == 2, \
            "The shape of c_vector passed to TwoCompDoubleWellFHCrossQuadratic.calculate_fe() is not 2x1"
        assert hasattr(c_vector[0], "grad"), \
            "The instance c_vector[0] has no attribute grad associated with it"
        assert hasattr(c_vector[1], "grad"), \
            "The instance c_vector[1] has no function grad associated with it"
        assert hasattr(c_vector[0].grad, 'mag'), \
            "The instance c_vector[0].grad has no attribute mag associated with it"
        assert hasattr(c_vector[1].grad, 'mag'), \
            "The instance c_vector[1].grad has no attribute mag associated with it"

        # Calculate the free energy
        fe = (1.0/(6.0*self._c_bar_1**3) * (c_vector[0] - self._c_bar_1) ** 4
              + 1.0/(15.0*self._c_bar_1**5) * (c_vector[0] - self._c_bar_1) ** 6
              - self._beta * (c_vector[0] - self._c_bar_1) ** 2  
              + self._chi * c_vector[0] * c_vector[1]
              + c_vector[1] * np.log(c_vector[1])
              + 0.5 * self._kappa * c_vector[1].grad.mag ** 2)

        return fe

    def calculate_mu(self, c_vector):
        """Calculate chemical potential of the species.

        Chemical potential of species 1:

        .. math::

            \\mu_1[c_1, c_2] = \\delta f / \\delta c_1 = 2.0/(3.0 \\bar{c}_1^3) (c_1-\\bar{c}_1)^3 - 2.0 (\\beta) (c_1-\\bar{c}_1)
                               + \\chi c_2 - 2.0 \\beta w^2 \\nabla^2 c_1

        Chemical potential of species 2:

        .. math::

            \\mu_2[c_1, c_2] = \\delta f / \\delta c_2 = \\chi c_1 + \\log c_2


        Args:
            c_vector (numpy.ndarray): A 2x1 vector of species concentrations that looks like :math:`[c_1, c_2]`. The
            concentration variables :math:`c_1` and :math:`c_2` must be instances of the class
            :class:`fipy.CellVariable` or equivalent. These instances should have an attribute called
            :attr:`.faceGrad.divergence` that returns the Laplacian of the concentration field for every position in the
            mesh to compute the surface tension contribution to the chemical potential of species 1

        Returns:
            mu (list): A 2x1 vector of chemical potentials that looks like :math:`[\\mu_1, \\mu_2]`
        """

        # Check that c_vector satisfies the necessary conditions
        assert len(c_vector) == 2, \
            "The shape of c_vector passed to TwoCompDoubleWellFHCrossQuadratic.calculate_mu() is not 2x1"
        assert hasattr(c_vector[0], "faceGrad"), \
            "The instance c_vector[0] has no attribute faceGrad associated with it"
        assert hasattr(c_vector[1], "faceGrad"), \
            "The instance c_vector[1] has no attribute faceGrad associated with it"
        assert hasattr(c_vector[0].faceGrad, "divergence"), \
            "The instance c_vector[0].faceGrad has no attribute divergence associated with it"
        assert hasattr(c_vector[1].faceGrad, "divergence"), \
            "The instance c_vector[1].faceGrad has no attribute divergence associated with it"

        # Calculate the chemical potentials
        mu_1 = (2.0/(3.0*self._c_bar_1**3) * (c_vector[0] - self._c_bar_1) ** 3
                + 2.0/(5.0*self._c_bar_1**5) * (c_vector[0] - self._c_bar_1) ** 5
                - 2.0 * (self._beta) * (c_vector[0] - self._c_bar_1)
                + self._chi * c_vector[1]
                - self._kappa * c_vector[0].faceGrad.divergence)
        mu_2 = self._chi * c_vector[0] + np.log(c_vector[1])
        mu = [mu_1, mu_2]

        return mu

    def calculate_osmotic_pressure(self, c_vector):
        """Calculate the osmotic pressure at each position in the domain

        The expression for osmotic pressure is

        .. math::

            \\Pi[c_1, c_2] = f[c_1, c_2] - \\mu_1 c_1 - \\mu_2 c_2

        Args:
            c_vector (numpy.ndarray): A 2x1 vector of species concentrations that looks like :math:`[c_1, c_2]`. The
            concentration variables :math:`c_1` and :math:`c_2` must be instances of the class
            :class:`fipy.CellVariable` or equivalent. 

        Returns:
            osmotic_pressure (list): An instance of the class of the class :class:`fipy.CellVariable` which has same
            dimensions as :math:`c_1` and :math:`c_2`
        """

        # Check that c_vector satisfies the necessary conditions
        assert len(c_vector) == 2, \
            "The shape of c_vector passed to TwoCompDoubleWellFHCrossQuadratic.calculate_mu() is not 2x1"

        # Calculate the osmotic pressure
        mu = self.calculate_mu(c_vector)
        f = self.calculate_fe(c_vector)
        osmotic_pressure = f - c_vector[0]*mu[0] - c_vector[1]*mu[1]

        return osmotic_pressure
    
    def calculate_pre_factor(self, c_vector):
        """Calculate the pre-factors that multiply the diffusivity of each species
        
        For the species 1 which is concentrated, pre-factor = c_0 * (c_bar_1 - c_0)
        For the species 2 which is dilute, pre-factor = c_1
        """
        
        pre_factor = [c_vector[0]*(self._c_bar_1 - c_vector[0])/self._c_bar_1, c_vector[1]]
        return pre_factor
    
    def calculate_jacobian(self, c_vector):
        """Calculate the Jacobian matrix of coefficients to feed to the transport equations.

        In calculating the Jacobian, we ignore the surface tension and only take the bulk part of the free energy 
        that depends on the concentration fields:

        .. math::
            J_{11} = \\delta^2 f_{bulk} / \\delta c^2_1 = 

        .. math::
            J_{12} = \\delta^2 f_{bulk} / \\delta c_1 \\delta c_2 = \\chi

        .. math::
            J_{21} = \\delta^2 f_{bulk} / \\delta c_1 \\delta c_2 = \\chi

        .. math::
            J_{22} = \\delta^2 f_{bulk} / \\delta c^2_2 = \\frac{1.0}{c_2}

        Args:
            c_vector (numpy.ndarray): A 2x1 vector of species concentrations that looks like :math:`[c_1, c_2]`.The
            concentration variables :math:`c_1` and :math:`c_2` must be instances of the class
            :class:`fipy.CellVariable` or equivalent
        Returns:
            jacobian (numpy.ndarray): A 2x2 Jacobian matrix, with each entry itself being a vector of the same size as
            c_vector[0]
        """

        # Check that c_vector satisfies the necessary conditions
        assert len(c_vector) == 2, \
            "The shape of c_vector passed to TwoCompDoubleWellFHCrossQuadratic.calculate_mu() is not 2x1"
        
        # Calculate the Jacobian matrix
        # jacobian = np.array([[2.0/self._c_bar_1**3 * (c_vector[0] - self._c_bar_1) ** 2 - 2.0 * self._beta, self._chi],
        #                      [self._chi, 1.0]])
        # jacobian = np.array([[2.0*self._c_bar_1/((2.0*self._c_bar_1 - c_vector[0])*c_vector[0]) - 2.0 * self._beta, self._chi],
        #                      [self._chi, 1.0]])
        jacobian = np.array([[self._c_bar_1/(c_vector[0]*(self._c_bar_1 - c_vector[0])) - 2.0 * self._beta, 
                              self._chi], 
                             [self._chi, 1.0]])
        return jacobian