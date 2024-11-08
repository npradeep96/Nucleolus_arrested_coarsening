"""Module that contains classes to implement different reaction rates
"""

import numpy as np
import fipy as fp


class FirstOrderReaction(object):
    """Rate law for a simple first order reaction with constant rate coefficient

    .. math::
        rate(c) = k c
    """

    def __init__(self, k):
        """Initialize an object of :class:`first_order_reaction`.

        Args:
             k (float): Rate constant for the first order reaction
        """
        self._k = k

    def rate(self, concentration):
        """Calculate and return the reaction rate given a concentration value.

        Args:
            concentration (fipy.CellVariable): Concentration variable

        Returns:
             reaction_rate (fipy.ImplicitSourceTerm): Reaction rate
        """
        # return self._k * concentration
        return fp.ImplicitSourceTerm(coeff=self._k, var=concentration)