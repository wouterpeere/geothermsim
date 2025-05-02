# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from functools import partial

from jax import Array, jit, vmap


class _GroundHeatExchanger(ABC):
    """Ground heat exchanger cross-section.

    Attributes
    ----------
    n_pipes : int
        Number of pipes.

    """

    @abstractmethod
    def thermal_resistances(self, m_flow: float) -> Array:
        """Evaluate delta-circuit thermal resistances.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of delta-circuit thermal
            resistances (in m-K/W).

        """
        ...
