# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

from jax import numpy as jnp
from jax import Array, jit


class _TemporalSuperposition(ABC):
    """Temporal superposition of thermal response factors.

    """

    @abstractmethod
    def next_time_step(self) -> float:
        """Advance to next simulation time step.

        Returns
        -------
        float
            Time (in seconds) of the new time step.

        """
        ...

    @abstractmethod
    def reset_history(self):
        """Reset the history to its initial condition.

        """
        ...

    @abstractmethod
    def set_current_load(self, q: Array):
        """Set the current heat extraction rate.

        Parameters
        ----------
        q : array
            (`n_boreholes`, `n_nodes`,) array of heat extraction rates
            (in W/m).

        """
        ...

    @abstractmethod
    def temperature(self) -> Array:
        """Evaluate temperatures at nodes.

        Returns
        ----------
        T_b : array
            (`n_boreholes`, `n_nodes`,) array of borehole wall
            temperature variations (in degree Celcius).

        """
        ...

    @abstractmethod
    def temperature_to_point(self) -> Array:
        """Evaluate temperatures at point.

        Returns
        ----------
        T : array
            (`n_points`,) array of ground temperature variations
            (in degree Celcius).

        """
        ...

    @staticmethod
    @jit
    def _temperature(h_to_self: Array, q: Array) -> Array:
        """Spatial and temporal superposition at nodes.

        Parameters
        ----------
        h_to_self : array
            (`n_times`, `n_boreholes`, `n_nodes`, `n_boreholes`,
            `n_nodes`,) array of thermal response factors.
        q : array
            (`n_times`, `n_boreholes`, `n_nodes`,) array of heat
            extraction rates (in W/m).

        Returns
        ----------
        T_b : array
            (`n_boreholes`, `n_nodes`,) array of borehole wall
            temperatures (in degree Celcius).

        """
        T = jnp.tensordot(h_to_self, q, axes=([0, -2, -1], [0, -2, -1]))
        return T

    @staticmethod
    @jit
    def _temperature_to_point(h_to_point: Array, q: Array) -> Array:
        """Spatial and temporal superposition at points.

        Parameters
        ----------
        h_to_point : array
            (`n_times`, `n_points`, `n_boreholes`, `n_nodes`,) array of
            thermal response factors.
        q : array
            (`n_times`, `n_boreholes`, `n_nodes`,) array of heat
            extraction rates (in W/m).

        Returns
        ----------
        T : array
            (`n_points`,) array of ground temperatures (in degree
            Celcius).

        """
        T = jnp.tensordot(h_to_point, q, axes=([0, -2, -1], [0, -2, -1]))
        return T
