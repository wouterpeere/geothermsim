# -*- coding: utf-8 -*-
from time import perf_counter

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from ..borefield.borefield import Borefield
from ..borefield.network import Network
from ._temporal_superposition import _TemporalSuperposition


class LoadHistoryReconstruction(_TemporalSuperposition):
    """Load history reconstruction.

    Parameters
    ----------
    borefield : borefield or network
        The borefield.
    time : array
        Array of simulation times (in seconds).
    alpha : float
        Ground thermal diffusivity (in m^2/s).
    p : array_like or None, default: ``None``
        (`n_points`, 3,) array of positions to evaluate the ground
        temperature. If `p` is ``None``, the ground temperature is not
        evaluated.
    disp : bool, default: ``True``
        Set to ``True`` to print initialization progression messages.

    Attributes
    ----------
    n_times : int
        Number of time steps.
    n_points : int
        Number of points to evaluate ground temperatures.
    h_to_self : array
        (`n_times`, `n_boreholes`, `n_nodes`, `n_boreholes`,
        `n_nodes`,) array of thermal response factors at the nodes.
    h_to_point : array
        (`n_times`, `n_points`, `n_boreholes`, `n_nodes`,) array of
        thermal response factors at points.

    """

    def __init__(self, borefield: Borefield | Network, time: ArrayLike, alpha: float, p: ArrayLike | None = None, disp: bool = True):
        tic = perf_counter()
        # Runtime type validation
        if not isinstance(time, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {time}")
        if not isinstance(p, ArrayLike) and p is not None:
            raise TypeError(f"Expected arraylike or None input; got {p}")
        # Convert input to jax.Array
        time = jnp.asarray(time)
        if p is not None:
            p = jnp.asarray(p)

        self.borefield = borefield
        self.time = time
        self.n_times = len(time)
        self.alpha = alpha
        self.p = p
        self.q = jnp.zeros((len(self.time), borefield.n_boreholes, borefield.n_nodes))
        self.q_reconstructed = jnp.zeros((len(self.time), borefield.n_boreholes, borefield.n_nodes))
        self._time = 0.
        self._k = -1
        if disp:
            print('Initialization start.')
        self.h_to_self = borefield.h_to_self(self.time, alpha)
        if disp:
            toc = perf_counter()
            print(
                f'Completed thermal response factors to nodes. '
                f'Elapsed time: {toc-tic:.2f} seconds.'
                    )
        if p is not None:
            self.n_points = p.shape[0]
            self.h_to_point = borefield.h_to_point(p, self.time, alpha)
            if disp:
                toc = perf_counter()
                print(
                    f'Completed thermal response factors to ground. '
                    f'Elapsed time: {toc-tic:.2f} seconds.'
                        )
        else:
            self.n_points = 0
            self.h_to_point = jnp.zeros((0, borefield.n_boreholes, borefield.n_nodes))
        if disp:
            toc = perf_counter()
            print(
                f'Initialization end. Elapsed time: {toc-tic:.2f} seconds.'
            )

    def next_time_step(self) -> float:
        """Advance to next simulation time step.

        Returns
        -------
        float
            Time (in seconds) of the new time step.

        """
        self._k += 1
        self._time = self.time[self._k]
        return self._time

    def reset_history(self):
        """Reset the history to its initial condition.

        """
        self.q = self.q.at[:].set(0.)
        self.q_reconstructed = self.q_reconstructed.at[:].set(0.)
        self._time = 0.
        self._k = -1
        return

    def set_current_load(self, q: Array):
        """Set the current heat extraction rate.

        Parameters
        ----------
        q : array
            (`n_boreholes`, `n_nodes`,) array of heat extraction rates
            (in W/m).

        """
        self.q = self._current_load(self.q, q, self._k)
        return

    def temperature(self) -> Array:
        """Evaluate temperatures at nodes.

        Returns
        ----------
        T_b : array
            (`n_boreholes`, `n_nodes`,) array of borehole wall
            temperature variations (in degree Celcius).

        """
        self.q_reconstructed = self._reconstruct_load_history(self._time, self.time, self.q)
        T = self._temperature(self.h_to_self, self.q_reconstructed)
        return T

    def temperature_to_point(self) -> Array:
        """Evaluate temperatures at point.

        Returns
        ----------
        T : array
            (`n_points`,) array of ground temperature variations
            (in degree Celcius).

        """
        self.q_reconstructed = self._reconstruct_load_history(self._time, self.time, self.q)
        T = self._temperature_to_point(self.h_to_point, self.q_reconstructed)
        return T

    @staticmethod
    def _current_load(q_history: Array, q: Array, k: int) -> Array:
        """Add current load to load history

        Parameters
        ----------
        q_history : array
            (`n_times`, `n_boreholes`, `n_nodes`,) array of loads
            (in W/m).
        q : array
            (`n_boreholes`, `n_nodes`,) array of current loads (in W/m).
        k : int
            Current time step.

        Returns
        -------
        array
            (`n_times`, `n_boreholes`, `n_nodes`,) array of updated
            loads (in W/m).

        """
        return q_history.at[k].set(q)

    @staticmethod
    @jit
    def _reconstruct_load_history(current_time: float, time: Array, q: Array) -> Array:
        """Reconstruct the load history

        Parameters
        ----------
        current_time : float
            Current time (in seconds).
        time : array
            (`n_times`,) array of time steps (in seconds).
        q : array
            (`n_times`,`n_boreholes`, `n_nodes`,) array of loads (in W/m).

        Returns
        -------
        array
            (`n_times`,`n_boreholes`, `n_nodes`,) array of reconstructed
            loads (in W/m).

        """
        time = jnp.concatenate((jnp.zeros(1), time))
        dtime = jnp.diff(time)
        q_accumulated = jnp.concatenate((jnp.zeros((1, ) + q.shape[1:]), jnp.cumsum((q.T * dtime).T, axis=0)))
        time_reconstructed = (current_time - time)[::-1]
        dtime_reconstructed = dtime[::-1]
        time_reconstructed = jnp.maximum(0., time_reconstructed)
        q_reconstructed = vmap(
            vmap(
                lambda _q: jnp.interp(time_reconstructed[1:], time, _q) - jnp.interp(time_reconstructed[:-1], time, _q),
                in_axes=1,
                out_axes=1),
            in_axes=2,
            out_axes=2
        )(q_accumulated)
        q_reconstructed = (q_reconstructed.T / dtime_reconstructed).T
        q_reconstructed = q_reconstructed.at[1:].subtract(q_reconstructed[:-1])[::-1]
        return q_reconstructed
