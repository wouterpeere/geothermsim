# -*- coding: utf-8 -*-
from time import perf_counter

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from ..borefield.borefield import Borefield
from ..borefield.network import Network
from ._temporal_superposition import _TemporalSuperposition


class LoadAggregation(_TemporalSuperposition):
    """Load aggregation.

    Parameters
    ----------
    borefield : borefield or network
        The borefield.
    dt : float
        Time step (in seconds).
    tmax : float
        Maximum time of the simulation (in seconds).
    alpha : float
        Ground thermal diffusivity (in m^2/s).
    cells_per_level : int, default: ``5``
        Number of cells per aggregation level.
    p : array_like or None, default: ``None``
        (`n_points`, 3,) array of positions to evaluate the ground
        temperature. If `p` is ``None``, the ground temperature is not
        evaluated.
    disp : bool, default: ``True``
        Set to ``True`` to print initialization progression messages.

    Attributes
    ----------
    n_cells : int
        Number of aggregation cells.
    n_points : int
        Number of points to evaluate ground temperatures.
    time : array
        (`n_cells`+1,) array of start/end times of the aggregation cells
        (in seconds).
    A : array
        Load shifting matrix.
    h_to_self : array
        (`n_cells`, `n_boreholes`, `n_nodes`, `n_boreholes`,
        `n_nodes`,) array of thermal response factors at the nodes.
    h_to_point : array
        (`n_cells`, `n_points`, `n_boreholes`, `n_nodes`,) array of
        thermal response factors at points.

    """

    def __init__(self, borefield: Borefield | Network, dt: float, tmax: float, alpha: float, cells_per_level: int = 5, p: ArrayLike | None = None, disp: bool = True):
        tic = perf_counter()
        # Runtime type validation
        if not isinstance(p, ArrayLike) and p is not None:
            raise TypeError(f"Expected arraylike or None input; got {p}")
        # Convert input to jax.Array
        if p is not None:
            p = jnp.asarray(p)

        self.borefield = borefield
        self.dt = dt
        self.tmax = tmax
        self.cells_per_level = cells_per_level
        self.p = p
        self._time = 0.
        self._k = -1
        self.time = self._load_aggregation_cells(dt, tmax, cells_per_level)
        self.n_cells = len(self.time) - 1
        self.A = self._load_shifting_matrix(self.time)
        self.q = jnp.zeros((len(self.time) - 1, borefield.n_boreholes, borefield.n_nodes))
        if disp:
            print('Initialization start.')
        self.h_to_self = borefield.h_to_self(self.time[1:], alpha)
        self.h_to_self = self.h_to_self.at[1:].set(jnp.diff(self.h_to_self, axis=0))
        if disp:
            toc = perf_counter()
            print(
                f'Completed thermal response factors to nodes. '
                f'Elapsed time: {toc-tic:.2f} seconds.'
                    )
        if p is not None:
            self.n_points = p.shape[0]
            self.h_to_point = borefield.h_to_point(p, self.time[1:], alpha)
            self.h_to_point = self.h_to_point.at[1:].set(jnp.diff(self.h_to_point, axis=0))
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
        self.q = self._next_time_step(self.A, self.q)
        self._time += self.dt
        return self._time

    def reset_history(self):
        """Reset the history to its initial condition.

        """
        self.q = self.q.at[:].set(0.)
        self._time = 0.
        self._k = -1

    def set_current_load(self, q: Array):
        """Set the current heat extraction rate.

        Parameters
        ----------
        q : array
            (`n_boreholes`, `n_nodes`,) array of heat extraction rates
            (in W/m).

        """
        self.q = self._current_load(self.q, q)

    def temperature(self) -> Array:
        """Evaluate temperatures at nodes.

        Returns
        ----------
        T_b : array
            (`n_boreholes`, `n_nodes`,) array of borehole wall
            temperature variations (in degree Celcius).

        """
        T_b = self._temperature(self.h_to_self, self.q)
        return T_b

    def temperature_to_point(self) -> Array:
        """Evaluate temperatures at point.

        Returns
        ----------
        T : array
            (`n_points`,) array of ground temperature variations
            (in degree Celcius).

        """
        T = self._temperature_to_point(self.h_to_point, self.q)
        return T

    @staticmethod
    @jit
    def _current_load(q_history: Array, q: Array) -> Array:
        """Add current load to load history

        Parameters
        ----------
        q_history : array
            (`n_cells`, `n_boreholes`, `n_nodes`,) array of aggregated
            loads (in W/m).
        q : array
            (`n_boreholes`, `n_nodes`,) array of current loads (in W/m).

        Returns
        -------
        array
            (`n_cells`, `n_boreholes`, `n_nodes`,) array of updated
            aggregated loads (in W/m).

        """
        return q_history.at[0].set(q)

    @staticmethod
    @jit
    def _next_time_step(A: Array, q: Array) -> Array:
        """Shift load history.

        Parameters
        ----------
        A : array
            Load shifting matrix.
        q : array
            (`n_cells`, `n_boreholes`, `n_nodes`,) array of aggregated
            loads in (W/m).

        Returns
        -------
        array
            (`n_cells`, `n_boreholes`, `n_nodes`,) array of aggregated
            loads at next time step (in W/m).

        """
        return jnp.tensordot(A, q, axes=(1, 0))

    @staticmethod
    def _load_aggregation_cells(dt: float, tmax: float, cells_per_level: int) -> Array:
        """Build load aggregation cells.

        Parameters
        ----------
        dt : float
            Time step (in seconds).
        tmax : float
            Maximum time of the simulation (in seconds).
        cells_per_level : int, default: ``5``
            Number of cells per aggregation level.

        Returns
        -------
        array
            Array of start/end times of the aggregation cells
            (in seconds).

        """
        time = [0.]
        i = 0
        t = 0.
        while time[-1] < tmax:
            # Increment cell count
            i += 1
            # Cell size doubles every (cells_per_level) time steps
            v = jnp.ceil(i / cells_per_level)
            width = 2**(v - 1)
            t += width * dt
            # Append time vector
            time.append(t)
        return jnp.array(time)

    @staticmethod
    def _load_shifting_matrix(time: Array) -> Array:
        """Load shifting matrix

        Parameters
        ----------
        time : array
            Array of start/end times of the aggregation cells
            (in seconds).

        Returns
        -------
        array
            Load shifting matrix.
        """
        width = jnp.diff(time) / time[1]
        n_times = len(width)
        A = (1. - 1. / width) * jnp.eye(n_times) + jnp.diag(1. / width[1:], k=-1)
        return A
