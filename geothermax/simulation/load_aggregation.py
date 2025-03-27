# -*- coding: utf-8 -*-
import jax
from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from ..borefield.borefield import Borefield
from ..borefield.network import Network
from ._temporal_superposition import _TemporalSuperposition


class LoadAggregation(_TemporalSuperposition):

    def __init__(self, borefield: Borefield | Network, dt: float, tmax: float, alpha: float, cells_per_level: int = 5, p: ArrayLike | None = None):
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
        self.A = self._load_shifting_matrix(self.time)
        self.q = jnp.zeros((len(self.time) - 1, borefield.n_boreholes, borefield.n_nodes))
        self.h_to_self = borefield.h_to_self(self.time[1:], alpha)
        self.h_to_self = self.h_to_self.at[1:].set(jnp.diff(self.h_to_self, axis=0))
        if p is not None:
            self.h_to_point = borefield.h_to_point(p, self.time[1:], alpha)
            self.h_to_point = self.h_to_point.at[1:].set(jnp.diff(self.h_to_point, axis=0))
        else:
            self.h_to_point = jnp.zeros((0, borefield.n_boreholes, borefield.n_nodes))

    def next_time_step(self) -> float:
        self.q = self._next_time_step(self.A, self.q)
        self._time += self.dt
        return self._time

    def reset_history(self):
        self.q = self.q.at[:].set(0.)
        self._time = 0.
        self._k = -1
        return

    def set_current_load(self, q: Array):
        self.q = self._current_load(self.q, q)
        return

    def temperature(self) -> Array:
        T = self._temperature(self.h_to_self, self.q)
        return T

    def temperature_to_point(self) -> Array:
        T = self._temperature_to_point(self.h_to_point, self.q)
        return T

    @staticmethod
    @jit
    def _current_load(q_history: Array, q: Array) -> Array:
        return q_history.at[0].set(q)

    @staticmethod
    @jit
    def _next_time_step(A: Array, q: Array) -> Array:
        return jnp.tensordot(A, q, axes=(1, 0))

    @staticmethod
    def _load_aggregation_cells(dt: float, tmax: float, cells_per_level: int) -> Array:
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
        width = jnp.diff(time) / time[1]
        n_times = len(width)
        A = (1. - 1. / width) * jnp.eye(n_times) + jnp.diag(1. / width[1:], k=-1)
        return A
