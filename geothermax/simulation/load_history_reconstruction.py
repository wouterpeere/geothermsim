# -*- coding: utf-8 -*-
from functools import partial

from jax import jit, vmap
from jax import numpy as jnp
import jax

from ._temporal_superposition import _TemporalSuperposition


class LoadHistoryReconstruction(_TemporalSuperposition):

    def __init__(self, borefield, time, alpha, p=None):
        self.borefield = borefield
        self.time = time
        self.alpha = alpha
        self.p = p
        self.q = jnp.zeros((len(self.time), borefield.n_boreholes, borefield.n_nodes))
        self.q_reconstructed = jnp.zeros((0, borefield.n_boreholes, borefield.n_nodes))
        self._time = 0.
        self._k = -1
        self.h_to_self = borefield.h_to_self(self.time, alpha)
        if p is not None:
            self.h_to_point = borefield.h_to_point(p, self.time, alpha)
        else:
            self.h_to_point = jnp.zeros((0, borefield.n_boreholes, borefield.n_nodes))

    def next_time_step(self):
        self._k += 1
        self._time = self.time[self._k]
        return self._time

    def reset_history(self):
        self.q = self.q.at[:].set(0.)
        self.q_reconstructed = jnp.zeros((0, self.borefield.n_boreholes, self.borefield.n_nodes))
        self._time = 0.
        self._k = -1
        return

    def set_current_load(self, q):
        self.q = self._current_load(self.q, q, self._k)
        return

    def temperature(self):
        self.q_reconstructed = self._reconstruct_load_history(self.time[:self._k+1], self.q[:self._k+1])
        T = self._temperature(self.h_to_self[:self._k+1], self.q_reconstructed)
        return T

    def temperature_to_point(self):
        self.q_reconstructed = self._reconstruct_load_history(self.time[:self._k+1], self.q[:self._k+1])
        T = self._temperature_to_point(self.h_to_point[:self._k+1], self.q_reconstructed)
        return T

    @staticmethod
    def _current_load(q_history, q, k):
        return q_history.at[k].set(q)

    @staticmethod
    @jit
    def _reconstruct_load_history(time, q):
        time = jnp.concatenate((jnp.zeros(1), time))
        dtime = jnp.diff(time)
        q_accumulated = jnp.concatenate((jnp.zeros((1, ) + q.shape[1:]), jnp.cumsum((q.T * dtime).T, axis=0)))
        dtime_reconstructed = dtime[::-1]
        time_reconstructed = jnp.concatenate((jnp.zeros(1), jnp.cumsum(dtime_reconstructed)))
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

