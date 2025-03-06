# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

from jax import jit
from jax import numpy as jnp


class _TemporalSuperposition(ABC):

    @abstractmethod
    def next_time_step(self):
        ...

    @abstractmethod
    def reset_history(self):
        ...

    @abstractmethod
    def set_current_load(self, q):
        ...

    @abstractmethod
    def temperature(self):
        ...

    @abstractmethod
    def temperature_to_point(self):
        ...

    @staticmethod
    @jit
    def _temperature(h, q):
        T = jnp.tensordot(h, q, axes=([0, -2, -1], [0, -2, -1]))
        return T

    @staticmethod
    @jit
    def _temperature_to_point(h, q):
        T = jnp.tensordot(h, q, axes=([0, -2, -1], [0, -2, -1]))
        return T
