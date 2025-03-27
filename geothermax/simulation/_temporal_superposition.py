# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import jax
from jax import numpy as jnp
from jax import Array, jit


class _TemporalSuperposition(ABC):

    @abstractmethod
    def next_time_step(self) -> float:
        ...

    @abstractmethod
    def reset_history(self):
        ...

    @abstractmethod
    def set_current_load(self, q: Array):
        ...

    @abstractmethod
    def temperature(self) -> Array:
        ...

    @abstractmethod
    def temperature_to_point(self) -> Array:
        ...

    @staticmethod
    @jit
    def _temperature(h: Array, q: Array) -> Array:
        T = jnp.tensordot(h, q, axes=([0, -2, -1], [0, -2, -1]))
        return T

    @staticmethod
    @jit
    def _temperature_to_point(h: Array, q: Array) -> Array:
        T = jnp.tensordot(h, q, axes=([0, -2, -1], [0, -2, -1]))
        return T
