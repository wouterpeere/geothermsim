# -*- coding: utf-8 -*-
from functools import partial

from jax import numpy as jnp
from jax import jit, vmap
import jax
import numpy as np
from quadax import quadgk
from scipy.special import roots_legendre


class Basis:

    def __init__(self, xi, order=21):
        # --- Class atributes ---
        self.xi = xi
        self.n_nodes = len(xi)
        self.order = order

        # --- Basis functions (psi) ---
        power = jnp.arange(self.n_nodes)
        coefs = jnp.linalg.inv(
            np.power.outer(self.xi, power)
        )
        f_psi = lambda _eta: _eta**power @ coefs
        self.f_psi = jit(
            lambda _eta: vmap(f_psi, in_axes=0)(_eta) if len(jnp.shape(_eta)) > 0 else f_psi(_eta)
        )

        # --- Initialize quadrature methods ---
        self._initialize_quad_methods(order)

    def _initialize_quad_methods(self, order):
        self._initialize_fixed_quad_ts(order)
        self._initialize_fixed_quad_gl(order)

    def _initialize_fixed_quad_ts(self, order):
        one_minus_eps = jnp.array(1.0) - 10 * jnp.finfo(jnp.array(1.0).dtype).eps
        tanh_inverse = lambda x: 0.5 * jnp.log((1 + x) / (1 - x))
        sinh_inverse = lambda x: jnp.log(x + jnp.sqrt(x**2 + 1))
        t_max = sinh_inverse(2. / jnp.pi * tanh_inverse(one_minus_eps))
        t = jnp.linspace(-t_max, t_max, num=order)

        x = jnp.tanh(0.5 * jnp.pi * jnp.sinh(t))
        w = 0.5 * jnp.pi * jnp.cosh(t) / jnp.cosh(0.5 * jnp.pi * jnp.sinh(t))**2
        w = w * jnp.diff(t)[0]
        w = 2 * w / jnp.sum(w)
        interval = jnp.concatenate([-jnp.ones(1), self.xi, jnp.ones(1)])
        low, high = interval[:-1], interval[1:]
        self._x_ts = jnp.concatenate([0.5 * (b + a) + 0.5 * (b - a) * x for a, b in zip(low, high)])
        self._w_ts = jnp.concatenate([0.5 * (b - a) * w for a, b in zip(low, high)])
        self._psi_ts = self.f_psi(self._x_ts)

    def _initialize_fixed_quad_gl(self, order):
        x, w = roots_legendre(order)
        x = jnp.array(x)
        w = jnp.array(w)
        interval = jnp.concatenate([-jnp.ones(1), self.xi, jnp.ones(1)])
        low, high = interval[:-1], interval[1:]
        self._x_gl = jnp.concatenate([0.5 * (b + a) + 0.5 * (b - a) * x for a, b in zip(low, high)])
        self._w_gl = jnp.concatenate([0.5 * (b - a) * w for a, b in zip(low, high)])
        self._psi_gl = self.f_psi(self._x_gl)

    def integrate_fixed_gl(self, fun, a, b):
        x, w = roots_legendre(self.order)
        x = 0.5 * (b + a) + 0.5 * (b - a) * jnp.array(x)
        w = 0.5 * (b - a) * jnp.array(w)
        _psi = self.f_psi(x)
        integral = lambda xi, w, psi: (fun(xi) * psi) @ w
        return vmap(integral, in_axes=(None, None, 1), out_axes=-1)(x, w, _psi)

    def integrate_subintervals_fixed_gl(self, fun):
        integral = lambda xi, w, psi: (fun(xi) * psi) @ w
        return vmap(integral, in_axes=(None, None, 1), out_axes=-1)(self._x_gl, self._w_gl, self._psi_gl)

    def integrate_subintervals_fixed_ts(self, fun):
        integral = lambda xi, w, psi: (fun(xi) * psi) @ w
        return vmap(integral, in_axes=(None, None, 1), out_axes=-1)(self._x_ts, self._w_ts, self._psi_ts)

    @classmethod
    def Legendre(cls, n_nodes, order=21):
        return cls(jnp.array(roots_legendre(n_nodes)[0]), order=order)
