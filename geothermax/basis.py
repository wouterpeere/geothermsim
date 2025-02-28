# -*- coding: utf-8 -*-
from functools import partial

from jax import numpy as jnp
from jax import jit, vmap
import jax
import numpy as np
from quadax import quadgk
from scipy.special import roots_legendre


class Basis:

    def __init__(self, xi, order=101, order_nodes=21):
        # --- Class atributes ---
        self.xi = xi
        self.n_nodes = len(xi)
        self.order = order
        self.order_nodes = order_nodes

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
        self._initialize_quad_methods(order, order_nodes)

    def _initialize_quad_methods(self, order, order_nodes):
        self._x_gl, self._w_gl = self._gauss_legendre_rule(order)
        self._x_gl_nodes, self._w_gl_nodes, self._psi_gl_nodes = self._initialize_quad_nodes(*self._gauss_legendre_rule(order_nodes))
        self._x_ts, self._w_ts = self._tanh_sinh_rule(order)
        self._x_ts_nodes, self._w_ts_nodes, self._psi_ts_nodes = self._initialize_quad_nodes(*self._tanh_sinh_rule(order_nodes))

    def _initialize_quad_nodes(self, x, w):
        interval = jnp.concatenate([-jnp.ones(1), self.xi, jnp.ones(1)])
        low, high = interval[:-1], interval[1:]
        x_nodes = jnp.concatenate([0.5 * (b + a) + 0.5 * (b - a) * x for a, b in zip(low, high)])
        w_nodes = jnp.concatenate([0.5 * (b - a) * w for a, b in zip(low, high)])
        psi_nodes = self.f_psi(x_nodes)
        return x_nodes, w_nodes, psi_nodes

    def _quad(self, fun, a, b, x, w):
        x = 0.5 * (b + a) + 0.5 * (b - a) * jnp.array(x)
        w = 0.5 * (b - a) * jnp.array(w)
        psi = self.f_psi(x)
        return self._quad_psi(fun, x, w, psi)

    def _quad_psi(self, fun, x, w, psi):
        integral = lambda _eta, _w, _psi: (fun(_eta) * _psi) @ _w
        return vmap(integral, in_axes=(None, None, 1), out_axes=-1)(x, w, psi)

    def quad_gl(self, fun, a, b):
        x, w = self._x_gl, self._w_gl
        return self._quad(fun, a, b, x, w)

    def quad_gl_nodes(self, fun):
        x, w, psi = self._x_gl_nodes, self._w_gl_nodes, self._psi_gl_nodes
        return self._quad_psi(fun, x, w, psi)

    def quad_ts_nodes(self, fun):
        x, w, psi = self._x_ts_nodes, self._w_ts_nodes, self._psi_ts_nodes
        return self._quad_psi(fun, x, w, psi)

    @staticmethod
    def _gauss_legendre_rule(order):
        x, w = roots_legendre(order)
        x = jnp.array(x)
        w = jnp.array(w)
        return x, w

    @staticmethod
    def _tanh_sinh_rule(order):
        one_minus_eps = jnp.array(1.0) - 10 * jnp.finfo(jnp.array(1.0).dtype).eps
        tanh_inverse = lambda x: 0.5 * jnp.log((1 + x) / (1 - x))
        sinh_inverse = lambda x: jnp.log(x + jnp.sqrt(x**2 + 1))
        t_max = sinh_inverse(2. / jnp.pi * tanh_inverse(one_minus_eps))
        t = jnp.linspace(-t_max, t_max, num=order)

        x = jnp.tanh(0.5 * jnp.pi * jnp.sinh(t))
        w = 0.5 * jnp.pi * jnp.cosh(t) / jnp.cosh(0.5 * jnp.pi * jnp.sinh(t))**2
        w = w * jnp.diff(t)[0]
        w = 2 * w / jnp.sum(w)
        return x, w

    @classmethod
    def Legendre(cls, n_nodes, order=101, order_nodes=21):
        xi = jnp.array(roots_legendre(n_nodes)[0])
        return cls(xi, order=order, order_nodes=order_nodes)
