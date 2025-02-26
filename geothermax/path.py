# -*- coding: utf-8 -*-
from functools import partial

from jax import numpy as jnp
from jax.scipy.special import erfc
import jax
from scipy.special import roots_legendre


class Path:

    def __init__(self, xi, p, order=None, s_order=None):
        self.xi = xi
        self.p = p
        self.order = order
        self.n_nodes = len(xi)
        if s_order is None:
            s_order = 2 * (self.n_nodes - 1)
        self.s_order = s_order
        self._power = jnp.arange(self.n_nodes)
        _A = jax.vmap(lambda eta: eta**self._power, in_axes=0, out_axes=0)(xi)
        self.coefs = jnp.linalg.solve(_A, p)
        _p = lambda eta: eta**self._power @ self.coefs
        self._p = jax.jit(_p)
        self._p_mapped = jax.jit(jax.vmap(_p, in_axes=0))
        coord = lambda eta, coefs: eta**self._power @ coefs
        _dp_dxi = lambda eta: jax.vmap(jax.grad(coord), in_axes=(None, 1))(eta, self.coefs)
        self._dp_dxi = jax.jit(_dp_dxi)
        self._dp_dxi_mapped = jax.jit(jax.vmap(_dp_dxi, in_axes=0))
        _J = lambda eta: jnp.linalg.norm(_dp_dxi(eta))
        self._J = jax.jit(_J)
        self._J_mapped = jax.jit(jax.vmap(_J, in_axes=0))
        x, w = roots_legendre(self.s_order)
        x = jnp.array(x)
        w = jnp.array(w)
        low = jnp.concatenate([-jnp.ones(1), x[:-1]])
        high = x
        s = jnp.cumsum(jax.vmap(lambda a, b: self._J_mapped(0.5 * (a + b) + 0.5 * x * (b - a)) @ w * 0.5 * (b - a), in_axes=(0, 0))(low, high))
        _s_power = jnp.arange(self.s_order)
        _A = jax.vmap(lambda eta: eta**_s_power, in_axes=0)(x)
        _s_coefs = jnp.linalg.solve(_A, s)
        _s = lambda eta: eta**_s_power @ _s_coefs
        self._s = jax.jit(_s)
        self._s_mapped = jax.jit(jax.vmap(_s, in_axes=0))

    @partial(jax.jit, static_argnames=['self'])
    def point_heat_source(self, xi, p, time, alpha, r_min=0.):
        p_jv = self._p(xi)
        r = jnp.sqrt(jnp.linalg.norm(p_jv - p)**2 + r_min**2)
        r_mirror = jnp.linalg.norm(p_jv - p * jnp.array([1, 1, -1]))
        h = 0.5 * erfc(r / jnp.sqrt(4 * alpha * time)) / r - 0.5 * erfc(r_mirror / jnp.sqrt(4 * alpha * time)) / r_mirror
        return h * self._J(xi)
