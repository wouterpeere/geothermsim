# -*- coding: utf-8 -*-
from functools import partial

from jax import grad, jit, vmap
from jax import numpy as jnp
from jax.scipy.special import erfc
import jax
from scipy.special import roots_legendre


class Path:

    def __init__(self, xi, p, order=None, s_order=None):
        # --- Class atributes ---
        self.xi = xi
        self.p = p
        self.order = order
        n_nodes = len(xi)
        self.n_nodes = n_nodes
        if s_order is None:
            s_order = 2 * (self.n_nodes - 1)
        self.s_order = s_order

        # --- Path functions ---
        # Position (p)
        p_power = jnp.arange(n_nodes)
        A = vmap(lambda _eta: _eta**p_power, in_axes=0, out_axes=0)(xi)
        p_coefs = jnp.linalg.solve(A, p)
        f_p = lambda _eta: _eta**p_power @ p_coefs
        self.f_p = jit(f_p)
        F_p = vmap(f_p, in_axes=0)
        self.F_p = jit(F_p)
        # Derivative of position (dp/dxi)
        f_p_i = lambda _eta, _coefs: _eta**p_power @ _coefs
        f_dp_dxi = lambda _eta: vmap(
            grad(f_p_i),
            in_axes=(None, 1)
        )(_eta, p_coefs)
        self.f_dp_dxi = jit(f_dp_dxi)
        F_dp_dxi = vmap(f_dp_dxi, in_axes=0)
        self.F_dp_dxi = jit(F_dp_dxi)
        # Norm of the Jacobian (J)
        f_J = lambda _eta: jnp.linalg.norm(f_dp_dxi(_eta))
        self.f_J = jit(f_J)
        F_J = vmap(f_J, in_axes=0)
        self.F_J = jit(F_J)
        # Longitudinal position (s)
        x_s, w_s = roots_legendre(s_order)
        x_s = jnp.array(x_s)
        w_s = jnp.array(w_s)
        low = jnp.concatenate([-jnp.ones(1), x_s[:-1]])
        high = x_s
        s = jnp.cumsum(
            vmap(
                lambda _a, _b: self.F_J(0.5 * (_a + _b) + 0.5 * x_s * (_b - _a)) @ w_s * 0.5 * (_b - _a),
                in_axes=(0, 0)
            )(low, high)
        )
        s_power = jnp.arange(s_order)
        A = vmap(lambda eta: eta**s_power, in_axes=0)(x_s)
        s_coefs = jnp.linalg.solve(A, s)
        f_s = lambda _eta: _eta**s_power @ s_coefs
        self.f_s = jit(f_s)
        F_s = vmap(f_s, in_axes=0)
        self.F_s = jit(F_s)

    @partial(jit, static_argnames=['self'])
    def point_heat_source(self, xi, p, time, alpha, r_min=0.):
        p_jv = self.f_p(xi)
        r = jnp.sqrt(jnp.linalg.norm(p_jv - p)**2 + r_min**2)
        r_mirror = jnp.linalg.norm(p_jv - p * jnp.array([1, 1, -1]))
        h = 0.5 * erfc(r / jnp.sqrt(4 * alpha * time)) / r - 0.5 * erfc(r_mirror / jnp.sqrt(4 * alpha * time)) / r_mirror
        return h * self.f_J(xi)
