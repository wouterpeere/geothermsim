# -*- coding: utf-8 -*-
from functools import partial

import jax
from jax import numpy as jnp
from jax import Array, grad, jit, vmap
from jax.typing import ArrayLike
from jax.scipy.special import erfc
import numpy as np
from scipy.special import roots_legendre


class Path:

    def __init__(self, xi: ArrayLike, p: ArrayLike, order: int | None = None, s_order: int | None = None):
        # Runtime type validation
        if not isinstance(xi, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {xi}")
        if not isinstance(p, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {p}")
        # Convert input to jax.Array
        xi = jnp.asarray(xi)
        p = jnp.atleast_2d(p)

        # --- Class atributes ---
        self.xi = xi
        self.p = p
        n_nodes = len(xi)
        self.n_nodes = n_nodes
        if order is None:
            order = n_nodes
        order = jnp.maximum(2, jnp.minimum(order, n_nodes))
        self.order = order
        if s_order is None:
            s_order = 2 * (self.n_nodes - 1)
        self.s_order = s_order

        # --- Path functions ---
        # Position (p)
        p_w = np.ones(n_nodes)
        if order < n_nodes:
            p_w[0] = 1e4
        p_coefs = jnp.array(np.polyfit(xi, p, order-1, w=p_w))
        f_p = lambda _eta: jnp.polyval(p_coefs, _eta)
        self.f_p = jit(
            lambda _eta: vmap(f_p, in_axes=0)(_eta) if len(jnp.shape(_eta)) > 0 else f_p(_eta)
        )
        # Derivative of position (dp/dxi)
        f_dp_dxi = lambda _eta: jax.jacobian(f_p)(_eta)
        self.f_dp_dxi = jit(
            lambda _eta: vmap(f_dp_dxi, in_axes=0)(_eta) if len(jnp.shape(_eta)) > 0 else f_dp_dxi(_eta)
        )
        # Norm of the Jacobian (J)
        f_J = lambda _eta: jnp.linalg.norm(f_dp_dxi(_eta))
        self.f_J = jit(
            lambda _eta: vmap(f_J, in_axes=0)(_eta) if len(jnp.shape(_eta)) > 0 else f_J(_eta)
        )
        # Longitudinal position (s)
        x_s, w_s = roots_legendre(s_order)
        x_s = jnp.array(x_s)
        w_s = jnp.array(w_s)
        low = jnp.concatenate([-jnp.ones(1), x_s[:-1]])
        high = x_s
        s = jnp.cumsum(
            vmap(
                lambda _a, _b: self.f_J(0.5 * (_a + _b) + 0.5 * x_s * (_b - _a)) @ w_s * 0.5 * (_b - _a),
                in_axes=(0, 0)
            )(low, high)
        )
        s_coefs = jnp.array(np.polyfit(x_s, s, s_order-1))
        f_s = lambda _eta: jnp.polyval(s_coefs, _eta)
        self.f_s = jit(
            lambda _eta: vmap(f_s, in_axes=0)(_eta) if len(jnp.shape(_eta)) > 0 else f_s(_eta)
        )

    @partial(jit, static_argnames=['self'])
    def point_heat_source(self, xi: Array | float, p: Array, time: Array | float, alpha: float, r_min: float = 0.) -> Array | float:
        if len(jnp.shape(time)) > 0:
            return vmap(
                self.point_heat_source,
                in_axes=(None, None, -1, None, None)
            )(xi, p, time, alpha, r_min)
        if len(jnp.shape(p)) > 1:
            return vmap(
                self.point_heat_source,
                in_axes=(None, -2, None, None, None)
            )(xi, p, time, alpha, r_min)
        if len(jnp.shape(xi)) > 0:
            return vmap(
                self.point_heat_source,
                in_axes=(-1, None, None, None, None)
            )(xi, p, time, alpha, r_min)
        # Current position of the point source
        p_source = self.f_p(xi)
        # Distance to the real point (p)
        r = jnp.sqrt(jnp.linalg.norm(p_source - p)**2 + r_min**2)
        # Distance to the mirror point (p')
        r_mirror = jnp.linalg.norm(p_source - p * jnp.array([1, 1, -1]))
        # Point heat source solution
        h = 0.5 * erfc(r / jnp.sqrt(4 * alpha * time)) / r - 0.5 * erfc(r_mirror / jnp.sqrt(4 * alpha * time)) / r_mirror
        return h * self.f_J(xi)
