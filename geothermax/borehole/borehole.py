# -*- coding: utf-8 -*-
from functools import partial

from jax import numpy as jnp
import jax


class Borehole:

    def __init__(self, r_b, path, basis, n_segments, segment_ratios=None):
        self.r_b = r_b
        self.path = path
        self.basis = basis
        self.n_segments = n_segments
        self.n_nodes = self.basis.n_nodes * self.n_segments
        if segment_ratios is None:
            segment_ratios = jnp.full(self.n_segments, 1. / self.n_segments)
        self.segment_ratios = segment_ratios
        self.xi_edges = jnp.concatenate((-jnp.ones(1), 2 * jnp.cumsum(self.segment_ratios) - 1))
        a, b = self.xi_edges[:-1], self.xi_edges[1:]
        self._f_sb = lambda eta: 0.5 * (b + a) + 0.5 * eta * (b - a)
        self._f_bs = lambda eta: (2 * eta - (b + a)) / (b - a)
        self._f_sb_mapped = jax.vmap(self._f_sb, in_axes=0, out_axes=0)
        self._f_bs_mapped = jax.vmap(self._f_bs, in_axes=0, out_axes=0)
        self.xi = self._f_sb_mapped(self.basis.xi).T.flatten()
        self.p = self.path._p_mapped(self.xi)
        self.L = jnp.diff(self.path._s_mapped(jnp.array([-1., 1.])))[0]
        w_limits = jnp.concatenate([jnp.array([1.]), jnp.full(self.n_segments - 1, 0.5), jnp.array([1.])])
        in_segment = lambda eta: jnp.heaviside(b - eta, w_limits[1:]) * jnp.heaviside(eta - a, w_limits[:-1])
        self._psi = lambda eta: jax.vmap(lambda eta_p, _in_segment: self.basis._psi(eta_p) * _in_segment, in_axes=(0, 0))(self._f_bs(eta), in_segment(eta)).flatten()
        self._psi_mapped = jax.vmap(self._psi, in_axes=0)

    @partial(jax.jit, static_argnames=['self'])
    def point_heat_source(self, xi, p, time, alpha, r_min=0.):
        xi_segments = self._f_sb(xi)
        return jax.vmap(lambda eta: self.path.point_heat_source(eta, p, time, alpha, r_min=r_min), in_axes=0)(xi_segments) * self.segment_ratios

    def h_to_borehole(self, borehole, time, alpha):
        return self.h_to_point(borehole.p, time, alpha)

    @partial(jax.jit, static_argnames=['self'])
    def h_to_self(self, time, alpha):
        point_heat_source = jax.vmap(
            jax.vmap(
                partial(self.point_heat_source, r_min=self.r_b),
                in_axes=(None, 0, None, None)
                ),
            in_axes=(None, None, 0, None)
            )
        fun = jax.vmap(lambda xi: point_heat_source(xi, self.p, time, alpha), in_axes=0, out_axes=-1)
        # return self.basis.integrate_fixed_ts(fun)
        n_times = len(time)
        return self.basis.integrate_subintervals_fixed_ts(fun).reshape(n_times, self.n_nodes, self.n_nodes)

    @partial(jax.jit, static_argnames=['self'])
    def h_to_point(self, p, time, alpha, r_min=0):
        point_heat_source = jax.vmap(
            jax.vmap(
                partial(self.point_heat_source, r_min=r_min),
                in_axes=(None, 0, None, None)
                ),
            in_axes=(None, None, 0, None)
            )
        fun = jax.vmap(lambda xi: point_heat_source(xi, p, time, alpha), in_axes=0, out_axes=-1)
        n_times = len(time)
        return self.basis.integrate_subintervals_fixed_gl(fun).reshape(n_times, -1, self.n_nodes)
