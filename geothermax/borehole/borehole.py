# -*- coding: utf-8 -*-
from functools import partial

from jax import jit, vmap
from jax import numpy as jnp
import jax


class Borehole:

    def __init__(self, r_b, path, basis, n_segments, segment_ratios=None):
        # --- Class attributes ---
        self.r_b = r_b
        self.path = path
        self.basis = basis
        self.n_segments = n_segments
        self.n_nodes = basis.n_nodes * n_segments
        if segment_ratios is None:
            segment_ratios = jnp.full(n_segments, 1. / n_segments)
        self.segment_ratios = segment_ratios
        # Segment edges
        xi_edges = jnp.concatenate((-jnp.ones(1), 2 * jnp.cumsum(segment_ratios) - 1))
        self.xi_edges = xi_edges
        # Borehole length
        self.L = jnp.diff(path.F_s(jnp.array([-1., 1.])))[0]

        # --- Changes of coordinates ---
        a, b = xi_edges[:-1], xi_edges[1:]
        # Segments --> Borehole
        f_xi_sb = lambda _eta: 0.5 * (b + a) + 0.5 * _eta * (b - a)
        self.f_xi_sb = jit(f_xi_sb)
        F_xi_sb = vmap(f_xi_sb, in_axes=0)
        self.F_xi_sb = jit(F_xi_sb)
        # Borehole --> Segments
        f_xi_bs = lambda _eta: (2 * _eta - (b + a)) / (b - a)
        self.f_xi_bs = jit(f_xi_bs)
        F_xi_bs = vmap(f_xi_bs, in_axes=0)
        self.F_xi_bs = jit(F_xi_bs)

        # --- Basis functions ---
        # Weights at interfaces between segments
        w_interfaces = jnp.concatenate([jnp.array([1.]), jnp.full(n_segments - 1, 0.5), jnp.array([1.])])
        f_w_interfaces = lambda _eta: jnp.heaviside(b - _eta, w_interfaces[1:]) * jnp.heaviside(_eta - a, w_interfaces[:-1])
        f_psi = lambda _eta: jax.vmap(
            lambda _eta_p, _in_segment: basis.f_psi(_eta_p) * _in_segment,
            in_axes=(0, 0)
        )(f_xi_bs(_eta), f_w_interfaces(_eta)).flatten()
        self.f_psi = f_psi
        F_psi = vmap(f_psi, in_axes=0)
        self.F_psi = jit(F_psi)

        # --- Nodal values of path and basis functions ---
        # Borehole coordinates (xi)
        xi = F_xi_sb(basis.xi).T.flatten()
        self.xi = xi
        # Positions (p)
        self.p = path.F_p(xi)
        # Derivatives of position (dp/dxi)
        self.dp_dxi = path.F_dp_dxi(xi)
        # Norms of the Jacobian (J)
        self.J = path.F_J(xi)
        # Longitudinal positions (s)
        self.s = path.F_s(xi)

    def h_to_borehole(self, borehole, time, alpha):
        return self.h_to_point(borehole.p, time, alpha)

    @partial(jit, static_argnames=['self'])
    def h_to_coordinate_on_self(self, xi, time, alpha):
        # Positions (p) of points on self
        p = self.path.F_p(xi)
        return self.h_to_point(p, time, alpha, r_min=self.r_b)

    
    @partial(jit, static_argnames=['self'])
    def h_to_point(self, p, time, alpha, r_min=0):
        # Integrand of point heat source
        integrand = vmap(
            lambda _eta: self._point_heat_source(_eta, p, time, alpha, r_min=r_min),
            in_axes=0,
            out_axes=-1)
        n_times = len(time)
        n_nodes = self.n_nodes
        h_to_point = self.basis.integrate_subintervals_fixed_gl(
            integrand
        ).reshape(n_times, -1, n_nodes)
        return h_to_point
        

    @partial(jit, static_argnames=['self'])
    def h_to_self(self, time, alpha):
        # Integrand of point heat source evaluated at borehole nodes
        integrand = vmap(
            lambda _eta: self._point_heat_source(_eta, self.p, time, alpha, r_min=self.r_b),
            in_axes=0,
            out_axes=-1)
        # Integral of the point heat source
        n_times = len(time)
        n_nodes = self.n_nodes
        h_to_self = self.basis.integrate_subintervals_fixed_ts(
            integrand
        ).reshape(n_times, n_nodes, n_nodes)
        return h_to_self


    @partial(jit, static_argnames=['self'])
    def _point_heat_source(self, xi_p, p, time, alpha, r_min=0.):
        # Coordinates (xi) of all sources at local segment coordinates (xi')
        xi = self.f_xi_sb(xi_p)
        # Point heat source solutions
        point_heat_source = partial(
            self.path.point_heat_source,
            r_min=r_min)
        h = vmap(
            point_heat_source,
            in_axes=(0, None, None, None)
        )
        # Add vmaps over (p) and (time) if they are arrays
        # The output shape wil be: (n_time, n_p, n_segments)
        if len(jnp.shape(p)) == 2:
            h = vmap(
                h,
                in_axes=(None, 0, None, None)
            )
        if len(jnp.shape(time)) == 1:
            h = vmap(
                h,
                in_axes=(None, None, 0, None)
            )
        return h(xi, p, time, alpha) * self.segment_ratios
