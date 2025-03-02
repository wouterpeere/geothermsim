# -*- coding: utf-8 -*-
from functools import partial

from jax import jit, vmap
from jax import numpy as jnp
import jax

from .borefield import Borefield
from .borehole import SingleUTube
from .path import Path


class Network(Borefield):

    def __init__(self, boreholes):
        super().__init__(boreholes)

    @partial(jit, static_argnames=['self'])
    def g(self, xi, m_flow, cp_f):
        return self._heat_extraction_rate(xi, m_flow, cp_f)

    @partial(jit, static_argnames=['self'])
    def g_to_self(self, m_flow, cp_f):
        return self._heat_extraction_rate(self.xi, m_flow, cp_f)

    @partial(jit, static_argnames=['self'])
    def fluid_temperature(self, xi, T_f_in, T_b, m_flow, cp_f):
        a_in, a_b = self._fluid_temperature(xi, m_flow, cp_f)
        T_f = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)
        return T_f

    @partial(jit, static_argnames=['self'])
    def heat_extraction_rate(self, xi, T_f_in, T_b, m_flow, cp_f):
        a_in, a_b = self._heat_extraction_rate(xi, m_flow, cp_f)
        q = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)
        return q

    @partial(jit, static_argnames=['self'])
    def heat_extraction_rate_to_self(self, T_f_in, T_b, m_flow, cp_f):
        return self.heat_extraction_rate(self.xi, T_f_in, T_b, m_flow, cp_f)

    @partial(jit, static_argnames=['self'])
    def outlet_fluid_temperature(self, T_f_in, T_b, m_flow, cp_f):
        a_in, a_b = self._outlet_fluid_temperature(m_flow, cp_f)
        q = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)
        return q

    def _fluid_temperature(self, xi, m_flow, cp_f):
        m_flow_borehole = m_flow / self.n_boreholes
        a_in = jnp.stack(
            [
                borehole._fluid_temperature_a_in(xi, m_flow_borehole, cp_f)
                for borehole in self.boreholes
            ],
            axis=0
        )
        a_b = jnp.stack(
            [
                borehole._fluid_temperature_a_b(xi, m_flow_borehole, cp_f)
                for borehole in self.boreholes
            ],
            axis=0
        )
        return a_in, a_b

    def _heat_extraction_rate(self, xi, m_flow, cp_f):
        m_flow_borehole = m_flow / self.n_boreholes
        a_in = jnp.stack(
            [
                borehole._heat_extraction_rate_a_in(xi, m_flow_borehole, cp_f)
                for borehole in self.boreholes
            ],
            axis=0
        )
        a_b = jnp.stack(
            [
                borehole._heat_extraction_rate_a_b(xi, m_flow_borehole, cp_f)
                for borehole in self.boreholes
            ],
            axis=0
        )
        return a_in, a_b

    def _outlet_fluid_temperature(self, m_flow, cp_f):
        m_flow_borehole = m_flow / self.n_boreholes
        a_in = jnp.stack(
            [
                borehole._outlet_fluid_temperature_a_in(m_flow_borehole, cp_f)
                for borehole in self.boreholes
            ],
            axis=0
        )
        a_b = jnp.stack(
            [
                borehole._outlet_fluid_temperature_a_b(m_flow_borehole, cp_f)
                for borehole in self.boreholes
            ],
            axis=0
        )
        return a_in, a_b

    @classmethod
    def from_positions(cls, L, D, r_b, x, y, R_d, basis, n_segments, tilt=0., orientation=0., segment_ratios=None, order=None):
        n_boreholes = len(x)
        L = jnp.broadcast_to(L, n_boreholes)
        D = jnp.broadcast_to(D, n_boreholes)
        r_b = jnp.broadcast_to(r_b, n_boreholes)
        tilt = jnp.broadcast_to(tilt, n_boreholes)
        orientation = jnp.broadcast_to(orientation, n_boreholes)
        boreholes = []
        xi = jnp.array([-1., 1.])
        for j in range(n_boreholes):
            p = jnp.array(
                [
                    [x[j], y[j], -D[j]],
                    [x[j] + L[j] * jnp.sin(tilt[j]) * jnp.cos(orientation[j]), y[j] + L[j] * jnp.sin(tilt[j]) * jnp.sin(orientation[j]), -D[j] - L[j] * jnp.cos(tilt[j])],
                ]
            )
            path = Path(xi, p, order=order)
            boreholes.append(SingleUTube(R_d, r_b[j], path, basis, n_segments, segment_ratios=segment_ratios))
        return cls(boreholes)

    @classmethod
    def rectangle_field(cls, N_1, N_2, B_1, B_2, L, D, r_b, R_d, basis, n_segments, segment_ratios=None, order=None):
        # Borehole positions and orientation
        x = jnp.tile(jnp.arange(N_1), N_2) * B_1
        y = jnp.repeat(jnp.arange(N_2), N_1) * B_2
        return cls.from_positions(L, D, r_b, x, y, R_d, basis, n_segments, segment_ratios=segment_ratios)
        
