# -*- coding: utf-8 -*-
from functools import partial

from jax import numpy as jnp
import jax
from scipy.special import roots_legendre

from .borehole import Borehole


class SingleUTube(Borehole):

    def __init__(self, R_d, r_b, path, basis, n_segments, segment_ratios=None):
        super().__init__(r_b, path, basis, n_segments, segment_ratios=segment_ratios)
        self.R_d = R_d

    def G(self, m_flow, cp_f):
        return self._heat_extraction_rate(self.xi, m_flow, cp_f)

    def _general_solution_a_in(self, xi, m_flow, cp_f):
        s = self.path._s(xi)
        a_in = jnp.stack(
            (
                self._f1(s, m_flow, cp_f),
                -self._f2(s, m_flow, cp_f)
            ),
            axis=-1
        )
        return a_in

    def _general_solution_a_out(self, xi, m_flow, cp_f):
        s = self.path._s(xi)
        a_out = jnp.stack(
            (
                self._f2(s, m_flow, cp_f),
                self._f3(s, m_flow, cp_f)
            ),
            axis=-1
        )
        return a_out

    def _general_solution_a_b(self, xi, m_flow, cp_f):
        f1 = jax.vmap(lambda _eta: self._f4(self.path._s(xi) - self.path._s(_eta), m_flow, cp_f) * self.path._J(_eta), in_axes=0)
        f2 = jax.vmap(lambda _eta: -self._f5(self.path._s(xi) - self.path._s(_eta), m_flow, cp_f) * self.path._J(_eta), in_axes=0)
        high = jnp.maximum(-1., jnp.minimum(1., self._f_bs(xi)))
        a, b = self.xi_edges[:-1], self.xi_edges[1:]
        a_b = jnp.stack(
            [
                jnp.stack(
                    [
                        self.basis.integrate_fixed_gl(
                            lambda _eta: f1(0.5 * (b[v] + a[v]) + 0.5 * _eta * (b[v] - a[v])) * self.segment_ratios[v],
                            -1.,
                            high[v])
                        for v in range(self.n_segments)
                    ]
                ),
                    jnp.stack(
                    [
                        self.basis.integrate_fixed_gl(
                            lambda _eta: f2(0.5 * (b[v] + a[v]) + 0.5 * _eta * (b[v] - a[v])) * self.segment_ratios[v],
                            -1.,
                            high[v])
                        for v in range(self.n_segments)
                    ]
                )
            ],
            axis=0)
        return a_b
        
    def _fluid_temperature(self, xi, m_flow, cp_f):
        b_in, b_b = self._outlet_fluid_temperature(m_flow, cp_f)
        c_in, c_out, c_b = self._general_solution(xi, m_flow, cp_f)
        a_in = c_in + b_in * c_out
        a_b = c_b + jax.vmap(jnp.outer, in_axes=(0, None))(c_out, b_b)
        return a_in, a_b

    def _general_solution(self, xi, m_flow, cp_f):
        a_in = jax.vmap(self._general_solution_a_in, in_axes=(0, None, None))(xi, m_flow, cp_f)
        a_out = jax.vmap(self._general_solution_a_out, in_axes=(0, None, None))(xi, m_flow, cp_f)
        a_b = jax.vmap(self._general_solution_a_b, in_axes=(0, None, None))(xi, m_flow, cp_f).reshape(-1, 2, self.n_nodes)
        return a_in, a_out, a_b

    def _heat_extraction_rate(self, xi, m_flow, cp_f):
        b_in, b_b = self._fluid_temperature(xi, m_flow, cp_f)
        R_b = self.R_d[0, 0] * self.R_d[1, 1] / (self.R_d[0, 0] + self.R_d[1, 1])
        a_in = -(b_in[:, 0] / self.R_d[0, 0] + b_in[:, 1] / self.R_d[1, 1])
        a_b = -(b_b[:, 0, :] / self.R_d[0, 0] + b_b[:, 1, :] / self.R_d[1, 1]) + self._psi_mapped(xi) / R_b
        return a_in, a_b

    def _outlet_fluid_temperature(self, m_flow, cp_f):
        L = self.L
        a_in = (self._f1(L, m_flow, cp_f) + self._f2(L, m_flow, cp_f)) / (self._f3(L, m_flow, cp_f) - self._f2(L, m_flow, cp_f))
        f = jax.vmap(
            lambda _eta: (
                self._f4(L - self.path._s_mapped(self._f_sb(_eta)), m_flow, cp_f)
                + self._f5(L - self.path._s_mapped(self._f_sb(_eta)), m_flow, cp_f)
            ) / (
                self._f3(L, m_flow, cp_f)
                - self._f2(L, m_flow, cp_f)
            ) * self.path._J_mapped(self._f_sb(_eta)) * self.segment_ratios,
            in_axes=0,
            out_axes=-1)
        a_b = self.basis.integrate_fixed_gl(f, -1, 1.).flatten()
        return a_in, a_b

    def _beta1(self, m_flow, cp_f):
        return 1. / (m_flow * cp_f * self.R_d[0, 0])

    def _beta2(self, m_flow, cp_f):
        return 1. / (m_flow * cp_f * self.R_d[1, 1])

    def _beta12(self, m_flow, cp_f):
        return 1. / (m_flow * cp_f * self.R_d[0, 1])

    def _beta(self, m_flow, cp_f):
        beta1 = self._beta1(m_flow, cp_f)
        beta2 = self._beta2(m_flow, cp_f)
        beta = 0.5 * (beta2 - beta1)
        return beta

    def _gamma(self, m_flow, cp_f):
        beta1 = self._beta1(m_flow, cp_f)
        beta2 = self._beta2(m_flow, cp_f)
        beta12 = self._beta12(m_flow, cp_f)
        gamma = jnp.sqrt(0.25 * (beta1 + beta2)**2 + beta12 * (beta1 + beta2))
        return gamma

    def _delta(self, m_flow, cp_f):
        beta1 = self._beta1(m_flow, cp_f)
        beta2 = self._beta2(m_flow, cp_f)
        beta12 = self._beta12(m_flow, cp_f)
        gamma = self._gamma(m_flow, cp_f)
        delta = 1. / gamma * (beta12 + 0.5 * (beta1 + beta2))
        return delta

    def _f1(self, s, m_flow, cp_f):
        beta = self._beta(m_flow, cp_f)
        gamma = self._gamma(m_flow, cp_f)
        delta = self._delta(m_flow, cp_f)
        f1 = jnp.exp(beta * s) * (jnp.cosh(gamma * s) - delta * jnp.sinh(gamma * s))
        return f1

    def _f2(self, s, m_flow, cp_f):
        beta = self._beta(m_flow, cp_f)
        gamma = self._gamma(m_flow, cp_f)
        beta12 = self._beta12(m_flow, cp_f)
        f2 = jnp.exp(beta * s) * beta12 / gamma * jnp.sinh(gamma * s)
        return f2

    def _f3(self, s, m_flow, cp_f):
        beta = self._beta(m_flow, cp_f)
        gamma = self._gamma(m_flow, cp_f)
        delta = self._delta(m_flow, cp_f)
        f3 = jnp.exp(beta * s) * (jnp.cosh(gamma * s) + delta * jnp.sinh(gamma * s))
        return f3

    def _f4(self, s, m_flow, cp_f):
        beta1 = self._beta1(m_flow, cp_f)
        beta2 = self._beta2(m_flow, cp_f)
        beta12 = self._beta12(m_flow, cp_f)
        beta = self._beta(m_flow, cp_f)
        gamma = self._gamma(m_flow, cp_f)
        delta = self._delta(m_flow, cp_f)
        f4 = jnp.exp(beta * s) * (beta1 * jnp.cosh(gamma * s) - (delta * beta1 + beta2 * beta12 / gamma) * jnp.sinh(gamma * s))
        return f4

    def _f5(self, s, m_flow, cp_f):
        beta1 = self._beta1(m_flow, cp_f)
        beta2 = self._beta2(m_flow, cp_f)
        beta12 = self._beta12(m_flow, cp_f)
        beta = self._beta(m_flow, cp_f)
        gamma = self._gamma(m_flow, cp_f)
        delta = self._delta(m_flow, cp_f)
        f5 = jnp.exp(beta * s) * (beta2 * jnp.cosh(gamma * s) + (delta * beta2 + beta1 * beta12 / gamma) * jnp.sinh(gamma * s))
        return f5
