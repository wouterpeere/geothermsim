# -*- coding: utf-8 -*-
from collections.abc import Callable
from functools import partial
from typing import Self, Tuple

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from ._tube import _Tube
from ..basis import Basis
from ..path import Path


class SingleUTube(_Tube):
    """Single U-Tube geothermal borehole.

    Parameters
    ----------
    R_d : array_like or callable
        (2, 2,) array of thermal resistances (in m-K/W), or callable that
        takes the mass flow rate as input (in kg/s) and returns a (2, 2,)
        array.
    r_b : float
        Borehole radius (in meters).
    path : path
        Path of the borehole.
    basis : basis
        Basis functions.
    n_segments : int
        Number of segments.
    segment_ratios : array_like or None, default: None
        Normalized size of the segments. Should total ``1``
        (i.e. ``sum(segment_ratios) = 1``). If `segment_ratios` is
        ``None``, segments of equal size are considered (i.e.
        ``segment_ratios[v] = 1 / n_segments``).

    Attributes
    ----------
    n_pipes : int
        Number of pipes in the borehole.
    n_nodes : int
        Total number of nodes along the borehole.
    xi_edges : array
        (`n_segments`+1,) array of the coordinates of the edges of the
        segments.
    L : float
        Borehole length (in meters).
    xi : array
        (`n_nodes`,) array of node coordinates.
    p : array
        (`n_nodes`, 3,) array of node positions.
    dp_dxi : array
        (`n_nodes`, 3,) array of the derivatives of the position at the
        node coordinates.
    J : array
        (`n_nodes`,) array of the norm of the Jacobian at the node
        coordinates.
    s : array
        (`n_nodes`,) array of the longitudinal position at the node
        coordinates.
    w : array
        (`n_nodes`,) array of quadrature weights at the node coordinates.
        These quadrature weights take into account the norm of the
        Jacobian.

    """

    def _fluid_temperature_a_in(self, xi: Array | float, beta_ij: Array) -> Array:
        """Inlet coefficient to evaluate the fluid temperatures.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        array
            (M, `n_pipes`,) array of coefficients for the inlet fluid
            temperature.

        """
        b_in = self._outlet_fluid_temperature_a_in(beta_ij)
        c_in = self._general_solution_a_in(xi, beta_ij)
        c_out = self._general_solution_a_out(xi, beta_ij)
        a_in = c_in + b_in * c_out
        return a_in

    def _fluid_temperature_a_b(self, xi: Array | float, beta_ij: Array) -> Array:
        """Borehole wall coefficient to evaluate the fluid temperatures.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        array
            (M, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        b_b = self._outlet_fluid_temperature_a_b(beta_ij)
        c_out = self._general_solution_a_out(xi, beta_ij)
        c_b = self._general_solution_a_b(xi, beta_ij)
        a_b = c_b + vmap(jnp.outer, in_axes=(0, None))(c_out, b_b)
        return a_b

    def _general_solution(self, xi: Array | float, m_flow: float, cp_f: float) -> Tuple[Array, Array, Array]:
        """Coefficients to evaluate the general solution.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        a_in : array or float
            (M, `n_pipes`,) array of coefficients for the inlet fluid
            temperature.
        a_out : array or float
            (M, `n_pipes`,) array of coefficients for the outlet fluid
            temperature.
        a_b : array
            (M, `n_pipes`, `n_nodes`,) array of coefficients for the
            borehole wall temperature.

        """
        beta_ij = self._beta_ij(m_flow, cp_f)
        a_in = self._general_solution_a_in(xi, beta_ij)
        a_out = self._general_solution_a_out(xi, beta_ij)
        a_b = self._general_solution_a_b(xi, beta_ij)
        return a_in, a_out, a_b

    def _general_solution_a_in(self, xi: Array | float, beta_ij: Array) -> Array:
        """Inlet coefficient to evaluate the general solution.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array or float
            (M, 2,) array of coefficients for the inlet fluid temperature.

        """
        s = self.path.f_s(xi)
        a_in = jnp.stack(
            (
                self._f1(s, beta_ij),
                -self._f2(s, beta_ij)
            ),
            axis=-1
        )
        return a_in

    def _general_solution_a_out(self, xi: Array | float, beta_ij: Array) -> Array:
        """Outlet coefficient to evaluate the general solution.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array or float
            (M, 2,) array of coefficients for the outlet fluid
            temperature.

        """
        s = self.path.f_s(xi)
        a_out = jnp.stack(
            (
                self._f2(s, beta_ij),
                self._f3(s, beta_ij)
            ),
            axis=-1
        )
        return a_out

    def _general_solution_a_b(self, xi: Array | float, beta_ij: Array) -> Array:
        """Borehole wall coefficient to evaluate the general solution.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array
            (M, 2, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        if len(jnp.shape(xi)) == 1:
            a_b = vmap(
                self._general_solution_a_b,
                in_axes=(0, None)
            )(xi, beta_ij)
        else:
            s = self.path.f_s(xi)
            f1 = lambda _eta: self._f4(s - self.path.f_s(_eta), beta_ij) * self.path.f_J(_eta)
            f2 = lambda _eta: -self._f5(s - self.path.f_s(_eta), beta_ij) * self.path.f_J(_eta)
            high = jnp.maximum(-1., jnp.minimum(1., self.f_xi_bs(xi)))
            a, b = self.xi_edges[:-1], self.xi_edges[1:]
            f_xi_bs = lambda _eta, _a, _b: 0.5 * (_b + _a) + 0.5 * _eta * (_b - _a)
            integrand = lambda _eta, _a, _b, _ratio: jnp.stack(
                [
                    f1(f_xi_bs(_eta, _a, _b)) * _ratio,
                    f2(f_xi_bs(_eta, _a, _b)) * _ratio,
                ]
            )
            integral = lambda _a, _b, _ratio, _high: self.basis.quad_gl(
                    vmap(
                        lambda _eta: integrand(_eta, _a, _b, _ratio),
                        in_axes=0,
                        out_axes=-1
                    ),
                -1.,
                _high
                )
            a_b = vmap(
                    integral,
                    in_axes=(0, 0, 0, 0),
                    out_axes=1
                )(a, b, self.segment_ratios, high).reshape(2, self.n_nodes)
        return a_b

    def _outlet_fluid_temperature_a_in(self, beta_ij: Array) -> float:
        """Inlet coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        float
            Coefficient for the inlet fluid temperature.

        """
        L = self.L
        a_in = (self._f1(L, beta_ij) + self._f2(L, beta_ij)) / (self._f3(L, beta_ij) - self._f2(L, beta_ij))
        return a_in

    def _outlet_fluid_temperature_a_b(self, beta_ij: Array) -> Array:
        """Borehole coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array
            (`n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        L = self.L
        f = vmap(
            lambda _eta: (
                self._f4(L - self.path.f_s(self.f_xi_sb(_eta)), beta_ij)
                + self._f5(L - self.path.f_s(self.f_xi_sb(_eta)), beta_ij)
            ) / (
                self._f3(L, beta_ij)
                - self._f2(L, beta_ij)
            ) * self.path.f_J(self.f_xi_sb(_eta)) * self.segment_ratios,
            in_axes=0,
            out_axes=-1)
        a_b = self.basis.quad_gl(f, -1, 1.).flatten()
        return a_b

    def _beta(self, beta_ij: Array) -> float:
        """Coefficient ``beta`` from Hellström (1991).

        Parameters
        ----------
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        float

        """
        beta1 = beta_ij[0, 0]
        beta2 = beta_ij[1, 1]
        beta = 0.5 * (beta2 - beta1)
        return beta

    def _gamma(self, beta_ij: Array) -> float:
        """Coefficient ``gamma`` from Hellström (1991).

        Parameters
        ----------
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        float

        """
        beta1 = beta_ij[0, 0]
        beta2 = beta_ij[1, 1]
        beta12 = beta_ij[0, 1]
        gamma = jnp.sqrt(0.25 * (beta1 + beta2)**2 + beta12 * (beta1 + beta2))
        return gamma

    def _delta(self, beta_ij: Array) -> float:
        """Coefficient ``delta`` from Hellström (1991).

        Parameters
        ----------
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        float

        """
        beta1 = beta_ij[0, 0]
        beta2 = beta_ij[1, 1]
        beta12 = beta_ij[0, 1]
        gamma = self._gamma(beta_ij)
        delta = 1. / gamma * (beta12 + 0.5 * (beta1 + beta2))
        return delta

    def _f1(self, s: Array | float, beta_ij: Array) -> Array | float:
        """Function ``f1`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array or float
            (M,) array.

        """
        beta = self._beta(beta_ij)
        gamma = self._gamma(beta_ij)
        delta = self._delta(beta_ij)
        f1 = jnp.exp(beta * s) * (jnp.cosh(gamma * s) - delta * jnp.sinh(gamma * s))
        return f1

    def _f2(self, s: Array | float, beta_ij: Array) -> Array | float:
        """Function ``f2`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array or float
            (M,) array.

        """
        beta = self._beta(beta_ij)
        gamma = self._gamma(beta_ij)
        beta12 = beta_ij[0, 1]
        f2 = jnp.exp(beta * s) * beta12 / gamma * jnp.sinh(gamma * s)
        return f2

    def _f3(self, s: Array | float, beta_ij: Array) -> Array | float:
        """Function ``f3`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array or float
            (M,) array.

        """
        beta = self._beta(beta_ij)
        gamma = self._gamma(beta_ij)
        delta = self._delta(beta_ij)
        f3 = jnp.exp(beta * s) * (jnp.cosh(gamma * s) + delta * jnp.sinh(gamma * s))
        return f3

    def _f4(self, s: Array | float, beta_ij: Array) -> Array | float:
        """Function ``f4`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array or float
            (M,) array.

        """
        beta1 = beta_ij[0, 0]
        beta2 = beta_ij[1, 1]
        beta12 = beta_ij[0, 1]
        beta = self._beta(beta_ij)
        gamma = self._gamma(beta_ij)
        delta = self._delta(beta_ij)
        f4 = jnp.exp(beta * s) * (beta1 * jnp.cosh(gamma * s) - (delta * beta1 + beta2 * beta12 / gamma) * jnp.sinh(gamma * s))
        return f4

    def _f5(self, s: Array | float, beta_ij: Array) -> Array | float:
        """Function ``f5`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array or float
            (M,) array.

        """
        beta1 = beta_ij[0, 0]
        beta2 = beta_ij[1, 1]
        beta12 = beta_ij[0, 1]
        beta = self._beta(beta_ij)
        gamma = self._gamma(beta_ij)
        delta = self._delta(beta_ij)
        f5 = jnp.exp(beta * s) * (beta2 * jnp.cosh(gamma * s) + (delta * beta2 + beta1 * beta12 / gamma) * jnp.sinh(gamma * s))
        return f5
