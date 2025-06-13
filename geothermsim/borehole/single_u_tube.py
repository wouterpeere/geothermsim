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

    def _fluid_temperature_a_in(self, xi: float, beta_ij: Array) -> Array:
        """Inlet coefficient to evaluate the fluid temperatures.

        Parameters
        ----------
        xi : float
            Coordinate along the borehole.
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        array
            (`n_pipes`,) array of coefficients for the inlet fluid
            temperature.

        """
        b_in = self._outlet_fluid_temperature_a_in(beta_ij)
        c_in = self._general_solution_a_in(xi, beta_ij)
        c_out = self._general_solution_a_out(xi, beta_ij)
        a_in = c_in + b_in * c_out
        return a_in

    def _fluid_temperature_a_b(self, xi: float, beta_ij: Array) -> Array:
        """Borehole wall coefficient to evaluate the fluid temperatures.

        Parameters
        ----------
        xi : float
            Coordinate along the borehole.
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        array
            (`n_pipes`, `n_nodes`,) array of coefficients for the
            borehole wall temperature.

        """
        b_b = self._outlet_fluid_temperature_a_b(beta_ij)
        c_out = self._general_solution_a_out(xi, beta_ij)
        c_b = self._general_solution_a_b(xi, beta_ij)
        a_b = c_b + jnp.outer(c_out, b_b)
        return a_b

    def _general_solution_a_in(self, xi: float, beta_ij: Array) -> Array:
        """Inlet coefficient to evaluate the general solution.

        Parameters
        ----------
        xi : float
            Coordinate along the borehole.
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array
            (2,) array of coefficients for the inlet fluid temperature.

        """
        s = self.path.f_s(xi)
        a_in = jnp.array(
            [self._f1(s, beta_ij), -self._f2(s, beta_ij)]
        )
        return a_in

    def _general_solution_a_out(self, xi: float, beta_ij: Array) -> Array:
        """Outlet coefficient to evaluate the general solution.

        Parameters
        ----------
        xi : float
            Coordinate along the borehole.
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array
            (2,) array of coefficients for the outlet fluid
            temperature.

        """
        s = self.path.f_s(xi)
        a_out = jnp.array(
            [self._f2(s, beta_ij), self._f3(s, beta_ij)]
        )
        return a_out

    def _general_solution_a_b(self, xi: float, beta_ij: Array) -> Array:
        """Borehole wall coefficient to evaluate the general solution.

        Parameters
        ----------
        xi : float
            Coordinate along the borehole.
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array
            (2, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        # Longitudinal position to evaluate the solution
        s_xi = self.path.f_s(xi)
        # Upper bound of integration along each segment
        high = jnp.maximum(-1., jnp.minimum(1., self.f_xi_bs(xi)))
        # Coordinates of the top and bottom edges of the segments
        a, b = self.xi_edges[:-1], self.xi_edges[1:]

        def f_xi_sb(_zeta_p: float, _a: float, _b: float) -> float:
            """Borehole coordinate from segment coordinate and edges"""
            return 0.5 * (_b + _a) + 0.5 * _zeta_p * (_b - _a)

        def integral_f4(_a: float, _b: float, _high: float, _ratio: float) -> Array:
            """Integral involving the function f4."""
            def integrand_f4(_zeta_p: float) -> float:
                """Integrand involving the function f4."""
                _eta = f_xi_sb(_zeta_p, _a, _b)
                s_eta = self.path.f_s(_eta)
                return self._f4(s_xi - s_eta, beta_ij) * self.path.f_J(_eta)

            return self.basis.quad_gl(integrand_f4, -1, _high) * _ratio

        def integral_f5(_a: float, _b: float, _high: float, _ratio: float) -> Array:
            """Integral involving the function f5."""
            def integrand_f5(_zeta_p: float) -> float:
                """Integrand involving the function f5."""
                _eta = f_xi_sb(_zeta_p, _a, _b)
                s_eta = self.path.f_s(_eta)
                return -self._f5(s_xi - s_eta, beta_ij) * self.path.f_J(_eta)

            return self.basis.quad_gl(integrand_f5, -1, _high) * _ratio

        a_b = jnp.stack(
            [
                vmap(
                    integral_f4,
                    in_axes=(0, 0, 0, 0),
                    out_axes=0
                )(a, b, high, self.segment_ratios),
                vmap(
                    integral_f5,
                    in_axes=(0, 0, 0, 0),
                    out_axes=0
                )(a, b, high, self.segment_ratios)
            ],
            axis=0
        ).reshape(self.n_pipes, self.n_nodes)
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
        # Borehole length
        L = self.L
        # Coordinates of the top and bottom edges of the segments
        a, b = self.xi_edges[:-1], self.xi_edges[1:]

        def f_xi_sb(_eta_p: float, _a: float, _b: float) -> float:
            """Borehole coordinate from segment coordinate and edges"""
            return 0.5 * (_b + _a) + 0.5 * _eta_p * (_b - _a)

        one_over_f3_minus_f2 = 1 / (self._f3(L, beta_ij) - self._f2(L, beta_ij))
        def integral_outlet_fluid_temperature(_a: float, _b: float, _ratio: float):
            """Integral for the evaluation of the outlet fluid temperature."""
            def integrand_outlet_fluid_temperature(_eta_p: float):
                """Integrand for the evaluation of the outlet fluid temperature."""
                _eta = f_xi_sb(_eta_p, _a, _b)
                s_eta = self.path.f_s(_eta)
                J_eta = self.path.f_J(_eta)
                integrand = one_over_f3_minus_f2 * J_eta * (
                    self._f4(L - s_eta, beta_ij)
                    + self._f5(L - s_eta, beta_ij)
                    )
                return integrand

            integral = self.basis.quad_gl(
                integrand_outlet_fluid_temperature, -1, 1
            ) * _ratio
            return integral

        a_b = vmap(
            integral_outlet_fluid_temperature,
            in_axes=(0, 0, 0),
            out_axes=0
            )(a, b, self.segment_ratios)
        return a_b

    @staticmethod
    def _beta(beta_ij: Array) -> float:
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

    @staticmethod
    def _gamma(beta_ij: Array) -> float:
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

    @staticmethod
    def _delta(beta_ij: Array) -> float:
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
        gamma = SingleUTube._gamma(beta_ij)
        delta = 1. / gamma * (beta12 + 0.5 * (beta1 + beta2))
        return delta

    @staticmethod
    @jit
    def _f1(s: Array | float, beta_ij: Array) -> Array | float:
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
        beta = SingleUTube._beta(beta_ij)
        gamma = SingleUTube._gamma(beta_ij)
        delta = SingleUTube._delta(beta_ij)
        f1 = jnp.exp(beta * s) * (jnp.cosh(gamma * s) - delta * jnp.sinh(gamma * s))
        return f1

    @staticmethod
    @jit
    def _f2(s: Array | float, beta_ij: Array) -> Array | float:
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
        beta = SingleUTube._beta(beta_ij)
        gamma = SingleUTube._gamma(beta_ij)
        beta12 = beta_ij[0, 1]
        f2 = jnp.exp(beta * s) * beta12 / gamma * jnp.sinh(gamma * s)
        return f2

    @staticmethod
    @jit
    def _f3(s: Array | float, beta_ij: Array) -> Array | float:
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
        beta = SingleUTube._beta(beta_ij)
        gamma = SingleUTube._gamma(beta_ij)
        delta = SingleUTube._delta(beta_ij)
        f3 = jnp.exp(beta * s) * (jnp.cosh(gamma * s) + delta * jnp.sinh(gamma * s))
        return f3

    @staticmethod
    @jit
    def _f4(s: Array | float, beta_ij: Array) -> Array | float:
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
        beta = SingleUTube._beta(beta_ij)
        gamma = SingleUTube._gamma(beta_ij)
        delta = SingleUTube._delta(beta_ij)
        f4 = jnp.exp(beta * s) * (beta1 * jnp.cosh(gamma * s) - (delta * beta1 + beta2 * beta12 / gamma) * jnp.sinh(gamma * s))
        return f4

    @staticmethod
    @jit
    def _f5(s: Array | float, beta_ij: Array) -> Array | float:
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
        beta = SingleUTube._beta(beta_ij)
        gamma = SingleUTube._gamma(beta_ij)
        delta = SingleUTube._delta(beta_ij)
        f5 = jnp.exp(beta * s) * (beta2 * jnp.cosh(gamma * s) + (delta * beta2 + beta1 * beta12 / gamma) * jnp.sinh(gamma * s))
        return f5
