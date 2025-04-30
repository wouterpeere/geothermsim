# -*- coding: utf-8 -*-
from collections.abc import Callable
from functools import partial
from typing import Self, Tuple

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.scipy.linalg import expm
from jax.typing import ArrayLike

from ._tube import _Tube
from ..basis import Basis
from ..path import Path


class MultipleUTube(_Tube):
    """Multiple U-Tube geothermal borehole.

    Parameters
    ----------
    R_d : array_like or callable
        (`n_pipes`, `n_pipes`,) array of thermal resistances (in m-K/W),
        or callable that takes the mass flow rate as input (in kg/s) and
        returns a (`n_pipes`, `n_pipes`,) array.
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
        # General solution at coordinates `xi`
        E_0_xi = self._general_solution_a_0(xi, beta_ij)
        # General solution at the bottom-end
        E_0 = self._general_solution_a_0(1., beta_ij)
        # Coefficients for connectivity at the top of the borehole
        # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
        c_in = self._top_connectivity_in
        c_u = self._top_connectivity_u

        # Intermediate coefficients
        delta_E_0_xi = (
            E_0_xi[..., :self._n_pipes_over_two] @ c_u
            + E_0_xi[..., self._n_pipes_over_two:]
        )
        delta_E_0_u = (
            E_0[:self._n_pipes_over_two]
            - E_0[self._n_pipes_over_two:]
        )
        delta_E_0_u = (
            delta_E_0_u[:, :self._n_pipes_over_two] @ c_u
            + delta_E_0_u[:, self._n_pipes_over_two:]
        )
        delta_E_0_in = (
            E_0[self._n_pipes_over_two:]
            - E_0[:self._n_pipes_over_two]
        )
        delta_E_0_in = delta_E_0_in[:, :self._n_pipes_over_two] @ c_in

        # Inlet coefficient to evaluate the fluid temperatures
        a_in = (
            delta_E_0_xi @ jnp.linalg.solve(delta_E_0_u, delta_E_0_in)
            + E_0_xi[..., :self._n_pipes_over_two] @ c_in
        )
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
        # General solution at coordinates `xi`
        E_0_xi = self._general_solution_a_0(xi, beta_ij)
        E_b_xi = self._general_solution_a_b(xi, beta_ij)
        # General solution at the bottom-end
        E_0 = self._general_solution_a_0(1., beta_ij)
        E_b = self._general_solution_a_b(1., beta_ij)
        # Coefficients for connectivity at the top of the borehole
        # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
        c_in = self._top_connectivity_in
        c_u = self._top_connectivity_u

        # Intermediate coefficients
        delta_E_0_xi = (
            E_0_xi[..., :self._n_pipes_over_two] @ c_u
            + E_0_xi[..., self._n_pipes_over_two:]
        )
        delta_E_0_u = (
            E_0[:self._n_pipes_over_two]
            - E_0[self._n_pipes_over_two:]
        )
        delta_E_0_u = (
            delta_E_0_u[:, :self._n_pipes_over_two] @ c_u
            + delta_E_0_u[:, self._n_pipes_over_two:]
        )
        delta_E_b = (
            E_b[self._n_pipes_over_two:]
            - E_b[:self._n_pipes_over_two]
        )

        # Borehole wall coefficient to evaluate the fluid temperatures
        a_b = (
            delta_E_0_xi @ jnp.linalg.solve(delta_E_0_u, delta_E_b)
            + E_b_xi
        )
        return a_b

    def _general_solution(self, xi: Array | float, m_flow: float, cp_f: float) -> Tuple[Array, Array]:
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
        a_0 : array or float
            (M, `n_pipes`,) array of coefficients for the top-end fluid
            temperature.
        a_b : array
            (M, `n_pipes`, `n_nodes`,) array of coefficients for the
            borehole wall temperature.

        """
        beta_ij = self._beta_ij(m_flow, cp_f)
        a_0 = self._general_solution_a_0(xi, beta_ij)
        a_b = self._general_solution_a_b(xi, beta_ij)
        return a_0, a_b

    def _general_solution_a_0(self, xi: Array | float, beta_ij: Array) -> Array:
        """Top-end coefficient to evaluate the general solution.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        array or float
            (M, `n_pipes`, `n_pipes`) array of coefficients for the
            top-end fluid temperature.

        """
        if len(jnp.shape(xi)) == 1:
            a_0 = vmap(
                self._general_solution_a_0,
                in_axes=(0, None),
                out_axes=0
            )(xi, beta_ij)
        else:
            s = self.path.f_s(xi)
            a_0 = expm(self._ode_coefficients(beta_ij) * s)
        return a_0

    def _general_solution_a_b(self, xi: Array | float, beta_ij: Array) -> Array:
        """Borehole wall coefficient to evaluate the general solution.

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
            (M, `n_pipes`, `n_nodes`,) array of coefficients for the
            borehole wall temperature.

        """
        if len(jnp.shape(xi)) == 1:
            a_b = vmap(
                self._general_solution_a_b,
                in_axes=(0, None),
                out_axes=0
            )(xi, beta_ij)
        else:
            s = self.path.f_s(xi)
            high = jnp.maximum(-1., jnp.minimum(1., self.f_xi_bs(xi)))
            a, b = self.xi_edges[:-1], self.xi_edges[1:]
            f_xi_bs = lambda _eta, _a, _b: 0.5 * (_b + _a) + 0.5 * _eta * (_b - _a)

            ode_coefficients = self._ode_coefficients(beta_ij)
            ode_coefficients_sum = ode_coefficients.sum(axis=1)
            expm_delta_s = lambda _eta: expm(ode_coefficients * (s - self.path.f_s(_eta)))
            integrand = lambda _eta: -expm_delta_s(_eta) @ ode_coefficients_sum * self.path.f_J(_eta)
            integrand_segment = lambda _eta, _a, _b, _ratio: (
                integrand(f_xi_bs(_eta, _a, _b)) * _ratio
            )
            integral = lambda _a, _b, _ratio, _high: self.basis.quad_gl(
                    vmap(
                        lambda _eta: integrand_segment(_eta, _a, _b, _ratio),
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
                )(a, b, self.segment_ratios, high).reshape(self.n_pipes, self.n_nodes)
        return a_b

    def _ode_coefficients(self, beta_ij: Array) -> Array:
        """Coefficients of the ordinary differential equations.

        Parameters
        ----------
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of coefficients of the ordinary
            differential equations.

        """
        diag_indices = jnp.diag_indices_from(beta_ij)
        ode_coefficients = beta_ij.at[diag_indices].set(
            -beta_ij.sum(axis=1)
        )
        ode_coefficients = ode_coefficients.at[self._n_pipes_over_two:, :].multiply(-1)
        return ode_coefficients

    def _outlet_fluid_temperature_a_in(self, beta_ij: Array) -> float:
        """Inlet coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        float
            Coefficient for the inlet fluid temperature.

        """
        # General solution at the bottom-end
        E_0 = self._general_solution_a_0(1., beta_ij)
        # Coefficients for connectivity at the top of the borehole
        # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
        c_in = self._top_connectivity_in
        c_u = self._top_connectivity_u
        # Coefficient for mixing of the fluid at the outlet
        # T_f_out = m_u @ T_f_u(-1)
        m_u = self._mixing_u

        # Intermediate coefficients
        delta_E_0_u = (
            E_0[:self._n_pipes_over_two]
            - E_0[self._n_pipes_over_two:]
        )
        delta_E_0_u = (
            delta_E_0_u[:, :self._n_pipes_over_two] @ c_u
            + delta_E_0_u[:, self._n_pipes_over_two:]
        )
        delta_E_0_in = (
            E_0[self._n_pipes_over_two:]
            - E_0[:self._n_pipes_over_two]
        )
        delta_E_0_in = delta_E_0_in[:, :self._n_pipes_over_two] @ c_in

        # Inlet coefficient to evaluate the outlet fluid temperature
        a_in = m_u @ jnp.linalg.solve(delta_E_0_u, delta_E_0_in)
        return a_in

    def _outlet_fluid_temperature_a_b(self, beta_ij: Array) -> Array:
        """Borehole coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        array
            (`n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        # General solution at the bottom-end
        E_0 = self._general_solution_a_0(1., beta_ij)
        E_b = self._general_solution_a_b(1., beta_ij)
        # Coefficients for connectivity at the top of the borehole
        # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
        c_in = self._top_connectivity_in
        c_u = self._top_connectivity_u
        # Coefficient for mixing of the fluid at the outlet
        # T_f_out = m_u @ T_f_u(-1)
        m_u = self._mixing_u

        # Intermediate coefficients
        delta_E_0_u = (
            E_0[:self._n_pipes_over_two]
            - E_0[self._n_pipes_over_two:]
        )
        delta_E_0_u = (
            delta_E_0_u[:, :self._n_pipes_over_two] @ c_u
            + delta_E_0_u[:, self._n_pipes_over_two:]
        )
        delta_E_b = (
            E_b[self._n_pipes_over_two:]
            - E_b[:self._n_pipes_over_two]
        )

        # Borehole coefficient to evaluate the outlet fluid temperature
        a_b = m_u @ jnp.linalg.solve(delta_E_0_u, delta_E_b)
        return a_b
