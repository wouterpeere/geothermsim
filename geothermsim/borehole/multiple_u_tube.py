# -*- coding: utf-8 -*-
from typing import Tuple

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.lax import fori_loop
from jax.scipy.linalg import expm

from ._tube import _Tube
from ..basis import Basis


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

    @staticmethod
    @jit
    def _fluid_temperature_a_in(xi_p: float, index: int, beta_ij: Array, top_connectivity: Tuple[Array, Array], s_coefs: Array) -> Array:
        """Inlet coefficient to evaluate the fluid temperatures.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        beta_ij : array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole as a function of `xi_p`.

        Returns
        -------
        array
            (`n_pipes`,) array of coefficients for the inlet fluid
            temperature.

        """
        n_pipes = jnp.shape(beta_ij)[0]
        n_pipes_over_two = n_pipes // 2
        n_segments = jnp.shape(s_coefs)[1]
        # General solution at coordinates `xi`
        E_0_xi = MultipleUTube._general_solution_a_0(
            xi_p, index, beta_ij, s_coefs)
        # General solution at the bottom-end
        E_0 = MultipleUTube._general_solution_a_0(
            1., n_segments, beta_ij, s_coefs)
        # Coefficients for connectivity at the top of the borehole
        # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
        c_in, c_u = top_connectivity

        # Intermediate coefficients
        delta_E_0_xi = (
            E_0_xi[:, :n_pipes_over_two] @ c_u
            + E_0_xi[:, n_pipes_over_two:]
        )
        delta_E_0_u = (
            E_0[:n_pipes_over_two]
            - E_0[n_pipes_over_two:]
        )
        delta_E_0_u = (
            delta_E_0_u[:, :n_pipes_over_two] @ c_u
            + delta_E_0_u[:, n_pipes_over_two:]
        )
        delta_E_0_in = (
            E_0[n_pipes_over_two:]
            - E_0[:n_pipes_over_two]
        )
        delta_E_0_in = delta_E_0_in[:, :n_pipes_over_two] @ c_in

        # Inlet coefficient to evaluate the fluid temperatures
        a_in = (
            delta_E_0_xi @ jnp.linalg.solve(delta_E_0_u, delta_E_0_in)
            + E_0_xi[:, :n_pipes_over_two] @ c_in
        )
        return a_in

    @staticmethod
    @jit
    def _fluid_temperature_a_b(xi_p: float, index: int, beta_ij: Array, top_connectivity: Tuple[Array, Array], s_coefs: Array, J_coefs: Array, psi_coefs: Array, x: Array, w: Array) -> Array:
        """Borehole wall coefficient to evaluate the fluid temperatures.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        beta_ij : array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole segments as a function of `xi_p`.
        J_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the norm of the jacobian (in meters) along
            the borehole segments as a function of `xi_p`.
        psi_coefs : array
            (`n_nodes`, `n_nodes`,) array of polynomial coefficients
            for the evaluation of the polynomial basis functions along
            the borehole segments as a function of `xi_p`.
        x : array
            Array of coordinates along the borehole segment to evaluate the
            integrand function.
        w : array
            Integration weights associated with coordinates `x`.

        """
        n_pipes = jnp.shape(beta_ij)[0]
        n_pipes_over_two = n_pipes // 2
        n_segments = jnp.shape(s_coefs)[1]
        # General solution at coordinates `xi`
        E_0_xi = MultipleUTube._general_solution_a_0(
            xi_p, index, beta_ij, s_coefs)
        E_b_xi = MultipleUTube._general_solution_a_b(
            xi_p, index, beta_ij, s_coefs, J_coefs, psi_coefs, x, w)
        # General solution at the bottom-end
        E_0 = MultipleUTube._general_solution_a_0(
            1., n_segments, beta_ij, s_coefs)
        E_b = MultipleUTube._general_solution_a_b(
            1., n_segments, beta_ij, s_coefs, J_coefs, psi_coefs, x, w)
        # Coefficients for connectivity at the top of the borehole
        # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
        c_in, c_u = top_connectivity

        # Intermediate coefficients
        delta_E_0_xi = (
            E_0_xi[:, :n_pipes_over_two] @ c_u
            + E_0_xi[:, n_pipes_over_two:]
        )
        delta_E_0_u = (
            E_0[:n_pipes_over_two]
            - E_0[n_pipes_over_two:]
        )
        delta_E_0_u = (
            delta_E_0_u[:, :n_pipes_over_two] @ c_u
            + delta_E_0_u[:, n_pipes_over_two:]
        )
        delta_E_b = (
            E_b[n_pipes_over_two:]
            - E_b[:n_pipes_over_two]
        )

        # Borehole wall coefficient to evaluate the fluid temperatures
        a_b = (
            delta_E_0_xi @ jnp.linalg.solve(delta_E_0_u, delta_E_b)
            + E_b_xi
        )
        return a_b

    @staticmethod
    @jit
    def _general_solution_a_0(xi_p: float, index: int, beta_ij: Array, s_coefs: Array) -> Array:
        """Top-end coefficient to evaluate the general solution.

        Parameters
        ----------
        xi : float
            Coordinates along the borehole.
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`) array of coefficients for the
            top-end fluid temperature.

        """
        s = MultipleUTube._longitudinal_position(xi_p, index, s_coefs)
        A = MultipleUTube._ode_coefficients(beta_ij)
        a_0 = MultipleUTube._phi(s, A)
        return a_0

    @staticmethod
    @jit
    def _general_solution_a_b(xi_p: float, index: int, beta_ij: Array, s_coefs: Array, J_coefs: Array, psi_coefs: Array, x: Array, w: Array) -> Array:
        """Borehole wall coefficient to evaluate the general solution.

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
        # Initialize coefficients
        n_pipes = jnp.shape(beta_ij)[0]
        n_segments = jnp.shape(s_coefs)[1]
        n_nodes = jnp.shape(psi_coefs)[0]
        a_b = jnp.zeros((n_pipes, n_segments, n_nodes))
        # Longitudinal position corresponding to coordinate `xi_p`
        s = MultipleUTube._longitudinal_position(xi_p, index, s_coefs)
        # Coefficients of the ODE
        A = MultipleUTube._ode_coefficients(beta_ij)
        b = -A.sum(axis=1)

        # Evaluation of the integral along segments above the segment `index`
        def integral_segment_above_point(i, a_b):
            # Longitudinal positions, norms of Jacobian and basis functions
            # at integration points
            t = MultipleUTube._longitudinal_position(x, i, s_coefs)
            J = MultipleUTube._norm_of_jacobian(x, i, J_coefs)
            psi = vmap(
                Basis._f_psi,
                in_axes=(0, None),
                out_axes=-1
            )(x, psi_coefs)

            # Integral of the function
            phi = vmap(
                MultipleUTube._phi,
                in_axes=(0, None),
                out_axes=-1
            )(s - t, A)
            phi_b = vmap(
                jnp.dot,
                in_axes=(-1, None),
                out_axes=-1
            )(phi, b)
            phi_b_psi = vmap(
                jnp.multiply,
                in_axes=(0, None),
                out_axes=0
            )(phi_b, psi)
            a_b = a_b.at[:, i, :].set(
                (J * phi_b_psi) @ w)
            return a_b
        a_b = fori_loop(0, index, integral_segment_above_point, a_b)

        # Evaluation of the integral along segment `index`
        # Ratio of the segment from top to the evaluation point
        factor = 0.5 * (xi_p + 1)
        # Rescale integration points and weights
        x = factor * (x + 1) - 1
        w = factor * w
        # Longitudinal positions, norms of Jacobian and basis functions
        # at integration points
        t = MultipleUTube._longitudinal_position(x, index, s_coefs)
        J = MultipleUTube._norm_of_jacobian(x, index, J_coefs)
        psi = vmap(
            Basis._f_psi,
            in_axes=(0, None),
            out_axes=-1
        )(x, psi_coefs)

        # Integral of the function
        phi = vmap(
            MultipleUTube._phi,
            in_axes=(0, None),
            out_axes=-1
        )(s - t, A)
        phi_b = vmap(
            jnp.dot,
            in_axes=(-1, None),
            out_axes=-1
        )(phi, b)
        phi_b_psi = vmap(
            jnp.multiply,
            in_axes=(0, None),
            out_axes=0
        )(phi_b, psi)
        a_b = a_b.at[:, index, :].set(
            (J * phi_b_psi) @ w)
        return a_b.reshape(n_pipes, -1)

    @staticmethod
    @jit
    def _ode_coefficients(beta_ij: Array) -> Array:
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
        n_pipes_over_two = jnp.shape(beta_ij)[0] // 2
        diag_indices = jnp.diag_indices_from(beta_ij)
        ode_coefficients = beta_ij.at[diag_indices].set(
            -beta_ij.sum(axis=1)
        )
        ode_coefficients = ode_coefficients.at[n_pipes_over_two:, :].multiply(-1)
        return ode_coefficients

    @staticmethod
    @jit
    def _outlet_fluid_temperature_a_in(beta_ij: Array, top_connectivity: Tuple[Array, Array], mixing: Array, s_coefs: Array) -> float:
        """Inlet coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        mixing : array
            (`n_pipes`/2,) array of coefficients to evaluate the outlet fluid
            temperature from the fluid temperatures at the top-end of the
            borehole in the upward flowing pipes following the relation:
            ``T_f_out = mixing @ T_fu``.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole segments as a function of `xi_p`.

        Returns
        -------
        float
            Coefficient for the inlet fluid temperature.

        """
        n_pipes = jnp.shape(beta_ij)[0]
        n_pipes_over_two = n_pipes // 2
        n_segments = jnp.shape(s_coefs)[1]
        # General solution at the bottom-end
        E_0 = MultipleUTube._general_solution_a_0(
            1., n_segments, beta_ij, s_coefs)
        # Coefficients for connectivity at the top of the borehole
        # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
        c_in, c_u = top_connectivity
        # Coefficient for mixing of the fluid at the outlet
        # T_f_out = m_u @ T_f_u(-1)
        m_u = mixing

        # Intermediate coefficients
        delta_E_0_u = (
            E_0[:n_pipes_over_two]
            - E_0[n_pipes_over_two:]
        )
        delta_E_0_u = (
            delta_E_0_u[:, :n_pipes_over_two] @ c_u
            + delta_E_0_u[:, n_pipes_over_two:]
        )
        delta_E_0_in = (
            E_0[n_pipes_over_two:]
            - E_0[:n_pipes_over_two]
        )
        delta_E_0_in = delta_E_0_in[:, :n_pipes_over_two] @ c_in

        # Inlet coefficient to evaluate the outlet fluid temperature
        a_in = m_u @ jnp.linalg.solve(delta_E_0_u, delta_E_0_in)
        return a_in

    @staticmethod
    @jit
    def _outlet_fluid_temperature_a_b(beta_ij: Array, top_connectivity: Tuple[Array, Array], mixing: Array, s_coefs: Array, J_coefs: Array, psi_coefs: Array, x: Array, w: Array) -> Array:
        """Borehole coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        mixing : array
            (`n_pipes`/2,) array of coefficients to evaluate the outlet fluid
            temperature from the fluid temperatures at the top-end of the
            borehole in the upward flowing pipes following the relation:
            ``T_f_out = mixing @ T_fu``.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole segments as a function of `xi_p`.
        J_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the norm of the jacobian (in meters) along
            the borehole segments as a function of `xi_p`.
        psi_coefs : array
            (`n_nodes`, `n_nodes`,) array of polynomial coefficients
            for the evaluation of the polynomial basis functions along
            the borehole segments as a function of `xi_p`.
        x : array
            Array of coordinates along the borehole segment to evaluate the
            integrand function.
        w : array
            Integration weights associated with coordinates `x`.

        Returns
        -------
        array
            (`n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        n_pipes = jnp.shape(beta_ij)[0]
        n_pipes_over_two = n_pipes // 2
        n_segments = jnp.shape(s_coefs)[1]
        # General solution at the bottom-end
        E_0 = MultipleUTube._general_solution_a_0(
            1., n_segments, beta_ij, s_coefs)
        E_b = MultipleUTube._general_solution_a_b(
            1., n_segments, beta_ij, s_coefs, J_coefs, psi_coefs, x, w)
        # Coefficients for connectivity at the top of the borehole
        # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
        c_in, c_u = top_connectivity
        # Coefficient for mixing of the fluid at the outlet
        # T_f_out = m_u @ T_f_u(-1)
        m_u = mixing

        # Intermediate coefficients
        delta_E_0_u = (
            E_0[:n_pipes_over_two]
            - E_0[n_pipes_over_two:]
        )
        delta_E_0_u = (
            delta_E_0_u[:, :n_pipes_over_two] @ c_u
            + delta_E_0_u[:, n_pipes_over_two:]
        )
        delta_E_b = (
            E_b[n_pipes_over_two:]
            - E_b[:n_pipes_over_two]
        )

        # Borehole coefficient to evaluate the outlet fluid temperature
        a_b = m_u @ jnp.linalg.solve(delta_E_0_u, delta_E_b)
        return a_b

    @staticmethod
    @jit
    def _phi(s: float, ode_coefficients: Array) -> Array:
        """State-transition matrix.

        Parameters
        ----------
        s : float
            Longitudinal position along the borehole (in meters).
        ode_coefficients : array
            (`n_pipes`, `n_pipes`,) array of coefficients of the ordinary
            differential equation.

        Returns
        -------
        array
            State-trasition matrix of shape (`n_pipes`, `n_pipes`,).

        """
        return expm(ode_coefficients * s)
