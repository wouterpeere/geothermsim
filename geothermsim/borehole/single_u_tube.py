# -*- coding: utf-8 -*-
from functools import partial
from typing import Tuple

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.lax import fori_loop, switch

from ._tube import _Tube
from ..basis import Basis


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
            the borehole as a function of `xi_p`.

        Returns
        -------
        array
            (`n_pipes`,) array of coefficients for the inlet fluid
            temperature.

        """
        b_in = SingleUTube._outlet_fluid_temperature_a_in(
            beta_ij, top_connectivity, None, s_coefs)
        c_in = SingleUTube._general_solution_a_in(
            xi_p, index, beta_ij, s_coefs)
        c_out = SingleUTube._general_solution_a_out(
            xi_p, index, beta_ij, s_coefs)
        a_in = c_in + b_in * c_out
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

        Returns
        -------
        array
            (`n_pipes`, `n_nodes`,) array of coefficients for the
            borehole wall temperature.

        """
        b_b = SingleUTube._outlet_fluid_temperature_a_b(
            beta_ij, top_connectivity, None, s_coefs, J_coefs, psi_coefs, x, w)
        c_out = SingleUTube._general_solution_a_out(
            xi_p, index, beta_ij, s_coefs)
        c_b = SingleUTube._general_solution_a_b(
            xi_p, index, beta_ij, s_coefs, J_coefs, psi_coefs, x, w)
        a_b = c_b + jnp.outer(c_out, b_b)
        return a_b

    @staticmethod
    @jit
    def _general_solution_a_in(xi_p: float, index: int, beta_ij: Array, s_coefs: Array) -> Array:
        """Inlet coefficient to evaluate the general solution.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole segments as a function of `xi_p`.

        Returns
        -------
        array
            (2,) array of coefficients for the inlet fluid temperature.

        """
        s = SingleUTube._longitudinal_position(xi_p, index, s_coefs)
        a_in = jnp.array(
            [SingleUTube._f1(s, beta_ij), -SingleUTube._f2(s, beta_ij)]
        )
        return a_in

    @staticmethod
    @jit
    def _general_solution_a_out(xi_p: float, index: int, beta_ij: Array, s_coefs: Array) -> Array:
        """Outlet coefficient to evaluate the general solution.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole segments as a function of `xi_p`.

        Returns
        -------
        array
            (2,) array of coefficients for the outlet fluid
            temperature.

        """
        s = SingleUTube._longitudinal_position(xi_p, index, s_coefs)
        a_out = jnp.array(
            [SingleUTube._f2(s, beta_ij), SingleUTube._f3(s, beta_ij)]
        )
        return a_out

    @staticmethod
    @jit
    def _general_solution_a_b(xi_p: float, index: int, beta_ij: Array, s_coefs: Array, J_coefs: Array, psi_coefs: Array, x: Array, w: Array) -> Array:
        """Borehole wall coefficient to evaluate the general solution.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.
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
            (2, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        # Initialize coefficients
        n_pipes = jnp.shape(beta_ij)[0]
        n_segments = jnp.shape(s_coefs)[1]
        n_nodes = jnp.shape(psi_coefs)[0]
        a_b = jnp.zeros((n_pipes, n_segments, n_nodes))
        # Longitudinal position corresponding to coordinate `xi_p`
        s = SingleUTube._longitudinal_position(xi_p, index, s_coefs)

        def _integral(_i: int, _factor: float) -> Array:
            """Integrals over a portion of a segment."""
            # Rescale integration points and weights
            _x = _factor * (x + 1) - 1
            _w = _factor * w
            # Longitudinal positions, norms of Jacobian and basis functions
            # at integration points
            _t = SingleUTube._longitudinal_position(_x, _i, s_coefs)
            _J = SingleUTube._norm_of_jacobian(_x, _i, J_coefs)
            _psi = vmap(
                Basis._f_psi,
                in_axes=(0, None),
                out_axes=-1
            )(_x, psi_coefs)

            # Integral of the function f4
            _integrand_f4 = _J * SingleUTube._f4(s - _t, beta_ij)
            # Integral of the function f5
            _integrand_f5 = -_J * SingleUTube._f5(s - _t, beta_ij)
            _a_b = jnp.stack([(_integrand_f4 * _psi) @ _w, (_integrand_f5 * _psi) @ _w])
            return _a_b

        def _zeros(_i: int) -> Array:
            """Array of zeros for segments below `index`."""
            return jnp.zeros((n_pipes, n_nodes))

        # Ratio of the segment from top to the evaluation point
        factor = 0.5 * (xi_p + 1)
        branches = [
            partial(_integral, _factor=1.),
            partial(_integral, _factor=factor),
            _zeros
        ]

        # Evaluation of the integral along all segments
        def _evaluate_integrals(_i, _a_b):
            _a_b = _a_b.at[:, _i, :].set(switch(_i - index + 1, branches, _i))
            return _a_b
        a_b = fori_loop(0, n_segments, _evaluate_integrals, a_b)

        return a_b.reshape(n_pipes, -1)

    @staticmethod
    @jit
    def _outlet_fluid_temperature_a_in(beta_ij: Array, top_connectivity: Tuple[Array, Array], mixing: Array, s_coefs: Array) -> float:
        """Inlet coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.
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
        n_segments = jnp.shape(s_coefs)[1]
        L = SingleUTube._longitudinal_position(1., n_segments, s_coefs)
        a_in = (
            SingleUTube._f1(L, beta_ij) + SingleUTube._f2(L, beta_ij)
        ) / (SingleUTube._f3(L, beta_ij) - SingleUTube._f2(L, beta_ij))
        return a_in

    @staticmethod
    @jit
    def _outlet_fluid_temperature_a_b(beta_ij: Array, top_connectivity: Tuple[Array, Array], mixing: Array, s_coefs: Array, J_coefs: Array, psi_coefs: Array, x: Array, w: Array) -> Array:
        """Borehole coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.
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
        n_segments = jnp.shape(s_coefs)[1]
        L = SingleUTube._longitudinal_position(1., n_segments, s_coefs)
        one_over_f3_minus_f2 = 1 / (SingleUTube._f3(L, beta_ij) - SingleUTube._f2(L, beta_ij))

        # Longitudinal positions, norms of Jacobian and basis functions
        t = vmap(
            SingleUTube._longitudinal_position,
            in_axes=(None, 0, None),
            out_axes=0
        )(x, jnp.arange(n_segments), s_coefs)
        J = vmap(
            SingleUTube._norm_of_jacobian,
            in_axes=(None, 0, None),
            out_axes=0
        )(x, jnp.arange(n_segments), J_coefs)
        psi = vmap(
            Basis._f_psi,
            in_axes=(0, None),
            out_axes=-1
        )(x, psi_coefs)

        # Integral of the function
        integrand = one_over_f3_minus_f2 * J * (
            SingleUTube._f4(L - t, beta_ij)
            + SingleUTube._f5(L - t, beta_ij)
            )
        a_b = vmap(
            jnp.outer,
            in_axes=(-1, -1),
            out_axes=-1
        )(integrand, psi) @ w
        return a_b.flatten()

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
