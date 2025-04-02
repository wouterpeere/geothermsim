# -*- coding: utf-8 -*-
from functools import partial
from typing import Self, Tuple

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from ..basis import Basis
from .borehole import Borehole
from ..path import Path


class SingleUTube(Borehole):
    """Single U-Tube geothermal borehole.

    Parameters
    ----------
    R_d : array_like
        (2, 2,) array of thermal resistances (in m-K/W).
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

    def __init__(self, R_d: ArrayLike, r_b: float, path: Path, basis: Basis, n_segments: int, segment_ratios: ArrayLike | None = None):
        # Runtime type validation
        if not isinstance(R_d, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {R_d}")
        # Convert input to jax.Array
        R_d = jnp.atleast_2d(R_d)

        super().__init__(r_b, path, basis, n_segments, segment_ratios=segment_ratios)
        self.R_d = R_d

    @partial(jit, static_argnames=['self'])
    def effective_borehole_thermal_resistance(self, m_flow: float, cp_f: float) -> float:
        """Effective borehole thermal resistance.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        float
            Effective borehole thermal resistance (in m-K/W).

        """
        a = self._outlet_fluid_temperature_a_in(m_flow, cp_f)
        b = self._heat_extraction_rate_a_in(self.xi, m_flow, cp_f) @ self.w
        # Effective borehole thermal resistance
        R_b = -0.5 * self.L * (1. + a) / b
        return R_b

    @partial(jit, static_argnames=['self'])
    def g(self, xi: Array | float, m_flow: float, cp_f: float) -> Tuple[Array | float, Array]:
        """Coefficients to evaluate the heat extraction rate.

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
            (M,) array of coefficients for the inlet fluid temperature.
        a_b : array
            (M, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        return self._heat_extraction_rate(xi, m_flow, cp_f)

    @partial(jit, static_argnames=['self'])
    def g_to_self(self, m_flow: float, cp_f: float) -> Tuple[Array, Array]:
        """Coefficients to evaluate the heat extraction rate at nodes.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        a_in : array or float
            (`n_nodes`,) array of coefficients for the inlet fluid
            temperature.
        a_b : array
            (`n_nodes`, `n_nodes`,) array of coefficients for the borehole
            wall temperature.

        """
        return self._heat_extraction_rate(self.xi, m_flow, cp_f)

    @partial(jit, static_argnames=['self'])
    def fluid_temperature(self, xi: Array | float, T_f_in: float, T_b: Array, m_flow: float, cp_f: float) -> Array:
        """Fluid temperatures.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_nodes`,) array of borehole wall temperatures (in degree
            Celsius).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array
            (M, 2,) array of fluid temperatures (in degree Celsius).

        """
        a_in, a_b = self._fluid_temperature(xi, m_flow, cp_f)
        T_f = a_in * T_f_in + a_b @ T_b
        return T_f

    @partial(jit, static_argnames=['self'])
    def heat_extraction_rate(self, xi: Array | float, T_f_in: float, T_b: Array, m_flow: float, cp_f: float) -> Array | float:
        """Heat extraction rate.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_nodes`,) array of borehole wall temperatures (in degree
            Celsius).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array
            (M,) array of heat extraction rate (in W/m). If `xi` is a
            ``float``, then ``M=0``.

        """
        a_in, a_b = self._heat_extraction_rate(xi, m_flow, cp_f)
        q = a_in * T_f_in + a_b @ T_b
        return q

    @partial(jit, static_argnames=['self'])
    def heat_extraction_rate_to_self(self, T_f_in: float, T_b: Array, m_flow: float, cp_f: float) -> Array:
        """Heat extraction rate at nodes.

        Parameters
        ----------
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_nodes`,) array of borehole wall temperatures (in degree
            Celsius).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array
            (`n_nodes`,) array of heat extraction rate (in W/m). If `xi`
            is a ``float``, then ``M=0``.

        """
        return self.heat_extraction_rate(self.xi, T_f_in, T_b, m_flow, cp_f)

    @partial(jit, static_argnames=['self'])
    def outlet_fluid_temperature(self, T_f_in: float, T_b: Array, m_flow: float, cp_f: float) -> float:
        """Outlet fluid temperature.

        Parameters
        ----------
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_nodes`,) array of borehole wall temperatures (in degree
            Celsius).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        float
            Outlet fluid temperature (in degree Celsius).

        """
        a_in, a_b = self._outlet_fluid_temperature(m_flow, cp_f)
        T_f_out = a_in * T_f_in + a_b @ T_b
        return T_f_out

    def _fluid_temperature(self, xi: Array | float, m_flow: float, cp_f: float) -> Tuple[Array, Array]:
        """Coefficients to evaluate the fluid temperatures.

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
        a_in : array
            (M, 2,) array of coefficients for the inlet fluid temperature.
        a_b : array
            (M, 2, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        a_in = self._fluid_temperature_a_in(xi, m_flow, cp_f)
        a_b = self._fluid_temperature_a_b(xi, m_flow, cp_f)
        return a_in, a_b

    def _fluid_temperature_a_in(self, xi: Array | float, m_flow: float, cp_f: float) -> Array:
        """Inlet coefficient to evaluate the fluid temperatures.

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
        array
            (M, 2,) array of coefficients for the inlet fluid temperature.

        """
        b_in = self._outlet_fluid_temperature_a_in(m_flow, cp_f)
        c_in = self._general_solution_a_in(xi, m_flow, cp_f)
        c_out = self._general_solution_a_out(xi, m_flow, cp_f)
        a_in = c_in + b_in * c_out
        return a_in

    def _fluid_temperature_a_b(self, xi: Array | float, m_flow: float, cp_f: float) -> Array:
        """Borehole wall coefficient to evaluate the fluid temperatures.

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
        array
            (M, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        b_b = self._outlet_fluid_temperature_a_b(m_flow, cp_f)
        c_out = self._general_solution_a_out(xi, m_flow, cp_f)
        c_b = self._general_solution_a_b(xi, m_flow, cp_f)
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
            (M, 2,) array of coefficients for the inlet fluid temperature.
        a_out : array or float
            (M, 2,) array of coefficients for the outlet fluid
            temperature.
        a_b : array
            (M, 2, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        a_in = self._general_solution_a_in(xi, m_flow, cp_f)
        a_out = self._general_solution_a_out(xi, m_flow, cp_f)
        a_b = self._general_solution_a_b(xi, m_flow, cp_f)
        return a_in, a_out, a_b

    def _general_solution_a_in(self, xi: Array | float, m_flow: float, cp_f: float) -> Array:
        """Inlet coefficient to evaluate the general solution.

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
        array or float
            (M, 2,) array of coefficients for the inlet fluid temperature.

        """
        s = self.path.f_s(xi)
        a_in = jnp.stack(
            (
                self._f1(s, m_flow, cp_f),
                -self._f2(s, m_flow, cp_f)
            ),
            axis=-1
        )
        return a_in

    def _general_solution_a_out(self, xi: Array | float, m_flow: float, cp_f: float) -> Array:
        """Outlet coefficient to evaluate the general solution.

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
        array or float
            (M, 2,) array of coefficients for the outlet fluid
            temperature.

        """
        s = self.path.f_s(xi)
        a_out = jnp.stack(
            (
                self._f2(s, m_flow, cp_f),
                self._f3(s, m_flow, cp_f)
            ),
            axis=-1
        )
        return a_out

    def _general_solution_a_b(self, xi: Array | float, m_flow: float, cp_f: float) -> Array:
        """Borehole wall coefficient to evaluate the general solution.

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
        array
            (M, 2, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        if len(jnp.shape(xi)) == 1:
            a_b = vmap(
                self._general_solution_a_b,
                in_axes=(0, None, None)
            )(xi, m_flow, cp_f)
        else:
            s = self.path.f_s(xi)
            f1 = lambda _eta: self._f4(s - self.path.f_s(_eta), m_flow, cp_f) * self.path.f_J(_eta)
            f2 = lambda _eta: -self._f5(s - self.path.f_s(_eta), m_flow, cp_f) * self.path.f_J(_eta)
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

    def _heat_extraction_rate(self, xi: Array | float, m_flow: float, cp_f: float) -> Tuple[Array | float, Array]:
        """Coefficients to evaluate the heat extraction rate.

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
            (M,) array of coefficients for the inlet fluid temperature.
        a_b : array
            (M, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        a_in = self._heat_extraction_rate_a_in(xi, m_flow, cp_f)
        a_b = self._heat_extraction_rate_a_b(xi, m_flow, cp_f)
        return a_in, a_b

    def _heat_extraction_rate_a_in(self, xi: Array | float, m_flow: float, cp_f: float) -> Array | float:
        """Inlet coefficient to evaluate the heat extraction rate.

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
        array or float
            (M,) array of coefficients for the inlet fluid temperature.

        """
        b_in = self._fluid_temperature_a_in(xi, m_flow, cp_f)
        a_in = -(b_in[:, 0] / self.R_d[0, 0] + b_in[:, 1] / self.R_d[1, 1])
        return a_in

    def _heat_extraction_rate_a_b(self, xi: Array | float, m_flow: float, cp_f: float) -> Array:
        """Borehole wall coefficient to evaluate the heat extraction rate.

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
        array
            (M, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        b_b = self._fluid_temperature_a_b(xi, m_flow, cp_f)
        R_b = self.R_d[0, 0] * self.R_d[1, 1] / (self.R_d[0, 0] + self.R_d[1, 1])
        a_b = -(b_b[:, 0, :] / self.R_d[0, 0] + b_b[:, 1, :] / self.R_d[1, 1]) + vmap(self.f_psi, in_axes=0)(xi) / R_b
        return a_b

    def _outlet_fluid_temperature(self, m_flow: float, cp_f: float) -> Tuple[float, Array]:
        """Coefficients to evaluate the outlet fluid temperature.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        a_in : float
            Coefficient for the inlet fluid temperature.
        a_b : array
            (`n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        a_in = self._outlet_fluid_temperature_a_in(m_flow, cp_f)
        a_b = self._outlet_fluid_temperature_a_b(m_flow, cp_f)
        return a_in, a_b

    def _outlet_fluid_temperature_a_in(self, m_flow: float, cp_f: float) -> float:
        """Inlet coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        float
            Coefficient for the inlet fluid temperature.

        """
        L = self.L
        a_in = (self._f1(L, m_flow, cp_f) + self._f2(L, m_flow, cp_f)) / (self._f3(L, m_flow, cp_f) - self._f2(L, m_flow, cp_f))
        return a_in

    def _outlet_fluid_temperature_a_b(self, m_flow: float, cp_f: float) -> Array:
        """Borehole coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array
            (`n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        L = self.L
        f = vmap(
            lambda _eta: (
                self._f4(L - self.path.f_s(self.f_xi_sb(_eta)), m_flow, cp_f)
                + self._f5(L - self.path.f_s(self.f_xi_sb(_eta)), m_flow, cp_f)
            ) / (
                self._f3(L, m_flow, cp_f)
                - self._f2(L, m_flow, cp_f)
            ) * self.path.f_J(self.f_xi_sb(_eta)) * self.segment_ratios,
            in_axes=0,
            out_axes=-1)
        a_b = self.basis.quad_gl(f, -1, 1.).flatten()
        return a_b

    def _beta1(self, m_flow: float, cp_f: float) -> float:
        """Pipe 1 to wall thermal conductance coefficient.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        float

        """
        return 1. / (m_flow * cp_f * self.R_d[0, 0])

    def _beta2(self, m_flow: float, cp_f: float) -> float:
        """Pipe 2 to wall thermal conductance coefficient.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        float

        """
        return 1. / (m_flow * cp_f * self.R_d[1, 1])

    def _beta12(self, m_flow: float, cp_f: float) -> float:
        """Pipe to pipe thermal conductance coefficient.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        float

        """
        return 1. / (m_flow * cp_f * self.R_d[0, 1])

    def _beta(self, m_flow: float, cp_f: float) -> float:
        """Coefficient ``beta`` from Hellström (1991).

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        float

        """
        beta1 = self._beta1(m_flow, cp_f)
        beta2 = self._beta2(m_flow, cp_f)
        beta = 0.5 * (beta2 - beta1)
        return beta

    def _gamma(self, m_flow: float, cp_f: float) -> float:
        """Coefficient ``gamma`` from Hellström (1991).

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        float

        """
        beta1 = self._beta1(m_flow, cp_f)
        beta2 = self._beta2(m_flow, cp_f)
        beta12 = self._beta12(m_flow, cp_f)
        gamma = jnp.sqrt(0.25 * (beta1 + beta2)**2 + beta12 * (beta1 + beta2))
        return gamma

    def _delta(self, m_flow: float, cp_f: float) -> float:
        """Coefficient ``delta`` from Hellström (1991).

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        float

        """
        beta1 = self._beta1(m_flow, cp_f)
        beta2 = self._beta2(m_flow, cp_f)
        beta12 = self._beta12(m_flow, cp_f)
        gamma = self._gamma(m_flow, cp_f)
        delta = 1. / gamma * (beta12 + 0.5 * (beta1 + beta2))
        return delta

    def _f1(self, s: Array | float, m_flow: float, cp_f: float) -> Array | float:
        """Function ``f1`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array or float
            (M,) array.

        """
        beta = self._beta(m_flow, cp_f)
        gamma = self._gamma(m_flow, cp_f)
        delta = self._delta(m_flow, cp_f)
        f1 = jnp.exp(beta * s) * (jnp.cosh(gamma * s) - delta * jnp.sinh(gamma * s))
        return f1

    def _f2(self, s: Array | float, m_flow: float, cp_f: float) -> Array | float:
        """Function ``f2`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array or float
            (M,) array.

        """
        beta = self._beta(m_flow, cp_f)
        gamma = self._gamma(m_flow, cp_f)
        beta12 = self._beta12(m_flow, cp_f)
        f2 = jnp.exp(beta * s) * beta12 / gamma * jnp.sinh(gamma * s)
        return f2

    def _f3(self, s: Array | float, m_flow: float, cp_f: float) -> Array | float:
        """Function ``f3`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array or float
            (M,) array.

        """
        beta = self._beta(m_flow, cp_f)
        gamma = self._gamma(m_flow, cp_f)
        delta = self._delta(m_flow, cp_f)
        f3 = jnp.exp(beta * s) * (jnp.cosh(gamma * s) + delta * jnp.sinh(gamma * s))
        return f3

    def _f4(self, s: Array | float, m_flow: float, cp_f: float) -> Array | float:
        """Function ``f4`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array or float
            (M,) array.

        """
        beta1 = self._beta1(m_flow, cp_f)
        beta2 = self._beta2(m_flow, cp_f)
        beta12 = self._beta12(m_flow, cp_f)
        beta = self._beta(m_flow, cp_f)
        gamma = self._gamma(m_flow, cp_f)
        delta = self._delta(m_flow, cp_f)
        f4 = jnp.exp(beta * s) * (beta1 * jnp.cosh(gamma * s) - (delta * beta1 + beta2 * beta12 / gamma) * jnp.sinh(gamma * s))
        return f4

    def _f5(self, s: Array | float, m_flow: float, cp_f: float) -> Array | float:
        """Function ``f5`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array or float
            (M,) array.

        """
        beta1 = self._beta1(m_flow, cp_f)
        beta2 = self._beta2(m_flow, cp_f)
        beta12 = self._beta12(m_flow, cp_f)
        beta = self._beta(m_flow, cp_f)
        gamma = self._gamma(m_flow, cp_f)
        delta = self._delta(m_flow, cp_f)
        f5 = jnp.exp(beta * s) * (beta2 * jnp.cosh(gamma * s) + (delta * beta2 + beta1 * beta12 / gamma) * jnp.sinh(gamma * s))
        return f5

    @classmethod
    def from_dimensions(cls, R_d: ArrayLike, L: float, D: float, r_b: float, x: float, y: float, basis: Basis, n_segments: int, tilt: float = 0., orientation: float = 0., segment_ratios: ArrayLike | None = None, order: int | None = None) -> Self:
        """Straight borehole from its dimensions.

        Parameters
        ----------
        R_d : array_like
            (2, 2,) array of thermal resistances (in m-K/W).
        L : float
            Borehole length (in meters).
        D : float
            Borehole buried depth (in meters).
        r_b : float
            Borehole radius (in meters).
        x, y : float
            Horizontal position (in meters) of the top end of the
            borehole.
        basis : basis
            Basis functions.
        n_segments : int
            Number of segments.
        tilt : float, default: ``0.``
            Tilt angle (in radians) of the borehole with respect to
            vertical.
        orientation : float, default: ``0.``
            Orientation (in radians) of the inclined borehole. An
            inclination toward the x-axis corresponds to an orientation
            of zero.
        segment_ratios : array_like or None, default: None
            Normalized size of the segments. Should total ``1``
            (i.e. ``sum(segment_ratios) = 1``). If `segment_ratios` is
            ``None``, segments of equal size are considered (i.e.
            ``segment_ratios[v] = 1 / n_segments``).

        Returns
        -------
        borehole
            Instance of the `Borehole` class.

        """
        xi = jnp.array([-1., 1.])
        p = jnp.array(
            [
                [x, y, -D],
                [x + L * jnp.sin(tilt) * jnp.cos(orientation), y + L * jnp.sin(tilt) * jnp.sin(orientation), -D - L * jnp.cos(tilt)],
            ]
        )
        path = Path(xi, p, order=order)
        return cls(R_d, r_b, path, basis, n_segments, segment_ratios=segment_ratios)
